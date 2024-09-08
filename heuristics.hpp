// ***********************************************************************
// 
// Jet: Multilevel Graph Partitioning
//
// Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). 
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
#pragma once
#include <limits>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosGraph_MIS2.hpp"
#include "ExperimentLoggerUtil.hpp"

namespace jet_partitioner {

template<typename ordinal_t>
KOKKOS_INLINE_FUNCTION ordinal_t xorshiftHash(ordinal_t key) {
  ordinal_t x = key;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

template<class crsMat>
class coarsen_heuristics {
public:
    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_view_t = typename Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = typename Kokkos::View<scalar_t*, Device>;
    using edge_view_t = typename Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = typename Kokkos::View<edge_offset_t, Device>;
    using rand_view_t = typename Kokkos::View<uint64_t*, Device>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = typename Kokkos::RangePolicy<exec_space>;
    using team_policy_t = typename Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    using part_view_t = typename Kokkos::View<int*, Device>;
    using pool_t = Kokkos::Random_XorShift64_Pool<Device>;
    using gen_t = typename pool_t::generator_type;
    using hasher_t = Kokkos::pod_hash<ordinal_t>;
    using argmax_reducer_t = Kokkos::MaxLoc<uint32_t, edge_offset_t, Device>;
    using argmax_t = typename argmax_reducer_t::value_type;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr ordinal_t ORD_MIN = std::numeric_limits<ordinal_t>::min();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;

    struct coarse_map {
        ordinal_t coarse_vtx;
        vtx_view_t map;
    };

    //hn is a list of vertices such that vertex i wants to aggregate with vertex hn(i)
    ordinal_t parallel_map_construct(vtx_view_t vcmap, const ordinal_t n, const vtx_view_t vperm, const vtx_view_t hn) {

        ordinal_t perm_length = n;
        Kokkos::View<ordinal_t, Device> nvertices_coarse("nvertices");

        //construct mapping using heaviest edges
        int swap = 1;
        vtx_view_t curr_perm = vperm;
        while (perm_length > 0) {
            vtx_view_t next_perm("next perm", perm_length);
            Kokkos::View<ordinal_t, Device> next_length("next_length");

            Kokkos::parallel_for(policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i) {
                ordinal_t u = perm_length == n ? i : curr_perm(i);
                ordinal_t v = hn(u);
                int condition = u < v;
                //need to enforce an ordering condition to allow hard-stall conditions to be broken
                if (condition ^ swap) {
                    if (Kokkos::atomic_compare_exchange_strong(&vcmap(u), ORD_MAX, ORD_MAX - 1)) {
                        if (u == v || Kokkos::atomic_compare_exchange_strong(&vcmap(v), ORD_MAX, ORD_MAX - 1)) {
                            ordinal_t cv = u;
                            if(v < u){
                                cv = v;
                            }
                            vcmap(u) = cv;
                            vcmap(v) = cv;
                        }
                        else {
                            if (vcmap(v) < n) {
                                vcmap(u) = vcmap(v);
                            }
                            else {
                                vcmap(u) = ORD_MAX;
                            }
                        }
                    }
                }
            });
            Kokkos::fence();
            //add the ones that failed to be reprocessed next round
            //maybe count these then create next_perm to save memory?
            Kokkos::parallel_scan(policy_t(0, perm_length), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final) {
                ordinal_t u = curr_perm(i);
                if (vcmap(u) >= n) {
                    if(final){
                        next_perm(update) = u;
                    }
                    update++;
                }
                if(final && (i + 1) == perm_length){
                    next_length() = update;
                }
            });
            Kokkos::fence();
            swap = swap ^ 1;
            Kokkos::deep_copy(perm_length, next_length);
            curr_perm = next_perm;
        }
        Kokkos::parallel_scan("assign aggregates", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t u, ordinal_t& update, const bool final){
            if(vcmap(u) == u){
                if(final){
                    vcmap(u) = update;
                }
                update++;
            } else if(final){
                vcmap(u) = vcmap(u) + n;
            }
            if(final && (u + 1) == n){
                nvertices_coarse() = update;
            }
        });
        Kokkos::parallel_for("propagate aggregates", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t u) {
            if(vcmap(u) >= n) {
                ordinal_t c_id = vcmap(u) - n;
                vcmap(u) = vcmap(c_id);
            }
        });
        ordinal_t nc = 0;
        Kokkos::deep_copy(nc, nvertices_coarse);
        return nc;
    }

    coarse_map coarsen_HEC(const matrix_t& g,
        const wgt_view_t& vtx_w,
        bool uniform_weights,
        pool_t& rand_pool,
        ExperimentLoggerUtil<scalar_t>& experiment) {

        ordinal_t n = g.numRows();

        vtx_view_t hn("heavies", n);

        vtx_view_t vcmap("vcmap", n);

        Kokkos::parallel_for("initialize vcmap", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            vcmap(i) = ORD_MAX;
        });

        Kokkos::Timer timer;

        vtx_view_t vperm("vperm", n);
        Kokkos::parallel_for("initialize vperm", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            vperm(i) = i;
        });

        experiment.addMeasurement(Measurement::Permute, timer.seconds());
        timer.reset();

        if (uniform_weights) {
            //all weights equal at this level so choose heaviest edge randomly
            Kokkos::parallel_for("Random HN", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
                gen_t generator = rand_pool.get_state();
                ordinal_t adj_size = g.graph.row_map(i + 1) - g.graph.row_map(i);
                if(adj_size > 0){
                edge_offset_t offset = g.graph.row_map(i) + (generator.urand64() % adj_size);
                hn(i) = g.graph.entries(offset);
                } else {
                    hn(i) = generator.urand64() % n;
                }
                rand_pool.free_state(generator);
            });
        }
        else {
            scalar_t sum_v_w = 0;
            Kokkos::parallel_reduce("calc max", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
                scalar_t wgt = vtx_w(i);
                update += wgt;
            }, sum_v_w);
            scalar_t max_allowed = 6*sum_v_w / n;
            Kokkos::parallel_for("Heaviest HN", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
                ordinal_t i = thread.league_rank();
                ordinal_t adj_size = g.graph.row_map(i + 1) - g.graph.row_map(i);
                if(adj_size > 0 && vtx_w(i) < max_allowed){
                    edge_offset_t end = g.graph.row_map(i + 1);
                    edge_offset_t start = g.graph.row_map(i);
                    typename Kokkos::MaxLoc<scalar_t,edge_offset_t,Device>::value_type argmax{0, end};
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx, Kokkos::ValLocScalar<scalar_t,edge_offset_t>& local) {
                        scalar_t wgt = g.values(idx);
                        ordinal_t v = g.graph.entries(idx);
                        if(wgt >= local.val && vtx_w(v) < max_allowed){
                            local.val = wgt;
                            local.loc = idx;
                        }
                    
                    }, Kokkos::MaxLoc<scalar_t, edge_offset_t,Device>(argmax));
                    Kokkos::single(Kokkos::PerTeam(thread), [=](){
                        if(argmax.loc >= start && argmax.loc < end){
                            ordinal_t h = g.graph.entries(argmax.loc);
                            hn(i) = h;
                        } else {
                            hn(i) = i;
                        }
                    });
                } else {
                    hn(i) = i;
                    //gen_t generator = rand_pool.get_state();
                    //hn(i) = generator.urand64() % n;
                    //rand_pool.free_state(generator);
                }
            });
        }
        experiment.addMeasurement(Measurement::Heavy, timer.seconds());
        timer.reset();
        ordinal_t nc = 0;
        nc = parallel_map_construct(vcmap, n, vperm, hn);
        experiment.addMeasurement(Measurement::MapConstruct, timer.seconds());
        timer.reset();

        coarse_map out;
        out.coarse_vtx = nc;
        out.map = vcmap;

        return out;
    }

    ordinal_t countUnmatched(vtx_view_t target) {
        ordinal_t total = 0;

        Kokkos::parallel_reduce("count unmatched", policy_t(0, target.extent(0)), KOKKOS_LAMBDA(ordinal_t i, ordinal_t& update) {
            if (target(i) == ORD_MAX) {
                update++;
            }
        }, total);

        return total;
    }

    template<typename hash_t>
    void matchHash(const vtx_view_t unmappedVtx, const Kokkos::View<hash_t*, Device> hashes, const hash_t nullkey, vtx_view_t vcmap){
        ordinal_t mappable = unmappedVtx.extent(0);
        Kokkos::View<hash_t*, Device> htable(Kokkos::ViewAllocateWithoutInitializing("hashes hash table"), mappable);
        vtx_view_t twins(Kokkos::ViewAllocateWithoutInitializing("twin table"), mappable);
        Kokkos::deep_copy(htable, nullkey);
        Kokkos::deep_copy(twins, -1);
        Kokkos::parallel_for("match by hash", policy_t(0, mappable), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = unmappedVtx(x);
            hash_t h = hashes(x);
            ordinal_t key = h % mappable;
            bool found = false;
            //find the slot already owned by key
            //or claim ownership of a slot for this key
            while(!found){
                if(htable(key) == nullkey){
                    Kokkos::atomic_compare_exchange(&htable(key), nullkey, h);
                }
                if(htable(key) == h){
                    found = true;
                } else {
                    key++;
                    if(key >= mappable) key -= mappable;
                }
            }
            found = false;
            //check if another vertex with same digest is in slot
            //if so, match with it
            //else, insert into slot
            while(!found){
                ordinal_t twin = twins(key);
                if(twin == -1){
                    if(Kokkos::atomic_compare_exchange_strong(&twins(key), twin, i)) found = true;
                } else {
                    if(Kokkos::atomic_compare_exchange_strong(&twins(key), twin, -1)){
                        ordinal_t cv = twin < i ? twin : i;
                        vcmap(twin) = cv;
                        vcmap(i) = cv;
                        found = true;
                    }
                }
            }
        });
    }

    template<bool is_initial, bool is_uniform>
    struct pickMatch {
        matrix_t g;
        vtx_view_t vcmap;
        vtx_view_t hn;
        pool_t rand_pool;
        vtx_view_t vperm;
        ordinal_t n;
        ordinal_t perm_length;

        pickMatch(matrix_t _g,
            vtx_view_t _vcmap,
            vtx_view_t _hn,
            pool_t _rand_pool,
            vtx_view_t _vperm,
            ordinal_t _n,
            ordinal_t _perm_length) :
                g(_g),
                vcmap(_vcmap),
                hn(_hn),
                rand_pool(_rand_pool),
                vperm(_vperm),
                n(_n),
                perm_length(_perm_length) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const {
            const ordinal_t i = thread.league_rank();
            ordinal_t u = perm_length == n ? i : vperm(i);
            if(!is_initial && (vcmap(u) != ORD_MAX || hn(u) == ORD_MAX || vcmap(hn(u)) == ORD_MAX)) return;
            scalar_t max_ewt = 0;
            edge_offset_t start = g.graph.row_map(u);
            edge_offset_t end = g.graph.row_map(u+1);
            uint32_t r = 0;
            Kokkos::single(Kokkos::PerTeam(thread), [=](uint32_t& update){
                gen_t generator = rand_pool.get_state();
                update = generator.urand();
                rand_pool.free_state(generator);
            }, r);
            if(!is_uniform){
                // find max edge weight
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j, scalar_t& update){
                    if(!is_initial && vcmap(g.graph.entries(j)) != ORD_MAX) return;
                    if(g.values(j) > update){
                        update = g.values(j);
                    }
                }, Kokkos::Max<scalar_t, Device>(max_ewt));
            }
            thread.team_barrier();
            argmax_t argmax{0, end};
            // select a random adjacent vertex having the max edge weight
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j, argmax_t& local) {
                //v must be unmatched to be considered
                if(!is_initial && vcmap(g.graph.entries(j)) != ORD_MAX) return;
                if(is_uniform || g.values(j) == max_ewt){
                    uint32_t v = g.graph.entries(j);
                    uint32_t tiebreaker = xorshiftHash<uint32_t>(v + r);
                    if(tiebreaker >= local.val){
                        local.val = tiebreaker;
                        local.loc = j;
                    }
                }
            }, argmax_reducer_t(argmax));
            thread.team_barrier();
            if(argmax.loc >= start && argmax.loc < end){
                ordinal_t hn_u = g.graph.entries(argmax.loc);
                hn(u) = hn_u;
            } else {
                hn(u) = ORD_MAX;
            }
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& i) const {
            ordinal_t u = perm_length == n ? i : vperm(i);
            if(!is_initial && (vcmap(u) != ORD_MAX || hn(u) == ORD_MAX || vcmap(hn(u)) == ORD_MAX)) return;
            ordinal_t h = ORD_MAX;
            gen_t generator = rand_pool.get_state();
            uint32_t r = generator.urand();
            rand_pool.free_state(generator);
            scalar_t max_ewt = 0;
            uint32_t tiebreaker = 0;
            // select a random adjacent vertex having the max edge weight
            for (edge_offset_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                ordinal_t v = g.graph.entries(j);
                //v must be unmatched to be considered
                if (is_initial || vcmap(v) == ORD_MAX) {
                    if (!is_uniform && max_ewt < g.values(j)) {
                        max_ewt = g.values(j);
                        h = v;
                        tiebreaker = xorshiftHash<uint32_t>(v + r);
                    } else if(is_uniform || max_ewt == g.values(j)){
                        uint32_t sim_wgt = xorshiftHash<uint32_t>(v + r);
                        if(tiebreaker < sim_wgt){
                            h = v;
                            tiebreaker = sim_wgt;
                        }
                    }
                }
            }
            hn(u) = h;
        }
    };

    coarse_map coarsen_match(const matrix_t& g,
        bool uniform_weights, pool_t& rand_pool,
        int match_choice) {

        ordinal_t n = g.numRows();

        vtx_view_t hn(Kokkos::ViewAllocateWithoutInitializing("heavies"), n);
        vtx_view_t vcmap(Kokkos::ViewAllocateWithoutInitializing("vcmap"), n);
        Kokkos::deep_copy(hn, ORD_MAX);
        Kokkos::deep_copy(vcmap, ORD_MAX);
        vtx_view_t vperm_scratch(Kokkos::ViewAllocateWithoutInitializing("vperm"), n);
        vtx_view_t vperm = vperm_scratch;

        if (uniform_weights) {
            //all weights equal at this level so choose heaviest edge randomly
            Kokkos::parallel_for("Potential matches (random)", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
                ordinal_t adj_size = g.graph.row_map(i + 1) - g.graph.row_map(i);
                if(adj_size == 0) return;
                gen_t generator = rand_pool.get_state();
                edge_offset_t offset = generator.urand(g.graph.row_map(i), g.graph.row_map(i+1));
                hn(i) = g.graph.entries(offset);
                rand_pool.free_state(generator);
            });
        }
        else {
            pickMatch<true, false> matcher(g, vcmap, hn, rand_pool, vperm, n, n);
            if(!is_host_space && g.nnz() / g.numRows() > 32){
                Kokkos::parallel_for("Potential matches (heavy)", team_policy_t(n, Kokkos::AUTO), matcher);
            } else {
                Kokkos::parallel_for("Potential matches (heavy)", policy_t(0, n), matcher);
            }
        }
        ordinal_t perm_length = n;
        //construct mapping using heaviest edges
        int swap = 1;
        vtx_view_t perm_scratch(Kokkos::ViewAllocateWithoutInitializing("next perm"), n);
        while (perm_length > 0) {
            //match vertices with heaviest unmatched edge
            Kokkos::parallel_for("commit matches (part 1)", policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i){
                ordinal_t u = perm_length == n ? i : vperm(i);
                ordinal_t v = hn(u);
                if(v == ORD_MAX) return;
                int condition = (u < v) ^ swap;
                //need to enforce an ordering condition to allow hard-stall conditions to be broken
                if (!condition) {
                    vcmap(u) = ORD_MAX - 1;
                }
            });
            Kokkos::parallel_for("commit matches (part 2)", policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i){
                ordinal_t u = perm_length == n ? i : vperm(i);
                ordinal_t v = hn(u);
                if(v == ORD_MAX) return;
                int condition = (u < v) ^ swap;
                //need to enforce an ordering condition to allow hard-stall conditions to be broken
                if (condition) {
                    ordinal_t cv = u < v ? u : v;
                    if (Kokkos::atomic_compare_exchange_strong(&vcmap(v), ORD_MAX - 1, cv)) {
                        vcmap(u) = cv;
                    }
                }
            });
            Kokkos::parallel_for("commit matches (part 3)", policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i){
                ordinal_t u = perm_length == n ? i : vperm(i);
                if(vcmap(u) == ORD_MAX - 1){
                    vcmap(u) = ORD_MAX;
                }
            });

            // find new matches for unmatched vertices
            if(uniform_weights){
                pickMatch<false, true> matcher(g, vcmap, hn, rand_pool, vperm, n, perm_length);
                if(!is_host_space && g.nnz() / g.numRows() > 32){
                    Kokkos::parallel_for("Potential matches (random)", team_policy_t(perm_length, Kokkos::AUTO), matcher);
                } else {
                    Kokkos::parallel_for("Potential matches (random)", policy_t(0, perm_length), matcher);
                }
            } else {
                pickMatch<false, false> matcher(g, vcmap, hn, rand_pool, vperm, n, perm_length);
                if(!is_host_space && g.nnz() / g.numRows() > 32){
                    Kokkos::parallel_for("Potential matches (heavy)", team_policy_t(perm_length, Kokkos::AUTO), matcher);
                } else {
                    Kokkos::parallel_for("Potential matches (heavy)", policy_t(0, perm_length), matcher);
                }
            }
            vtx_view_t perm = perm_scratch;
            if(perm_length != n){
                perm = Kokkos::subview(perm_scratch, std::make_pair((ordinal_t)0, perm_length));
                Kokkos::deep_copy(exec_space(), perm, vperm);
            }
            Kokkos::parallel_scan("scan remaining", policy_t(0, perm_length), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
                ordinal_t u = perm_length == n ? i : perm(i);
                if(vcmap(u) == ORD_MAX && hn(u) != ORD_MAX){
                    if(final){
                        vperm_scratch(update) = u;
                    }
                    update++;
                }
            }, perm_length);
            swap = swap ^ 1;
            vperm = Kokkos::subview(vperm_scratch, std::make_pair((ordinal_t)0, perm_length));
        }

        if (match_choice == 1) {
            ordinal_t unmapped = countUnmatched(vcmap);
            double unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

            //leaf matches
            if (unmappedRatio > 0.25) {
                vtx_view_t unmappedVtx(Kokkos::ViewAllocateWithoutInitializing("unmapped vertices"), unmapped);
                ordinal_t mappable;
                Kokkos::parallel_scan("scan unmapped", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
                    if(vcmap(i) == ORD_MAX && g.graph.row_map(i+1) - g.graph.row_map(i) == 1){
                        if(final){
                            unmappedVtx(update) = i;
                        }
                        update++;
                    }
                }, mappable);
                unmappedVtx = Kokkos::subview(unmappedVtx, std::make_pair((ordinal_t)0, mappable));
                vtx_view_t hashes(Kokkos::ViewAllocateWithoutInitializing("hashes"), mappable);
                Kokkos::parallel_for("create digests", policy_t(0, mappable), KOKKOS_LAMBDA(ordinal_t i) {
                    ordinal_t u = unmappedVtx(i);
                    ordinal_t v = g.graph.entries(g.graph.row_map(u));
                    hashes(i) = v;
                });
                ordinal_t nullkey = ORD_MAX;
                matchHash<ordinal_t>(unmappedVtx, hashes, nullkey, vcmap);
            }

            unmapped = countUnmatched(vcmap);
            unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

            //twin matches
            if (unmappedRatio > 0.25) {
                vtx_view_t unmappedVtx(Kokkos::ViewAllocateWithoutInitializing("unmapped vertices"), unmapped);
                Kokkos::View<uint64_t*, Device> hashes(Kokkos::ViewAllocateWithoutInitializing("hashes"), unmapped);
                Kokkos::parallel_scan("scan unmapped", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
                    if(vcmap(i) == ORD_MAX){
                        if(final){
                            unmappedVtx(update) = i;
                        }
                        update++;
                    }
                });

                //compute (order independent) digests of adjacency lists
                //if two digests are equal, we assume the two adjacency lists are equal (may not always be true)
                Kokkos::parallel_for("create digests", team_policy_t(unmapped, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
                    ordinal_t u = unmappedVtx(thread.league_rank());
                    uint64_t hash = 0;
                    hasher_t hasher;
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, g.graph.row_map(u), g.graph.row_map(u + 1)), [=](const edge_offset_t j, uint64_t& thread_sum) {
                        uint64_t x = g.graph.entries(j);
                        uint64_t y = hasher(x);
                        //I think hasher returns 32 bits so we need to extend it to 64
                        y = y*y + y;
                        thread_sum += y;
                    }, hash);
                    Kokkos::single(Kokkos::PerTeam(thread), [=]() {
                        hashes(thread.league_rank()) = hash;
                    });
                });
                uint64_t nullkey = 0;
                matchHash<uint64_t>(unmappedVtx, hashes, nullkey, vcmap);
            }

            unmapped = countUnmatched(vcmap);
            unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

            //relative matches
            if (unmappedRatio > 0.25) {
                vtx_view_t unmappedVtx(Kokkos::ViewAllocateWithoutInitializing("unmapped vertices"), unmapped);
                ordinal_t mappable;
                Kokkos::parallel_scan("scan unmapped", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
                    if(vcmap(i) == ORD_MAX){
                        if(final){
                            unmappedVtx(update) = i;
                        }
                        update++;
                    }
                }, mappable);
                vtx_view_t hashes(Kokkos::ViewAllocateWithoutInitializing("hashes"), mappable);
                Kokkos::parallel_for("create digests", policy_t(0, mappable), KOKKOS_LAMBDA(ordinal_t i) {
                    ordinal_t u = unmappedVtx(i);
                    ordinal_t h = ORD_MAX;
                    scalar_t max_wgt = 0;
                    ordinal_t min_deg = ORD_MAX;

                    // select the lowest degree adjacent vertex
                    for (edge_offset_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                        ordinal_t v = g.graph.entries(j);
                        ordinal_t vdeg = g.graph.row_map(v+1) - g.graph.row_map(v);
                        if (min_deg > vdeg) {
                            min_deg = vdeg;
                            max_wgt = g.values(j);
                            h = v;
                        } else if(min_deg == vdeg){
                            if(max_wgt < g.values(j)){
                                h = v;
                                max_wgt = g.values(j);
                            }
                        }
                    }
                    hashes(i) = h;
                });
                ordinal_t nullkey = ORD_MAX;
                matchHash<ordinal_t>(unmappedVtx, hashes, nullkey, vcmap);
            }
        }

        //create singleton aggregates of remaining unmatched vertices
        Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i){
            if (vcmap(i) == ORD_MAX) {
                vcmap(i) = i;
            }
        });
        ordinal_t nc = 0;
        //if something breaks here it's probably cuz adding n causes overflow
        Kokkos::parallel_scan("set coarse ids", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
            if(vcmap(i) == i){
                if(final){
                    vcmap(i) = update;
                }
                update++;
            } else if(final) {
                vcmap(i) += n;
            }
        }, nc);
        Kokkos::parallel_for("prop coarse ids", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            if(vcmap(i) >= n){
                vcmap(i) = vcmap(vcmap(i) - n);
            }
        });

        coarse_map out;
        out.coarse_vtx = nc;
        out.map = vcmap;
        return out;
    }
};

}
