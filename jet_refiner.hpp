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
#include <type_traits>
#include <limits>
#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "ExperimentLoggerUtil.hpp"
#include "part_stat.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class jet_refiner {
public:

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    // need some trickery because make_signed is undefined for floating point types
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using vtx_svt = Kokkos::View<ordinal_t, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_svt = Kokkos::View<part_t, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    using stat = part_stat<matrix_t, part_t>;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr gain_t GAIN_MIN = std::numeric_limits<gain_t>::lowest();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    static constexpr part_t NULL_PART = -1;
    static constexpr part_t HASH_RECLAIM = -2;

    static const ordinal_t max_sections = 128;
    static const int max_buckets = 50;
    static const int mid_bucket = 25;

//data that is preserved between levels in the multilevel scheme
struct refine_data {
    gain_vt part_sizes;
    scalar_t total_size = 0;
    gain_t cut = 0;
    gain_t total_imb = 0;
    bool init = false;
};

struct problem {
    matrix_t g;
    wgt_view_t vtx_w;
    part_t k;
    double imb;
    ordinal_t opt;
    ordinal_t size_max;
};

//vertex-part connectivity data
struct conn_data {
    gain_vt conn_vals;
    edge_view_t conn_offsets;
    vtx_view_t lock_bit;
    part_vt dest_cache;
    part_vt conn_entries;
    part_vt conn_table_sizes;
};

//this struct contains all the scratch memory used by the refinement iterations
struct scratch_mem {
    gain_vt gain1, gain2, gain_persistent, evict_start, evict_end;
    vtx_view_t vtx1, vtx2, zeros1;
    part_vt dest_part, undersized;
    vtx_svt counter1;
    gain_svt cut_change1, cut_change2, max_part;
    gain_vt reduce_locs;
    typename gain_vt::HostMirror reduce_copy;
    part_svt total_undersized;

    scratch_mem(const ordinal_t n, const ordinal_t min_size, const part_t k) {
        gain1 = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain scratch 1"), std::max(n, min_size));
        gain2 = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain scratch 2"), n);
        gain_persistent = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
        evict_start = gain_vt("evict start", k);
        evict_end = gain_vt("evict end", k);
        undersized = part_vt("undersized parts", k);
        vtx1 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 1"), n);
        vtx2 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 2"), std::max(n, min_size));
        dest_part = part_vt(Kokkos::ViewAllocateWithoutInitializing("destination scratch"), n);
        zeros1 = vtx_view_t("zeros 1", n);
        counter1 = vtx_svt("counter 1");
        total_undersized = part_svt("total undersized");
        reduce_locs = gain_vt("reduce to here", 3);
        reduce_copy = Kokkos::create_mirror_view(reduce_locs);
        cut_change1 = Kokkos::subview(reduce_locs, 0);
        cut_change2 = Kokkos::subview(reduce_locs, 1);
        max_part = Kokkos::subview(reduce_locs, 2);
    }
};

    scratch_mem perm_scratch;
    conn_data perm_cdata;

    //find maximum size for conn_entries and conn_vals
    edge_offset_t count_gain_size(const matrix_t largest, part_t k){
        edge_offset_t gain_size = 0;
        Kokkos::parallel_reduce("comp offsets", policy_t(0, largest.numRows()), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update){
            ordinal_t degree = largest.graph.row_map(i + 1) - largest.graph.row_map(i);
            if(degree > static_cast<ordinal_t>(k)) degree = k;
            update += degree;
        }, gain_size);
        return gain_size;
    }

    jet_refiner(const matrix_t largest, part_t k) :
        perm_scratch(largest.numRows(), k*max_sections*max_buckets, k) {
        ordinal_t n = largest.numRows();
        edge_view_t conn_offsets("gain offsets", n + 1);
        edge_offset_t gain_size = count_gain_size(largest, k);
        perm_cdata.conn_vals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("conn vals"), gain_size);
        perm_cdata.conn_entries = part_vt(Kokkos::ViewAllocateWithoutInitializing("conn entries"), gain_size);
        perm_cdata.conn_offsets = conn_offsets;
        perm_cdata.dest_cache = part_vt(Kokkos::ViewAllocateWithoutInitializing("best connected part for each vertex"), n);
        perm_cdata.conn_table_sizes = part_vt(Kokkos::ViewAllocateWithoutInitializing("map size"), n);
        perm_cdata.lock_bit = vtx_view_t("lock bit", n);
    }

void copy_refine_data(refine_data& lhs, refine_data& rhs){
    Kokkos::deep_copy(exec_space(), lhs.part_sizes, rhs.part_sizes);
    lhs.total_size = rhs.total_size;
    lhs.cut = rhs.cut;
    lhs.total_imb = rhs.total_imb;
    lhs.init = rhs.init;
}

refine_data clone_refine_data(refine_data& rhs){
    refine_data clone;
    clone.part_sizes = gain_vt(Kokkos::ViewAllocateWithoutInitializing("part sizes"), rhs.part_sizes.extent(0));
    copy_refine_data(clone, rhs);
    return clone;
}

//determines which vertices (if any) should be moved to another part to decrease cutsize
//8 kernels, 2 device-host syncs
vtx_view_t jet_lp(const problem& prob, const part_vt& part, const conn_data& cdata, scratch_mem& scratch, double filter_ratio){
    const matrix_t& g = prob.g;
    ordinal_t n = g.numRows();
    ordinal_t num_pos = 0;
    vtx_view_t swap_scratch = scratch.vtx1;
    part_vt dest_part = scratch.dest_part;
    part_vt conn_entries = cdata.conn_entries;
    edge_view_t conn_offsets = cdata.conn_offsets;
    gain_vt conn_vals = cdata.conn_vals;
    gain_vt save_gains = scratch.gain_persistent;
    vtx_view_t lock_bit = cdata.lock_bit;
    Kokkos::parallel_for("select destination part (lp)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        part_t best = cdata.dest_cache(i);
        if(best != NULL_PART) {
            dest_part(i) = best;
            return;
        }
        part_t p = part(i);
        best = p;
        gain_t b_conn = 0;
        gain_t p_conn = 0;
        edge_offset_t start = conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        edge_offset_t end = start + size;
        //finds potential destination as most connected part excluding p
        for(edge_offset_t j = start; j < end; j++){
            gain_t j_conn = conn_vals(j);
            if(j_conn > b_conn && conn_entries(j) != p){
                best = conn_entries(j);
                b_conn = j_conn;
            } else if(j_conn > 0 && conn_entries(j) == p){
                p_conn = j_conn;
            }
        }
        save_gains(i) = 0;
        if(best != p){
            // vertices must pass this filter in order to be considered further
            // b_conn >= p_conn may seem redundant but it is important
            // to address an edge case where floor(filter_ratio*p_conn) rounds to zero
            if(b_conn >= p_conn || ((p_conn - b_conn) < floor(filter_ratio*p_conn))){
                save_gains(i) = b_conn - p_conn;
            } else {
                best = p;
            }
        }
        cdata.dest_cache(i) = best;
        //a vertex is not considered further if best == p
        dest_part(i) = best;
    });
    //need to store the pre-afterburn gains into a separate view
    //than savegains, because we write new values into it that may not be overwritten
    //if a vertex has its best neighbor cached
    gain_vt pregain = scratch.gain1;
    //write all unlocked vertices that passed the above filter into an unordered list
    //output count of such vertices into num_pos
    Kokkos::parallel_scan("filter potentially viable moves", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        part_t p = part(i);
        part_t best = dest_part(i);
        if(p != best && lock_bit(i) == 0){
            if(final){
                swap_scratch(update) = i;
                pregain(i) = save_gains(i);
            }
            update++;
        } else if(final){
            pregain(i) = GAIN_MIN;
            lock_bit(i) = 0;
        }
    }, num_pos);
    //truncate scratch views by num_pos
    vtx_view_t pos_moves = Kokkos::subview(swap_scratch, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //in this kernel every potential move from the previous filters
    //is reevaluated by considering the effect of the other potential moves
    //a move is considered to occur before another according to their potential gains
    //and the vertex ids
    Kokkos::parallel_for("afterburner heuristic", team_policy_t(num_pos, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        gain_t change = 0;
        ordinal_t i = pos_moves(t.league_rank());
        part_t best = dest_part(i);
        part_t p = part(i);
        gain_t igain = pregain(i);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&](const edge_offset_t j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            gain_t vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if(vgain > igain || (vgain == igain && v < i)){
                part_t vpart = dest_part(v);
                scalar_t wgt = g.values(j);
                if(vpart == p){
                    update -= wgt;
                } else if(vpart == best){
                    update += wgt;
                }
                vpart = part(v);
                if(vpart == p){
                    update += wgt;
                } else if(vpart == best){
                    update -= wgt;
                }
            }
        }, change);
        t.team_barrier();
        Kokkos::single(Kokkos::PerTeam(t), [&](){
            if(igain + change >= 0){
                lock_bit(i) = 1;
            }
        });
    });
    vtx_view_t swaps2 = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //scan all vertices that passed the post filter
    Kokkos::parallel_scan("filter beneficial moves", policy_t(0, num_pos), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t v = pos_moves(i);
        if(lock_bit(v)){
            if(final){
                swaps2(update) = v;
            }
            update++;
        }
    }, num_pos);
    pos_moves = Kokkos::subview(swaps2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    return pos_moves;
}

KOKKOS_INLINE_FUNCTION
static ordinal_t gain_bucket(const gain_t& gx, const scalar_t& vwgt){
    //cast to float so we can approximate log_2
    float gain = static_cast<float>(gx) / static_cast<float>(vwgt);
    ordinal_t gain_type = 0;
    if(gain > 0.0){
        gain_type = 0;
    } else if(gain == 0.0) {
        gain_type = 1;
    } else {
        gain_type = mid_bucket;
        gain = abs(gain);
        if(gain < 1.0){
            while(gain < 1.0){
                gain *= 1.5;
                gain_type--;
            }
            if(gain_type < 2){
                gain_type = 2;
            }
        } else {
            while(gain > 1.0){
                gain /= 1.5;
                gain_type++;
            }
            if(gain_type > max_buckets){
                gain_type = max_buckets - 1;
            }
        }
    }
    return gain_type;
}

//determines vertices to move out of oversized parts to satisfy balance constraint
//performs evictions before assigning destinations
//at most 14 kernels, 2 device-host syncs
vtx_view_t rebalance_strong(const problem& prob, const part_vt& part, const conn_data& cdata, scratch_mem& scratch, gain_vt part_sizes){
    const matrix_t& g = prob.g;
    const part_t k = prob.k;
    const gain_t opt_size = prob.opt;
    const wgt_view_t& vtx_w = prob.vtx_w;
    ordinal_t n = g.numRows();
    ordinal_t sections = max_sections;
    ordinal_t section_size = (n + sections*k) / (sections*k);
    if(section_size < 4096){
        section_size = 4096;
        sections = (n + section_size*k) / (section_size*k);
    }
    //use minibuckets within each gain bucket to reduce atomic contention
    //because the number of gain buckets is small
    ordinal_t t_minibuckets = max_buckets*k*sections;
    vtx_view_t bucket_sizes = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), t_minibuckets + 2));
    Kokkos::deep_copy(exec_space(), bucket_sizes, 0);
    //atomically count vertices in each gain bucket
    gain_t size_max = prob.size_max;
    gain_t max_dest = opt_size + 1;
    Kokkos::parallel_for("assign move scores part1", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = part(i);
        if(part_sizes(p) > size_max){
            uint64_t tk = 0;
            uint64_t tg = 0;
            edge_offset_t start = cdata.conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            gain_t p_gain = 0;
            //calculate average loss from moving to an adjacent part
            for(edge_offset_t j = start; j < end; j++){
                part_t pj = cdata.conn_entries(j);
                if(pj == p){
                    p_gain = cdata.conn_vals(j);
                } else if(pj > NULL_PART) {
                    if(part_sizes(pj) < max_dest){
                        tg += cdata.conn_vals(j);
                        tk += 1;
                    }
                }
            }
            if(tk == 0) tk = 1;
            gain_t gain = (tg / tk) - p_gain;
            ordinal_t gain_type = gain_bucket(gain, vtx_w(i));
            //add to count of appropriate bucket
            if(gain_type < max_buckets && vtx_w(i) < 2*(part_sizes(p) - opt_size)){
                ordinal_t g_id = (max_buckets*p + gain_type) * sections + (i % sections) + 1;
                Kokkos::atomic_increment(&bucket_sizes(g_id));
            }
        }
    });
    vtx_view_t bucket_offsets = bucket_sizes;
    //scan bucket sizes to compute offsets
    if(t_minibuckets < 10000 && !is_host_space){
        Kokkos::parallel_for("scan scores", team_policy_t(1, 1024), KOKKOS_LAMBDA(const member& t){
            //this scan is small so do it within a team instead of an entire grid to save kernel launch time
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets + 2), [&] (const ordinal_t i, ordinal_t& update, const bool final) {
                ordinal_t curr = bucket_sizes(i);
                if(final){
                    bucket_offsets(i) = update;
                }
                update += curr;
            });
        });
    } else {
        Kokkos::parallel_scan("scan scores", policy_t(0, t_minibuckets + 2), KOKKOS_LAMBDA(const ordinal_t& i, ordinal_t& update, const bool final){
            ordinal_t curr = bucket_sizes(i);
            if(final){
                bucket_offsets(i) = update;
            }
            update += curr;
        });
    }
    vtx_view_t least_bad_moves = scratch.vtx1;
    Kokkos::parallel_for("assign move scores part2", Kokkos::Experimental::require(policy_t(0, n), Kokkos::Experimental::WorkItemProperty::HintHeavyWeight), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = part(i);
        if(part_sizes(p) > size_max){
            uint64_t tk = 0;
            uint64_t tg = 0;
            edge_offset_t start = cdata.conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            gain_t p_gain = 0;
            //calculate average loss from moving to an adjacent part
            for(edge_offset_t j = start; j < end; j++){
                part_t pj = cdata.conn_entries(j);
                if(pj == p){
                    p_gain = cdata.conn_vals(j);
                } else if(pj > NULL_PART) {
                    if(part_sizes(pj) < max_dest){
                        tg += cdata.conn_vals(j);
                        tk += 1;
                    }
                }
            }
            if(tk == 0) tk = 1;
            gain_t gain = (tg / tk) - p_gain;
            ordinal_t gain_type = gain_bucket(gain, vtx_w(i));
            //atomically count vertices again, write vertices to appropriate location
            if(gain_type < max_buckets && vtx_w(i) < 2*(part_sizes(p) - opt_size)){
                ordinal_t g_id = (max_buckets*p + gain_type) * sections + (i % sections) + 1;
                ordinal_t insert = Kokkos::atomic_fetch_add(&bucket_offsets(g_id), 1);
                least_bad_moves(insert) = i;
            }
        }
    });
    vtx_svt t_vtx_s = Kokkos::subview(bucket_offsets, t_minibuckets);
    ordinal_t t_vtx = 0;
    Kokkos::deep_copy(exec_space(), t_vtx, t_vtx_s);
    least_bad_moves = Kokkos::subview(least_bad_moves, std::make_pair(static_cast<ordinal_t>(0), t_vtx));
    gain_vt balance_scan = Kokkos::subview(scratch.gain1, std::make_pair(static_cast<ordinal_t>(0), t_vtx + 1));
    //scan vwgts of possible moves after gathering into buckets
    Kokkos::parallel_scan("assign move scores part3", policy_t(0, t_vtx), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update, const bool final){
        ordinal_t x = least_bad_moves(i);
        update += vtx_w(x);
        if(final){
            balance_scan(i + 1) = update;
            if(i == 0){
                balance_scan(i) = 0;
            }
        }
    });
    gain_vt evict_start = scratch.evict_start;
    gain_vt evict_end = scratch.evict_end;
    Kokkos::parallel_for("find score cutoffs", policy_t(0, k), KOKKOS_LAMBDA(const int idx){
        evict_start(idx) = bucket_offsets(idx*max_buckets*sections);
        if(part_sizes(idx) > size_max){
            gain_t evict_total = part_sizes(idx) - size_max;
            ordinal_t start = bucket_offsets(idx*max_buckets*sections);
            ordinal_t end = bucket_offsets((idx + 1)*max_buckets*sections);
            gain_t find = balance_scan(start) + evict_total;
            ordinal_t mid = (start + end) / 2;
            //binary search to find eviction cutoffs for each k
            while(start + 1 < end){
                if(balance_scan(mid) >= find){
                    end = mid;
                } else {
                    start = mid;
                }
                mid = (start + end) / 2;
            }
            //if(abs(balance_scan(end) - find) < abs(balance_scan(start) - find)){
                evict_end(idx) = end;
            //} else {
            //    evict_end(idx) = start;
            //}
        } else {
            evict_end(idx) = bucket_offsets(idx*max_buckets*sections);
        }
    });
    vtx_view_t moves = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), t_vtx));
    ordinal_t num_moves = 0;
    Kokkos::parallel_scan("filter below cutoffs", policy_t(0, t_vtx), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t u = least_bad_moves(i);
        part_t p = part(u);
        //vertices below cutoff are chosen for eviction
        if(i < evict_end(p)){
            if(final){
                moves(update) = u;
            }
            update++;
        }
    }, num_moves);
    t_vtx = num_moves;
    vtx_view_t only_moves = Kokkos::subview(moves, std::make_pair(static_cast<ordinal_t>(0), t_vtx));
    part_vt dest_part = scratch.dest_part;
    vtx_view_t unassigned = only_moves;
    //scan vwgts of evicted vertices
    Kokkos::parallel_scan("balance scan", policy_t(0, t_vtx), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update, const bool final){
        ordinal_t x = unassigned(i);
        update += vtx_w(x);
        if(final){
            balance_scan(i + 1) = update;
            if(i == 0){
                balance_scan(i) = 0;
            }
        }
    });
    //assign consecutive chunks of vertices to undersized parts using scan result
    Kokkos::parallel_for("cookie cutter", policy_t(0, 1), KOKKOS_LAMBDA(const int idx){
        if(idx == 0){
            evict_start(0) = 0;
            for(int p = 0; p < k; p++){
                gain_t select = 0;
                if(max_dest > part_sizes(p)){
                    select = max_dest - part_sizes(p);
                }
                ordinal_t start = evict_start(p);
                ordinal_t end = t_vtx;
                gain_t find = balance_scan(start) + select;
                ordinal_t mid = (start + end) / 2;
                //binary search to find eviction cutoffs for each k
                while(start + 1 < end){
                    if(balance_scan(mid) >= find){
                        end = mid;
                    } else {
                        start = mid;
                    }
                    mid = (start + end) / 2;
                }
                if(abs(balance_scan(end) - find) < abs(balance_scan(start) - find)){
                    evict_end(p) = end;
                } else {
                    evict_end(p) = start;
                }
                if(p + 1 < k){
                    evict_start(p+1) = evict_end(p);
                }
            }
        }
    });
    Kokkos::parallel_for("select destination parts (rs)", policy_t(0, t_vtx), KOKKOS_LAMBDA(const ordinal_t i){
        int p = 0;
        //find chunk that contains i
        while(p < k && evict_start(p) <= i){
            p++;
        }
        p--;
        if(i < evict_end(p)){
            dest_part(unassigned(i)) = p;
        } else {
            dest_part(unassigned(i)) = part(unassigned(i));
        }
    });
    return only_moves;
}

//an unused version of rebalance_weak that fills all undersized parts instead of draining all oversized parts
vtx_view_t rebalance_pull(const matrix_t& g, const part_t k, const wgt_view_t& vtx_w, const part_vt& part, const conn_data& cdata, const gain_t opt_size, const double imb_ratio, scratch_mem& scratch, gain_vt part_sizes){
    ordinal_t n = g.numRows();
    //GPU version
    ordinal_t sections = max_sections;
    ordinal_t section_size = (n + sections*k) / (sections*k);
    if(section_size < 4096){
        section_size = 4096;
        sections = (n + section_size*k) / (section_size*k);
    }
    //use minibuckets within each gain bucket to reduce atomic contention
    //because the number of gain buckets is small
    ordinal_t t_minibuckets = max_buckets*k*sections;
    gain_vt bucket_offsets = Kokkos::subview(scratch.gain1, std::make_pair(static_cast<ordinal_t>(0), t_minibuckets));
    gain_vt bucket_sizes = bucket_offsets;
    Kokkos::deep_copy(exec_space(), bucket_sizes, 0);
    part_vt dest_part = scratch.dest_part;
    gain_t size_max = (2.0 - imb_ratio)*opt_size;
    gain_vt save_gains = scratch.gain2;
    part_vt undersized = scratch.undersized;
    part_svt total_undersized = scratch.total_undersized;
    Kokkos::parallel_for("init undersized parts list", team_policy_t(1, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        //this scan is small so do it within a team instead of an entire grid to save kernel launch time
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, k), [&] (const part_t i, part_t& update, const bool final) {
            if(part_sizes(i) < size_max){
                if(final){
                    undersized(update) = i;
                }
                update++;
            }
            if(final && i + 1 == k){
                total_undersized() = update;
            }
        });
    });
    Kokkos::parallel_for("select destination parts (rw)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i) {
        part_t p = part(i);
        gain_t p_gain = 0;
        part_t best = p;
        gain_t gain = 0;
        if(part_sizes(p) > opt_size){
            edge_offset_t start = cdata.conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            for(edge_offset_t j = start; j < end; j++){
                part_t pj = cdata.conn_entries(j);
                if(pj > NULL_PART && part_sizes(pj) < size_max){
                    gain_t jgain = cdata.conn_vals(j);
                    if(jgain > gain){
                            best = pj;
                            gain = jgain;
                    }
                }
                if(pj == p){
                    p_gain = cdata.conn_vals(j);
                }
            }
            if(gain > 0){
                dest_part(i) = best;
                save_gains(i) = gain - p_gain;
            } else {
                best = undersized(i % total_undersized());
                dest_part(i) = best;
                save_gains(i) = -p_gain;
            }
        } else {
            dest_part(i) = p;
        }
    });
    gain_vt vscore = save_gains;
    vtx_view_t bid = scratch.vtx2;
    //atomically count vertices in each gain bucket
    Kokkos::parallel_for("assign move scores", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = part(i);
        part_t best = dest_part(i);
        bid(i) = -1;
        if(p != best){
            //cast to int so we can approximate log_2
            int gain = save_gains(i);
            ordinal_t gain_type = gain_bucket(gain, vtx_w(i));
            if(gain_type < max_buckets){
                ordinal_t g_id = (max_buckets*best + gain_type) * sections + (i % sections);
                bid(i) = g_id;
                vscore(i) = Kokkos::atomic_fetch_add(&bucket_sizes(g_id), vtx_w(i));
            }
        }
    });
    //exclusive prefix sum to compute offsets
    //bucket_sizes is an alias of bucket_offsets
    if(t_minibuckets < 10000 && !is_host_space){
        Kokkos::parallel_for("scan score buckets", team_policy_t(1, 1024), KOKKOS_LAMBDA(const member& t){
            //this scan is small so do it within a team instead of an entire grid to save kernel launch time
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets), [&] (const ordinal_t i, gain_t& update, const bool final) {
                gain_t x = bucket_sizes(i);
                if(final){
                    bucket_offsets(i) = update;
                }
                update += x;
            });
        });
    } else {
        Kokkos::parallel_scan("scan score buckets", policy_t(0, t_minibuckets), KOKKOS_LAMBDA(const ordinal_t& i, gain_t& update, const bool final){
            gain_t x = bucket_sizes(i);
            if(final){
                bucket_offsets(i) = update;
            }
            update += x;
        });
    }
    vtx_view_t moves = scratch.vtx1;
    ordinal_t num_moves = 0;
    Kokkos::parallel_scan("filter scores below cutoff", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t b = bid(i);
        if(b != -1){
            part_t p = dest_part(i);
            ordinal_t begin_bucket = max_buckets*p*sections;
            gain_t score = vscore(i) + bucket_offsets(b) - bucket_offsets(begin_bucket);
            gain_t limit = size_max - part_sizes(p);
            if(score < limit){
                if(final){
                    moves(update) = i;
                }
                update++;
            }
        }
    }, num_moves);
    vtx_view_t only_moves = Kokkos::subview(moves, std::make_pair(static_cast<ordinal_t>(0), num_moves));
    return only_moves;
}

//determines vertices to move out of oversized parts to satisfy balance constraint
//performs evictions after assigning destinations
//at most 8 kernels, 1 device-host sync
vtx_view_t rebalance_weak(const problem& prob, const part_vt& part, const conn_data& cdata, scratch_mem& scratch, gain_vt part_sizes){
    const matrix_t& g = prob.g;
    const part_t k = prob.k;
    const gain_t opt_size = prob.opt;
    const wgt_view_t& vtx_w = prob.vtx_w;
    ordinal_t n = g.numRows();
    ordinal_t sections = max_sections;
    ordinal_t section_size = (n + sections*k) / (sections*k);
    if(section_size < 4096){
        section_size = 4096;
        sections = (n + section_size*k) / (section_size*k);
    }
    //use minibuckets within each gain bucket to reduce atomic contention
    //because the number of gain buckets is small
    ordinal_t t_minibuckets = max_buckets*k*sections;
    gain_vt bucket_offsets = Kokkos::subview(scratch.gain1, std::make_pair(static_cast<ordinal_t>(0), t_minibuckets + 1));
    gain_vt bucket_sizes = bucket_offsets;
    Kokkos::deep_copy(exec_space(), bucket_sizes, 0);
    part_vt dest_part = scratch.dest_part;
    gain_t size_max = prob.size_max;
    gain_vt save_gains = scratch.gain2;
    part_vt undersized = scratch.undersized;
    gain_t max_dest = size_max*0.99;
    if(max_dest < size_max - 100){
        max_dest = size_max - 100;
    }
    part_svt total_undersized = scratch.total_undersized;
    Kokkos::parallel_for("init undersized parts list", team_policy_t(1, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        //this scan is small so do it within a team instead of an entire grid to save kernel launch time
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, k), [&] (const part_t i, part_t& update, const bool final) {
            if(part_sizes(i) < max_dest){
                if(final){
                    undersized(update) = i;
                }
                update++;
            }
            if(final && i + 1 == k){
                total_undersized() = update;
            }
        });
    });
    Kokkos::parallel_for("select destination parts (rw)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i) {
        part_t p = part(i);
        gain_t p_gain = 0;
        part_t best = p;
        gain_t gain = 0;
        if(part_sizes(p) > size_max && vtx_w(i) < 1.5*(part_sizes(p) - opt_size)){
            edge_offset_t start = cdata.conn_offsets(i);
            part_t size = cdata.conn_table_sizes(i);
            edge_offset_t end = start + size;
            //find most connected undersized part
            for(edge_offset_t j = start; j < end; j++){
                part_t pj = cdata.conn_entries(j);
                if(pj > NULL_PART && part_sizes(pj) < max_dest){
                    gain_t jgain = cdata.conn_vals(j);
                    if(jgain > gain){
                            best = pj;
                            gain = jgain;
                    }
                }
                if(pj == p){
                    p_gain = cdata.conn_vals(j);
                }
            }
            if(gain > 0){
                dest_part(i) = best;
                save_gains(i) = gain - p_gain;
            } else {
                //choose arbitrary undersized part
                best = undersized(i % total_undersized());
                dest_part(i) = best;
                save_gains(i) = -p_gain;
            }
        } else {
            dest_part(i) = p;
        }
    });
    gain_vt vscore = save_gains;
    vtx_view_t bid = scratch.vtx2;
    //atomically add vwgts in each gain bucket
    //use atomic_fetch_add to get score
    Kokkos::parallel_for("assign move scores", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = part(i);
        part_t best = dest_part(i);
        bid(i) = -1;
        if(p != best){
            gain_t gain = save_gains(i);
            ordinal_t gain_type = gain_bucket(gain, vtx_w(i));
            ordinal_t g_id = (max_buckets*p + gain_type) * sections + (i % sections);
            bid(i) = g_id;
            vscore(i) = Kokkos::atomic_fetch_add(&bucket_sizes(g_id), vtx_w(i));
        }
    });
    //exclusive prefix sum to compute offsets
    //bucket_sizes is an alias of bucket_offsets
    if(t_minibuckets < 10000 && !is_host_space){
        Kokkos::parallel_for("scan score buckets", team_policy_t(1, 1024), KOKKOS_LAMBDA(const member& t){
            //this scan is small so do it within a team instead of an entire grid to save kernel launch time
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets), [&] (const ordinal_t i, gain_t& update, const bool final) {
                gain_t x = bucket_sizes(i);
                if(final){
                    bucket_offsets(i) = update;
                }
                update += x;
            });
        });
    } else {
        Kokkos::parallel_scan("scan score buckets", policy_t(0, t_minibuckets), KOKKOS_LAMBDA(const ordinal_t& i, gain_t& update, const bool final){
            gain_t x = bucket_sizes(i);
            if(final){
                bucket_offsets(i) = update;
            }
            update += x;
        });
    }
    vtx_view_t moves = scratch.vtx1;
    ordinal_t num_moves = 0;
    Kokkos::parallel_scan("filter scores below cutoff", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t b = bid(i);
        if(b != -1){
            part_t p = part(i);
            ordinal_t begin_bucket = max_buckets*p*sections;
            gain_t score = vscore(i) + bucket_offsets(b) - bucket_offsets(begin_bucket);
            gain_t limit = part_sizes(p) - size_max;
            if(score < limit){
                if(final){
                    moves(update) = i;
                }
                update++;
            }
        }
    }, num_moves);
    vtx_view_t only_moves = Kokkos::subview(moves, std::make_pair(static_cast<ordinal_t>(0), num_moves));
    return only_moves;
}

KOKKOS_INLINE_FUNCTION
void build_row_cdata_large(const conn_data& cdata, const matrix_t& g, const part_vt part, const part_t k, const member& t){
    ordinal_t i = t.league_rank();
    edge_offset_t g_start = cdata.conn_offsets(i);
    edge_offset_t g_end = cdata.conn_offsets(i + 1);
    part_t size = g_end - g_start;
    gain_t* s_conn_vals = (gain_t*) t.team_shmem().get_shmem(sizeof(gain_t) * size);
    part_t* s_conn_entries = (part_t*) t.team_shmem().get_shmem(sizeof(part_t) * size);
    part_t* used_cap = (part_t*) t.team_shmem().get_shmem(sizeof(part_t));
    *used_cap = 0;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
        s_conn_vals[j] = 0;
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
        s_conn_entries[j] = NULL_PART;
    });
    t.team_barrier();
    //construct conn table in shared memory
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j){
        ordinal_t v = g.graph.entries(j);
        gain_t wgt = g.values(j);
        part_t p = part(v);
        part_t p_o = p % size;
        if(size == k){
            if(s_conn_entries[p_o] == NULL_PART && Kokkos::atomic_compare_exchange_strong(s_conn_entries + p_o, NULL_PART, p)) Kokkos::atomic_add(used_cap, 1);
        } else {
            bool success = false;
            while(!success){
                part_t px = s_conn_entries[p_o];
                // the comparisons to p and NULL_PART need to be atomic
                while(px != p && px != NULL_PART){
                    p_o = (p_o + 1) % size;
                    px = s_conn_entries[p_o];
                }
                if(s_conn_entries[p_o] == p){
                    success = true;
                } else {
                    if(Kokkos::atomic_compare_exchange_strong(s_conn_entries + p_o, NULL_PART, p)) Kokkos::atomic_add(used_cap, 1);
                    if(s_conn_entries[p_o] == p){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % size;
                    }
                }
            }
        }
        Kokkos::atomic_add(s_conn_vals + p_o, wgt);
    });
    t.team_barrier();
    part_t old_size = size;
    size = *used_cap;
    part_t quarter_size = size / 4;
    part_t min_inc = 3;
    if(quarter_size < min_inc) quarter_size = min_inc;
    size += quarter_size;
    if(size < old_size){
        cdata.conn_table_sizes(i) = size;
        //copy conn table into smaller conn table in global memory
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, old_size), [&] (const edge_offset_t& j){
            part_t p = s_conn_entries[j];
            if(p > NULL_PART){
                part_t p_o = p % size;
                bool success = false;
                while(!success){
                    while(cdata.conn_entries(g_start + p_o) != NULL_PART){
                        p_o = (p_o + 1) % size;
                    }
                    Kokkos::atomic_compare_exchange(&cdata.conn_entries(g_start + p_o), NULL_PART, p);
                    if(cdata.conn_entries(g_start + p_o) == p){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % size;
                    }
                }
                cdata.conn_vals(g_start + p_o) = s_conn_vals[j];
            }
        });
    } else {
        size = old_size;
        cdata.conn_table_sizes(i) = size;
        //copy conn table into global memory
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
            cdata.conn_vals(g_start + j) = s_conn_vals[j];
        });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
            cdata.conn_entries(g_start + j) = s_conn_entries[j];
        });
    }
}

//updates datastructures assuming a "large" number of vertices are moved
//2 kernels, 0 device-host syncs
void update_large(const problem& prob, part_vt part, const vtx_view_t swaps, scratch_mem& scratch, conn_data& cdata){
    const matrix_t& g = prob.g;
    const part_t k = prob.k;
    ordinal_t total_moves = swaps.extent(0);
    vtx_view_t swap_bit = scratch.zeros1;
    Kokkos::parallel_for("mark adjacent", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        //mark adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            if(swap_bit(v) == 0) swap_bit(v) = 1;
        });
    });
    //recompute conn tables for each vertex adjacent to a moved vertex
    Kokkos::parallel_for("reset conn DS", team_policy_t(g.numRows(), Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(k*sizeof(gain_t) + k*sizeof(part_t) + 4*sizeof(part_t))), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = t.league_rank();
        if(swap_bit(i) == 1){
            edge_offset_t g_start = cdata.conn_offsets(i);
            edge_offset_t g_end = cdata.conn_offsets(i + 1);
            part_t size = g_end - g_start;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                cdata.conn_vals(g_start + j) = 0;
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t j) {
                cdata.conn_entries(g_start + j) = NULL_PART;
            });
            build_row_cdata_large(cdata, g, part, k, t);
            Kokkos::single(Kokkos::PerTeam(t), [=](){
                //reset swap bit to 0 so memory can be reused
                swap_bit(i) = 0;
                cdata.dest_cache(i) = NULL_PART;
            });
        }
    });
}

//update datastructures assuming a "small" number of vertices are moved
//2 kernels, 0 device-host syncs
void update_small(const problem& prob, const part_vt part, const vtx_view_t swaps, const part_vt dest_part, conn_data& cdata){
    const matrix_t& g = prob.g;
    const part_t k = prob.k;
    ordinal_t total_moves = swaps.extent(0);
    Kokkos::parallel_for("update conns (subtract) (high degree)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        //dest_part stores old part at this point
        part_t p = dest_part(i);
        //subtract i's contribution to p connectivity for adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            edge_offset_t v_start = cdata.conn_offsets(v);
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = p % v_size;
            //v is always adjacent to p because it is adjacent to i which was in p
            while(cdata.conn_entries(v_start + p_o) != p){
                p_o = (p_o + 1) % v_size;
            }
            //DO NOT USE ATOMIC_ADD_FETCH HERE IT IS WAY SLOWER
            gain_t x = Kokkos::atomic_fetch_add(&cdata.conn_vals(v_start + p_o), -wgt);
            //parts have locked locations if v_size == k (even when not originally allocated to size k)
            if(v_size < k && x == wgt){
                //free this gain slot
                cdata.conn_entries(v_start + p_o) = HASH_RECLAIM;
            }
        });
    });
    Kokkos::parallel_for("update conns (add) (high degree)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        //part contains new part at this point
        part_t best = part(i);
        //add i's contribution to best connectivity for adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            cdata.dest_cache(v) = NULL_PART;
            edge_offset_t v_start = cdata.conn_offsets(v);
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = best % v_size;
            bool success = false;
            //check if best in conn table
            //can only determine best is absent if NULL_PART is found or v_size reached
            for(part_t q = 0; q < v_size; q++){
                part_t p_i = (best + q) % v_size;
                if(cdata.conn_entries(v_start + p_i) == best){
                    success = true;
                    p_o = p_i;
                    break;
                } else if(cdata.conn_entries(v_start + p_i) == NULL_PART){
                    break;
                }
            }
            part_t count = 0;
            //insert best into conn table
            //needs to find either HASH_RECLAIM or NULL_PART to make insertion
            while(!success && count < v_size){
                part_t px = cdata.conn_entries(v_start + p_o);
                // comparisons to best and NULL_PART need to be atomic
                while(px != best && px > NULL_PART && count++ < v_size){
                    p_o = (p_o + 1) % v_size;
                    px = cdata.conn_entries(v_start + p_o);
                }
                if(px == best){
                    success = true;
                } else if(px <= NULL_PART){
                    //don't care if this thread succeeds if another thread succeeds with the same value
                    Kokkos::atomic_compare_exchange(&cdata.conn_entries(v_start + p_o), px, best);
                    if(cdata.conn_entries(v_start + p_o) == best){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % v_size;
                        count++;
                    }
                }
            }
            //if we run out of space, start densely allocating after end of current hash table
            //this has the side-effect of making future lookups in this row O(v_size)
            if(!success){
                p_o = v_size;
                while(!success){
                    part_t px = cdata.conn_entries(v_start + p_o);
                    // comparisons to best and NULL_PART need to be atomic
                    while(px != best && px > NULL_PART){
                        p_o++;
                        px = cdata.conn_entries(v_start + p_o);
                    }
                    if(px == best){
                        success = true;
                    } else {
                        if(Kokkos::atomic_compare_exchange_strong(&cdata.conn_entries(v_start + p_o), px, best)){
                            Kokkos::atomic_add(&cdata.conn_table_sizes(v), 1);
                        }
                        //don't care if this thread succeeded if another thread succeeded with the same value
                        if(cdata.conn_entries(v_start + p_o) == best){
                            success = true;
                        } else {
                            p_o++;
                        }
                    }
                }
            }
            Kokkos::atomic_add(&cdata.conn_vals(v_start + p_o), wgt);
        });
    });
}

KOKKOS_INLINE_FUNCTION
static gain_t lookup(const part_t* keys, const gain_t* vals, const part_t& target, const part_t& size){
    for(part_t q = 0; q < size; q++){
        part_t p_i = (target + q) % size;
        if(keys[p_i] == target){
            return vals[p_i];
        } else if(keys[p_i] == NULL_PART){
            return 0;
        }
    }
    return 0;
}

//perform swaps, update gains, and compute change to cut and imbalance
//4 kernels, 1 device-host syncs
void perform_moves(const problem& prob, part_vt part, const vtx_view_t swaps, const part_vt dest_part, scratch_mem& scratch, conn_data cdata, refine_data& curr_state){
    const matrix_t& g = prob.g;
    const wgt_view_t& vtx_w = prob.vtx_w;
    ordinal_t total_moves = swaps.extent(0);
    //total change in cutsize = (sum over all moves) -((new_b_con - new_p_con) + (old_b_con - old_p_con))
    Kokkos::parallel_reduce("count cutsize change part1", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x, gain_t& gain_update){
        ordinal_t i = swaps(x);
        part_t best = dest_part(i);
        part_t p = part(i);
        edge_offset_t start = cdata.conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        gain_t p_con = lookup(cdata.conn_entries.data() + start, cdata.conn_vals.data() + start, p, size);
        gain_t b_con = lookup(cdata.conn_entries.data() + start, cdata.conn_vals.data() + start, best, size);
        gain_update += b_con - p_con;
    }, scratch.cut_change1);
    //change part assignments and update part sizes
    Kokkos::parallel_for("perform moves", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = swaps(x);
        part_t p = part(i);
        part_t best = dest_part(i);
        cdata.dest_cache(i) = NULL_PART;
        Kokkos::atomic_add(&curr_state.part_sizes(p), -vtx_w(i));
        Kokkos::atomic_add(&curr_state.part_sizes(best), vtx_w(i));
        part(i) = best;
        //update needs to know old part assignment
        dest_part(i) = p;
    });
    if(total_moves > static_cast<ordinal_t>(g.numRows() / 10)){
        update_large(prob, part, swaps, scratch, cdata);
    } else {
        update_small(prob, part, swaps, dest_part, cdata);
    }
    Kokkos::parallel_reduce("count cutsize change part2", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x, gain_t& gain_update){
        ordinal_t i = swaps(x);
        part_t p = dest_part(i);
        part_t best = part(i);
        edge_offset_t start = cdata.conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        gain_t p_con = lookup(cdata.conn_entries.data() + start, cdata.conn_vals.data() + start, p, size);
        gain_t b_con = lookup(cdata.conn_entries.data() + start, cdata.conn_vals.data() + start, best, size);
        gain_update += b_con - p_con;
    }, scratch.cut_change2);
    stat::stash_largest(curr_state.part_sizes, scratch.max_part);
    //this whole mess avoids 2 stream-syncs on the 3 reductions in this function
    //by combining their data movements to host together
    Kokkos::deep_copy(exec_space(), scratch.reduce_copy, scratch.reduce_locs);
    int64_t cut_change = scratch.reduce_copy(1) + scratch.reduce_copy(0);
    gain_t max_size = scratch.reduce_copy(2);
    curr_state.total_imb = max_size > prob.opt ? max_size - prob.opt : 0;
    curr_state.cut -= cut_change;
} 

//initializes datastructures
conn_data init_conn_data(const conn_data& scratch_cdata, const matrix_t& g, const part_vt& part, part_t k){
    ordinal_t n = g.numRows();
    conn_data cdata;
    cdata.conn_offsets = Kokkos::subview(scratch_cdata.conn_offsets, std::make_pair(static_cast<ordinal_t>(0), n + 1));
    Kokkos::parallel_for("comp conn row size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(degree > static_cast<ordinal_t>(k)) degree = k;
        cdata.conn_offsets(i + 1) = degree;
    });
    edge_offset_t gain_size = 0;
    Kokkos::parallel_scan("comp conn offsets", policy_t(0, n + 1), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update, const bool final){
        update += cdata.conn_offsets(i);
        if(final){
            cdata.conn_offsets(i) = update;
        }
    }, gain_size);
    cdata.conn_vals = Kokkos::subview(scratch_cdata.conn_vals, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.conn_entries = Kokkos::subview(scratch_cdata.conn_entries, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.dest_cache = Kokkos::subview(scratch_cdata.dest_cache, std::make_pair(static_cast<ordinal_t>(0), n));
    cdata.conn_table_sizes = Kokkos::subview(scratch_cdata.conn_table_sizes, std::make_pair(static_cast<ordinal_t>(0), n));
    cdata.lock_bit = Kokkos::subview(scratch_cdata.lock_bit, std::make_pair(static_cast<ordinal_t>(0), n));
    Kokkos::deep_copy(exec_space(), cdata.conn_vals, 0);
    Kokkos::deep_copy(exec_space(), cdata.conn_entries, NULL_PART);
    Kokkos::deep_copy(exec_space(), cdata.dest_cache, NULL_PART);
    Kokkos::deep_copy(exec_space(), cdata.lock_bit, 0);
    //initialize conn tables for each vertex
    //conn tables are resized to be small so that traversal is faster, but large enough so that updates have few collisions
    if((g.nnz() / g.numRows()) < 8) {
        //low degree version
        Kokkos::parallel_for("init conn DS", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t& i){
            edge_offset_t g_start = cdata.conn_offsets(i);
            edge_offset_t g_end = cdata.conn_offsets(i + 1);
            part_t size = g_end - g_start;
            part_t used_cap = 0;
            for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                part_t p = part(v);
                part_t p_o = p % size;
                if(size < k){
                    while(cdata.conn_entries(g_start + p_o) != NULL_PART && cdata.conn_entries(g_start + p_o) != p){
                        p_o = (p_o + 1) % size;
                    }
                }
                cdata.conn_vals(g_start + p_o) += wgt;
                if(cdata.conn_entries(g_start + p_o) == NULL_PART){
                    cdata.conn_entries(g_start + p_o) = p;
                    used_cap++;
                }
            }
            part_t old_size = size;
            size = used_cap;
            part_t quarter_size = size / 4;
            part_t min_inc = 3;
            if(quarter_size < min_inc) quarter_size = min_inc;
            size += quarter_size;
            if(size < old_size){
                //don't get fancy just redo it with a smaller conn table
                for(edge_offset_t j = g_start; j < g_start + old_size; j++){
                    cdata.conn_entries(j) = NULL_PART;
                    cdata.conn_vals(j) = 0;
                }
                for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                    ordinal_t v = g.graph.entries(j);
                    gain_t wgt = g.values(j);
                    part_t p = part(v);
                    part_t p_o = p % size;
                    if(size < k){
                        while(cdata.conn_entries(g_start + p_o) != NULL_PART && cdata.conn_entries(g_start + p_o) != p){
                            p_o = (p_o + 1) % size;
                        }
                    }
                    cdata.conn_vals(g_start + p_o) += wgt;
                    if(cdata.conn_entries(g_start + p_o) == NULL_PART){
                        cdata.conn_entries(g_start + p_o) = p;
                    }
                }
            } else {
                size = old_size;
            }
            cdata.conn_table_sizes(i) = size;
        });
    } else {
        //high-degree version
        //add 4*sizeof(part_t) for alignment reasons I think
        Kokkos::parallel_for("init conn DS (team)", team_policy_t(g.numRows(), Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(k*sizeof(gain_t) + k*sizeof(part_t) + 4*sizeof(part_t))), KOKKOS_LAMBDA(const member& t){
            build_row_cdata_large(cdata, g, part, k, t);
        });
    }
    return cdata;
}

void jet_refine(const matrix_t g, const part_t k, const double imb_ratio, wgt_view_t vtx_w, part_vt best_part, int level, refine_data& best_state, ExperimentLoggerUtil<scalar_t>& experiment){
    Kokkos::Timer y;
    //contains several scratch views that are reused in each iteration
    //reallocating in each iteration would be expensive (GPU memory is often slow to deallocate)
    scratch_mem& scratch = perm_scratch;
    //initialize metadata if this is first level being refined
    //ie. if this is the coarsest level
    if(!best_state.init){
        best_state.cut = stat::get_total_cut(g, best_part);
        best_state.part_sizes = stat::get_part_sizes(g, vtx_w, best_part, k);
        best_state.total_size = stat::get_total_size(g, vtx_w);
    }
    problem prob;
    prob.g = g;
    prob.k = k;
    prob.imb = imb_ratio;
    prob.vtx_w = vtx_w;
    prob.opt = stat::optimal_size(best_state.total_size, k);
    prob.size_max = prob.opt*imb_ratio;
    if(!best_state.init){
        best_state.init = true;
        gain_t max_size = stat::largest_part_size(best_state.part_sizes);
        best_state.total_imb = max_size > prob.opt ? max_size - prob.opt : 0;
        std::cout << "Initial " << std::fixed << (best_state.cut / 2) << " " << std::setprecision(6) << (static_cast<double>(best_state.total_imb) / static_cast<double>(prob.opt)) << " ";
        std::cout << g.numRows() << std::endl;
    }
    refine_data curr_state = clone_refine_data(best_state);
    gain_t imb_max = prob.size_max - prob.opt;
    part_vt part(Kokkos::ViewAllocateWithoutInitializing("current partition"), g.numRows());
    Kokkos::deep_copy(exec_space(), part, best_part);
    conn_data cdata = init_conn_data(perm_cdata, g, part, k);
    int count = 0;
    int iter_count = 0;
    Kokkos::fence();
    Kokkos::Timer iter_t;
    int balance_counter = 0;
    int lab_counter = 0;
    double tol = 0.999;
#ifdef FOUR9
    tol = 0.9999;
#elif defined TWO9
    tol = 0.99;
#endif
    //repeat until 12 phases since a significant
    //improvement in cut or balance
    //this accounts for at least 3 full lp+rebalancing cycles
    double filter_ratio = 0.75;
    if(level == 0) filter_ratio = 0.25;
    while(count++ <= 11){
        iter_count++;
        vtx_view_t moves;
        if(curr_state.total_imb <= imb_max){
            moves = jet_lp(prob, part, cdata, scratch, filter_ratio);
            balance_counter = 0;
            lab_counter++;
        } else {
            if(balance_counter < 2){
                moves = rebalance_weak(prob, part, cdata, scratch, curr_state.part_sizes);
            } else {
                moves = rebalance_strong(prob, part, cdata, scratch, curr_state.part_sizes);
            }
            balance_counter++;
        }
        perform_moves(prob, part, moves, scratch.dest_part, scratch, cdata, curr_state);
        //copy current partition and relevant data to output partition if following conditions pass
        if(best_state.total_imb > imb_max && curr_state.total_imb < best_state.total_imb){
            copy_refine_data(best_state, curr_state);
            Kokkos::deep_copy(exec_space(), best_part, part);
            count = 0;
        } else if(curr_state.cut < best_state.cut && (curr_state.total_imb <= imb_max || curr_state.total_imb <= best_state.total_imb)){
            //do not reset counter if cut improvement is too small
            if(curr_state.cut < tol*best_state.cut){
                count = 0;
            }
            copy_refine_data(best_state, curr_state);
            Kokkos::deep_copy(exec_space(), best_part, part);
        }
    }
    Kokkos::fence();
    double best_imb_ratio = static_cast<double>(best_state.total_imb) / static_cast<double>(prob.opt);
    //divide cut by 2 because each cut edge is counted from both sides
    typename ExperimentLoggerUtil<scalar_t>::CoarseLevel cl(best_state.cut / 2, best_imb_ratio, g.nnz(), g.numRows(), y.seconds(), iter_t.seconds(), iter_count, lab_counter);
    experiment.addCoarseLevel(cl);
    y.reset();
    iter_t.reset();
}
};

}
