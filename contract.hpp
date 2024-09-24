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
#include <list>
#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_HashmapAccumulator.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"
#include "ExperimentLoggerUtil.hpp"
#include "heuristics.hpp"

namespace jet_partitioner {

template<class crsMat> //typename ordinal_t, typename edge_offset_t, typename scalar_t, class Device>
class contracter {
public:

    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using dyn_policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using dyn_team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using member = typename team_policy_t::member_type;
    using pool_t = Kokkos::Random_XorShift64_Pool<Device>;
    using coarse_map = typename coarsen_heuristics<matrix_t>::coarse_map;
    static constexpr ordinal_t get_null_val() {
        // this value must line up with the null value used by the hashmap
        // accumulator
        if (std::is_signed<ordinal_t>::value) {
            return -1;
        } else {
            return std::numeric_limits<ordinal_t>::max();
        }
    }
    static constexpr ordinal_t ORD_MAX  = get_null_val();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    // contains matrix and vertex weights corresponding to current level
    // interp matrix maps previous level to this level
    struct coarse_level_triple {
        matrix_t mtx;
        wgt_view_t vtx_w;
        coarse_map interp_mtx;
        int level;
        bool uniform_weights = false;
    };

    // define behavior-controlling enums
    enum Heuristic { HECv1, HECv2, HECv3, Match, MtMetis };

    // internal parameters and data
    // default heuristic is MtMetis
    Heuristic h = MtMetis;
    coarsen_heuristics<matrix_t> mapper;
    ordinal_t coarse_vtx_cutoff = 1000;
    ordinal_t min_allowed_vtx = 250;
    unsigned int max_levels = 200;
    const ordinal_t large_row_threshold = 1000;
    
bool has_large_row(const matrix_t g){
    ordinal_t max_row = 0;
    Kokkos::parallel_reduce("find max row", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update){
        ordinal_t degree = g.graph.row_map(i+1) - g.graph.row_map(i);
        if(degree > update) update = degree;
    }, Kokkos::Max<ordinal_t, Kokkos::HostSpace>(max_row));
    return (max_row >= large_row_threshold);
}

bool should_use_dyn(const ordinal_t n, const Kokkos::View<const edge_offset_t*, Device> work, int t_count){
    bool use_dyn = false;
    edge_offset_t max = 0;
    edge_offset_t min = std::numeric_limits<edge_offset_t>::max();
    if(is_host_space){
        ordinal_t static_size = (n + t_count) / t_count;
        for(ordinal_t i = 0; i < t_count; i++){
            ordinal_t start = i * static_size;
            ordinal_t end = start + static_size;
            if(start > n) start = n;
            if(end > n) end = n;
            edge_offset_t size = work(end) - work(start);
            if(size > max){
                max = size;
            }
            if(size < min) {
                min = size;
            }
        }
        //printf("min size: %i, max size: %i\n", min, max);
        if(n > 500000 && max > 5*min){
            use_dyn = true;
        }
    }
    return use_dyn;
}

struct countingFunctor {

    matrix_t g;
    vtx_view_t vcmap;
    edge_view_t degree_initial;
    wgt_view_t c_vtx_w, f_vtx_w;
    ordinal_t workLength;

    countingFunctor(matrix_t _g,
            vtx_view_t _vcmap,
            edge_view_t _degree_initial,
            wgt_view_t _c_vtx_w,
            wgt_view_t _f_vtx_w) :
        g(_g),
        vcmap(_vcmap),
        degree_initial(_degree_initial),
        c_vtx_w(_c_vtx_w),
        f_vtx_w(_f_vtx_w),
        workLength(_g.numRows()) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& i) const 
    {
        ordinal_t u = vcmap(i);
        edge_offset_t start = g.graph.row_map(i);
        edge_offset_t end = g.graph.row_map(i + 1);
        ordinal_t nonLoopEdgesTotal = end - start;
        Kokkos::atomic_add(&degree_initial(u), nonLoopEdgesTotal);
        Kokkos::atomic_add(&c_vtx_w(u), f_vtx_w(i));
    }
};

struct combineAndDedupe {
    matrix_t g;
    vtx_view_t vcmap;
    vtx_view_t htable;
    wgt_view_t hvals;
    edge_view_t hrow_map;

    combineAndDedupe(matrix_t _g,
            vtx_view_t _vcmap,
            vtx_view_t _htable,
            wgt_view_t _hvals,
            edge_view_t _hrow_map) :
            g(_g),
            vcmap(_vcmap),
            htable(_htable),
            hvals(_hvals),
            hrow_map(_hrow_map) {}

    KOKKOS_INLINE_FUNCTION
        edge_offset_t insert(const edge_offset_t& hash_start, const edge_offset_t& size, const ordinal_t& u) const {
            edge_offset_t offset = abs(xorshiftHash<ordinal_t>(u)) % size;
            while(true){
                ordinal_t v = htable(hash_start + offset);
                if(v == -1){
                    v = Kokkos::atomic_compare_exchange(&htable(hash_start + offset), -1, u);
                }
                if(v == u || v == -1){
                    return offset;
                }
                offset++;
                if(offset >= size) offset -= size;
            }
        }

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const
    {
        const ordinal_t x = thread.league_rank();
        const ordinal_t i = vcmap(x);
        const edge_offset_t start = g.graph.row_map(x);
        const edge_offset_t end = g.graph.row_map(x + 1);
        const edge_offset_t hash_start = hrow_map(i);
        const edge_offset_t size = hrow_map(i + 1) - hash_start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            ordinal_t u = vcmap(g.graph.entries(j));
            if(i != u){
                edge_offset_t offset = insert(hash_start, size, u);
                Kokkos::atomic_add(&hvals(hash_start + offset), g.values(j));
            }
        });
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& x) const
    {
        const ordinal_t i = vcmap(x);
        const edge_offset_t start = g.graph.row_map(x);
        const edge_offset_t end = g.graph.row_map(x + 1);
        const edge_offset_t hash_start = hrow_map(i);
        const edge_offset_t size = hrow_map(i + 1) - hash_start;
        for(edge_offset_t j = start; j < end; j++){
            ordinal_t u = vcmap(g.graph.entries(j));
            if(i != u){
                edge_offset_t offset = insert(hash_start, size, u);
                Kokkos::atomic_add(&hvals(hash_start + offset), g.values(j));
            }
        }
    }
};

struct countUnique {
    vtx_view_t htable;
    edge_view_t hrow_map, coarse_row_map_f;

    countUnique(vtx_view_t _htable,
            edge_view_t _hrow_map,
            edge_view_t _coarse_row_map_f) :
            htable(_htable),
            hrow_map(_hrow_map),
            coarse_row_map_f(_coarse_row_map_f) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const
    {
        const ordinal_t i = thread.league_rank();
        const edge_offset_t start = hrow_map(i);
        const edge_offset_t end = hrow_map(i + 1);
        ordinal_t uniques = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j, ordinal_t& update){
            if(htable(j) != -1){
                update++;
            }
        }, uniques);
        Kokkos::single(Kokkos::PerTeam(thread), [=](){
            coarse_row_map_f(i) = uniques;
        });
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t i) const
    {
        const edge_offset_t start = hrow_map(i);
        const edge_offset_t end = hrow_map(i + 1);
        ordinal_t uniques = 0;
        for(edge_offset_t j = start; j < end; j++) {
            if(htable(j) != -1){
                uniques++;
            }
        }
        coarse_row_map_f(i) = uniques;
    }
};

struct consolidateUnique {
    vtx_view_t htable, entries_coarse;
    wgt_view_t hvals, wgts_coarse;
    edge_view_t hrow_map, coarse_row_map_f;

    consolidateUnique(vtx_view_t _htable,
            vtx_view_t _entries_coarse,
            wgt_view_t _hvals,
            wgt_view_t _wgts_coarse,
            edge_view_t _hrow_map,
            edge_view_t _coarse_row_map_f) :
            htable(_htable),
            entries_coarse(_entries_coarse),
            hvals(_hvals),
            wgts_coarse(_wgts_coarse),
            hrow_map(_hrow_map),
            coarse_row_map_f(_coarse_row_map_f) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const
    {
        const ordinal_t i = thread.league_rank();
        const edge_offset_t start = hrow_map(i);
        const edge_offset_t end = hrow_map(i + 1);
        const edge_offset_t write_to = coarse_row_map_f(i);
        ordinal_t* total = (ordinal_t*) thread.team_shmem().get_shmem(sizeof(ordinal_t));
        *total = 0;
        thread.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            if(htable(j) != -1){
                //we don't care about the insertion order
                //this is faster than a scan
                ordinal_t insert = Kokkos::atomic_fetch_add(total, 1);
                entries_coarse(write_to + insert) = htable(j);
                wgts_coarse(write_to + insert) = hvals(j);
            }
        });
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t i) const
    {
        const edge_offset_t start = hrow_map(i);
        const edge_offset_t end = hrow_map(i + 1);
        edge_offset_t write_to = coarse_row_map_f(i);
        for (edge_offset_t j = start; j < end; j++){
            if(htable(j) != -1){
                entries_coarse(write_to) = htable(j);
                wgts_coarse(write_to) = hvals(j);
                write_to++;
            }
        };
    }
};

coarse_level_triple build_coarse_graph(const coarse_level_triple level,
    const coarse_map vcmap,
    ExperimentLoggerUtil<scalar_t>& experiment) {

    matrix_t g = level.mtx;
    ordinal_t n = g.numRows();
    ordinal_t nc = vcmap.coarse_vtx;

    Kokkos::Timer timer;
    edge_view_t hrow_map("hashtable row map", nc + 1);
    wgt_view_t f_vtx_w = level.vtx_w;
    wgt_view_t c_vtx_w = wgt_view_t("coarse vertex weights", nc);
    countingFunctor countF(g, vcmap.map, hrow_map, c_vtx_w, f_vtx_w);
    Kokkos::parallel_for("count edges per coarse vertex (also compute coarse vertex weights)", policy_t(0, n), countF);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Count, timer.seconds());
    timer.reset();
    edge_offset_t hash_size = 0;
    //exclusive prefix sum
    Kokkos::parallel_scan("scan offsets", policy_t(0, nc + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = hrow_map(i);
        if(final){
            hrow_map(i) = update;
        }
        update += val;
    }, hash_size);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Prefix, timer.seconds());
    timer.reset();
    vtx_view_t htable(Kokkos::ViewAllocateWithoutInitializing("hashtable keys"), hash_size);
    Kokkos::deep_copy(htable, -1);
    wgt_view_t hvals("hashtable values", hash_size);
    // use thread teams on gpu when graph has decent average degree or very large max degree
    bool use_team = (!is_host_space && (hash_size / n >= 12 || has_large_row(g)));
    //insert each coarse vertex into a bucket determined by a hash
    //use linear probing to resolve conflicts
    //combine weights using atomic addition
    combineAndDedupe cnd(g, vcmap.map, htable, hvals, hrow_map);
    if(use_team) {
        Kokkos::parallel_for("deduplicate", team_policy_t(n, Kokkos::AUTO), cnd);
    } else {
        bool use_dyn = should_use_dyn(n, g.graph.row_map, exec_space().concurrency());
        if(use_dyn){
            Kokkos::parallel_for("deduplicate", dyn_policy_t(0, n), cnd);
        } else {
            Kokkos::parallel_for("deduplicate", policy_t(0, n), cnd);
        }
    }
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Dedupe, timer.seconds());
    timer.reset();
    edge_view_t coarse_row_map_f("edges_per_source", nc + 1);
    countUnique cu(htable, hrow_map, coarse_row_map_f);
    if(use_team) {
        Kokkos::parallel_for("count unique", team_policy_t(nc, Kokkos::AUTO), cu);
    } else {
        Kokkos::parallel_for("count unique", policy_t(0, nc), cu);
    }
    Kokkos::fence();
    experiment.addMeasurement(Measurement::WriteGraph, timer.seconds());
    timer.reset();
    Kokkos::parallel_scan("scan offsets", policy_t(0, nc + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = coarse_row_map_f(i);
        if(final){
            coarse_row_map_f(i) = update;
        }
        update += val;
    }, hash_size);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Prefix, timer.seconds());
    timer.reset();
    vtx_view_t entries_coarse(Kokkos::ViewAllocateWithoutInitializing("coarse entries"), hash_size);
    wgt_view_t wgts_coarse(Kokkos::ViewAllocateWithoutInitializing("coarse weights"), hash_size);
    consolidateUnique consolidate(htable, entries_coarse, hvals, wgts_coarse, hrow_map, coarse_row_map_f);
    if(use_team) {
        Kokkos::parallel_for("consolidate", team_policy_t(nc, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(4*sizeof(ordinal_t))), consolidate);
    } else {
        bool use_dyn = should_use_dyn(nc, hrow_map, exec_space().concurrency());
        if(use_dyn){
            Kokkos::parallel_for("consolidate", dyn_policy_t(0, nc), consolidate);
        } else {
            Kokkos::parallel_for("consolidate", policy_t(0, nc), consolidate);
        }
    }
    graph_type gc_graph(entries_coarse, coarse_row_map_f);
    matrix_t gc("gc", nc, wgts_coarse, gc_graph);
    coarse_level_triple next_level;
    next_level.mtx = gc;
    next_level.vtx_w = c_vtx_w;
    next_level.level = level.level + 1;
    next_level.interp_mtx = vcmap;
    next_level.uniform_weights = false;
    Kokkos::fence();
    experiment.addMeasurement(Measurement::WriteGraph, timer.seconds());
    timer.reset();
    return next_level;
}

coarse_map generate_coarse_mapping(const matrix_t g,
    const wgt_view_t& vtx_w,
    bool uniform_weights,
    pool_t& rand_pool,
    ExperimentLoggerUtil<scalar_t>& experiment) {

    Kokkos::Timer timer;
    coarse_map interpolation_graph;
    int choice = 0;

    switch (h) {
        case HECv1:
            choice = 0;
            break;
        case HECv2:
            choice = 1;
            break;
        case HECv3:
            choice = 2;
            break;
        case Match:
            choice = 0;
            break;
        case MtMetis:
            choice = 1;
            break;
        default:
            choice = 0;
    }

    switch (h) {
        case HECv1:
        case HECv2:
        case HECv3:
            interpolation_graph = mapper.coarsen_HEC(g, vtx_w, uniform_weights, rand_pool, experiment);
            break;
        case Match:
        case MtMetis:
            interpolation_graph = mapper.coarsen_match(g, uniform_weights, rand_pool, choice);
            break;
    }
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Map, timer.seconds());
    return interpolation_graph;
}

std::list<coarse_level_triple> generate_coarse_graphs(const matrix_t fine_g, const wgt_view_t vweights, ExperimentLoggerUtil<scalar_t>& experiment, bool uniform_eweights = false) {

    Kokkos::Timer timer;
    std::list<coarse_level_triple> levels;
    coarse_level_triple finest;
    finest.mtx = fine_g;
    //1-indexed, not zero indexed
    finest.level = 1;
    finest.uniform_weights = uniform_eweights;
    finest.vtx_w = vweights;
    levels.push_back(finest);
    pool_t rand_pool(std::time(nullptr));
    while (levels.rbegin()->mtx.numRows() > coarse_vtx_cutoff) {

        coarse_level_triple current_level = *levels.rbegin();

        coarse_map interp_graph = generate_coarse_mapping(current_level.mtx, current_level.vtx_w, current_level.uniform_weights, rand_pool, experiment);

        if (interp_graph.coarse_vtx < min_allowed_vtx) {
            break;
        }

        timer.reset();
        coarse_level_triple next_level = build_coarse_graph(current_level, interp_graph, experiment);
        Kokkos::fence();
        experiment.addMeasurement(Measurement::Build, timer.seconds());
        timer.reset();

        levels.push_back(next_level);

        if(levels.size() > max_levels) break;
#ifdef DEBUG
        double coarsen_ratio = (double) levels.rbegin()->mtx.numRows() / (double) (++levels.rbegin())->mtx.numRows();
        printf("Coarsening ratio: %.8f\n", coarsen_ratio);
#endif
    }
    return levels;
}

void set_heuristic(Heuristic _h) {
    this->h = _h;
}

void set_coarse_vtx_cutoff(ordinal_t _coarse_vtx_cutoff) {
    this->coarse_vtx_cutoff = _coarse_vtx_cutoff;
}

void set_min_allowed_vtx(ordinal_t _min_allowed_vtx) {
    this->min_allowed_vtx = _min_allowed_vtx;
}

void set_max_levels(unsigned int _max_levels) {
    this->max_levels = _max_levels;
}

};

}
