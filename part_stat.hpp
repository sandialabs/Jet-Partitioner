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

namespace jet_partitioner {

template<class crsMat, typename part_t>
class part_stat {
public:

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    // need some trickery because make_signed is undefined for floating point types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using scalar_t = typename matrix_t::value_type;
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using gain_2vt = Kokkos::View<gain_t**, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;

static gain_t get_total_cut(const matrix_t g, const part_vt partition){
    gain_t total_cut = 0;
    if(!is_host_space ){
        Kokkos::parallel_reduce("find total cut (team)", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t, gain_t& update){
            gain_t local_cut = 0;
            ordinal_t i = t.league_rank();
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j, gain_t& local_update){
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                if(partition(i) != partition(v)){
                    local_update += wgt;
                }
            }, local_cut);
            Kokkos::single(Kokkos::PerTeam(t), [&] (){
                update += local_cut;
            });
        }, total_cut);
    } else {
        Kokkos::parallel_reduce("find total cut", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update){
            gain_t local_cut = 0;
            for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                if(partition(i) != partition(v)){
                    local_cut += wgt;
                }
            }
            update += local_cut;
        }, total_cut);
    }
    return total_cut;
}

// this is needed in a few different places so it is best to have one implementation for consistency
static ordinal_t optimal_size(const ordinal_t total_size, const part_t k){
    //round up as per convention
    return (total_size + k - 1) / k;
}

static gain_2vt cut_heatmap(const matrix_t g, const part_vt partition, const part_t k){
    gain_2vt heatmap("heatmap", k, k);
    Kokkos::parallel_for("create cut heatmap (team)", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = t.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            if(partition(i) != partition(v)){
                Kokkos::atomic_add(&heatmap(partition(i), partition(v)), wgt);
            }
        });
    });
    return heatmap;
}

static gain_vt cut_per_part(const matrix_t g, const part_vt partition, const part_t k){
    gain_vt heatmap("heatmap", k);
    Kokkos::parallel_for("find cut per part (team)", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = t.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            if(partition(i) != partition(v)){
                Kokkos::atomic_add(&heatmap(partition(i)), wgt);
            }
        });
    });
    return heatmap;
}

static gain_vt get_part_sizes(const matrix_t g, const wgt_view_t vtx_w, const part_vt partition, part_t k){
    gain_vt part_size("part sizes", k);
    Kokkos::parallel_for("calc part sizes", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = partition(i);
        Kokkos::atomic_add(&part_size(p), vtx_w(i));
    });
    return part_size;
}

//get sum of vertex weights
static scalar_t get_total_size(const matrix_t g, const wgt_view_t vtx_w){
    scalar_t total_size = 0;
    Kokkos::parallel_reduce("sum of vertex weights", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
        update += vtx_w(i);
    }, total_size);
    return total_size;
}

static gain_t largest_part_size(const gain_vt& ps){
    gain_t result = 0;
    Kokkos::parallel_reduce("get max part size", policy_t(0, ps.extent(0)), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update){
        if(ps(i) > update){
            update = ps(i);
        }
    }, Kokkos::Max<gain_t, Kokkos::HostSpace>(result));
    return result;
}

static void stash_largest(const gain_vt& ps, gain_svt& result){
    Kokkos::parallel_reduce("get max part size (store in view)", policy_t(0, ps.extent(0)), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update){
        if(ps(i) > update){
            update = ps(i);
        }
    }, Kokkos::Max<gain_t, typename gain_svt::memory_space>(result));
}

static gain_t max_part_cut(const matrix_t g, part_vt part, const part_t k){
    return largest_part_size(cut_per_part(g, part, k));
}

static scalar_t comm_size(const matrix_t& g, const part_vt& part, part_t k){
    ordinal_t n = g.numRows();
    edge_view_t conn_offsets("comp offsets", n + 1);
    Kokkos::parallel_for("comp conn row size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(degree > static_cast<ordinal_t>(k)) degree = k;
        conn_offsets(i + 1) = degree;
    });
    edge_offset_t gain_size = 0;
    Kokkos::parallel_scan("comp conn offsets", policy_t(0, n + 1), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update, const bool final){
        update += conn_offsets(i);
        if(final){
            conn_offsets(i) = update;
        }
    }, gain_size);
    part_vt conn_entries("conn entries", gain_size);
    part_t NULL_PART = -1;
    Kokkos::deep_copy(exec_space(), conn_entries, NULL_PART);
    scalar_t result = 0;
    Kokkos::parallel_reduce("find communication volume", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t& i, scalar_t& update){
        edge_offset_t g_start = conn_offsets(i);
        edge_offset_t g_end = conn_offsets(i + 1);
        part_t size = g_end - g_start;
        part_t used_cap = 0;
        part_t local = part(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            ordinal_t v = g.graph.entries(j);
            part_t p = part(v);
            if(p == local) continue;
            part_t p_o = p % size;
            if(size < k){
                while(conn_entries(g_start + p_o) != NULL_PART && conn_entries(g_start + p_o) != p){
                    p_o = (p_o + 1) % size;
                }
            }
            if(conn_entries(g_start + p_o) == NULL_PART){
                conn_entries(g_start + p_o) = p;
                used_cap++;
            }
        }
        update += used_cap;
    }, result);
    return result;
}

static int64_t least_squares(const matrix_t g, part_vt part, const part_t k){
    gain_vt ex_cut = cut_per_part(g, part, k);
    int64_t result = 0;
    Kokkos::parallel_reduce("least squares", policy_t(0, k), KOKKOS_LAMBDA(const part_t p, int64_t& res){
        int64_t ex = ex_cut(p);
        res += ex*ex;
    }, result);
    return result;
}

};

}
