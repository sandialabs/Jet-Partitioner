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
#include "jet_refiner.hpp"
#include "contract.hpp"
#include <limits>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "part_stat.hpp"
#include "io.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class uncoarsener {
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
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_mt = typename part_vt::HostMirror;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using coarsener_t = contracter<matrix_t>;
    using clt = typename coarsener_t::coarse_level_triple;
    using ref_t = jet_refiner<matrix_t, part_t>; 
    using rfd_t = typename ref_t::refine_data;
    using gain_t = typename ref_t::gain_t;
    using gain_vt = typename ref_t::gain_vt;
    using stat = part_stat<matrix_t, part_t>;

static double get_max_imb(gain_vt part_sizes, part_t k){
    typename gain_vt::HostMirror ps_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), part_sizes);
    gain_t imb = 0;
    gain_t fine_n = 0;
    for(int p = 0; p < k; p++){
        if(ps_host(p) > imb){
            imb = ps_host(p);
        }
        fine_n += ps_host(p);
    }
    return static_cast<double>(imb) / static_cast<double>(stat::optimal_size(fine_n, k));
}

static void project(ordinal_t fine_n, vtx_view_t map, part_vt input, part_vt output){
    Kokkos::parallel_for("project", policy_t(0, fine_n), KOKKOS_LAMBDA(const ordinal_t i){
        output(i) = input(map(i));
    });
}

static part_vt multilevel_jet(std::list<clt> cg_list, part_vt coarse_guess, part_t k, const double imb_ratio, rfd_t& rfd, ExperimentLoggerUtil<scalar_t>& experiment, Kokkos::Timer& t){
    ref_t refiner(cg_list.front().mtx, k);

    //this is used for outputting the coarse data for use by another program
    //timing data is reset after dumping for comparison with other program
    bool is_dumped = true;
#ifdef EXP
    is_dumped = false;
#endif
    while (!cg_list.empty()) {
        clt cg = cg_list.back();
        if(!is_dumped){
            //dumps the hierarchy starting from the coarsest graph that is balanced before refinement
            double imb = 0;
            if(!rfd.init){
                imb = get_max_imb(stat::get_part_sizes(cg.mtx, cg.vtx_w, coarse_guess, k), k);
            } else {
                imb = get_max_imb(rfd.part_sizes, k);
            }
            if(imb <= imb_ratio){
                dump_coarse_part(coarse_guess);
                dump_coarse(cg_list);
                is_dumped = true;
                Kokkos::fence();
                t.reset();
            }
        }
        refiner.jet_refine(cg.mtx, k, imb_ratio, cg.vtx_w, coarse_guess, cg_list.size() - 1, rfd, experiment);
        cg_list.pop_back();
        if(!cg_list.empty()){
            clt next_cg = cg_list.back();
            // project solution onto finer level graph
            part_vt fine_vec(Kokkos::ViewAllocateWithoutInitializing("fine vec"), next_cg.mtx.numRows());
            project(next_cg.mtx.numRows(), cg.interp_mtx.map, coarse_guess, fine_vec);
            coarse_guess = fine_vec;
        }
    }

    return coarse_guess;
}

static part_vt uncoarsen(std::list<clt> cg_list, part_vt coarsest, part_t k, double imb_ratio
    , scalar_t& ec, ExperimentLoggerUtil<scalar_t>& experiment) {

    Kokkos::Timer t;
    rfd_t rfd;
    part_vt res = multilevel_jet(cg_list, coarsest, k, imb_ratio, rfd, experiment, t);
    Kokkos::fence();
    double rtime = t.seconds();
    t.reset();
    ec = rfd.cut / 2;
    gain_vt part_sizes = rfd.part_sizes;
    typename gain_vt::HostMirror ps_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), part_sizes);
    gain_t largest = 0;
    gain_t total = rfd.total_size;
    double opt = stat::optimal_size(total, k);
    gain_t smallest = total;
    for(int p = 0; p < k; p++){
        if(ps_host(p) > largest){
            largest = ps_host(p);
        }
        if(ps_host(p) < smallest){
            smallest = ps_host(p);
        }
    }
    experiment.addMeasurement(Measurement::Refine, rtime);
    experiment.setFinestImbRatio(static_cast<double>(largest) / opt);
    experiment.setFinestEdgeCut(ec);
    experiment.setLargestPartSize(largest);
    experiment.setSmallestPartSize(smallest);
    experiment.setMaxPartCut(stat::max_part_cut(cg_list.front().mtx, res, k));
    experiment.setObjective(stat::comm_size(cg_list.front().mtx, res, k));
    return res;
}
};

}
