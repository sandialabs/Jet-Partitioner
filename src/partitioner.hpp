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
#include "contract.hpp"
#include "uncoarsen.hpp"
#include "initial_partition.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class partitioner {
public:

    using matrix_t = crsMat;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using scalar_t = typename matrix_t::value_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using coarsener_t = contracter<matrix_t>;
    using init_t = initial_partitioner<matrix_t, part_t>;
    using uncoarsener_t = uncoarsener<matrix_t, part_t>;
    using coarse_level_triple = typename coarsener_t::coarse_level_triple;
    using stat = part_stat<matrix_t, part_t>;

static part_vt partition(scalar_t& edge_cut,
                                  const config_t& config,
                                  const matrix_t g,
                                  const wgt_vt vweights,
                                  bool uniform_ew,
                                  experiment_data<scalar_t>& experiment) {

    coarsener_t coarsener;

    Kokkos::fence();
    Kokkos::Timer t;
    double start_time = t.seconds();
    part_t k = config.num_parts;

    switch(config.coarsening_alg){
        case 0:
            coarsener.set_heuristic(coarsener_t::MtMetis);
            break;
        case 1:
            coarsener.set_heuristic(coarsener_t::HECv1);
            break;
        case 2:
            coarsener.set_heuristic(coarsener_t::Match);
            break;
        default:
            coarsener.set_heuristic(coarsener_t::MtMetis);
    }
    int cutoff = k*8;
    if(cutoff > 1024){
        cutoff = k*2;
        cutoff = std::max(1024, cutoff);
    }
    coarsener.set_coarse_vtx_cutoff(cutoff);
    coarsener.set_min_allowed_vtx(cutoff / 4);
    std::list<coarse_level_triple> cg_list = coarsener.generate_coarse_graphs(g, vweights, experiment, uniform_ew);
    Kokkos::fence();
    double fin_coarsening_time = t.seconds();
    double imb_ratio = config.max_imb_ratio;
    part_vt coarsest_p = init_t::metis_init(cg_list.back().mtx, cg_list.back().vtx_w, k, imb_ratio);
    //part_vt coarsest_p = init_t::random_init(cg_list.back().vtx_w, k, imb_ratio);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::InitPartition, t.seconds() - fin_coarsening_time);
    part_vt part = uncoarsener_t::uncoarsen(cg_list, coarsest_p, config,
        edge_cut, experiment);

    Kokkos::fence();
    double fin_uncoarsening = t.seconds();
    cg_list.clear();
    Kokkos::fence();
    double fin_time = t.seconds();
    experiment.addMeasurement(Measurement::Total, fin_time - start_time);
    experiment.addMeasurement(Measurement::Coarsen, fin_coarsening_time - start_time);
    experiment.addMeasurement(Measurement::FreeGraph, fin_time - fin_uncoarsening);
    
    if(config.verbose){
        // additional partition statistics
        experiment.setMaxPartCut(stat::max_part_cut(g, part, k));
        experiment.setObjective(stat::comm_size(g, part, k));

        experiment.refinementReport();
        experiment.verboseReport();
    }

    return part;
}
};

}
