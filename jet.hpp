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
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using coarsener_t = contracter<matrix_t>;
    using init_t = initial_partitioner<matrix_t, part_t>;
    using uncoarsener_t = uncoarsener<matrix_t, part_t>;
    using coarse_level_triple = typename coarsener_t::coarse_level_triple;

static part_vt partition(matrix_t g, wgt_view_t vweights, const part_t k, const double imb_ratio, bool uniform_ew,
                                  ExperimentLoggerUtil<scalar_t>& experiment) {

    coarsener_t coarsener;

    std::list<coarse_level_triple> cg_list;
    Kokkos::Timer t;
    double start_time = t.seconds();

    //coarsener.set_heuristic(coarsener_t::HECv1);
    coarsener.set_heuristic(coarsener_t::MtMetis);
    int cutoff = k*8;
    if(cutoff > 1024){
        cutoff = k*2;
        cutoff = std::max(1024, cutoff);
    }
    coarsener.set_coarse_vtx_cutoff(cutoff);
    coarsener.set_min_allowed_vtx(cutoff / 4);
    cg_list = coarsener.generate_coarse_graphs(g, vweights, experiment, uniform_ew);
    Kokkos::fence();
    double fin_coarsening_time = t.seconds();
    experiment.addMeasurement(Measurement::Coarsen, fin_coarsening_time - start_time);
    part_vt coarsest_p = init_t::metis_init(cg_list.back().mtx, cg_list.back().vtx_w, k, imb_ratio);
    //part_vt coarsest_p = init_t::random_init(cg_list.back().vtx_w, k, imb_ratio);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::InitPartition, t.seconds() - fin_coarsening_time);
    scalar_t edge_cut = 0;
    part_vt part = uncoarsener_t::uncoarsen(cg_list, coarsest_p, k, imb_ratio
        , edge_cut, experiment);

    Kokkos::fence();
    cg_list.clear();
    Kokkos::fence();
    double fin_time = t.seconds();
    experiment.addMeasurement(Measurement::Total, fin_time - start_time);

    return part;
}
};

}
