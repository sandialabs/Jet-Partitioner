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
#include "defs.h"
#include "io.hpp"
#include <limits>

using namespace jet_partitioner;

part_vt partition(value_t& edge_cut,
                    const config_t& config,
                    ExperimentLoggerUtil<value_t>& experiment) {

    using coarsener_t = contracter<matrix_t>;
    using uncoarsener_t = uncoarsener<matrix_t, part_t>;
    using coarse_level_triple = typename coarsener_t::coarse_level_triple;

    std::list<coarse_level_triple> cg_list = load_coarse();
    Kokkos::fence();
    Kokkos::Timer t;
    double start_time = t.seconds();
    part_t k = config.num_parts;
    double fin_coarsening_time = t.seconds();
    double imb_ratio = config.max_imb_ratio;
    part_vt coarsest_p = load_coarse_part(cg_list.back().mtx.numRows());
    Kokkos::fence();
    experiment.addMeasurement(Measurement::InitPartition, t.seconds() - fin_coarsening_time);
    part_vt part = uncoarsener_t::uncoarsen(cg_list, coarsest_p, k, imb_ratio
        , edge_cut, experiment);

    Kokkos::fence();
    double fin_uncoarsening = t.seconds();
    cg_list.clear();
    Kokkos::fence();
    double fin_time = t.seconds();
    experiment.addMeasurement(Measurement::Total, fin_time - start_time);
    experiment.addMeasurement(Measurement::Coarsen, fin_coarsening_time - start_time);
    experiment.addMeasurement(Measurement::FreeGraph, fin_time - fin_uncoarsening);

    experiment.refinementReport();
    experiment.verboseReport();

    return part;
}

void degree_weighting(const matrix_t& g, wgt_view_t vweights){
    Kokkos::parallel_for("set v weights", r_policy(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        vweights(i) = g.graph.row_map(i + 1) - g.graph.row_map(i);
    });
}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << "Insufficient number of args provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <config_file> <optional partition_output_filename> <optional metrics_filename>" << std::endl;
        return -1;
    }
    config_t config;
    if(!load_config(config, argv[1])) return -1;
    char *part_file = nullptr;
    char *metrics = nullptr;
    if(argc >= 3){
        part_file = argv[2];
    }
    if(argc >= 4){
        metrics = argv[3];
    }

    Kokkos::initialize();
    //must scope kokkos-related data
    //so that it falls out of scope b4 finalize
    {
        part_vt best_part;

        value_t edgecut_min = std::numeric_limits<value_t>::max();
        for (int i=0; i < config.num_iter; i++) {
            Kokkos::fence();
            value_t edgecut = 0;
            ExperimentLoggerUtil<value_t> experiment;
            part_vt part = partition(edgecut, config, experiment);

            if (edgecut < edgecut_min) {
                edgecut_min = edgecut;
                best_part = part;
            }
            bool first = true, last = true;
            if (i > 0) {
                first = false;
            }
            if (i + 1 < config.num_iter) {
                last = false;
            }
            if(metrics != nullptr) experiment.log(metrics, first, last);
        }
        std::cout << "Imported coarse graphs, min edgecut found is " << edgecut_min << std::endl;

        if(part_file != nullptr && config.num_iter > 0) write_part(best_part, part_file);
    }
    Kokkos::finalize();

    return 0;
}
