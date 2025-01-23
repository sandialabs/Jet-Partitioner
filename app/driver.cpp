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

#include "jet_defs.h"
#include "io.hpp"
#include "jet.h"
#include "jet_config.h"
#include <limits>
#include <vector>
#include <algorithm>

using namespace jet_partitioner;

void degree_weighting(const matrix_t& g, wgt_vt vweights){
    Kokkos::parallel_for("set v weights", r_policy(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        vweights(i) = g.graph.row_map(i + 1) - g.graph.row_map(i);
    });
}

value_t median(std::vector<value_t>& cuts){
    std::sort(cuts.begin(), cuts.end());
    int count = cuts.size();
    if(count % 2 == 0){
        return (cuts[count / 2] + cuts[(count / 2) - 1]) / 2;
    } else {
        return cuts[count / 2];
    }
}

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "Insufficient number of args provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <metis_graph_file> <config_file> <optional partition_output_filename> <optional metrics_filename>" << std::endl;
        return -1;
    }
    config_t config;
    char *filename = argv[1];
    if(!load_config(config, argv[2])) return -1;
    char *part_file = nullptr;
    char *metrics = nullptr;
    if(argc >= 4){
        part_file = argv[3];
    }
    if(argc >= 5){
        metrics = argv[4];
    }
#ifdef FOUR9
    config.refine_tolerance = 0.9999;
#elif defined TWO9
    config.refine_tolerance = 0.99;
#endif
#ifdef EXP
    config.dump_coarse = true;
#endif
    config.verbose = true;

    Kokkos::initialize();
    //must scope kokkos-related data
    //so that it falls out of scope b4 finalize
    {
        matrix_t g;
        bool uniform_ew = false;
        if(!load_metis_graph(g, uniform_ew, filename)) return -1;
        std::cout << "vertices: " << g.numRows() << "; edges: " << g.nnz() / 2 << std::endl;
        wgt_vt vweights("vertex weights", g.numRows());
        Kokkos::deep_copy(vweights, 1);

        part_vt best_part;

        value_t edgecut_min = std::numeric_limits<value_t>::max();
        std::vector<value_t> cuts;
        int64_t avg = 0;
        for (int i=0; i < config.num_iter; i++) {
            Kokkos::fence();
            value_t edgecut = 0;
            experiment_data<value_t> experiment;
#ifdef HOST
            part_vt part = partition_host(edgecut, config, g, vweights, uniform_ew,
                experiment);
#elif defined SERIAL
            part_vt part = partition_serial(edgecut, config, g, vweights, uniform_ew,
                experiment);
#else
            part_vt part = partition(edgecut, config, g, vweights, uniform_ew,
                experiment);
#endif
            avg += edgecut;
            cuts.push_back(edgecut);

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
        std::cout << "graph " << filename << ", min edgecut found is " << edgecut_min << std::endl;
        std::cout << "average edgecut: " << (avg / config.num_iter) << std::endl;
        std::cout << "median edgecut: " << median(cuts) << std::endl;

        if(part_file != nullptr && config.num_iter > 0) write_part(best_part, part_file);
    }
    Kokkos::finalize();

    return 0;
}
