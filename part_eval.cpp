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
#include "part_stat.hpp"
#include "defs.h"
#include "io.hpp"
#include <limits>

using namespace jet_partitioner;

int main(int argc, char **argv) {

    if (argc < 4) {
        std::cerr << "Insufficient number of args provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <metis_graph_file> <part_file> <k>" << std::endl;
        return -1;
    }
    char *filename = argv[1];
    char *part_file = argv[2];
    part_t k = atoi(argv[3]);

    Kokkos::initialize();
    //must scope kokkos-related data
    //so that it falls out of scope b4 finalize
    {
        matrix_t g;
        bool uniform_ew = false;
        if(!load_metis_graph(g, uniform_ew, filename)) return -1;
        std::cout << "vertices: " << g.numRows() << "; edges: " << g.nnz() / 2 << std::endl;
        wgt_view_t vweights("vertex weights", g.numRows());
        Kokkos::deep_copy(vweights, 1);

        part_vt part = load_part(g.numRows(), part_file);
        using stat = part_stat<matrix_t, part_t>; 
        using h_t = stat::gain_2vt;
        value_t cut = stat::get_total_cut(g, part);
        h_t heatmap_d = stat::cut_heatmap(g, part, k);
        h_t::HostMirror heatmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), heatmap_d);
        std::vector<std::vector<gain_t>> hvec(k, std::vector<gain_t>(k, 0));
        gain_t max = 0;
        for(part_t i = 0; i < k; i++){
            gain_t row_total = 0;
            for(part_t j = 0; j < k; j++){
                hvec[i][j] = heatmap(i, j);
                row_total += heatmap(i, j);
            }
            max = max > row_total ? max : row_total;
        }
        std::cout << "Max part cut: " << max << std::endl;
        cut = cut / 2;
        std::cout << "Cutsize: " << cut << std::endl;
        value_t comm_size = stat::comm_size(g, part, k);
        std::cout << "Comm size: " << comm_size << std::endl;
        gain_vt part_sizes = stat::get_part_sizes(g, vweights, part, k);
        typename gain_vt::HostMirror ps_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), part_sizes);
        gain_t largest = 0;
        gain_t total = 0;
        gain_t smallest = g.numRows();
        for(int p = 0; p < k; p++){
            total += ps_host(p);
            if(ps_host(p) > largest){
                largest = ps_host(p);
            }
            if(ps_host(p) < smallest){
                smallest = ps_host(p);
            }
        }
        double opt = stat::optimal_size(total, k);
        double max_imb = static_cast<double>(largest) / opt;
        double min_imb = static_cast<double>(smallest) / opt;
        std::cout << std::setprecision(5);
        std::cout << "Largest: " << max_imb << std::endl;
        std::cout << "Smallest: " << min_imb << std::endl;
    }
    Kokkos::finalize();

    return 0;
}
