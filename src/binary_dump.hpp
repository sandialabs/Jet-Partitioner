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
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class binary_dump {
public:
    // define internal types
    using matrix_t = crsMat;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using part_vt = Kokkos::View<part_t*, Device>;
    using coarsener_t = contracter<matrix_t>;
    using clt = typename coarsener_t::coarse_level_triple;

//writes a sequence of coarse graphs and mappings between each graph to binary file
//used to control for coarsening when experimenting with refinement
static void dump_coarse(std::list<clt> levels){
    FILE* cgfp = fopen("coarse_graphs.out", "w");
    int size = levels.size();
    fwrite(&size, sizeof(int), 1, cgfp);
    ordinal_t prev_n = 0;
    for(auto level : levels){
        matrix_t g = level.mtx;
        ordinal_t N = g.numRows();
        edge_offset_t M = g.nnz();
        auto rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.graph.row_map);
        auto entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.graph.entries);
        auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.values);
        auto vtx_wgts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), level.vtx_w);
        fwrite(&N, sizeof(ordinal_t), 1, cgfp);
        fwrite(&M, sizeof(edge_offset_t), 1, cgfp);
        fwrite(rows.data(), sizeof(edge_offset_t), N+1, cgfp);
        fwrite(entries.data(), sizeof(ordinal_t), M, cgfp);
        fwrite(values.data(), sizeof(scalar_t), M, cgfp);
        fwrite(vtx_wgts.data(), sizeof(scalar_t), N, cgfp);
        if(level.level > 1){
            typename jet_partitioner::contracter<matrix_t>::coarse_map interp_mtx = level.interp_mtx;
            auto i_entries = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), interp_mtx.map);
            fwrite(i_entries.data(), sizeof(ordinal_t), prev_n, cgfp);
        }
        prev_n = N;
    }
    fclose(cgfp);
}

//writes part to binary file
static void dump_coarse_part(part_vt part){
    FILE* cgfp = fopen("coarse_part.out", "wb");
    ordinal_t n = part.extent(0);
    auto part_m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), part);
    fwrite(part_m.data(), sizeof(part_t), n, cgfp);
    fclose(cgfp);
}
};
}