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
#include <cstdlib>
#include <cmath>
#include <utility>
#include <numeric>
#include <random>
#include <type_traits>
#include "metis.h"
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class initial_partitioner {
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
    using vtx_mirror_t = typename vtx_view_t::HostMirror;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using wgt_mirror_t = typename wgt_view_t::HostMirror;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_mt = typename part_vt::HostMirror;
    using metis_int = int;
    using metis_vt = Kokkos::View<metis_int*, Device>;
    using metis_mt = typename metis_vt::HostMirror;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using coarsener_t = contracter<matrix_t>;

template <class dst_vt, class src_vt>
static void copy(dst_vt dst, src_vt src){
    Kokkos::parallel_for("copy", policy_t(0, src.extent(0)), KOKKOS_LAMBDA(const int i){
        dst(i) = src(i);
    });
}

template <class src_vt>
static metis_mt to_metis_int(src_vt src){
    using src_t = std::remove_cv_t<typename src_vt::value_type>;
    int n = src.extent(0);
    metis_mt data(Kokkos::ViewAllocateWithoutInitializing("metis int host"), n);
    if(std::is_same_v<metis_int, src_t>){
        Kokkos::deep_copy(data, src);
    } else {
        metis_vt data_dev(Kokkos::ViewAllocateWithoutInitializing("metis int dev"), n);
        copy<metis_vt, src_vt>(data_dev, src);
        Kokkos::deep_copy(data, data_dev);
    }
    return data;
}

static part_vt metis_init(matrix_t g, wgt_view_t vtx_w, int k, double imb_ratio){
    int n = g.numRows();
    metis_vt part_metis("part metis type", n);
    metis_mt pm = Kokkos::create_mirror_view(part_metis);
    metis_mt vtx_wm = to_metis_int<wgt_view_t>(vtx_w);
    metis_mt xadj = to_metis_int<typename matrix_t::row_map_type>(g.graph.row_map);
    metis_mt adjcwgt = to_metis_int<wgt_view_t>(g.values);
    metis_mt adjncy = to_metis_int<vtx_view_t>(g.graph.entries);
    real_t imbalance = imb_ratio;
    int ec = 0;
    int nweights = 1;
    int ret = METIS_PartGraphKway(&n, &nweights, xadj.data(), adjncy.data(),
				       vtx_wm.data(), NULL, adjcwgt.data(), &k, NULL,
				       &imbalance, NULL, &ec, pm.data());
    if(ret != METIS_OK){
        std::cerr << "Metis could not partition coarsest graph. Exiting..." << std::endl;
        exit(-1);
    }
    Kokkos::deep_copy(part_metis, pm);
    part_vt part("part", n);
    copy<part_vt, metis_vt>(part, part_metis);
    return part;
}

static part_vt random_init(wgt_view_t vtx_w, int k, double imb_ratio){
    ordinal_t n = vtx_w.extent(0);
    scalar_t total = 0;
    Kokkos::parallel_reduce("sum vtx", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
        update += vtx_w(i);
    }, total);
    scalar_t opt = total / k;
    scalar_t upper = opt * imb_ratio;
    wgt_mirror_t vw = Kokkos::create_mirror_view(vtx_w);
    Kokkos::deep_copy(vw, vtx_w);
    part_vt part_dev("part device", n);
    part_mt part = Kokkos::create_mirror_view(part_dev);
    wgt_mirror_t psizes("part sizes", k);
    std::random_device rd;
    std::mt19937 rg(rd());
    std::uniform_int_distribution<> range(0, k-1);
    for(ordinal_t i = 0; i < n; i++){
        scalar_t size = vw(i);
        part_t p = range(rg);
        int breaker = 0;
        while(!(psizes(p) + size < upper) && breaker < 2*k){
            p = range(rg);
            breaker++;
        }
        //this loop is needed in case vtx i can't find a valid fit in 2*k attempts
        while(!(psizes(p) < opt)){
            p = range(rg);
        }
        part(i) = p;
        psizes(p) += size;
    }
    Kokkos::deep_copy(part_dev, part);
    return part_dev;
}

};

}
