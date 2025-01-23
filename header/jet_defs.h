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

#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

namespace jet_partitioner {

using big_ordinal_t = int64_t;
using big_offset_t = int64_t;
using big_val_t = int64_t;

using ordinal_t = int32_t;
using edge_offset_t = int32_t;
using value_t = int32_t;

#if defined(SERIAL)
using Device = Kokkos::Serial;
#elif defined(HOST)
using Device = Kokkos::DefaultHostExecutionSpace;
#else
using Device = Kokkos::DefaultExecutionSpace;
#endif

// standard sized graphs
using matrix_t = typename KokkosSparse::CrsMatrix<value_t, ordinal_t, Device, void, edge_offset_t>;
using host_matrix_t = typename KokkosSparse::CrsMatrix<value_t, ordinal_t, Kokkos::DefaultHostExecutionSpace, void, edge_offset_t>;
using serial_matrix_t = typename KokkosSparse::CrsMatrix<value_t, ordinal_t, Kokkos::Serial, void, edge_offset_t>;
using graph_t = typename matrix_t::staticcrsgraph_type;
using edge_vt = Kokkos::View<edge_offset_t*, Device>;
using edge_mt = typename edge_vt::HostMirror;
using vtx_vt = Kokkos::View<ordinal_t*, Device>;
using vtx_mt = typename vtx_vt::HostMirror;
using wgt_vt = Kokkos::View<value_t*, Device>;
using wgt_mt = typename wgt_vt::HostMirror;
using wgt_host_vt = Kokkos::View<value_t*, Kokkos::DefaultHostExecutionSpace>;
using wgt_serial_vt = Kokkos::View<value_t*, Kokkos::Serial>;
using r_policy = Kokkos::RangePolicy<typename Device::execution_space>;
using part_t = int;
using part_vt = Kokkos::View<part_t*, Device>;
using part_mt = typename part_vt::HostMirror;

// bigger graphs (large edge counts and potentially large edge weights for coarse graphs)
using big_matrix_t = typename KokkosSparse::CrsMatrix<big_val_t, ordinal_t, Device, void, big_offset_t>;
using big_host_matrix_t = typename KokkosSparse::CrsMatrix<big_val_t, ordinal_t, Kokkos::DefaultHostExecutionSpace, void, big_offset_t>;
using big_serial_matrix_t = typename KokkosSparse::CrsMatrix<big_val_t, ordinal_t, Kokkos::Serial, void, big_offset_t>;
using big_wgt_vt = Kokkos::View<big_val_t*, Device>;
using big_wgt_host_vt = Kokkos::View<big_val_t*, Kokkos::DefaultHostExecutionSpace>;
using big_wgt_serial_vt = Kokkos::View<big_val_t*, Kokkos::Serial>;

// biggest graphs (bigger graphs + large vertex counts)
using biggest_matrix_t = typename KokkosSparse::CrsMatrix<big_val_t, big_ordinal_t, Device, void, big_offset_t>;
using biggest_host_matrix_t = typename KokkosSparse::CrsMatrix<big_val_t, big_ordinal_t, Kokkos::DefaultHostExecutionSpace, void, big_offset_t>;
using biggest_serial_matrix_t = typename KokkosSparse::CrsMatrix<big_val_t, big_ordinal_t, Kokkos::Serial, void, big_offset_t>;

};