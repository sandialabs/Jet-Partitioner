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
#include "experiment_data.hpp"
#include "jet_config.h"

namespace jet_partitioner {

part_vt partition(value_t& edge_cut,
                const config_t& config,
                const matrix_t g,
                const wgt_vt vweights,
                bool uniform_ew,
                experiment_data<value_t>& experiment);

part_mt partition_host(value_t& edge_cut,
                const config_t& config,
                const host_matrix_t g,
                const wgt_host_vt vweights,
                bool uniform_ew,
                experiment_data<value_t>& experiment);

part_mt partition_serial(value_t& edge_cut,
                const config_t& config,
                const serial_matrix_t g,
                const wgt_serial_vt vweights,
                bool uniform_ew,
                experiment_data<value_t>& experiment);

part_vt partition_big(big_val_t& edge_cut,
                const config_t& config,
                const big_matrix_t g,
                const big_wgt_vt vweights,
                bool uniform_ew,
                experiment_data<big_val_t>& experiment);

part_mt partition_host_big(big_val_t& edge_cut,
                const config_t& config,
                const big_host_matrix_t g,
                const big_wgt_host_vt vweights,
                bool uniform_ew,
                experiment_data<big_val_t>& experiment);

part_mt partition_serial_big(big_val_t& edge_cut,
                const config_t& config,
                const big_serial_matrix_t g,
                const big_wgt_serial_vt vweights,
                bool uniform_ew,
                experiment_data<big_val_t>& experiment);

part_vt partition_biggest(big_val_t& edge_cut,
                const config_t& config,
                const biggest_matrix_t g,
                const big_wgt_vt vweights,
                bool uniform_ew,
                experiment_data<big_val_t>& experiment);

part_mt partition_host_biggest(big_val_t& edge_cut,
                const config_t& config,
                const biggest_host_matrix_t g,
                const big_wgt_host_vt vweights,
                bool uniform_ew,
                experiment_data<big_val_t>& experiment);

part_mt partition_serial_biggest(big_val_t& edge_cut,
                const config_t& config,
                const biggest_serial_matrix_t g,
                const big_wgt_serial_vt vweights,
                bool uniform_ew,
                experiment_data<big_val_t>& experiment);

}