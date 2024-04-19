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
#include "defs.h"
#include <filesystem>
#include <sstream>
#include <string>
#include <iostream>

namespace jet_partitioner {

bool load_config(config_t& c, const char* config_f) {

    std::ifstream f(config_f);
    if (!f.is_open()) {
        std::cerr << "FATAL ERROR: Could not open config file " << config_f << std::endl;
        return false;
    }
    std::string lines[4];
    int reads = 0;
    // you might think that reading in four lines from a simple config file could be done like:
    // f >> c.coarsening_alg; 
    // f >> c.num_parts;
    // f >> c.num_iter;
    // f >> c.max_imb_ratio;
    // but that doesn't work if there are exactly 3 lines instead of 4 in the config file
    // because if the last line is a float like 3.14, then c.num_iter will contain the 3
    // and the c.max_imb_ratio will contain the .314
    for(int i = 0; i < 4; i++){
        if(f >> lines[i]) reads++;
    }
    f.close();
    if(reads != 4){
        std::cerr << "FATAL ERROR: Config file has less than 4 lines" << std::endl;
        return false;
    }
    c.coarsening_alg = std::stoi(lines[0]);
    c.num_parts = std::stoi(lines[1]);
    c.num_iter = std::stoi(lines[2]);
    c.max_imb_ratio = std::stod(lines[3]);
    return true;
}

bool load_binary_graph(matrix_t& g, const char *csr_filename) {

    FILE *infp = fopen(csr_filename, "rb");
    if (infp == NULL) {
        printf("Error: Could not open input file %s. Exiting ...\n", csr_filename);
        return false;
    }
    long n, m;
    long unused_vals[4];
    if(fread(&n, sizeof(long), 1, infp) == 0) return false;
    if(fread(&m, sizeof(long), 1, infp) == 0) return false;
    if(fread(unused_vals, sizeof(long), 4, infp) != 4) return false;
    edge_view_t row_map("row map", n + 1);
    edge_mirror_t row_map_m = Kokkos::create_mirror_view(row_map);
    vtx_view_t entries("entries", m);
    vtx_mirror_t entries_m = Kokkos::create_mirror_view(entries);
    size_t nitems_read = fread(row_map_m.data(), sizeof(edge_offset_t), n+1, infp);
    if(nitems_read != ((size_t)n+1)) return false;
    nitems_read = fread(entries_m.data(), sizeof(ordinal_t), m, infp);
    if(nitems_read != ((size_t) m)) return false;
    fclose(infp);
    Kokkos::deep_copy(row_map, row_map_m);
    Kokkos::deep_copy(entries, entries_m);
    wgt_view_t values(Kokkos::ViewAllocateWithoutInitializing("values"), m);
    Kokkos::deep_copy(values, 1);
    graph_t g_graph(entries, row_map);
    g = matrix_t("input graph", n, values, g_graph);
    return true;
}

template<typename t>
t fast_atoi( const char*& str )
{
    t val = 0;
    while(isdigit(*str)) {
        val = val*10 + static_cast<t>(*str - '0');
        str++;
    }
    return val;
}

void next_line(const char*& str){
    while(*str != '\n') str++;
    str++;
}

bool load_metis_graph(matrix_t& g, bool& uniform_ew, const char *fname) {
    Kokkos::Timer t;
    std::ifstream infp(fname);
    if (!infp.is_open()) {
        std::cerr << "FATAL ERROR: Could not open metis graph file " << fname << std::endl;
        return false;
    }
    size_t sz = std::filesystem::file_size(fname);
    char* s = new char[sz + 1];
    std::cout << "Reading " << sz << " bytes from " << fname << std::endl;
    infp.read(s, sz);
    infp.close();
    //append an endline to end of file in case one doesn't exist
    //needed to prevent parser from overshooting end of buffer
    if(s[sz - 1] != '\n'){
        s[sz] = '\n';
        sz++;
    }
    const char* f = s;
    const char* fmax = s + sz;
    size_t header[4] = {0, 0, 0, 0};
    //ignore commented lines
    while(*f == '%') next_line(f);
    while(!isdigit(*f)) f++;
    //read header data
    for(int i = 0; i < 4; i++){
        header[i] = fast_atoi<size_t>(f);
        while(!isdigit(*f)){
            if(*f == '\n'){
                //end for loop
                i = 4;
                f++;
                break;
            }
            f++;
        }
    }
    ordinal_t n = header[0];
    edge_offset_t m = header[1];
    int fmt = header[2];
    int ncon = header[3];
    bool has_ew = ((fmt % 10) == 1);
    if(fmt != 0 && fmt != 1){
        std::cerr << "FATAL ERROR: Unsupported format flags " << fmt << std::endl;
        std::cerr << "Graph parser does not currently support vertex weights" << std::endl;
        return false;
    }
    if(ncon != 0){
        std::cerr << "FATAL ERROR: Unsupported ncon " << ncon << std::endl;
        std::cerr << "Graph parser does not currently support vertex weights" << std::endl;
        return false;
    }
    vtx_view_t entries(Kokkos::ViewAllocateWithoutInitializing("entries"), m*2);
    vtx_mirror_t entries_m = Kokkos::create_mirror_view(entries);
    edge_view_t row_map(Kokkos::ViewAllocateWithoutInitializing("row map"), n + 1);
    edge_mirror_t row_map_m = Kokkos::create_mirror_view(row_map);
    wgt_view_t values(Kokkos::ViewAllocateWithoutInitializing("values"), 2*m);
    wgt_mirror_t values_m;
    if(has_ew){
        values_m = Kokkos::create_mirror_view(values);
    }
    edge_offset_t edges_read = 0;
    ordinal_t rows_read = 0;
    row_map_m(0) = 0;
    bool is_value = false;
    //ready edge information
    while(f < fmax){
        //increment past whitespace
        while(f < fmax && !isdigit(*f)){
            //ignore commented lines
            if(*f == '%'){
                next_line(f);
                continue;
            }
            if(*f == '\n'){
                //ignore extra trailing newlines
                if(rows_read < n) row_map_m(++rows_read) = edges_read;
            }
            f++;
        }
        if(f >= fmax) break;
        //fast_atoi also increments past numeric chars
        ordinal_t edge_info = fast_atoi<ordinal_t>(f);
        if(!is_value){
            //subtract 1 to convert to 0-indexed
            entries_m(edges_read) = edge_info - 1;
        } else {
            values_m(edges_read++) = edge_info;
        }
        if(has_ew){
            is_value = !is_value;
        } else {
            edges_read++;
        }
    }
    delete[] s;
    if(rows_read != n || edges_read != 2*m){
        std::cerr << "FATAL ERROR: Mismatch between expected and actual line/nonzero count in metis file" << std::endl;
        std::cerr << "Read " << rows_read << " lines and " << edges_read << " nonzeros" << std::endl;
        std::cerr << "Lines expected: " << n << "; Nonzeros expected: " << m*2 << std::endl;
        return false;
    }
    Kokkos::deep_copy(row_map, row_map_m);
    Kokkos::deep_copy(entries, entries_m);
    if(has_ew){
        Kokkos::deep_copy(values, values_m);
        uniform_ew = false;
    } else {
        Kokkos::deep_copy(values, 1);
        uniform_ew = true;
    }
    graph_t g_graph(entries, row_map);
    g = matrix_t("input graph", n, values, g_graph);
    std::cout << "Read graph from " << fname << " in " << std::setprecision(3) << t.seconds() << "s" << std::endl;
    return true;
}

void write_part(part_vt part_d, const char *fname){
    std::ofstream ofp(fname);
    if(!ofp.is_open()) return;
    part_mt part = Kokkos::create_mirror_view(part_d);
    Kokkos::deep_copy(part, part_d);
    size_t n = part.extent(0);
    std::stringstream ss;
    for(size_t x = 0; x < n; x++){
        ss << part(x) << std::endl;
    }
    ofp << ss.str();
    ofp.close();
}

part_vt load_part(ordinal_t n, const char *fname){
    std::ifstream ifp(fname);
    part_vt part_d("device part", n);
    if(!ifp.is_open()) return part_d;
    part_mt part = Kokkos::create_mirror_view(part_d);
    for(ordinal_t x = 0; x < n; x++){
        ifp >> part(x);
    }
    ifp.close();
    Kokkos::deep_copy(part_d, part);
    return part_d;
}

}
