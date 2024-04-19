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
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>

namespace jet_partitioner {

enum class Measurement : int {
	Map,
	Build,
	Count,
	Prefix,
	Bucket,
	Dedupe,
	RadixSort,
	RadixDedupe,
    WriteGraph,
	Permute,
	MapConstruct,
	Heavy,
    InitTransfer,
    HashmapAllocate,
    HashmapInsert,
    InitPartition,
    Coarsen,
    Refine,
    FreeGraph,
    Total,
	END
};

template <typename scalar_t>
class ExperimentLoggerUtil {

public:
	std::vector<std::string> measurementNames{
		"coarsen-map",
		"coarsen-build",
		"coarsen-count",
		"coarsen-prefix-sum",
		"coarsen-bucket",
		"coarsen-dedupe",
		"coarsen-radix-sort",
		"coarsen-radix-dedupe",
        "coarsen-write-graph",
		"coarsen-permute",
		"coarsen-map-construct",
		"heavy",
        "initial-transfer-to-device",
        "hashmap-allocate",
        "hashmap-insert",
        "initial-partition",
        "coarsen",
        "refine",
        "free-graph",
        "total",
	};
	std::vector<double> measurements;

	class CoarseLevel {
	public:
        int64_t edge_cut = 0;
        double imb = 0;
        uint64_t numEdges = 0;
		uint64_t numVertices = 0;
        double totalRefTime = 0;
        double iterationsTime = 0;
        int totalIterations = 0;
        int lpIterations = 0;

		CoarseLevel(int64_t _edge_cut, double _imb, uint64_t _numEdges, uint64_t _numVertices, double _totalRefTime, double _iterationsTime, int _totalIterations, int _lpIterations) :
            edge_cut(_edge_cut),
            imb(_imb),
            numEdges(_numEdges),
			numVertices(_numVertices),
            totalRefTime(_totalRefTime),
            iterationsTime(_iterationsTime),
            totalIterations(_totalIterations),
            lpIterations(_lpIterations) {}

	};

private:
	int numCoarseLevels = 0;
	std::vector<CoarseLevel> coarseLevels;
    double imb_ratio = 0;
    scalar_t fine_ec = 0;
    scalar_t max_part_cut = 0;
    scalar_t largest_part = 0;
    scalar_t smallest_part = 0;
    int64_t obj = 0;

public:
	ExperimentLoggerUtil() :
		measurements(static_cast<int>(Measurement::END), 0.0)
	{}

	void addCoarseLevel(CoarseLevel cl) {
		coarseLevels.push_back(cl);
		numCoarseLevels++;
	}

	void setFinestEdgeCut(scalar_t finestEdgeCut) {
		this->fine_ec = finestEdgeCut;
	}

    void setMaxPartCut(scalar_t x){
        this->max_part_cut = x;
    }
	
    void setFinestImbRatio(double _imb_ratio) {
		this->imb_ratio = _imb_ratio;
	}

    void setObjective(int64_t x){
        this->obj = x;
    }

    void setLargestPartSize(scalar_t x){
        this->largest_part = x;
    }

    void setSmallestPartSize(scalar_t x){
        this->smallest_part = x;
    }

	void addMeasurement(Measurement m, double val) {
		measurements[static_cast<int>(m)] += val;
	}

	double getMeasurement(Measurement m) {
		return measurements[static_cast<int>(m)];
	}

	void log(char* filename, bool first, bool last) {
		std::ofstream f;
		f.open(filename, std::ios::app);

		if (f.is_open()) {
			if (first) {
				f << "[";
			}
			f << "{";
            f << "\"edge-cut\":" << std::fixed << fine_ec << ",";
            f << "\"max-part-cut\":" << max_part_cut << ",";
            f << "\"objective\":" << obj << ",";
			f << "\"imbalance-ratio\":" << imb_ratio << ',';
			for (int i = 0; i < static_cast<int>(Measurement::END); i++) {
				f << "\"" << measurementNames[i] << "-duration-seconds\":" << measurements[i] << ",";
			}
			f << "\"number-coarse-levels\":" << numCoarseLevels << ",";
            f << "\"finest-refinement-duration-seconds\":" << coarseLevels.back().totalRefTime;
			f << "}";
			if (!last) {
				f << ",";
			}
			else {
				f << "]";
			}
			f.close();
		}
		else {
			std::cerr << "Could not open " << filename << std::endl;
		}
	}

    void verboseReport(){
        std::cout << "Final cut: " << std::fixed << fine_ec;
        std::cout << "; Max part cut: " << std::fixed << max_part_cut;
        std::cout << "; imb: " << imb_ratio;
        std::cout << "; largest: " << largest_part << "; smallest: " << smallest_part << std::endl;
        std::cout << std::setprecision(5);
        std::cout << "Coarsening time: " << getMeasurement(Measurement::Coarsen) << std::endl;
        std::cout << " - Coarsening aggregation time: " << getMeasurement(Measurement::Map) << std::endl;
        std::cout << " - Coarsening contraction time: " << getMeasurement(Measurement::Build) << std::endl;
        std::cout << "Initial partitioning time: " << getMeasurement(Measurement::InitPartition) << std::endl;
        std::cout << "Uncoarsening time: " << getMeasurement(Measurement::Refine) << std::endl;
        std::cout << "Coarse graph free time: " << getMeasurement(Measurement::FreeGraph) << std::endl;
        std::cout << "Total Partitioning Time: " << getMeasurement(Measurement::Total) << std::endl;
        std::cout << "Comm size: " << obj << std::endl;
    }

    void refinementReport(){
        std::cout << std::setprecision(6);
        std::cout << std::left << std::setw(6) << "Level" << std::setw(16) << "Edge Cut" << std::setw(10) << "Imbalance";
        std::cout << std::setw(13) << "Vertices" << std::setw(16) << "Edges" << std::setw(22) << "Total Refinement Time";
        std::cout << std::setw(17) << "Total Iterations" << std::setw(14) << "LP Iterations" << std::setw(23) << "Average Iteration Time" << std::endl;
        for(size_t i = 0; i < coarseLevels.size(); i++){
            CoarseLevel cl = coarseLevels[i];
            int level = coarseLevels.size() - 1 - i;
            std::cout << std::fixed << std::left << std::setw(6) << level << std::setw(16) << cl.edge_cut << std::setw(10) << cl.imb;
            std::cout << std::setw(13) << cl.numVertices << std::setw(16) << cl.numEdges << std::setw(22) << cl.totalRefTime;
            std::cout << std::setw(17) << cl.totalIterations << std::setw(14) << cl.lpIterations << std::setw(23) << (cl.iterationsTime / cl.totalIterations) << std::endl;
        }
    }
};

}
