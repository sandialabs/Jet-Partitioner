# A K-Way Graph Partitioner for the GPU

The Jet Partitioner is a parallel graph partitioner that runs on most CPU and GPU systems (via Kokkos, a required dependency).
This partitioner was developed in a collaboration between Sandia National Labs and Pennsylvania State University.
For details about the algorithm, please see https://arxiv.org/abs/2304.13194

## Dependencies

Kokkos (https://github.com/kokkos/kokkos): Enables performance portable parallelism.  
KokkosKernels (https://github.com/kokkos/kokkos-kernels): Necessary only for KokkosSparse::CrsMatrix class.  
Metis (https://github.com/KarypisLab/METIS): Used for initial partitioning of coarsest graph.

## Usage

### Executables

#### Partitioners
Each partitioner executable requires 2 parameters. The first is a graph file in metis format, the second is a config file. Multiple sample config files are provided in the "configs" directory. Optionally, a third parameter can be used to specify an output file for the partition, and a fourth parameter for runtime statistics in JSON format.  
Although the partitioner itself supports weighted edges and vertices, the import method currently does not support weighted vertices.  
jet: The primary partitioner exe. Coarsening algorithm can be set in config file. Runs on the default device.  
jet\_host: jet but runs on the host device.  
jet\_serial: jet but runs on the host on a single thread.

#### Helpers
pstat: Given a metis graph file, partition file, and k-value, will print out quality information on the partition.

### Using Jet in Your Code
We can not provide an option to compile a library due to the use of templates. However, you can import "jet.hpp" into your code to use the partitioner via the "jet\_partitioner::partition" method. Note that this requires you to add our source directory to your include path and also to link our dependencies. This method currently always uses the default coarsening algorithm.

### Input Format
We do not yet support vertex weights within metis graph files.

### Config File format:  
\<Coarsening algorithm\> (0 for 2-hop matching)/(1 for HEC)/(2 for pure matching)/(default is 2-hop matching)  
\<Number of parts\>  
\<Partitioning attempts\>  
\<Imbalance value\>  
