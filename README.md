# A K-Way Graph Partitioner for the GPU

The Jet Partitioner is a parallel graph partitioner that runs on most CPU and GPU systems (via Kokkos, a required dependency).
This partitioner was developed in a collaboration between Sandia National Labs and Pennsylvania State University.
For details about the algorithm, please see https://arxiv.org/abs/2304.13194  
If you intend to cite this partitioner in a publication, please cite https://doi.org/10.1137/23M1559129

## Dependencies

Kokkos (https://github.com/kokkos/kokkos): Enables performance portable parallelism.  
KokkosKernels (https://github.com/kokkos/kokkos-kernels): Necessary only for KokkosSparse::CrsMatrix class.  
Metis (https://github.com/KarypisLab/METIS): Used for initial partitioning of coarsest graph.  
(Circumstantial) GKLib (https://github.com/KarypisLab/GKlib.git): Needed to link against the github distribution of Metis. Not needed for older distributions of Metis.

## Usage

### Building

Standard CMake build process. CMake version >= 3.23 required. If your Metis build requires GKlib, add `-DLINK_GKLIB=True` to your cmake command when building Jet. You can add `-DMETIS_HINT=<metis install location>` to help cmake find metis (and gklib if it has the same install location) if you have not added metis to your relevant path variables. Example build scripts are provided for macOS with OpenMP and Linux systems with Cuda. These scripts handle all required dependencies.

### Executables

#### Partitioners
Each partitioner executable requires 2 parameters. The first is a graph file in metis format, the second is a config file. Multiple sample config files are provided in the "configs" directory. Optionally, a third parameter can be used to specify an output file for the partition, and a fourth parameter for runtime statistics in JSON format.  
Although the partitioner itself supports weighted edges and vertices, the import method currently does not support weighted vertices.  
jet: The primary partitioner exe. Coarsening algorithm can be set in config file. Runs on the default device.  
jet\_host: jet but runs on the host device.  
jet\_serial: jet but runs on the host on a single thread.

#### Helpers
pstat: Given a metis graph file, partition file, and k-value, will print out quality information on the partition.

### Using Jet Partitioner in Your Code
We provide a cmake package that you can install on your system. Add `find_package(jet CONFIG REQUIRED)` to your project's CMakeLists.txt file and link your executable/s to `jet::jet`. Include `jet.h` in your code to use one of the provided partitioning functions. Each function is distinguished by the target Kokkos execution space it will run in and the type of KokkosKernels CrsMatrix which it accepts. Reference `jet_defs.h` for the relevant template definitions of these parameters. You can set the desired part count and imbalance values on the input config_t struct (see `jet_config.h` for other parameters).

#### Tips
On Linux systems, you can create a file `~/.cmake/packages/jet/find.txt` that cmake will automatically use to find the jet cmake package.
Inside this file, add the full path to the jet install directory.

### Input Format
The partitioner executables accept graphs stored in the metis graph file format. We do not yet support vertex weights within metis graph files.

### Config File format:  
\<Coarsening algorithm\> (0 for 2-hop matching)/(1 for HEC)/(2 for pure matching)/(default is 2-hop matching)  
\<Number of parts\>  
\<Partitioning attempts\>  
\<Imbalance value\>  
\<Ultra quality settings\> (Optional) 1 to enable, 0 to disable (default)
