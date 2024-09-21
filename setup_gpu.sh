# This is an example script to build our code and its dependencies for an ubuntu system with H100 GPU
loc=$PWD/container
ipwd=$PWD
# need to use nvcc_wrapper to compile kokkos + kokkos-kernels
compiler=$loc/kokkos/bin/nvcc_wrapper
# Ampere86 is the closest arch option in kokkos for H100
arch=Ampere86
mkdir container
cd $loc
git clone https://github.com/kokkos/kokkos.git
git clone https://github.com/kokkos/kokkos-kernels.git
git clone https://github.com/KarypisLab/GKlib.git
git clone https://github.com/KarypisLab/METIS.git
mkdir install
cd kokkos
git checkout master
cd $loc/kokkos-kernels
git checkout master
mkdir build
cd build
# kokkos + kokkos-kernels build
./../cm_generate_makefile.bash \
--kokkos-path=$loc/kokkos --kokkoskernels-path=$loc/kokkos-kernels \
--release --kokkos-release \
--kokkos-prefix=$loc/install/kokkos \
--prefix=$loc/install/kokkos-kernels  --disable-tests --disable-perftests --disable-examples --no-default-eti \
--with-cuda --arch=$arch --compiler=$compiler \
--with-openmp --with-serial --enable-cuda-lambda
#if your kokkos-kernel build fails, it is probably due to running out of memory
#in this case you will need to change the -j to -j16 or something small
make install -j
cd $loc/GKlib
# gklib build
make config prefix=$loc/local
make install -j
cd $loc/METIS
# metis build
make config prefix=$loc/local
make install -j
cd $ipwd
mkdir build
cd build
#you need to have cmake version at least 3.18 for our build
cmake $ipwd -DCMAKE_PREFIX_PATH=$loc/install/kokkos-kernels/lib/cmake/KokkosKernels \
-DCMAKE_BUILD_TYPE=Release -DLINK_GKLIB=True -DMETIS_DIR="$loc/local"
make -j
make -j