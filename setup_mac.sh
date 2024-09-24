# This is an example script to build our code and its dependencies for macOS
# using homebrew clang (Apple clang doesn't support openmp)
loc=$PWD/container
ipwd=$PWD
compiler=/opt/homebrew/opt/llvm/bin/clang++
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
--compiler=$compiler \
--with-openmp --with-serial
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
# requires at least cmake version 3.18
cmake $ipwd -DCMAKE_PREFIX_PATH=$loc/install/kokkos-kernels/lib/cmake/KokkosKernels \
-DCMAKE_CXX_COMPILER=$compiler -DCMAKE_BUILD_TYPE=Release -DLINK_GKLIB=True -DMETIS_DIR="$loc/local"
make -j