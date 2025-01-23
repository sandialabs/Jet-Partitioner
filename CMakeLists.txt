cmake_minimum_required(VERSION 3.23)
project(jetpartition LANGUAGES CXX VERSION 1.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# both the apps and the library depend on kokkos and kokkoskernels
# find KokkosKernels also handles kokkos
find_package(KokkosKernels REQUIRED)
add_compile_options(-Wall -Wextra -Wshadow)
add_subdirectory(src)
add_subdirectory(app)