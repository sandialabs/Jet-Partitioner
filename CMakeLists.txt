cmake_minimum_required(VERSION 3.18)
project(jetpartition CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# GKlib is a dependency of Metis when built from its github repo
# Unfortunately there is no simple way to avoid also linking GKlib in this case
# Older distributions of metis do not create this dependency
SET(LINK_GKLIB False CACHE BOOL "Newer Metis distributions require us to link GKlib")

find_package(KokkosKernels REQUIRED)
add_compile_options(-Wall -Wextra -Wshadow)

# This is used by the build script
# to avoid putting metis and gklib in the global path
include_directories(${METIS_DIR}/include)
link_directories(${METIS_DIR}/lib)

add_executable(jet partition.cpp)
add_executable(jet4 partition.cpp)
add_executable(jet2 partition.cpp)
add_executable(jet_host partition.cpp)
add_executable(jet_import import_coarse.cpp)
add_executable(jet_export partition.cpp)
add_executable(jet_serial partition.cpp)
add_executable(pstat part_eval.cpp)

target_compile_definitions(jet PUBLIC HASHMAP_P)
target_compile_definitions(jet4 PUBLIC HASHMAP_P FOUR9)
target_compile_definitions(jet2 PUBLIC HASHMAP_P TWO9)
target_compile_definitions(jet_host PUBLIC HASHMAP_P HOST)
target_compile_definitions(jet_import PUBLIC HASHMAP_P HOST)
target_compile_definitions(jet_export PUBLIC HASHMAP_P HOST EXP)
target_compile_definitions(jet_serial PUBLIC HASHMAP_P SERIAL)
foreach(prog jet jet4 jet2 jet_host jet_import jet_export jet_serial pstat)
    target_link_libraries(${prog} Kokkos::kokkos Kokkos::kokkoskernels)
endforeach(prog)
foreach(prog jet jet4 jet2 jet_host jet_export jet_serial)
    target_link_libraries(${prog} metis)
endforeach(prog)
if(LINK_GKLIB)
foreach(prog jet jet4 jet2 jet_host jet_export jet_serial)
    target_link_libraries(${prog} GKlib)
endforeach(prog)
endif()