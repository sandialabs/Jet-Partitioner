# jet executables
add_executable(jet_ex driver.cpp)
set_property(TARGET jet_ex PROPERTY OUTPUT_NAME jet)
add_executable(jet4 driver.cpp)
add_executable(jet2 driver.cpp)
add_executable(jet_host driver.cpp)
add_executable(jet_import import_coarse.cpp)
add_executable(jet_export driver.cpp)
add_executable(jet_serial driver.cpp)
add_executable(pstat part_eval.cpp)

foreach(prog jet_import pstat)
    target_include_directories(${prog} PRIVATE ${CMAKE_SOURCE_DIR}/header ${CMAKE_SOURCE_DIR}/src)
endforeach(prog)

# compile definitions to set exe behavior
target_compile_definitions(jet4 PUBLIC FOUR9)
target_compile_definitions(jet2 PUBLIC TWO9)
target_compile_definitions(jet_host PUBLIC HOST)
target_compile_definitions(jet_import PUBLIC HOST)
target_compile_definitions(jet_export PUBLIC HOST EXP)
target_compile_definitions(jet_serial PUBLIC SERIAL)

# link executables
target_link_libraries(jet_import Kokkos::kokkos Kokkos::kokkoskernels)
target_link_libraries(pstat Kokkos::kokkos Kokkos::kokkoskernels)
# other executables get the kokkos dependencies via jet
foreach(prog jet_ex jet4 jet2 jet_host jet_export jet_serial)
    target_link_libraries(${prog} jet)
endforeach(prog)