cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(PhysSim LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package(CUDAToolkit)
find_package(yaml-cpp)
find_package(spdlog REQUIRED)

include_directories(include "/usr/local/cuda-11.7/include/")

# TODO/bmoody Not used yet, need to add support for replayer
file(GLOB_RECURSE
    sources
    src/*
    include/*)

add_executable(simulator
    include/Paths.h
    include/Types.h
    include/Simulate.cuh
    include/Operations.cuh
    src/InitSimulator.cpp
    src/Paths.cpp
    src/Types.cpp
    src/Core.cpp
    src/Simulate.cu
    src/Operations.cu
)

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE .)

target_link_libraries(simulator PUBLIC
    yaml-cpp
    spdlog::spdlog
)