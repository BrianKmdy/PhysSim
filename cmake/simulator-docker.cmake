
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit)
enable_language(CUDA)

if(UNIX)
    include_directories(include "/usr/local/cuda-11.7/include/")
endif()

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

target_link_libraries(simulator ${CONAN_LIBS})