set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
enable_language(CUDA)
if (UNIX)
    include_directories("/usr/local/cuda-11.7/include/")
endif ()

include_directories(include)

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

target_link_libraries(simulator
    PRIVATE
        cxxopts::cxxopts
        yaml-cpp
        spdlog::spdlog spdlog::spdlog_header_only
)
