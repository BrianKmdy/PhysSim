
project(Replayer LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# add_executable(test test.cpp)

# TODO/bmoody Not used yet, need to add support for replayer
# file(GLOB_RECURSE
#     sources
#     src/*
#     include/*)

include_directories(include)
add_executable(replayer
    include/Paths.h
    include/Types.h
    include/Replayer.h
    src/InitReplayer.cpp
    src/Paths.cpp
    src/Types.cpp
    src/Replayer.cpp
)
target_link_libraries(replayer ${CONAN_LIBS})

# add_library(cxxopts INTERFACE)
# target_include_directories(cxxopts INTERFACE .)
# 
# target_link_libraries(replayer PUBLIC
#     yaml-cpp
#     spdlog::spdlog
# )