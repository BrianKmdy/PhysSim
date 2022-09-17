cmake_minimum_required(VERSION 3.10)

project("PhysSim" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

include("cmake/replayer.cmake")
include("cmake/simulator-docker.cmake")