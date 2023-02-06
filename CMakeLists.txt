cmake_minimum_required(VERSION 3.10)

project(PhysSim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_FIND_DEBUG_MODE ON)

find_package(cxxopts CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)

include(cmake/simulator.cmake)
include(cmake/replayer.cmake)
include(cmake/pcat.cmake)