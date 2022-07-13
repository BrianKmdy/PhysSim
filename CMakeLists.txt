cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(PhysSim)

find_library(yaml-cpp yaml-cpp)
message(STATUS "foo include dir: ${lib}")

add_executable(ps test.cc)
target_link_libraries(ps PUBLIC yaml-cpp)