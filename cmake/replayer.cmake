
project(Replayer LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

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