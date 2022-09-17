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