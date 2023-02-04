include_directories(include)

add_executable(pcat
    src/Pcat.cpp
    include/Paths.h
    include/Types.h
    src/Types.cpp
)

target_link_libraries(pcat
    PRIVATE
        cxxopts::cxxopts
        yaml-cpp
        spdlog::spdlog spdlog::spdlog_header_only
)