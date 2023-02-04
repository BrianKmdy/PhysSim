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


target_link_libraries(replayer
    PRIVATE
        cxxopts::cxxopts
        yaml-cpp
        spdlog::spdlog spdlog::spdlog_header_only
        GLEW::GLEW
        glm::glm
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>
        ${OPENGL_LIBRARIES}
        glfw
)

# add_custom_command(
#     TARGET replayer POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#             ${CMAKE_CURRENT_SOURCE_DIR}/shaders
#             ${CMAKE_CURRENT_BINARY_DIR}/shaders
# )

add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            ${CMAKE_CURRENT_BINARY_DIR}/shaders
)

add_dependencies(replayer copy_shaders)