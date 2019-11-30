#version 330 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
layout (location = 0) in vec3 aPos;

out vec3 ourColor;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = vec3(0.0, 0.0, 0.0);
}