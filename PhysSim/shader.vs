#version 330 core
uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
layout (location = 0) in vec3 current;
layout (location = 1) in vec3 next;

out vec3 ourColor;

void main()
{
    vec3 delta = next - current;
    gl_Position = projection * view * model * vec4(current + (delta * time), 1.0);
    ourColor = vec3(0.0, 0.0, 0.0);
}