#version 330 core
uniform float time;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
layout (location = 0) in vec3 current;
layout (location = 1) in vec3 next;

void main()
{
    vec3 delta = next - current;
    vec3 pos = current + delta * time;
    gl_Position = projection * view * model * vec4(pos, 1.0);
}