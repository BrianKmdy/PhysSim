#version 330 core
uniform float time;
layout (location = 0) in vec3 progressBar;

void main()
{
    gl_Position = vec4(progressBar * time - vec3(1.0, 0.99, 0.0), 1.0);
}