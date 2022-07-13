#version 330 core
out vec4 FragColor;

in vec4 vertexPosition;

void main()
{
    float dist = distance(gl_PointCoord.xy, vec2(0.5f, 0.5f));
    float alpha = 0.0f;
    if (dist < 0.5f)
        alpha = 1 - dist * 4;
    else
        discard;
    FragColor = vec4(1.0f, 1.0f, 1.0f, alpha);
}