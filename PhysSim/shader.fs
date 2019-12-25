#version 330 core
out vec4 FragColor;

in vec4 vertexPosition;

void main()
{
    float dist = distance(gl_PointCoord.xy, vec2(0.5f, 0.5f));
    float alpha = 0.0f;
    if (dist < 0.1f)
        alpha = 1.0f;
    else if (dist < 0.5f)
        alpha = max(1.0f - (dist - 0.1f) * 3, 0.0f);
    else
        discard;
    FragColor = vec4(1.0f, 1.0f, 1.0f, alpha);
}