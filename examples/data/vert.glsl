#version 450

layout(location = 0) out vec2 UV;

// Vertices for fullscreen quad
vec4 vertices[] = {
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -1.0, 0.0, 0.0),
    vec4(1.0, -1.0, 1.0, 0.0),
    vec4(-1.0, 1.0, 0, 1.0),
    vec4(1.0, -1.0, 1.0, 0.0),
    vec4(1.0, 1.0, 1.0, 1.0)
};

void main() {
    UV = vertices[gl_VertexIndex].zw;
    gl_Position = vec4(vertices[gl_VertexIndex].xy, 0.0, 1.0);
}