#version 450

layout(location = 0) in vec2 iPos;
layout(location = 1) in vec2 iUV;

layout(location = 0) out vec2 UV;

void main() {
    UV = iUV;
    gl_Position = vec4(iPos, 0.0, 1.0);
}