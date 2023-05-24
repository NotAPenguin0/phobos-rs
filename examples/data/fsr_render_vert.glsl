#version 460

const vec3 vertices[] = {
vec3(-0.5f, -0.5f, -0.5f),
vec3(-0.5f, -0.5f, 0.5f),
vec3(-0.5f, 0.5f, 0.5f),
vec3(0.5f, 0.5f, -0.5f),
vec3(-0.5f, -0.5f, -0.5f),
vec3(-0.5f, 0.5f, -0.5f),
vec3(0.5f, -0.5f, 0.5f),
vec3(-0.5f, -0.5f, -0.5f),
vec3(0.5f, -0.5f, -0.5f),
vec3(0.5f, 0.5f, -0.5f),
vec3(0.5f, -0.5f, -0.5f),
vec3(-0.5f, -0.5f, -0.5f),
vec3(-0.5f, -0.5f, -0.5f),
vec3(-0.5f, 0.5f, 0.5f),
vec3(-0.5f, 0.5f, -0.5f),
vec3(0.5f, -0.5f, 0.5f),
vec3(-0.5f, -0.5f, 0.5f),
vec3(-0.5f, -0.5f, -0.5f),
vec3(-0.5f, 0.5f, 0.5f),
vec3(-0.5f, -0.5f, 0.5f),
vec3(0.5f, -0.5f, 0.5f),
vec3(0.5f, 0.5f, 0.5f),
vec3(0.5f, -0.5f, -0.5f),
vec3(0.5f, 0.5f, -0.5f),
vec3(0.5f, -0.5f, -0.5f),
vec3(0.5f, 0.5f, 0.5f),
vec3(0.5f, -0.5f, 0.5f),
vec3(0.5f, 0.5f, 0.5f),
vec3(0.5f, 0.5f, -0.5f),
vec3(-0.5f, 0.5f, -0.5f),
vec3(0.5f, 0.5f, 0.5f),
vec3(-0.5f, 0.5f, -0.5f),
vec3(-0.5f, 0.5f, 0.5f),
vec3(0.5f, 0.5f, 0.5f),
vec3(-0.5f, 0.5f, 0.5f),
vec3(0.5f, -0.5f, 0.5)
};

layout(set = 0, binding = 0) uniform Data {
    mat4 transform;
    mat4 view;
    mat4 projection;
    mat4 previous_matrix;
} data;

layout(location = 0) out vec4 ClipPos;
layout(location = 1) out vec4 PrevClipPos;

void main() {
    vec4 position = vec4(vertices[gl_VertexIndex], 1.0);
    ClipPos = data.projection * data.view * data.transform * position;
    PrevClipPos = data.previous_matrix * position;
    gl_Position = ClipPos;
}