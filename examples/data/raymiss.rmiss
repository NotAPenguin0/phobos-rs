#version 460

#extension GL_EXT_ray_tracing : require

struct Payload {
    vec3 hit_value;
};

layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload.hit_value = vec3(0.0, 1.0, 0.0);
}