#version 450

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer writeonly Output {
    float data[];
} outbuf;

void main() {
    outbuf.data[gl_GlobalInvocationID.x] = float(gl_GlobalInvocationID.x);
}