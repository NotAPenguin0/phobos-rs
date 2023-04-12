#version 460

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

layout(location = 0) in vec2 UV;
layout(location = 0) out vec4 FragColor;

layout(set = 0, binding = 0) uniform accelerationStructureEXT as;

layout(push_constant) uniform PC {
    mat4 view;
    mat4 projection;
} pc;

float intersect(vec3 origin, vec3 direction) {
    rayQueryEXT ray;

    rayQueryInitializeEXT(ray, as, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.001, direction, 10.0);

    rayQueryProceedEXT(ray);

    if (rayQueryGetIntersectionTypeEXT(ray, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
        return 1.0;
    }
    else return 0.0;
}

// walmart raygen shader
void main() {
    // Apply inverse of view and projection to find camera direction
    // This gives us the origin and direction of the ray
    vec4 origin = inverse(pc.view) * vec4(0, 0, 0, 1);
    vec2 normalized_uv = UV * 2.0 - 1.0;
    vec4 target = inverse(pc.projection) * vec4(normalized_uv.x, normalized_uv.y, 1, 1);
    vec4 direction = normalize(inverse(pc.view) * vec4(normalize(target.xyz), 0));
    FragColor = vec4(intersect(origin.xyz, direction.xyz).xxx, 1.0);
}