import common;

public struct GeometryInfo
{
    public uint index_offset;
    public uint index_count;
    public uint vertex_offset;
    public uint material_index;
};

public StructuredBuffer<GeometryInfo> instance_to_pool;
public StructuredBuffer<float3x4> transforms;
public StructuredBuffer<float3> positions;
public StructuredBuffer<float3> normals;
public StructuredBuffer<float3> tangents;
public StructuredBuffer<float2> uvs;
public StructuredBuffer<uint> indices;
public StructuredBuffer<uint> emissive_instance_indices;
public uniform uint emissive_instance_count;

RaytracingAccelerationStructure scene;

public struct Hit
{
    public uint instance_id;
    public uint primitive_index;
    public float2 barycentrics;
    public float distance;
    public bool is_front_face;
    public float3x4 object_to_world;

    public bool is_valid()
    {
        return instance_id != MAX_UINT;
    }
}

public Hit trace_ray(float3 ray_origin, float3 ray_direction)
{
    RayDesc primary_ray;
    primary_ray.Origin = ray_origin;
    primary_ray.Direction = ray_direction;
    primary_ray.TMin = 0.0;
    primary_ray.TMax = 1e30;

    RayQuery<RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
    query.TraceRayInline(scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, primary_ray);
    query.Proceed();

    Hit hit = (Hit)0;
    hit.instance_id = MAX_UINT;

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit.instance_id = query.CommittedInstanceID();
        hit.primitive_index = query.CommittedPrimitiveIndex();
        hit.barycentrics = query.CommittedTriangleBarycentrics();
        hit.distance = query.CommittedRayT();
        hit.is_front_face = query.CommittedTriangleFrontFace();
        hit.object_to_world = query.CommittedObjectToWorld3x4();
    }
    return hit;
}

// Returns true if there is occlusion
public bool trace_shadow_ray(
    float3 ray_origin,
    float3 ray_direction,
    float max_distance)
{
    RayDesc occlusion_ray;
    occlusion_ray.Origin = ray_origin;
    occlusion_ray.Direction = ray_direction;
    occlusion_ray.TMin = 0.0;
    occlusion_ray.TMax = max_distance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> query;
    query.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, occlusion_ray);
    query.Proceed();

    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

// RayTracing Gems, A Fast and Robust Method for Avoiding Self-Intersection
public float3 apply_normal_bias(float3 position, float3 normal)
{
    float3 a = abs(position);
    float distance_to_origin = max(a.x, max(a.y, a.z));
    float offset = (distance_to_origin < 1.0 / 32.0) ?
        1.192092896e-07 * 64.0 :
        1.192092896e-07 * 64.0 * distance_to_origin;
    return position + offset * normal;
}

// From cycles https://github.com/blender/cycles/blob/main/src/kernel/closure/bsdf_util.h
// Fixes the case where the shading normal results in specular reflection in the lower hemisphere
public float3 ensure_valid_specular_normal(float3 geometric_normal, float3 view, float3 normal)
{
    if (all(geometric_normal == normal))
        return normal;

    float3 reflection = 2.0 * dot(normal, view) * normal - view;
    float iz = dot(view, geometric_normal);
    float threshold = min(0.9 * iz, 0.01);
    if (dot(geometric_normal, reflection) >= threshold)
        return normal;

    float3 x = normal - dot(normal, geometric_normal) * geometric_normal;
    float x_length = length(x);
    x = x_length > 0.0 ? x / x_length : normal;

    float ix = dot(view, x);
    float a = ix * ix + iz * iz;
    float b = 2.0 * (a + iz * threshold);
    float c = (threshold + iz) * (threshold + iz);
    float discriminant = sqrt(max(b * b - 4.0 * a * c, 0.0));
    float n_z2 = 0.25 * (ix < 0.0 ? b + discriminant : b - discriminant) / a;

    return sqrt(max(1.0 - n_z2, 0.0)) * x + sqrt(max(n_z2, 0.0)) * geometric_normal;
}

public float4 triangle_normal_and_area(float3 v1v0, float3 v2v0)
{
    float3 cross_product = cross(v1v0, v2v0);
    float cross_length = length(cross_product);
    return float4(cross_product * rcp(cross_length), 0.5 * cross_length);
}

public float4 triangle_normal_and_area(float3 v0, float3 v1, float3 v2)
{
    return triangle_normal_and_area(v1 - v0, v2 - v0);
}

public void ortho_basis_from_vector(float3 n, out float3 b1, out float3 b2)
{
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = float3(b, sign + n.y * n.y * a, -n.y);
}

public float3x3 ortho_basis_from_vector(float3 n)
{
    float3 t, b;
    ortho_basis_from_vector(n, t, b);
    return transpose(float3x3(t, b, n));
}

public float3 cosine_hemisphere(float2 u, float3 normal)
{
    float a = sqrt(u.x);
    float b = 2.0f * PI * u.y;
    float3 localDir = float3(a * cos(b), a * sin(b), sqrt(1.0f - u.x));
    float3x3 basis = ortho_basis_from_vector(normal);
    return mul(basis, localDir);
}

// A Low-Distortion Map Between Triangle and Square, Heitz, 2019
// https://eheitzresearch.wordpress.com/749-2/
public void square_to_triangle(inout float x, inout float y)
{
    if (y > x) {
        x *= 0.5;
        y -= x;
    } else {
        y *= 0.5;
        x -= y;
    }
}

public float3 primary_ray_direction(
    float2 pixel,
    uint2 screen_size,
    float cam_pitch,
    float cam_yaw,
    float cam_fov)
{
    float2 uv = (pixel / float2(screen_size)) * 2.0 - 1.0;
    uv.y = -uv.y;
    float aspect = float(screen_size.x) / float(screen_size.y);
    float3 forward = float3(cos(cam_pitch) * sin(cam_yaw), sin(cam_pitch), -cos(cam_pitch) * cos(cam_yaw));
    float3 right = normalize(cross(forward, float3(0, 1, 0)));
    float3 up = cross(right, forward);
    float fov_scale = tan(cam_fov * 0.5 * PI / 180.0);
    return normalize(forward + (uv.x * aspect * right + uv.y * up) * fov_scale);
}