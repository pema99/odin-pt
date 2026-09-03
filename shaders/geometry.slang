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
    public float3x4 world_to_object;

    public bool is_valid()
    {
        return instance_id != MAX_UINT;
    }
}

public struct HitAttributes
{
    public float3 position;
    public float3 normal;
    public float3 geometric_normal;
    public float3 tangent;
    public float2 uv;
    public uint material_index;
    public uint triangle_count;
    public float triangle_area;
}

public Hit trace_ray(float3 ray_origin, float3 ray_direction)
{
    RayDesc primary_ray;
    primary_ray.Origin = ray_origin;
    primary_ray.Direction = ray_direction;
    primary_ray.TMin = 0.0;
    primary_ray.TMax = 1e30;

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, primary_ray);
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
        hit.world_to_object = query.CommittedWorldToObject3x4();
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

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, occlusion_ray);
    query.Proceed();

    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

public HitAttributes get_hit_attributes(Hit hit, float3 ray_origin, float3 ray_direction)
{
    HitAttributes attributes = (HitAttributes)0;

    GeometryInfo geo = instance_to_pool[hit.instance_id];
    uint base = geo.index_offset + hit.primitive_index * 3;
    uint i0 = geo.vertex_offset + indices[base + 0];
    uint i1 = geo.vertex_offset + indices[base + 1];
    uint i2 = geo.vertex_offset + indices[base + 2];
    float3 n0 = normals[i0];
    float3 n1 = normals[i1];
    float3 n2 = normals[i2];

    attributes.normal = normalize(
        n0 * (1.0 - hit.barycentrics.x - hit.barycentrics.y)
        + n1 * hit.barycentrics.x 
        + n2 * hit.barycentrics.y);

    float3x3 to_world = (float3x3)hit.object_to_world;
    attributes.normal = normalize(mul(transpose((float3x3)hit.world_to_object), attributes.normal));

    float3 p0 = positions[i0];
    float3 p1 = positions[i1];
    float3 p2 = positions[i2];
    float4 normal_and_area = triangle_normal_and_area(mul(to_world, p1 - p0), mul(to_world, p2 - p0));
    attributes.geometric_normal = normal_and_area.xyz;
    if (determinant(to_world) < 0.0)
        attributes.geometric_normal = -attributes.geometric_normal;

    // flip normals if they are pointing the wrong way
    if (dot(attributes.geometric_normal, ray_direction) > 0.0)
        attributes.geometric_normal = -attributes.geometric_normal;
    if (dot(attributes.normal, attributes.geometric_normal) < 0.0)
        attributes.normal = -attributes.normal;
    float3 t0 = tangents[i0];
    float3 t1 = tangents[i1];
    float3 t2 = tangents[i2];
    attributes.tangent = mul(to_world,
        t0 * (1.0 - hit.barycentrics.x - hit.barycentrics.y)
        + t1 * hit.barycentrics.x
        + t2 * hit.barycentrics.y);
    attributes.tangent = normalize(attributes.tangent - attributes.normal * dot(attributes.normal, attributes.tangent));

    float2 uv0 = uvs[i0];
    float2 uv1 = uvs[i1];
    float2 uv2 = uvs[i2];
    attributes.uv =
        uv0 * (1.0 - hit.barycentrics.x - hit.barycentrics.y)
        + uv1 * hit.barycentrics.x
        + uv2 * hit.barycentrics.y;

    attributes.position = hit.distance * ray_direction + ray_origin;

    attributes.material_index = geo.material_index;

    // needed for MIS
    attributes.triangle_count = geo.index_count / 3;
    attributes.triangle_area = normal_and_area.w;

    return attributes;
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