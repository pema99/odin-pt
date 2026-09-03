import geometry;
import material;
import common;
import sampling;
import spectrum;
import bsdf;

public enum NEEMode
{
    MIS,
    LightSampling,
    BSDFSampling,  
};

public struct LightSample
{
    public float3 emission;
    public float density; // in solid angle measure
    public float3 direction_to_light;

    public bool is_valid()
    {
        return any(emission > 0);
    }
};

public LightSample sample_emissive_triangle(
    float3 shading_point,
    float3 shading_geometric_normal,
    float2 pick_rng,
    float2 point_rng)
{
    LightSample light_sample = (LightSample)0;

    if (emissive_instance_count == 0)
        return light_sample;

    // Pick triangle
    const uint emissive_instance_index = uint(min(emissive_instance_count * pick_rng.x, emissive_instance_count - 1));
    const uint instance_index = emissive_instance_indices[emissive_instance_index];
    const GeometryInfo geo = instance_to_pool[instance_index];
    const uint triangle_count = geo.index_count / 3;
    const uint triangle_index = uint(min(triangle_count * pick_rng.y, triangle_count - 1));

    // Load triangle
    const float3x4 transform = transforms[instance_index];
    const uint i0 = geo.vertex_offset + indices[geo.index_offset + triangle_index * 3 + 0];
    const uint i1 = geo.vertex_offset + indices[geo.index_offset + triangle_index * 3 + 1];
    const uint i2 = geo.vertex_offset + indices[geo.index_offset + triangle_index * 3 + 2];
    const float3 p0 = mul(transform, float4(positions[i0], 1.0));
    const float3 p1 = mul(transform, float4(positions[i1], 1.0));
    const float3 p2 = mul(transform, float4(positions[i2], 1.0));
    const float2 uv0 = uvs[i0];
    const float2 uv1 = uvs[i1];
    const float2 uv2 = uvs[i2];

    // Pick a point on it
    float2 sq_point = point_rng;
    square_to_triangle(sq_point.x, sq_point.y);
    const float2 uv = (1.0 - sq_point.x - sq_point.y) * uv0 + sq_point.x * uv1 + sq_point.y * uv2;

    // Read emissive value
    const float3 emission = get_material_emission(geo.material_index, uv);
    if (all(emission == 0.0))
        return light_sample;

    const float3 light_point = (1.0 - sq_point.x - sq_point.y) * p0 + sq_point.x * p1 + sq_point.y * p2;

    // Calculate some geometry..
    const float3 light_to_shading = shading_point - light_point;
    const float distance_to_light = length(light_to_shading);
    const float3 light_direction = light_to_shading * rcp(distance_to_light);
    const float4 normal_and_area = triangle_normal_and_area(p0, p1, p2);
    const float triangle_area = normal_and_area.w;

    // Facing check
    float3 geometric_normal = normal_and_area.xyz;
    float cos_theta_light = dot(geometric_normal, light_direction);
    if (cos_theta_light < 0.0)
    {
        geometric_normal = -geometric_normal;
        cos_theta_light = -cos_theta_light;
    }

    // Shadow ray
    const float3 shadow_ray_origin = apply_normal_bias(light_point, geometric_normal);
    const float3 shadow_ray_target = apply_normal_bias(shading_point, shading_geometric_normal);
    const float3 shadow_ray_vector = shadow_ray_target - shadow_ray_origin;
    const float shadow_ray_length = length(shadow_ray_vector);
    if (trace_shadow_ray(shadow_ray_origin, shadow_ray_vector * rcp(shadow_ray_length), shadow_ray_length))
        return light_sample;

    float density = emissive_triangle_density(triangle_count, triangle_area, distance_to_light, cos_theta_light);

    // Fill out result
    light_sample.emission = emission;
    light_sample.density = density;
    light_sample.direction_to_light = -light_direction;

    return light_sample;
}

// density of sampling a point on an emissive triangle
public float emissive_triangle_density(
    uint triangle_count,
    float triangle_area, 
    float distance_to_light,
    float cos_theta_light)
{
    // dω = dA |cos(theta_l)| / dist^2
    // p_A dA = p_ω dω
    //
    // p_ω = p_A dA/dω = p_A dA/(dA |cos(theta_l)| / dist^2)
    //                 = p_A 1/(|cos(theta_l)| / dist^2)
    //                 = p_A (dist^2/|cos(theta_l)|)
    const float area_to_solid_angle_jacobian = (distance_to_light * distance_to_light) * rcp(cos_theta_light);

    const float light_pick_density = rcp(emissive_instance_count * triangle_count);
    const float point_pick_density = rcp(triangle_area);
    const float density = light_pick_density * point_pick_density * area_to_solid_angle_jacobian;

    return density;
}

public float mis_weight_light(NEEMode mode, float light_density, float bsdf_density)
{
    [forcecase] switch (mode)
    {
        case NEEMode.MIS:
            return power_heuristic(light_density, bsdf_density);
        case NEEMode.LightSampling:
            return 1;
        default:
            return 0;
    }
}

public float mis_weight_bsdf(NEEMode mode, float bsdf_density, float light_density)
{
    [forcecase] switch (mode)
    {
        case NEEMode.MIS:
            return power_heuristic(bsdf_density, light_density);
        case NEEMode.LightSampling:
            return 0;
        default:
            return 1;
    }
}