import geometry;
import scene;
import common;
import geometry;
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
    public float3 shadow_ray_origin;

    public bool is_valid()
    {
        return any(emission > 0);
    }
};

public LightSample sample_emissive_triangle(
    float3 shading_point,
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

    const float4 normal_and_area = triangle_normal_and_area(p0, p1, p2);
    const float triangle_area = normal_and_area.w;
    if (triangle_area <= 0)
        return light_sample;

    // Pick a point on it
    float2 sq_point = point_rng;
    square_to_triangle(sq_point.x, sq_point.y);

    // Read emissive value
    const float3 emission = get_material_emission_fast(geo.material_index, i0, i1, i2, float3(1.0 - sq_point.x - sq_point.y, sq_point.x, sq_point.y));
    if (all(emission == 0.0))
        return light_sample;

    const float3 light_point = (1.0 - sq_point.x - sq_point.y) * p0 + sq_point.x * p1 + sq_point.y * p2;

    // Get light direction
    const float3 light_to_shading = shading_point - light_point;
    const float distance_to_light = length(light_to_shading);
    const float3 light_direction = light_to_shading * rcp(distance_to_light);

    // Facing check
    float3 geometric_normal = normal_and_area.xyz;
    float cos_theta_light = dot(geometric_normal, light_direction);
    if (cos_theta_light < 0.0)
    {
        geometric_normal = -geometric_normal;
        cos_theta_light = -cos_theta_light;
    }

    float density = emissive_triangle_density(triangle_count, triangle_area, distance_to_light, cos_theta_light);

    // Fill out result
    light_sample.emission = emission;
    light_sample.density = density;
    light_sample.direction_to_light = -light_direction;
    light_sample.shadow_ray_origin = apply_normal_bias(light_point, geometric_normal);

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

float balance_heuristic(float density0, float density1)
{
    return 1.0 / (1.0 + density1 / density0);
}

float power_heuristic(float density0, float density1)
{
    float ratio = density1 / density0;
    return 1.0 / (1.0 + ratio * ratio);
}

public float calculate_mis_weight(NEEMode mode, float density0, float density1)
{
    if (mode == NEEMode.MIS)
        return power_heuristic(density0, density1);
    else
        return 1;
}