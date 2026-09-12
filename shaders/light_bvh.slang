import geometry;
import scene;
import common;

struct LightBounds
{
	float3 aabb_min;
	float3 aabb_max;
	float3 direction;
	float cos_cone_angle;
	float power;
	bool double_sided;
}

struct LightBVHNode
{
	LightBounds bounds;
	uint flags;

	uint child_index() { return flags & 0x7FFFFFFF; }
	uint light_index() { return flags & 0x7FFFFFFF; }
	bool is_leaf() { return bool(flags >> 31); }
}

struct EmissiveTriangle
{
	uint instance_index;
	uint primitive_index;
}

StructuredBuffer<EmissiveTriangle> lbvh_lights;
StructuredBuffer<LightBVHNode> lbvh_nodes;
StructuredBuffer<uint> lbvh_light_to_bit_trail;
StructuredBuffer<uint> lbvh_instance_to_light;
uniform uint lbvh_light_count;

// returns cos(max(0, theta_a - theta_b)) without inverse trig
float cos_sub_clamped(float sin_theta_a, float cos_theta_a, float sin_theta_b, float cos_theta_b)
{
    if (cos_theta_a > cos_theta_b)
        return 1;
    return cos_theta_a * cos_theta_b + sin_theta_a * sin_theta_b;
}

// returns sin(max(0, theta_a - theta_b)) without inverse trig
float sin_sub_clamped(float sin_theta_a, float cos_theta_a, float sin_theta_b, float cos_theta_b)
{
    if (cos_theta_a > cos_theta_b)
        return 0;
    return sin_theta_a * cos_theta_b - cos_theta_a * sin_theta_b;
}

// importance estimate for a cluster of lights
// https://github.com/mmp/pbrt-v4/blob/b4ce9687e6c695f5582997c61b0c66cf064bdb4a/src/pbrt/lightsamplers.h#L144
float light_bounds_importance(LightBounds bounds, float3 position, float3 normal)
{
    // get distance to shading point. further away lights are less important
    float3 centroid = (bounds.aabb_min + bounds.aabb_max) * 0.5;
    float3 to_shading = position - centroid;
    float sqr_distance = dot(to_shading, to_shading);
    float3 diagonal = bounds.aabb_max-bounds.aabb_min;
    float bounding_sphere_radius = length(diagonal)*0.5;
    // prevent weight from blowing up when cluster too close
    float sqr_distance_clamped = max(sqr_distance, bounding_sphere_radius);

    // compute cos, sin of angle between emission dir and shading dir,
    // call this angle theta_w. lights pointing away are less important
    float3 to_shading_dir = to_shading * rsqrt(sqr_distance);
    float cos_theta_w = dot(bounds.direction, to_shading_dir);
    if (bounds.double_sided)
        cos_theta_w = abs(cos_theta_w);
    float sin_theta_w = sqrt(max(1.0 - cos_theta_w*cos_theta_w, 0.0));

    // get smallest angle between the direction to shading point and any emitter
    // normal the cluster may contain
    float cos_cone_angle = bounds.cos_cone_angle;
    float sin_cone_angle = sqrt(max(1.0 - cos_cone_angle*cos_cone_angle, 0.0));
    float cos_bound = cos_sub_clamped(sin_theta_w, cos_theta_w, sin_cone_angle, cos_cone_angle);
    float sin_bound = sin_sub_clamped(sin_theta_w, cos_theta_w, sin_cone_angle, cos_cone_angle);

    // get the sin and cos of the angle subtended by the cluster from the shading point
    float cos_subtended = -1;
    float bounding_sphere_radius2 = bounding_sphere_radius*bounding_sphere_radius;
    if (sqr_distance >= bounding_sphere_radius2)
        cos_subtended = sqrt(max(1.0 - bounding_sphere_radius2*rcp(sqr_distance), 0.0));
    float sin_subtended = sqrt(max(1.0 - cos_subtended*cos_subtended, 0.0));

    // subtract to get the cos of the smallest angle achievable between any emitter normal
    // the cluster may contain, and any direction from within the cluster bounding sphere to the shading point
    float cos_min_emission_angle = cos_sub_clamped(sin_bound, cos_bound, sin_subtended, cos_subtended);

    // cosine term at the shading point. subtract subtended angle to get the largest receiver-side
    // cosine any point in the cluster bounding sphere could give
    float cos_receiver = abs(dot(to_shading_dir, normal));
    float sin_receiver = sqrt(max(1.0 - cos_receiver*cos_receiver, 0.0));
    float cos_receiver_bound = cos_sub_clamped(sin_receiver, cos_receiver, sin_subtended, cos_subtended);

    // get the importance estimate. we overestimate to avoid bias
    return max(
        bounds.power *              // power sum of cluster
        rcp(sqr_distance_clamped) * // distance falloff bound
        cos_min_emission_angle *    // light-side cosine bound
        cos_receiver_bound, 0);     // receiver-side cosine bound
}

// https://github.com/mmp/pbrt-v4/blob/b4ce9687e6c695f5582997c61b0c66cf064bdb4a/src/pbrt/lightsamplers.h#L266
// Pick a light from the bvh
bool pick_light_bvh(float3 position, float3 normal, float rng, out uint light_index, out float pick_density)
{
    light_index = 0;
    pick_density = 1;

    if (lbvh_light_count == 0)
        return false;

    uint node_index = 0;
    while (true)
    {
        LightBVHNode node = lbvh_nodes[node_index];
        if (!node.is_leaf())
        {
            LightBVHNode left = lbvh_nodes[node_index + 1];
            LightBVHNode right = lbvh_nodes[node.child_index()];

            // calc importance of both children
            float left_importance = light_bounds_importance(left.bounds, position, normal);
            float right_importance = light_bounds_importance(right.bounds, position, normal);
            float importance_sum = left_importance + right_importance;
            if (importance_sum == 0)
                return false;

            // pick with probability proportional to importance
            float rcp_importance_sum = rcp(importance_sum);
            float left_density = left_importance * rcp_importance_sum;
            if (rng < left_density)
            {
                pick_density *= left_density;

                // rescale to reuse rng
                rng *= rcp(left_density);
                rng = min(rng, ONE_MINUS_FLOAT_EPS);

                node_index = node_index + 1;
            }
            else
            {
                float right_density = right_importance * rcp_importance_sum;
                pick_density *= right_density;

                // rescale to reuse rng
                rng -= left_density;
                rng *= rcp(right_density);
                rng = min(rng, ONE_MINUS_FLOAT_EPS);

                node_index = node.child_index();
            }
        }
        else
        {
            light_index = node.light_index();
            return true;
        }
    }
}

public bool pick_light_bvh(float3 position, float3 normal, float rng, out uint instance_index, out uint triangle_index, out float pick_density)
{
    instance_index = 0;
    triangle_index = 0;

    uint light_index;
    bool result = pick_light_bvh(position, normal, rng, light_index, pick_density);
    if (result)
    {
        EmissiveTriangle triangle = lbvh_lights[light_index];
        instance_index = triangle.instance_index;
        triangle_index = triangle.primitive_index;
    }

    return result;
}
