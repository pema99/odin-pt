uniform uint frame;
uniform uint2 size;

cbuffer Camera {
    float3 cam_pos;
    float cam_yaw;
    float cam_pitch;
};

StructuredBuffer<float3> positions;
StructuredBuffer<uint>   indices;
RaytracingAccelerationStructure scene;
RWTexture2D<float4> output;

struct Hit
{
    float3 normal;
    float3 position;
    bool valid;
}

void ortho_basis_from_vector(float3 n, out float3 b1, out float3 b2)
{
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = float3(b, sign + n.y * n.y * a, -n.y);
}

float3x3 ortho_basis_from_vector(float3 n)
{
    float3 t, b;
    ortho_basis_from_vector(n, t, b);
    return transpose(float3x3(t, b, n));
}

float3 cosine_hemisphere(float2 u, float3 normal)
{
    float a = sqrt(u.x);
    float b = 2.0f * 3.14159 * u.y;
    float3 localDir = float3(a * cos(b), a * sin(b), sqrt(1.0f - u.x));
    float3x3 basis = ortho_basis_from_vector(normal);
    return mul(basis, localDir);
}

float3 uniform_hemisphere(float2 u, float3 normal)
{
    float z = u[0];
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * 3.14159 * u[1];
    float3 localDir = float3(r * cos(phi), r * sin(phi), z);
    float3x3 basis = ortho_basis_from_vector(normal);
    return mul(basis, localDir);
}

// From https://github.com/skeeto/hash-prospector/issues/19
uint low_bias_hash(uint value)
{
    value ^= value >> 16;
    value *= 0x21f0aaad;
    value ^= value >> 15;
    value *= 0xd35a2d97;
    value ^= value >> 15;
    return value;
}

uint pixel_hash(uint2 pixelCoord)
{
    return low_bias_hash((pixelCoord.x & 0xFFFF) | (pixelCoord.y << 16));
}

// Somewhat based on https://jcgt.org/published/0011/01/04/
float2 kronecker_2d(uint seed, uint sampleIndex, uint dimension)
{
    uint rot_x = low_bias_hash(seed + dimension);
    uint rot_y = low_bias_hash(rot_x);
    uint2 cranley_patterson = uint2(rot_x, rot_y);

    const uint2 alpha = uint2(3242174889u, 2447445414u);
    uint2 sequence_value = cranley_patterson + alpha * sampleIndex;

    const float jitter_strength = 0.0625;
    const uint2 alpha_jitter = uint2(2681146017u, 685201898u);
    float jitter_amplitude = 0.78 * jitter_strength / sqrt(2.0) * rsqrt(float(sampleIndex + 1));
    uint2 jitter_value = alpha_jitter * sampleIndex;
    sequence_value += uint2(float2(jitter_value) * jitter_amplitude);

    return float2(sequence_value) * 2.3283064365386963e-10;
}

float2 kronecker_2d(uint2 pixelCoord, uint sampleIndex, uint dimension)
{
    return kronecker_2d(pixel_hash(pixelCoord), sampleIndex, dimension);
}

Hit trace_ray(float3 ray_origin, float3 ray_direction)
{
    RayDesc primary_ray;
    primary_ray.Origin = ray_origin;
    primary_ray.Direction = ray_direction;
    primary_ray.TMin = 1e-3;
    primary_ray.TMax = 1e30;

    RayQuery<RAY_FLAG_NONE> primary_query;
    primary_query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, primary_ray);
    primary_query.Proceed();

    Hit hit = (Hit)0;
    hit.valid = primary_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;

    if (hit.valid)
    {
        uint prim = primary_query.CommittedPrimitiveIndex();
        float3 v0 = positions[indices[prim * 3 + 0]];
        float3 v1 = positions[indices[prim * 3 + 1]];
        float3 v2 = positions[indices[prim * 3 + 2]];
        float3x3 to_world = (float3x3)primary_query.CommittedObjectToWorld3x4();
        hit.normal = normalize(cross(mul(to_world, v1 - v0), mul(to_world, v2 - v0)));

        float t = primary_query.CommittedRayT();
        hit.position = t * ray_direction + ray_origin;
    }
    return hit;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= size))
        return;

    float2 uv = ((float2(id.xy) + 0.5) / float2(size)) * 2.0 - 1.0;
    uv.y = -uv.y;
    float aspect = float(size.x) / float(size.y);
    float3 forward = float3(cos(cam_pitch) * sin(cam_yaw), sin(cam_pitch), -cos(cam_pitch) * cos(cam_yaw));
    float3 right = normalize(cross(forward, float3(0, 1, 0)));
    float3 up = cross(right, forward);
    float3 dir = normalize(forward + (uv.x * aspect * right + uv.y * up) * 0.577);

    float3 color = lerp(float3(1.0, 1.0, 1.0), float3(0.35, 0.55, 1.0), dir.y * 0.5 + 0.5); // sky

    Hit hit = trace_ray(cam_pos, dir);
    if (hit.valid)
    {
        float hits = 0;
        uint samples = 128;
        for (int i = 0; i < samples; i++)
        {
            float2 rng = kronecker_2d(id.xy, i, 0);
            float3 cont_dir = uniform_hemisphere(rng, hit.normal);
            Hit cont_hit = trace_ray(hit.position, cont_dir);
            if (cont_hit.valid)
            {
                hits += 1;
            }
        }
        float ao = 1.0 - hits / samples;
        color = pow(ao,4);
    }
    
    output[id.xy] = float4(color, 1.0);
}
