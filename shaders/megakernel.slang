import sampling;
import raytracing;
import bsdf;
import material;
import common;
import rng;
import spectrum;

uniform uint2 screen_size;
uniform uint frame;
uniform uint sample_count;
uniform uint max_bounces;
uniform float fov;
cbuffer Camera
{
    float3 cam_pos;
    float cam_yaw;
    float cam_pitch;
};

RWTexture2D<float4> image;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= screen_size))
        return;

    float2 jitter = kronecker_2d(id.xy, frame, RNG_DIM_JITTER);
    float2 uv = ((id.xy + jitter) / float2(screen_size)) * 2.0 - 1.0;
    uv.y = -uv.y;
    float aspect = float(screen_size.x) / float(screen_size.y);
    float3 forward = float3(cos(cam_pitch) * sin(cam_yaw), sin(cam_pitch), -cos(cam_pitch) * cos(cam_yaw));
    float3 right = normalize(cross(forward, float3(0, 1, 0)));
    float3 up = cross(right, forward);
    float fov_scale = tan(fov * 0.5 * PI / 180.0);

    float3 ray_origin = cam_pos;
    float3 ray_direction = normalize(forward + (uv.x * aspect * right + uv.y * up) * fov_scale);
    float4 throughput = 1;
    float4 density_to_hero_ratio = 1;

    float4 wavelengths = 0;
    SpectrumBasis spectrum_basis = SpectrumBasis();
    if (spectral_mode != SpectralMode.RGB)
    {
        wavelengths = sample_wavelengths(kronecker_2d(id.xy, frame, RNG_DIM_WAVELENGTH).x);
        spectrum_basis = get_spectrum_basis(wavelengths);
    }

    float3 color = 0.0;
    for (uint depth = 0; depth < max_bounces; depth++)
    {
        Hit hit = trace_ray(ray_origin, ray_direction);
        if (hit.is_valid())
        {
            HitAttributes attributes = get_hit_attributes(hit, ray_origin, ray_direction);
            MaterialInfo mat = get_material(attributes);
            
            float3 view_direction = -ray_direction;
            float3 specular_normal = ensure_valid_specular_normal(attributes.geometric_normal, view_direction, attributes.normal);
            BSDF bsdf = get_material_bsdf(mat, spectrum_basis, hit.is_front_face, attributes.normal, specular_normal);

            // Add emission
            if (any(mat.emission > 0.0))
            {
                color += resolve_spectrum(
                    throughput * upsample_rgb(mat.emission, spectrum_basis),
                    density_to_hero_ratio,
                    spectrum_basis);
            }

            // Evaluate BSDF
            uint scatter_dim = RNG_DIM_SCATTER + depth * RNG_DIMS_PER_BOUNCE;
            float3 sample_direction;
            float4 bsdf_value_times_cosine;
            float4 bsdf_density;
            bsdf.sample(
                view_direction,
                attributes.geometric_normal,
                kronecker_2d(id.xy, frame, scatter_dim),
                sample_direction,
                bsdf_value_times_cosine,
                bsdf_density);
            if (mat.dispersion > 0 && spectral_mode != SpectralMode::RGB) {
                throughput.yzw = 0.0;
                density_to_hero_ratio.yzw = 0.0;
            }
            if (bsdf_density[0] <= 0.0)
                break;

            // Accumulate integrand
            throughput *= bsdf_value_times_cosine / bsdf_density[0];
            density_to_hero_ratio *= bsdf_density / bsdf_density[0];

            // Russian roulette
            if (depth >= 2)
            {
                uint russian_roulette_dim = RNG_DIM_RUSSIAN_ROULETTE + depth * RNG_DIMS_PER_BOUNCE;
                float russian_roulette_probability = min(1.0, max(throughput.x, max(throughput.y, max(throughput.z, throughput.w))));
                if (kronecker_2d(id.xy, frame, russian_roulette_dim).x > russian_roulette_probability)
                    break;
                throughput *= rcp(russian_roulette_probability);
            }

            // Update ray for next bounce
            float3 bias_normal = dot(sample_direction, attributes.geometric_normal) > 0.0 
                ? attributes.geometric_normal
                : -attributes.geometric_normal;
            ray_origin = apply_normal_bias(attributes.position, bias_normal);
            ray_direction = sample_direction;
        }
        else
        {
            float3 sky_color = float3(0.8, 0.7, 0.6);
            color += resolve_spectrum(
                throughput * upsample_rgb(sky_color, spectrum_basis),
                density_to_hero_ratio,
                spectrum_basis);
            break;
        }
    }
    float3 prev = sample_count == 0 ? color : image[id.xy].rgb;
    image[id.xy] = float4(lerp(prev, color, 1.0 / (sample_count + 1)), 1.0);
}
