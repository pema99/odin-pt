import sampling;
import geometry;
import bsdf;
import material;
import common;
import rng;
import spectrum;
import light_sampling;

uniform uint2 screen_size;
uniform uint frame;
uniform uint sample_count;
uniform uint max_bounces;
uniform float fov;
uniform NEEMode nee_mode;
cbuffer Camera
{
    float3 cam_pos;
    float cam_yaw;
    float cam_pitch;
};

RWTexture2D<float4> image;

void trace<let MODE : SpectralMode>(uint3 id)
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

    // Path state
    float3 ray_origin = cam_pos;
    float3 ray_direction = normalize(forward + (uv.x * aspect * right + uv.y * up) * fov_scale);
    float4 throughput = 1;
    float previous_bsdf_density = 0.0;
    bool previous_bounce_delta = false;
    float4 density_to_hero_ratio = 1;
    float4 medium_absorption = 0.0;

    float4 wavelengths = 0;
    SpectrumBasis spectrum_basis = SpectrumBasis();
    if (MODE != SpectralMode.RGB)
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
            bool incoming_outside = hit.is_front_face;
            
            float3 view_direction = -ray_direction;
            float3 specular_normal = ensure_valid_specular_normal(attributes.geometric_normal, view_direction, attributes.normal);
            BSDF bsdf = get_material_bsdf<MODE>(mat, spectrum_basis, incoming_outside, attributes.normal, specular_normal);

            // Lambert-beer
            throughput *= exp(-medium_absorption * hit.distance);

            // Add randomly hit emissive contribution
            if (any(mat.emission > 0.0) &&
                nee_mode != NEEMode.LightSampling)
            {
                float mis_weight = 1.0;
                if (depth > 0 && !previous_bounce_delta)
                {
                    float cos_theta_light = abs(dot(attributes.geometric_normal, ray_direction));
                    float light_density = emissive_triangle_density(attributes.triangle_count, attributes.triangle_area, hit.distance, cos_theta_light);
                    mis_weight = calculate_mis_weight(nee_mode, previous_bsdf_density, light_density);
                }
                float4 integrand = throughput 
                    * mis_weight 
                    * upsample_ergb<MODE>(mat.emission, spectrum_basis);
                color += resolve_spectrum<MODE>(integrand, density_to_hero_ratio, spectrum_basis);
            }

            // Next event estimation
            if (depth < max_bounces - 1 &&
                bsdf.has_continuous_lobe() &&
                nee_mode != NEEMode.BSDFSampling)
            {
                float2 pick_rng = kronecker_2d(id.xy, frame, RNG_DIMS_NEE_PICK + depth * RNG_DIMS_PER_BOUNCE);
                float2 point_rng = kronecker_2d(id.xy, frame, RNG_DIMS_NEE_POINT + depth * RNG_DIMS_PER_BOUNCE);
                LightSample nee_sample = sample_emissive_triangle(attributes.position, pick_rng, point_rng);
                if (nee_sample.is_valid())
                {
                    float4 nee_bsdf_density, nee_bsdf_value_times_cosine;
                    bsdf.evaluate(
                        view_direction,
                        nee_sample.direction_to_light,
                        nee_bsdf_value_times_cosine,
                        nee_bsdf_density);

                    float3 shadow_ray_bias_normal = dot(attributes.geometric_normal, nee_sample.direction_to_light) > 0.0
                        ? attributes.geometric_normal
                        : -attributes.geometric_normal;
                    float3 shadow_ray_target = apply_normal_bias(attributes.position, shadow_ray_bias_normal);
                    float3 shadow_ray_vector = shadow_ray_target - nee_sample.shadow_ray_origin;
                    float shadow_ray_length = length(shadow_ray_vector);
                    if (any(nee_bsdf_value_times_cosine > 0.0) &&
                        !trace_shadow_ray(nee_sample.shadow_ray_origin, shadow_ray_vector * rcp(shadow_ray_length), shadow_ray_length))
                    {
                        float4 upsampled_emission = upsample_ergb<MODE>(nee_sample.emission, spectrum_basis);

                        float mis_weight = calculate_mis_weight(nee_mode, nee_sample.density, nee_bsdf_density[0]);
                        float4 integrand = throughput 
                            * mis_weight 
                            * nee_bsdf_value_times_cosine 
                            * upsampled_emission 
                            * rcp(nee_sample.density);

                        color += resolve_spectrum<MODE>(integrand, density_to_hero_ratio, spectrum_basis);
                    }
                }
            }

            // Evaluate BSDF
            uint scatter_dim = RNG_DIM_SCATTER + depth * RNG_DIMS_PER_BOUNCE;
            float3 sample_direction;
            float4 bsdf_value_times_cosine;
            float4 bsdf_density;
            bool bsdf_is_delta;
            bsdf.sample(
                view_direction,
                attributes.geometric_normal,
                kronecker_2d(id.xy, frame, scatter_dim),
                sample_direction,
                bsdf_value_times_cosine,
                bsdf_density,
                bsdf_is_delta);
            if (bsdf_density[0] <= 0.0)
                break;

            previous_bsdf_density = bsdf_density[0];
            previous_bounce_delta = bsdf_is_delta;

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

            // Update medium
            bool outgoing_transmitted = dot(sample_direction, attributes.geometric_normal) < 0.0;
            if (outgoing_transmitted)
            {
                if (incoming_outside) // air -> inside volume
                    medium_absorption = get_material_absorption<MODE>(mat, spectrum_basis);
                else // inside volume -> air
                    medium_absorption = 0.0;
            }
            
            // Update ray for next bounce
            float3 bias_normal = outgoing_transmitted 
                ? -attributes.geometric_normal
                : attributes.geometric_normal;
            ray_origin = apply_normal_bias(attributes.position, bias_normal);
            ray_direction = sample_direction;
        }
        else
        {
            float3 sky_color = float3(0.8, 0.7, 0.6);
            color += resolve_spectrum<MODE>(
                throughput * upsample_ergb<MODE>(sky_color, spectrum_basis),
                density_to_hero_ratio,
                spectrum_basis);
            break;
        }
    }
    float3 prev = sample_count == 0 ? color : image[id.xy].rgb;
    image[id.xy] = float4(lerp(prev, color, 1.0 / (sample_count + 1)), 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main_rgb(uint3 id: SV_DispatchThreadID) { trace<SpectralMode.RGB>(id); }

[shader("compute")]
[numthreads(8, 8, 1)]
void main_spectral(uint3 id: SV_DispatchThreadID) { trace<SpectralMode.Spectral>(id); }
