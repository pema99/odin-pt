import common;
import geometry;
import spectrum;

public interface BSDF
{
    void evaluate(
        float3 view_direction,
        float3 sample_direction,
        out float4 value_times_cosine,
        out float4 density);

    void sample(
        float3 view_direction,
        float3 geometric_normal,
        float2 random_sample,
        out float3 sample_direction,
        out float4 value_times_cosine,
        out float4 density,
        out bool is_delta);

    bool has_continuous_lobe();
}

public enum BSDFType : uint
{
    Disney,
    Glass,
    Lambert,
}

// === Lambert ===

public struct LambertianBSDF : BSDF
{
    public float4 albedo;
    public float3 normal;

    public void evaluate(float3 view_direction, float3 sample_direction,
        out float4 value_times_cosine, out float4 density)
    {
        float n_dot_l = max(0.0, dot(normal, sample_direction));
        density = n_dot_l / PI;
        value_times_cosine = albedo / PI * n_dot_l;
    }

    public void sample(float3 view_direction, float3 geometric_normal, float2 random_sample,
        out float3 sample_direction, out float4 value_times_cosine, out float4 density, out bool is_delta)
    {
        is_delta = false;
        sample_direction = cosine_hemisphere(random_sample, normal);
        evaluate(view_direction, sample_direction, value_times_cosine, density);

        if (dot(geometric_normal, sample_direction) <= 0.0)
        {
            value_times_cosine = 0.0;
            density = 0.0;
        }
    }

    public bool has_continuous_lobe()
    {
        return true;
    }
}

// === PBR ===

// Microfacet Models for Refraction through Rough Surfaces, Walter et al., 2007, (eq. 33)
float ggx_ndf(float n_dot_h, float alpha)
{
    float alpha2 = alpha * alpha;
    float denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

float smith_ggx_helper(float n_dot_x, float alpha)
{
    float alpha2 = alpha * alpha;
    float cos2 = n_dot_x * n_dot_x;
    float tan2 = (1.0 - cos2) / cos2;
    return (sqrt(1.0 + alpha2 * tan2) - 1.0) * 0.5;
}

// Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz, 2014, (eq. 99)
float smith_ggx_geom(float n_dot_l, float n_dot_v, float alpha)
{
    return 1.0 / (1.0 + smith_ggx_helper(n_dot_l, alpha) + smith_ggx_helper(n_dot_v, alpha));
}

// https://en.wikipedia.org/wiki/Schlick's_approximation
float4 fresnel_schlick(float v_dot_h, float4 f0)
{
    float m = 1.0 - v_dot_h;
    float f = m * m * m * m * m;
    return f0 + (1.0 - f0) * f;
}

// Guesstimate an index of refraction for metals given their f0
// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_iridescence/README.md
float4 fresnel0_to_ior(float4 f0)
{
    float4 sqrt_f0 = sqrt(f0);
    return (1.0 + sqrt_f0) / (1.0 - sqrt_f0);
}

// A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence,
// Belcour & Barla, 2017, (eq. 10-11 + supplemental code)
// https://hal.science/hal-01518344/document
float4 fresnel_thin_film(
    float v_dot_h, float4 ior_outside, float ior_film, float4 eta_base, float thickness, float4 wavelengths)
{
    float4 eta_film = ior_film / ior_outside;

    // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
    float4 cos_theta_film_sq = 1.0 - (1.0 - v_dot_h * v_dot_h) / (eta_film * eta_film);
    float4 cos_theta_film = sqrt(max(cos_theta_film_sq, 0.0));
    float4 cos_theta_base = sqrt(max(1.0 - (1.0 - v_dot_h * v_dot_h) / (eta_base * eta_base), 0.0));

    // 12 = air->film, 23 = film->base
    // https://en.wikipedia.org/wiki/Fresnel_equations#Complex_amplitude_reflection_and_transmission_coefficients
    float4 r12_s = (v_dot_h - eta_film * cos_theta_film) / (v_dot_h + eta_film * cos_theta_film);
    float4 r12_p = (eta_film * v_dot_h - cos_theta_film) / (eta_film * v_dot_h + cos_theta_film);
    float4 r23_s = (eta_film * cos_theta_film - eta_base * cos_theta_base)
        / (eta_film * cos_theta_film + eta_base * cos_theta_base);
    float4 r23_p = (eta_base * cos_theta_film - eta_film * cos_theta_base)
        / (eta_base * cos_theta_film + eta_film * cos_theta_base);

    // signed amplitude -> phase, power
    float4 R12_s = r12_s * r12_s;
    float4 R12_p = r12_p * r12_p;
    float4 R23_s = r23_s * r23_s;
    float4 R23_p = r23_p * r23_p;
    float4 T121_s = 1.0 - R12_s;
    float4 T121_p = 1.0 - R12_p;
    float4 phi21_s = PI - select(r12_s < 0.0, PI, 0.0);
    float4 phi21_p = PI - select(r12_p < 0.0, PI, 0.0);
    float4 phi23_s = select(r23_s < 0.0, PI, 0.0);
    float4 phi23_p = select(r23_p < 0.0, PI, 0.0);

    // https://en.wikipedia.org/wiki/Thin-film_interference#Theory
    float4 optical_path_difference = 2.0 * ior_film * thickness * cos_theta_film;
    float4 Dphi = 2.0 * PI * optical_path_difference / wavelengths;

    // Start airy summation..
    float4 R123_s = min(R12_s * R23_s, 0.9999);
    float4 r123_s = sqrt(R123_s);
    float4 Rs_s = T121_s * T121_s * R23_s / (1.0 - R123_s);
    float4 R123_p = min(R12_p * R23_p, 0.9999);
    float4 r123_p = sqrt(R123_p);
    float4 Rs_p = T121_p * T121_p * R23_p / (1.0 - R123_p);

    // .. end of airy summation
    float4 cosP_s = cos(Dphi + phi23_s + phi21_s);
    float4 irid_s = (r123_s * cosP_s - r123_s * r123_s) / (1.0 - 2.0 * r123_s * cosP_s + r123_s * r123_s);
    float4 I_s = R12_s + Rs_s + 2.0 * (Rs_s - T121_s) * irid_s;
    float4 cosP_p = cos(Dphi + phi23_p + phi21_p);
    float4 irid_p = (r123_p * cosP_p - r123_p * r123_p) / (1.0 - 2.0 * r123_p * cosP_p + r123_p * r123_p);
    float4 I_p = R12_p + Rs_p + 2.0 * (Rs_p - T121_p) * irid_p;
    return select(cos_theta_film_sq < 0.0, 1.0, 0.5 * (I_s + I_p));
}

public interface Fresnel
{
    float4 evaluate(float v_dot_h);
    float4 base_ior();
    float4 normal_incidence();
}

public struct SchlickFresnel : Fresnel
{
    public float4 f0;

    public float4 evaluate(float v_dot_h)
    {
        return fresnel_schlick(v_dot_h, f0);
    }

    public float4 base_ior()
    {
        return fresnel0_to_ior(f0 + 0.0001);
    }

    public float4 normal_incidence()
    {
        return f0;
    }
}

public struct DielectricFresnel : Fresnel
{
    public float4 eta;

    public float4 evaluate(float v_dot_h)
    {
        return fresnel_dielectric(v_dot_h, eta);
    }

    public float4 base_ior()
    {
        return eta;
    }

    public float4 normal_incidence()
    {
        return fresnel_dielectric(1.0, eta);
    }
}

public struct ThinFilmFresnel<F : Fresnel> : Fresnel
{
    public F base;
    public float factor;
    public float4 ior_outside;
    public float ior_film;
    public float thickness;
    public float4 wavelengths;

    public float4 evaluate(float v_dot_h)
    {
        if (factor <= 0.0)
            return base.evaluate(v_dot_h);
        float4 thin_film = fresnel_thin_film(v_dot_h, ior_outside, ior_film, base.base_ior(), thickness, wavelengths);
        return lerp(base.evaluate(v_dot_h), thin_film, factor);
    }

    public float4 base_ior()
    {
        return base.base_ior();
    }

    public float4 normal_incidence()
    {
        return base.normal_incidence();
    }
}

// Microfacet Models for Refraction through Rough Surfaces, Walter et al., 2007, (eq. 20)
// without the fresnel term (reflection only)
float cook_torrance_reflection(float n_dot_l, float n_dot_v, float n_dot_h, float alpha)
{
    float distribution = ggx_ndf(n_dot_h, alpha);
    float geometry = smith_ggx_geom(n_dot_l, n_dot_v, alpha);
    return geometry * distribution / (4.0 * n_dot_l * n_dot_v);
}

// Microfacet Models for Refraction through Rough Surfaces, Walter et al., 2007, (eq. 20)
float4 cook_torrance(float n_dot_l, float n_dot_v, float n_dot_h, float v_dot_h, float4 f0, float alpha)
{
    return fresnel_schlick(v_dot_h, f0) * cook_torrance_reflection(n_dot_l, n_dot_v, n_dot_h, alpha);
}

// Directional albedo of the single-scattering GGX lobe is given by:
//   E(v, f0) = ∫Ω fr(f0, v, ω) dot(n, ω) dω
// This function calculates a scale and bias on f0 that approximate E(f0), i.e:
//   E(v, f0) ≈ f0 * scale + bias
// Computed using least squares with a 64x64 bake of the lobe and some tweaking.
float2 ggx_ss_directional_albedo_scale_bias(float roughness, float n_dot_v)
{
    float r = roughness;
    float m = n_dot_v;
    float white = 1.029460 + 0.176695 * r - 0.323018 * m - 0.423790 * r * m - 0.723949 * r * r
        + 0.962384 * m * m - 1.160124 * r * r * m + 0.906354 * r * m * m
        + 0.547459 * r * r * r - 0.680413 * m * m * m;
    float grazing = 1.0 - m;
    float tail = grazing * grazing * grazing * grazing * grazing;
    float bias = tail * (1.034784 - 2.327367 * r + 1.088382 * m - 1.029448 * r * m + 1.353766 * r * r);
    white = min(white, 1.0);
    bias = min(bias, white);
    return float2(white - bias, bias);
}

// Compensation factor for loss of energy with single-scattering microfacet model
// Turquin, Practical multiple scattering compensation for microfacet models, 2019, (eq. 16)
// https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
float4 turquin_ss_compensation(float4 f0, float white_albedo)
{
    return 1.0 + f0 * (1.0 - white_albedo) / white_albedo;
}

// Directional albedo for GGX lobe, compensated to approximate multiple-scattering, i.e:
// E(v, f0) = ∫Ω t * fr(f0, v, ω) dot(n, ω) dω    where t is the compensation
float4 ggx_ms_directional_albedo(float4 f0, float roughness, float n_dot_v)
{
    float2 ab = ggx_ss_directional_albedo_scale_bias(roughness, n_dot_v);
    return saturate((f0 * ab.x + ab.y) * turquin_ss_compensation(f0, 1.0 * ab.x + ab.y));
}

// Same as ggx_ms_directional_albedo but averaged over all view directions, i.e:
// E_avg(f0) = 2 ∫_0^1 E(μ, f0) μ dμ    where μ = dot(n, v)
float4 ggx_ms_average_albedo(float4 f0, float roughness)
{
    float r = roughness;
    float c0 = 0.062232 - 0.123081 * r + 0.064461 * r * r;
    float c1 = 0.930937 + r * (0.236909 + r * (-0.055513 + r * (-1.828272 + r * 1.143570)));
    float c2 = 0.006831 + r * (-0.113828 + r * (-0.008948 + r * (1.828266 + r * -1.143568)));
    return saturate(c0 + f0 * (c1 + f0 * c2));
}

// Sampling the GGX Distribution of Visible Normals, Heitz, 2018, (lst. 1)
float3 sample_ggx_vndf(float3 v, float alpha, float u1, float u2)
{
    // Section 3.2: transforming the view direction to the hemisphere configuration
    float3 vh = normalize(float3(alpha * v.x, alpha * v.y, v.z));

    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = vh.x * vh.x + vh.y * vh.y;
    float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 t2 = cross(vh, t1);

    // Section 4.2: parameterization of the projected area
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;

    // Section 4.3: reprojection onto hemisphere
    float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    return normalize(float3(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
}

// Sampling the GGX Distribution of Visible Normals, Heitz, 2018, (eq. 3, 17)
float ggx_vndf_pdf(float n_dot_v, float n_dot_h, float alpha)
{
    float g1 = 1.0 / (1.0 + smith_ggx_helper(n_dot_v, alpha));
    return g1 * ggx_ndf(n_dot_h, alpha) / (4.0 * n_dot_v);
}

public struct CookTorranceBSDF<F : Fresnel> : BSDF
{
    public F fresnel;
    public float roughness;
    public float3 normal;

    public void evaluate(float3 view_direction, float3 sample_direction,
        out float4 value_times_cosine, out float4 density)
    {
        value_times_cosine = 0.0;
        density = 0.0;
        float n_dot_l = dot(normal, sample_direction);
        float n_dot_v = dot(normal, view_direction);
        if (n_dot_l <= 0.0 || n_dot_v <= 0.0)
            return;

        float alpha = roughness * roughness;
        if (alpha < 1e-3)
            return;

        float3 half_vector = normalize(view_direction + sample_direction);
        float n_dot_h = dot(normal, half_vector);
        float v_dot_h = dot(view_direction, half_vector);
        density = ggx_vndf_pdf(n_dot_v, n_dot_h, alpha);

        float2 ab = ggx_ss_directional_albedo_scale_bias(roughness, n_dot_v);
        value_times_cosine =
            turquin_ss_compensation(fresnel.normal_incidence(), 1.0 * ab.x + ab.y)
            * fresnel.evaluate(v_dot_h)
            * cook_torrance_reflection(n_dot_l, n_dot_v, n_dot_h, alpha)
            * n_dot_l;
    }

    public void sample(float3 view_direction, float3 geometric_normal, float2 random_sample,
        out float3 sample_direction, out float4 value_times_cosine, out float4 density, out bool is_delta)
    {
        float alpha = roughness * roughness;
        is_delta = alpha < 1e-3;
        if (is_delta)
        {
            sample_direction = reflect(-view_direction, normal);
            value_times_cosine = fresnel.evaluate(dot(normal, view_direction));
            density = 1.0;
        }
        else
        {
            float3x3 basis = ortho_basis_from_vector(normal);
            float3 v_local = mul(transpose(basis), view_direction);
            float3 h_local = sample_ggx_vndf(v_local, alpha, random_sample.x, random_sample.y);
            float3 l_local = reflect(-v_local, h_local);
            sample_direction = mul(basis, l_local);
            evaluate(view_direction, sample_direction, value_times_cosine, density);
        }

        if (dot(geometric_normal, sample_direction) <= 0.0)
        {
            value_times_cosine = 0.0;
            density = 0.0;
        }
    }

    public bool has_continuous_lobe()
    {
        return roughness * roughness >= 1e-3;
    }
}

// === Mixture ===

public interface BSDFMixWeights
{
    // weight applied to each bsdf
    void weights(float3 view_direction, float3 sample_direction, out float4 weight_a, out float4 weight_b);
    // probability to sample the first bsdf (a)
    float probability_a(float3 view_direction);
}

public struct ConstantWeights : BSDFMixWeights
{
    public float4 weight_a;
    public float4 weight_b;
    public float probability;

    public void weights(float3 view_direction, float3 sample_direction, out float4 out_a, out float4 out_b)
    {
        out_a = weight_a;
        out_b = weight_b;
    }

    public float probability_a(float3 view_direction)
    {
        return probability;
    }
}

// fr = weight_a * fr_a + weight_b * fr_b
public struct MixtureBSDF<A : BSDF, B : BSDF, W : BSDFMixWeights> : BSDF
{
    public A bsdf_a;
    public B bsdf_b;
    public W weighting;

    public void evaluate(float3 view_direction, float3 sample_direction,
        out float4 value_times_cosine, out float4 density)
    {
        float4 weight_a, weight_b;
        weighting.weights(view_direction, sample_direction, weight_a, weight_b);
        float4 value_a, value_b;
        float4 density_a, density_b;
        bsdf_a.evaluate(view_direction, sample_direction, value_a, density_a);
        bsdf_b.evaluate(view_direction, sample_direction, value_b, density_b);
        float probability_a = weighting.probability_a(view_direction);
        density = probability_a * density_a + (1.0 - probability_a) * density_b;
        value_times_cosine = weight_a * value_a + weight_b * value_b;
    }

    public void sample(float3 view_direction, float3 geometric_normal, float2 random_sample,
        out float3 sample_direction, out float4 value_times_cosine, out float4 density, out bool is_delta)
    {
        float probability_a = weighting.probability_a(view_direction);
        float u1 = random_sample.x;
        float4 lobe_value;
        float4 lobe_density;
        bool sampled_a = u1 < probability_a;
        if (sampled_a)
        {
            u1 = saturate(u1 / probability_a);
            bsdf_a.sample(view_direction, geometric_normal, float2(u1, random_sample.y), sample_direction,
                lobe_value, lobe_density, is_delta);
        }
        else
        {
            u1 = saturate((u1 - probability_a) / (1.0 - probability_a));
            bsdf_b.sample(view_direction, geometric_normal, float2(u1, random_sample.y),
                sample_direction, lobe_value, lobe_density, is_delta);
        }

        if (lobe_density[0] <= 0.0)
        {
            value_times_cosine = 0.0;
            density = 0.0;
            return;
        }

        if (is_delta)
        {
            float4 weight_a, weight_b;
            weighting.weights(view_direction, sample_direction, weight_a, weight_b);
            value_times_cosine = (sampled_a ? weight_a : weight_b) * lobe_value;
            density = (sampled_a ? probability_a : 1.0 - probability_a) * lobe_density;
            return;
        }
        evaluate(view_direction, sample_direction, value_times_cosine, density);
    }

    public bool has_continuous_lobe()
    {
        return bsdf_a.has_continuous_lobe() || bsdf_b.has_continuous_lobe();
    }
}

// === Disney-like BSDF ===

// Lobe b sits under lobe a and receives only the energy a did not reflect away.
public struct CoatingWeights : BSDFMixWeights
{
    public float4 f0;
    public float roughness;
    public float4 base_weight;
    public float4 base_albedo;
    public float3 normal;

    // Revisiting Physically Based Shading at Imageworks, Kulla and Conty, 2017, page 58
    // https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
    // "Scale in a way that avoids energy gain while losing as little energy as possible"
    public void weights(float3 view_direction, float3 sample_direction, out float4 out_a, out float4 out_b)
    {
        float n_dot_v = saturate(dot(normal, view_direction));
        float n_dot_l = saturate(dot(normal, sample_direction));
        float4 transmitted_view = 1.0 - ggx_ms_directional_albedo(f0, roughness, n_dot_v);
        float4 transmitted_sample = 1.0 - ggx_ms_directional_albedo(f0, roughness, n_dot_l);
        out_a = 1.0;
        out_b = base_weight * min(transmitted_view, transmitted_sample);
        // TODO: Use ggx_ms_average_albedo over Kulla-Conty's "min weight" trick.
        //       Only works if we assume base layer is lambertian!
    }

    // Crash Course in BRDF Implementation, section 5 "Combining BRDFs", pick lobe based on estimated energy.
    // https://boksajak.github.io/files/CrashCourseBRDF.pdf
    public float probability_a(float3 view_direction)
    {
        float n_dot_v = saturate(dot(normal, view_direction));
        float4 coat_albedo = ggx_ms_directional_albedo(f0, roughness, n_dot_v);
        // float energy_coat = luminance(coat_albedo);
        // float energy_base = luminance(base_weight * (1.0 - coat_albedo) * base_albedo);
        // TODO: Compare above with max4^^
        float energy_coat = max4(coat_albedo);
        float energy_base = max4(base_weight * (1.0 - coat_albedo) * base_albedo);
        // https://pema.dev/obsidian/math/light-transport/ratio-for-diffuse-and-specular-bounce.html
        return clamp(energy_coat / max(energy_coat + energy_base, 1e-4), 0.05, 0.95);
    }
}

public typealias DisneyBSDF = MixtureBSDF<CookTorranceBSDF<ThinFilmFresnel<SchlickFresnel>>, LambertianBSDF, CoatingWeights>;

public DisneyBSDF make_disney_bsdf(
    float4 albedo,
    float metallic,
    float roughness,
    float ior,
    float thin_film_factor,
    float thin_film_ior,
    float thin_film_thickness,
    float3 normal,
    float3 specular_normal,
    float4 wavelengths)
{
    DisneyBSDF m;

    float f0_dielectric_sqrt = (ior - 1.0) / (ior + 1.0);
    float f0_dielectric = f0_dielectric_sqrt * f0_dielectric_sqrt;
    m.bsdf_a.fresnel.base.f0 = lerp(f0_dielectric, albedo, metallic);
    m.bsdf_a.fresnel.factor = thin_film_factor;
    m.bsdf_a.fresnel.ior_outside = 1.0;
    m.bsdf_a.fresnel.ior_film = thin_film_ior;
    m.bsdf_a.fresnel.thickness = thin_film_thickness;
    m.bsdf_a.fresnel.wavelengths = wavelengths;
    m.bsdf_a.roughness = roughness;
    m.bsdf_a.normal = specular_normal;

    m.bsdf_b.albedo = albedo;
    m.bsdf_b.normal = normal;

    m.weighting.f0 = m.bsdf_a.fresnel.base.f0;
    m.weighting.roughness = roughness;
    m.weighting.base_weight = 1.0 - metallic;
    m.weighting.base_albedo = albedo;
    m.weighting.normal = specular_normal;

    return m;
}

// === Glass ===
// Microfacet Models for Refraction through Rough Surfaces, Walter et al., 2007, (eq. 22)
float fresnel_dielectric(float c, float eta)
{
    float g2 = eta * eta - 1.0 + c * c;
    float g = sqrt(max(g2, 0.0));
    float a = (g - c) / (g + c);
    float b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

// Same as above but vectorized
float4 fresnel_dielectric(float c, float4 eta)
{
    float4 g2 = eta * eta - 1.0 + c * c;
    float4 g = sqrt(max(g2, 0.0));
    float4 a = (g - c) / (g + c);
    float4 b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

public struct GlassBSDF<F : Fresnel> : BSDF
{
    public F fresnel;
    public float roughness;
    public float4 color;
    public float3 normal;

    public void evaluate(float3 view_direction, float3 sample_direction,
        out float4 value_times_cosine, out float4 density)
    {
        float4 eta = fresnel.base_ior();
        value_times_cosine = 0.0;
        density = 0.0;
        if (all(eta == 1.0)) {
            // delta distribution, so chance of sampling is always 0
            return;
        }

        float alpha = roughness * roughness;
        if (alpha < 1e-3)
            return;

        float n_dot_v = dot(normal, view_direction);
        float n_dot_l = dot(normal, sample_direction);
        bool is_reflection = n_dot_l > 0.0;
        if (n_dot_v <= 0.0 || n_dot_l == 0.0)
            return;

        if (is_reflection)
        {
            float3 half_vector = normalize(view_direction + sample_direction);
            float n_dot_h = dot(normal, half_vector);
            float v_dot_h = dot(view_direction, half_vector);

            // Must match probability in sample()
            float4 reflectance = fresnel.evaluate(v_dot_h);
            density = reflectance * ggx_vndf_pdf(n_dot_v, n_dot_h, alpha);
            value_times_cosine = reflectance * cook_torrance_reflection(n_dot_l, n_dot_v, n_dot_h, alpha) * n_dot_l;
        }
        else
        {
            float geometry = smith_ggx_geom(-n_dot_l, n_dot_v, alpha);
            float masking = 1.0 / (1.0 + smith_ggx_helper(n_dot_v, alpha));

            // when wavelengths differ, each gets a different half-vector,
            // so we must process them in a loop
            uint wavelength_count = all(eta == eta[0]) ? 1 : 4;

            float4 transmission_without_fresnel = 0.0;
            float4 fresnel_transmission = 0.0;
            float4 fresnel_transmission_all_lanes = 0.0;
            for (uint i = 0; i < wavelength_count; i++)
            {
                float3 half_vector = -normalize(view_direction + eta[i] * sample_direction);
                if (dot(half_vector, normal) < 0.0)
                    half_vector = -half_vector;

                float n_dot_h = dot(normal, half_vector);
                float v_dot_h = dot(view_direction, half_vector);
                float l_dot_h = dot(sample_direction, half_vector);
                if (v_dot_h <= 0.0 || l_dot_h >= 0.0)
                    continue;

                // Tricky code.. this evaluates all 4 wavelengths at i's half-vector, so only i is valid!
                // Full vector kept for the wavelength_count==1 case, resolved after the loop
                fresnel_transmission_all_lanes = 1.0 - fresnel.evaluate(v_dot_h);
                fresnel_transmission[i] = fresnel_transmission_all_lanes[i];

                float distribution = ggx_ndf(n_dot_h, alpha);
                float denominator = v_dot_h + eta[i] * l_dot_h;

                // Walter et al., 2007, (eq. 21)
                transmission_without_fresnel[i] = abs(v_dot_h * l_dot_h) / abs(n_dot_v * n_dot_l)
                    * geometry * distribution
                    / (denominator * denominator);

                // Walter et al., 2007, (eq. 17)
                float visible_normal_density = masking * distribution * v_dot_h / n_dot_v;
                density[i] = fresnel_transmission[i] * visible_normal_density * eta[i] * eta[i]
                    * abs(l_dot_h) / (denominator * denominator);
            }

            // Splat the single result in the case where wavelengths are the same
            if (wavelength_count == 1)
            {
                fresnel_transmission = fresnel_transmission_all_lanes;
                transmission_without_fresnel = transmission_without_fresnel.x;
                density = density.x;
            }

            float4 transmission = fresnel_transmission * transmission_without_fresnel;
            value_times_cosine = transmission * abs(n_dot_l) * color;
        }
    }

    public void sample(float3 view_direction, float3 geometric_normal, float2 random_sample,
        out float3 sample_direction, out float4 value_times_cosine, out float4 density, out bool is_delta)
    {
        float4 eta = fresnel.base_ior();
        if (all(eta == 1.0)) {
            // "pass through" material, just tints
            sample_direction = -view_direction;
            value_times_cosine = color;
            density = 1.0;
            is_delta = true;
            return;
        }

        float alpha = roughness * roughness;
        is_delta = alpha < 1e-3;
        float3x3 basis = ortho_basis_from_vector(normal);
        float3 local_view = mul(transpose(basis), view_direction);
        float3 local_half = float3(0.0, 0.0, 1.0);
        if (!is_delta)
            local_half = sample_ggx_vndf(local_view, alpha, random_sample.x, random_sample.y);

        // Choose reflection vs refraction with fresnel
        float4 reflectance = fresnel.evaluate(dot(local_view, local_half));
        float reflection_probability = reflectance.x;
        bool is_reflection = frac(random_sample.x * 1024.0) < reflection_probability; // reuse rng
        float3 local_sample;
        if (is_reflection)
        {
            local_sample = reflect(-local_view, local_half);
        }
        else
        {
            // only hero sample decides where the ray continues
            local_sample = refract(-local_view, local_half, 1.0 / eta[0]);
        }
        sample_direction = mul(basis, local_sample);
        if (is_delta)
        {
            value_times_cosine = is_reflection ? reflectance : (1.0 - reflectance) / (eta * eta) * color;
            density = is_reflection ? reflection_probability : 1.0 - reflection_probability;
            if (!is_reflection)
            {
                value_times_cosine = select(eta == eta[0], value_times_cosine, 0.0);
                density = select(eta == eta[0], density, 0.0);
            }
        }
        else
        {
            evaluate(view_direction, sample_direction, value_times_cosine, density);
        }
        
        float geometric_side = dot(geometric_normal, sample_direction);
        if (dot(normal, sample_direction) > 0.0 ? geometric_side <= 0.0 : geometric_side >= 0.0)
        {
            value_times_cosine = 0.0;
            density = 0.0;
        }
    }

    public bool has_continuous_lobe()
    {
        return roughness * roughness >= 1e-3 && !all(fresnel.base_ior() == 1.0);
    }
}

public GlassBSDF<ThinFilmFresnel<DielectricFresnel>> make_glass_bsdf(
    float roughness,
    float4 color,
    float4 ior,
    bool entering,
    float thin_film_factor,
    float thin_film_ior,
    float thin_film_thickness,
    float3 normal,
    float4 wavelengths)
{
    GlassBSDF<ThinFilmFresnel<DielectricFresnel>> g;
    g.fresnel.base.eta = entering ? ior : rcp(ior);
    g.fresnel.factor = thin_film_factor;
    g.fresnel.ior_outside = entering ? 1.0 : ior;
    g.fresnel.ior_film = thin_film_ior;
    g.fresnel.thickness = thin_film_thickness;
    g.fresnel.wavelengths = wavelengths;
    g.roughness = roughness;
    g.color = color;
    g.normal = normal;
    return g;
}
