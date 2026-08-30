module bsdf;

import common;
import sampling;

public interface BSDF
{
    void evaluate(
        float3 view_direction,
        float3 sample_direction,
        out float3 value_times_cosine,
        out float density);

    void sample(
        float3 view_direction,
        float3 geometric_normal,
        float2 random_sample,
        out float3 sample_direction,
        out float3 value_times_cosine,
        out float density);
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
    public float3 albedo;
    public float3 normal;

    public void evaluate(float3 view_direction, float3 sample_direction,
        out float3 value_times_cosine, out float density)
    {
        float n_dot_l = max(0.0, dot(normal, sample_direction));
        density = n_dot_l / PI;
        value_times_cosine = albedo / PI * n_dot_l;
    }

    public void sample(float3 view_direction, float3 geometric_normal, float2 random_sample,
        out float3 sample_direction, out float3 value_times_cosine, out float density)
    {
        sample_direction = cosine_hemisphere(random_sample, normal);
        evaluate(view_direction, sample_direction, value_times_cosine, density);

        if (dot(geometric_normal, sample_direction) <= 0.0)
        {
            value_times_cosine = float3(0.0);
            density = 0.0;
        }
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
float3 fresnel_schlick(float v_dot_h, float3 f0)
{
    float f = pow(1.0 - v_dot_h, 5.0);
    return f0 + (1.0 - f0) * f;
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
float3 cook_torrance(float n_dot_l, float n_dot_v, float n_dot_h, float v_dot_h, float3 f0, float alpha)
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
    float white = 1.018665 + 0.148317 * r - 0.283689 * m - 0.420953 * r * m - 0.651543 * r * r
        + 0.917850 * m * m - 1.157466 * r * r * m + 0.923874 * r * m * m
        + 0.494024 * r * r * r - 0.683849 * m * m * m;
    float grazing = 1.0 - m;
    float tail = grazing * grazing * grazing * grazing * grazing;
    float bias = tail * (1.020995 - 2.210220 * r + 1.750230 * m - 1.805618 * r * m + 1.267224 * r * r);
    white = min(white, 1.0);
    bias = min(bias, white);
    return float2(white - bias, bias);
}

// Compensation factor for loss of energy with single-scattering microfacet model
// Turquin, Practical multiple scattering compensation for microfacet models, 2019, (eq. 16)
// https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
float3 turquin_ss_compensation(float3 f0, float white_albedo)
{
    return 1.0 + f0 * (1.0 - white_albedo) / white_albedo;
}

// Directional albedo for GGX lobe, compensated to approximate multiple-scattering, i.e:
// E(v, f0) = ∫Ω t * fr(f0, v, ω) dot(n, ω) dω    where t is the compensation
float3 ggx_ms_directional_albedo(float3 f0, float roughness, float n_dot_v)
{
    float2 ab = ggx_ss_directional_albedo_scale_bias(roughness, n_dot_v);
    return saturate((f0 * ab.x + ab.y) * turquin_ss_compensation(f0, 1.0 * ab.x + ab.y));
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

public struct CookTorranceBSDF : BSDF
{
    public float3 f0;
    public float roughness;
    public float3 normal;

    public void evaluate(float3 view_direction, float3 sample_direction,
        out float3 value_times_cosine, out float density)
    {
        value_times_cosine = float3(0.0);
        density = 0.0;
        float n_dot_l = dot(normal, sample_direction);
        float n_dot_v = dot(normal, view_direction);
        if (n_dot_l <= 0.0 || n_dot_v <= 0.0)
            return;

        float alpha = max(roughness * roughness, 1e-3);
        float3 half_vector = normalize(view_direction + sample_direction);
        float n_dot_h = dot(normal, half_vector);
        float v_dot_h = dot(view_direction, half_vector);
        density = ggx_vndf_pdf(n_dot_v, n_dot_h, alpha);

        float2 ab = ggx_ss_directional_albedo_scale_bias(roughness, n_dot_v);
        value_times_cosine = turquin_ss_compensation(f0, 1.0 * ab.x + ab.y)
             * cook_torrance(n_dot_l, n_dot_v, n_dot_h, v_dot_h, f0, alpha) * n_dot_l;
    }

    public void sample(float3 view_direction, float3 geometric_normal, float2 random_sample,
        out float3 sample_direction, out float3 value_times_cosine, out float density)
    {
        float alpha = max(roughness * roughness, 1e-3);
        float3x3 basis = ortho_basis_from_vector(normal);
        float3 v_local = mul(transpose(basis), view_direction);
        float3 h_local = sample_ggx_vndf(v_local, alpha, random_sample.x, random_sample.y);
        float3 l_local = reflect(-v_local, h_local);
        sample_direction = mul(basis, l_local);
        evaluate(view_direction, sample_direction, value_times_cosine, density);

        if (dot(geometric_normal, sample_direction) <= 0.0)
        {
            value_times_cosine = float3(0.0);
            density = 0.0;
        }
    }
}

// === Mixture ===

public interface BSDFMixWeights
{
    // weight applied to each bsdf
    void weights(float3 view_direction, float3 sample_direction, out float3 weight_a, out float3 weight_b);
    // probability to sample the first bsdf (a)
    float probability_a(float3 view_direction);
}

public struct ConstantWeights : BSDFMixWeights
{
    public float3 weight_a;
    public float3 weight_b;
    public float probability;

    public void weights(float3 view_direction, float3 sample_direction, out float3 out_a, out float3 out_b)
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
        out float3 value_times_cosine, out float density)
    {
        float3 weight_a, weight_b;
        weighting.weights(view_direction, sample_direction, weight_a, weight_b);
        float3 value_a, value_b;
        float density_a, density_b;
        bsdf_a.evaluate(view_direction, sample_direction, value_a, density_a);
        bsdf_b.evaluate(view_direction, sample_direction, value_b, density_b);
        float probability_a = weighting.probability_a(view_direction);
        density = probability_a * density_a + (1.0 - probability_a) * density_b;
        value_times_cosine = weight_a * value_a + weight_b * value_b;
    }

    public void sample(float3 view_direction, float3 geometric_normal, float2 random_sample,
        out float3 sample_direction, out float3 value_times_cosine, out float density)
    {
        float probability_a = weighting.probability_a(view_direction);
        float u1 = random_sample.x;
        float3 lobe_value;
        float lobe_density;
        if (u1 < probability_a)
        {
            u1 = saturate(u1 / probability_a);
            bsdf_a.sample(view_direction, geometric_normal, float2(u1, random_sample.y), sample_direction,
                lobe_value, lobe_density);
        }
        else
        {
            u1 = saturate((u1 - probability_a) / (1.0 - probability_a));
            bsdf_b.sample(view_direction, geometric_normal, float2(u1, random_sample.y),
                sample_direction, lobe_value, lobe_density);
        }

        if (lobe_density <= 0.0)
        {
            value_times_cosine = float3(0.0);
            density = 0.0;
            return;
        }
        evaluate(view_direction, sample_direction, value_times_cosine, density);
    }
}

// === Disney-like BSDF ===

// Lobe b sits under lobe a and receives only the energy a did not reflect away.
public struct CoatingWeights : BSDFMixWeights
{
    public float3 f0;
    public float roughness;
    public float3 base_weight;
    public float3 base_albedo;
    public float3 normal;

    // Revisiting Physically Based Shading at Imageworks, Kulla and Conty, 2017, page 58
    // https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
    // "Scale in a way that avoids energy gain while losing as little energy as possible"
    public void weights(float3 view_direction, float3 sample_direction, out float3 out_a, out float3 out_b)
    {
        float n_dot_v = saturate(dot(normal, view_direction));
        float n_dot_l = saturate(dot(normal, sample_direction));
        float3 transmitted_view = 1.0 - ggx_ms_directional_albedo(f0, roughness, n_dot_v);
        float3 transmitted_sample = 1.0 - ggx_ms_directional_albedo(f0, roughness, n_dot_l);
        out_a = float3(1.0);
        out_b = base_weight * min(transmitted_view, transmitted_sample);
    }

    // Crash Course in BRDF Implementation, section 5 "Combining BRDFs", pick lobe based on estimated energy.
    // https://boksajak.github.io/files/CrashCourseBRDF.pdf
    public float probability_a(float3 view_direction)
    {
        float n_dot_v = saturate(dot(normal, view_direction));
        float3 coat_albedo = ggx_ms_directional_albedo(f0, roughness, n_dot_v);
        float energy_coat = luminance(coat_albedo);
        float energy_base = luminance(base_weight * (1.0 - coat_albedo) * base_albedo);
        // https://pema.dev/obsidian/math/light-transport/ratio-for-diffuse-and-specular-bounce.html
        return clamp(energy_coat / max(energy_coat + energy_base, 1e-4), 0.05, 0.95);
    }
}

public typealias DisneyBSDF = MixtureBSDF<CookTorranceBSDF, LambertianBSDF, CoatingWeights>;

public DisneyBSDF make_disney_bsdf(float3 albedo, float metallic, float roughness, float ior, float3 normal, float3 specular_normal)
{
    DisneyBSDF m;

    float f0_dielectric_sqrt = (ior - 1.0) / (ior + 1.0);
    float f0_dielectric = f0_dielectric_sqrt * f0_dielectric_sqrt;
    m.bsdf_a.f0 = lerp(f0_dielectric, albedo, metallic);
    m.bsdf_a.roughness = roughness;
    m.bsdf_a.normal = specular_normal;

    m.bsdf_b.albedo = albedo;
    m.bsdf_b.normal = normal;

    m.weighting.f0 = m.bsdf_a.f0;
    m.weighting.roughness = roughness;
    m.weighting.base_weight = float3(1.0 - metallic);
    m.weighting.base_albedo = albedo;
    m.weighting.normal = normal;

    return m;
}

// === Glass ===
// Microfacet Models for Refraction through Rough Surfaces, Walter et al., 2007, (eq. 22)
float fresnel_dielectric(float c, float eta)
{
    float g2 = eta * eta - 1.0 + c * c;
    if (g2 < 0.0)
        return 1.0;

    float g = sqrt(g2);
    float a = (g - c) / (g + c);
    float b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

public struct GlassBSDF : BSDF
{
    public float roughness;
    public float3 color;
    public float eta;
    public float3 normal;

    public void evaluate(float3 view_direction, float3 sample_direction,
        out float3 value_times_cosine, out float density)
    {
        value_times_cosine = float3(0.0);
        density = 0.0;
        float alpha = max(roughness * roughness, 1e-3);
        float n_dot_v = dot(normal, view_direction);
        float n_dot_l = dot(normal, sample_direction);
        bool is_reflection = n_dot_l > 0.0;
        if (n_dot_v <= 0.0 || n_dot_l == 0.0)
            return;

        float3 half_vector;
        if (is_reflection)
            half_vector = normalize(view_direction + sample_direction);
        else
            half_vector = -normalize(view_direction + eta * sample_direction);

        if (dot(half_vector, normal) < 0.0)
            half_vector = -half_vector;

        float n_dot_h = saturate(dot(normal, half_vector));
        float v_dot_h = dot(view_direction, half_vector);
        float l_dot_h = dot(sample_direction, half_vector);
        if (v_dot_h <= 0.0 || (is_reflection ? l_dot_h <= 0.0 : l_dot_h >= 0.0))
            return;

        // Must match probability in sample()
        float fresnel = fresnel_dielectric(v_dot_h, eta);
        if (is_reflection)
        {
            density = fresnel * ggx_vndf_pdf(n_dot_v, n_dot_h, alpha);
            value_times_cosine = float3(fresnel * cook_torrance_reflection(n_dot_l, n_dot_v, n_dot_h, alpha) * n_dot_l);
        }
        else
        {
            // Walter et al., 2007, (eq. 21)
            float distribution = ggx_ndf(n_dot_h, alpha);
            float geometry = smith_ggx_geom(-n_dot_l, n_dot_v, alpha);
            float denominator = v_dot_h + eta * l_dot_h;
            float transmission = abs(v_dot_h * l_dot_h) / abs(n_dot_v * n_dot_l)
                * (1.0 - fresnel) * geometry * distribution
                / (denominator * denominator);

            // Walter et al., 2007, (eq. 17)
            float masking = 1.0 / (1.0 + smith_ggx_helper(n_dot_v, alpha));
            float visible_normal_density = masking * distribution * v_dot_h / n_dot_v;
            density = (1.0 - fresnel) * visible_normal_density * eta * eta
                * abs(l_dot_h) / (denominator * denominator);
            value_times_cosine = float3(transmission * abs(n_dot_l)) * color;
        }
    }

    public void sample(float3 view_direction, float3 geometric_normal, float2 random_sample,
        out float3 sample_direction, out float3 value_times_cosine, out float density)
    {
        float alpha = max(roughness * roughness, 1e-3);
        float3x3 basis = ortho_basis_from_vector(normal);
        float3 local_view = mul(transpose(basis), view_direction);
        float3 local_half = sample_ggx_vndf(local_view, alpha, random_sample.x, random_sample.y);

        // Choose reflection vs refraction with fresnel
        float reflection_probability = fresnel_dielectric(dot(local_view, local_half), eta);
        float3 local_sample;
        if (frac(random_sample.x * 1024.0) < reflection_probability) // reuse rng
        {
            local_sample = reflect(-local_view, local_half);
        }
        else
        {
            // Walter et al., 2007, (eq. 40)
            local_sample = refract(-local_view, local_half, 1.0 / eta);
        }
        sample_direction = mul(basis, local_sample);
        evaluate(view_direction, sample_direction, value_times_cosine, density);
        
        float geometric_side = dot(geometric_normal, sample_direction);
        if (dot(normal, sample_direction) > 0.0 ? geometric_side <= 0.0 : geometric_side >= 0.0)
        {
            value_times_cosine = float3(0.0);
            density = 0.0;
        }
    }
}