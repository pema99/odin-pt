import common;

// === Public API ===

public enum SpectralMode
{
    RGB = 0,
    Spectral = 1,
}

// cache some stuff so we can build it once per path..
public struct SpectrumBasis
{
    // The sampled wavelengths
    public float4 wavelengths;

    // The spectral primaries evaluated at the sampled wavelengths.
    // "How much light does each primary reflect at this wavelength"
    public float3 primaries[4];

    // CIE matching functions evaluated at the sampled wavelengths.
    // "How much does a unit of light at this wavelength contribute to X,Y,Z"
    public float3 xyz[4];

    public __init()
    {
        wavelengths = 0;
        for (int i = 0; i < 4; ++i)
        {
            primaries[i] = 0;
            xyz[i] = 0;
        }
    }
}

// (e)RGB -> Spectrum or RGB
// eRGB means RGB with illuminant E as the whitepoint
public float4 upsample_ergb<let MODE : SpectralMode>(float3 rgb, SpectrumBasis basis)
{
    if (MODE == SpectralMode.RGB)
        return float4(rgb, 0.0);
    return ergb_to_spectrum(rgb, basis);
}

// TODO: Can we avoid calling this so much
public float3 resolve_spectrum<let MODE : SpectralMode>(
    float4 contribution_over_hero_density,
    float4 density_over_hero_density,
    SpectrumBasis basis)
{
    if (MODE == SpectralMode.RGB)
        return contribution_over_hero_density.rgb;

    // Note to future Pema:
    // This is a single-sample MIS estimator using balance heuristic. Start with the heuristic:
    //   w_s(x) = n_s p_s(x) / Σ_i n_i p_i(x)
    // We have 4 strategies (treat each wavelength as the hero) and 1 single sample (n=1), so it becomes:
    //   w_s = p_s / Σ_i p_i = p_s / (p_0 + p_1 + p_2 + p_3)
    // The density of technique s is given by the product of the density of sampling the wavelength
    // of s (λ_s), and the probability of sampling the path given that we use λ_s to drive the BSDF:
    //   p_s = p(λ_s) p(X | λ_s)
    // The single-sample MIS estimator is given by:
    //   Σ_s w_s f_s / p_s =
    //   Σ_s (p_s / Σ_i p_i) f_s / p_s =
    //   Σ_s f_s / (Σ_i p_i)
    // We can pull p(λ_s) out of each p_i:
    //   Σ_s f_s / (p(λ_s) Σ_i p(X | λ_i))
    // And rewrite a bit:
    //   Σ_s f_s / ((1/VISIBLE_RANGE) Σ_i p(X | λ_i)) = 
    //   VISIBLE_RANGE * Σ_s f_s / Σ_i p(X | λ_i)
    // This function receives:
    //   contribution_over_hero_density_s = f_s / p(X | λ_0)    (call it c_s)
    //   density_over_hero_density_s = p(X | λ_s) / p(X | λ_0)  (call it d_s)
    // Substitute in these in the remaining estimator sum:
    //   Σ_s f_s / Σ_i p(X | λ_i) =
    //   Σ_s (c_s p(X | λ_0)) / (p(X | λ_0) Σ_i d_i) =
    //   Σ_s c_s / (Σ_i d_i)

    // weighted = c_s / (Σ_i d_i)
    float4 weighted = contribution_over_hero_density / dot(density_over_hero_density, 1.0);
    // weighted_xyz = Σ_s cie(λ_s) weighted 
    // (conversion does the sum, color matching omitted in derivation for brevity)
    float3 weighted_xyz = spectrum_to_xyz(weighted, basis);
    // divide by p(λ_s)
    float3 estimate = weighted_xyz * VISIBLE_WAVELENGTH_RANGE;

    return xyz_to_ergb(estimate);
}

// See tools/fit_rgb_to_spectrum_upsample.py
public SpectrumBasis get_spectrum_basis(float4 wavelengths)
{
    float4 normalize_wavelengths = (wavelengths - MIN_VISIBLE_WAVELENGTH) / VISIBLE_WAVELENGTH_RANGE;
    float4 position = normalize_wavelengths * (RGB_TO_SPECTRUM_LUT_SIZE - 1);
    int4 index = min(int4(position), RGB_TO_SPECTRUM_LUT_SIZE - 2);
    float4 fraction = position - float4(index);
    SpectrumBasis basis;
    basis.wavelengths = wavelengths;
    basis.primaries[0] = lerp(RGB_TO_SPECTRUM_LUT[index.x], RGB_TO_SPECTRUM_LUT[index.x + 1], fraction.x);
    basis.primaries[1] = lerp(RGB_TO_SPECTRUM_LUT[index.y], RGB_TO_SPECTRUM_LUT[index.y + 1], fraction.y);
    basis.primaries[2] = lerp(RGB_TO_SPECTRUM_LUT[index.z], RGB_TO_SPECTRUM_LUT[index.z + 1], fraction.z);
    basis.primaries[3] = lerp(RGB_TO_SPECTRUM_LUT[index.w], RGB_TO_SPECTRUM_LUT[index.w + 1], fraction.w);
    cie_matching(wavelengths, basis.xyz[0], basis.xyz[1], basis.xyz[2], basis.xyz[3]);
    return basis;
}

// Cheap approximation of rgb_to_spectrum using 3 gaussians
public SpectrumBasis get_spectrum_basis_gaussian(float4 wavelengths)
{
    float4 tr = (wavelengths - 671.6054) / 26.5297;
    float4 tg = (wavelengths - 508.8906) / 26.5518;
    float4 tb = (wavelengths - 442.8418) / 43.3015;
    float4 r = exp(-tr * tr);
    float4 g = exp(-tg * tg);
    float4 b = exp(-tb * tb);
    float4 weight = rcp(r + g + b);
    SpectrumBasis basis;
    basis.wavelengths = wavelengths;
    basis.primaries[0] = float3(r.x, g.x, b.x) * weight.x;
    basis.primaries[1] = float3(r.y, g.y, b.y) * weight.y;
    basis.primaries[2] = float3(r.z, g.z, b.z) * weight.z;
    basis.primaries[3] = float3(r.w, g.w, b.w) * weight.w;
    cie_matching(wavelengths, basis.xyz[0], basis.xyz[1], basis.xyz[2], basis.xyz[3]);
    return basis;
}

// Sample 1 "hero" wavelength and 3 additional evenly spaced wavelengths
// Hero Wavelength Spectral Sampling, Wilkie et al, 2014, (eq. 5)
// https://cgg.mff.cuni.cz/wp-content/uploads/2021/05/WNDWH14HWSS.pdf
public float4 sample_wavelengths(float rng)
{
    float min_w = MIN_VISIBLE_WAVELENGTH;
    float range = VISIBLE_WAVELENGTH_RANGE;

    float hero_wavelength = min_w + rng * range;

    return float4(
        hero_wavelength,
        min_w + ((hero_wavelength - min_w + 1 * range / 4) % range),
        min_w + ((hero_wavelength - min_w + 2 * range / 4) % range),
        min_w + ((hero_wavelength - min_w + 3 * range / 4) % range));
}

// Dispersive IOR using the convention defined by glTF spec:
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_dispersion#overview
public float4 dispersive_ior(float ior, float dispersion, float4 wavelengths)
{
    // glTF defines dispersion such that 1/abbe_number = dispersion/20
    float rcp_abbe_number = dispersion / 20.0;
    return max(ior + (ior - 1.0) * rcp_abbe_number * (523655 / (wavelengths * wavelengths) - 1.5168), 1.0);
}

// CIE XYZ -> linear sRGB, but scaled such that XYZ (1,1,1) = sRGB (1,1,1). Same primaries.
// White point is illuminant E (https://en.wikipedia.org/wiki/Standard_illuminant#Illuminant_E)
// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/cmd/rgb2spec_opt.cpp#L205
public float3 xyz_to_ergb(float3 xyz)
{
    float3 rgb;
    rgb.r = dot(float3(2.689989, -1.276020, -0.413844), xyz);
    rgb.g = dot(float3(-1.022095, 1.978261, 0.043821), xyz);
    rgb.b = dot(float3(0.061203, -0.224411, 1.162859), xyz);
    return rgb;
}

// === Helpers ===

static const float MIN_VISIBLE_WAVELENGTH = 380.0;
static const float MAX_VISIBLE_WAVELENGTH = 780.0;
static const float VISIBLE_WAVELENGTH_RANGE = MAX_VISIBLE_WAVELENGTH - MIN_VISIBLE_WAVELENGTH;

// Integral of CIE matching functions (the ones below) over the visible spectrum (as defined above)
// tools/integrate_cie_matching.py
static const float3 CIE_MATCHING_INTEGRAL = float3(106.713669, 106.943789, 106.806304);

float4 cie_x_matching(float4 wavelengths)
{
    float4 t1 = (wavelengths-442.0f)*select(wavelengths<442.0f, 0.0624f, 0.0374f);
    float4 t2 = (wavelengths-599.8f)*select(wavelengths<599.8f, 0.0264f, 0.0323f);
    float4 t3 = (wavelengths-501.1f)*select(wavelengths<501.1f, 0.0490f, 0.0382f);
    return 0.362f*exp(-0.5f*t1*t1) + 1.056f*exp(-0.5f*t2*t2)
                                   - 0.065f*exp(-0.5f*t3*t3);
}

float4 cie_y_matching(float4 wavelengths)
{
    float4 t1 = (wavelengths-568.8f)*select(wavelengths<568.8f, 0.0213f, 0.0247f);
    float4 t2 = (wavelengths-530.9f)*select(wavelengths<530.9f, 0.0613f, 0.0322f);
    return 0.821f*exp(-0.5f*t1*t1) + 0.286f*exp(-0.5f*t2*t2);
}

float4 cie_z_matching(float4 wavelengths)
{
    float4 t1 = (wavelengths-437.0f)*select(wavelengths<437.0f, 0.0845f, 0.0278f);
    float4 t2 = (wavelengths-459.0f)*select(wavelengths<459.0f, 0.0385f, 0.0725f);
    return 1.217f*exp(-0.5f*t1*t1) + 0.681f*exp(-0.5f*t2*t2);
}

// Wavelength -> CIE XYZ tristimulus values
// Simple Analytic Approximations to the CIE XYZ Color Matching Functions, Wyman et al, 2013
// https://jcgt.org/published/0002/02/01/
void cie_matching(float4 wavelengths,
    out float3 xyz_0, out float3 xyz_1,
    out float3 xyz_2, out float3 xyz_3)
{
    float4 x = cie_x_matching(wavelengths) * rcp(CIE_MATCHING_INTEGRAL.x);
    float4 y = cie_y_matching(wavelengths) * rcp(CIE_MATCHING_INTEGRAL.y);
    float4 z = cie_z_matching(wavelengths) * rcp(CIE_MATCHING_INTEGRAL.z);
    xyz_0 = float3(x.x, y.x, z.x);
    xyz_1 = float3(x.y, y.y, z.y);
    xyz_2 = float3(x.z, y.z, z.z);
    xyz_3 = float3(x.w, y.w, z.w);
}

// Given an (e)RGB color, how much light is reflected at each of the sampled wavelengths?
float4 ergb_to_spectrum(float3 rgb, SpectrumBasis basis)
{
    return float4(dot(rgb, basis.primaries[0]), dot(rgb, basis.primaries[1]),
                  dot(rgb, basis.primaries[2]), dot(rgb, basis.primaries[3]));
}

// Spectrum to CIE XYZ tristimulus values at the sampled wavelengths.
// https://en.wikipedia.org/wiki/CIE_1931_color_space#Analytical_approximation
float3 spectrum_to_xyz(float4 values, SpectrumBasis basis)
{
    return values.x * basis.xyz[0] + values.y * basis.xyz[1] +
           values.z * basis.xyz[2] + values.w * basis.xyz[3];
}

// tools/fit_rgb_to_spectrum_upsample.py
static const int RGB_TO_SPECTRUM_LUT_SIZE = 81;
static const float3 RGB_TO_SPECTRUM_LUT[RGB_TO_SPECTRUM_LUT_SIZE] = {
    float3(0.000000, 0.000000, 1.000000), float3(0.000000, 0.000000, 1.000000),
    float3(0.000000, 0.027893, 0.972107), float3(0.000000, 0.110509, 0.889491),
    float3(0.156819, 0.036305, 0.806875), float3(0.275741, 0.000000, 0.724259),
    float3(0.358357, 0.000000, 0.641643), float3(0.275741, 0.000000, 0.724259),
    float3(0.193125, 0.000000, 0.806875), float3(0.110509, 0.000000, 0.889491),
    float3(0.027893, 0.000000, 0.972107), float3(0.000000, 0.000000, 1.000000),
    float3(0.000000, 0.000000, 1.000000), float3(0.000000, 0.000000, 1.000000),
    float3(0.000000, 0.000000, 1.000000), float3(0.000000, 0.000000, 1.000000),
    float3(0.000000, 0.002307, 0.997693), float3(0.000000, 0.084923, 0.915077),
    float3(0.000000, 0.167538, 0.832462), float3(0.000000, 0.250154, 0.749846),
    float3(0.000000, 0.332770, 0.667230), float3(0.000000, 0.415386, 0.584614),
    float3(0.000000, 0.498002, 0.501998), float3(0.000000, 0.580618, 0.419382),
    float3(0.000000, 0.663234, 0.336766), float3(0.000000, 0.745850, 0.254150),
    float3(0.000000, 0.828466, 0.171534), float3(0.000000, 0.911082, 0.088918),
    float3(0.000000, 0.993698, 0.006302), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 0.997512, 0.002488), float3(0.073219, 0.855040, 0.071741),
    float3(0.230039, 0.712569, 0.057392), float3(0.386858, 0.570098, 0.043044),
    float3(0.543677, 0.427626, 0.028696), float3(0.700497, 0.285155, 0.014348),
    float3(0.857316, 0.142684, 0.000000), float3(0.999787, 0.000213, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(1.000000, 0.000000, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(1.000000, 0.000000, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(1.000000, 0.000000, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(1.000000, 0.000000, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(1.000000, 0.000000, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(1.000000, 0.000000, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(1.000000, 0.000000, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(1.000000, 0.000000, 0.000000),
    float3(1.000000, 0.000000, 0.000000), float3(0.857529, 0.142471, 0.000000),
    float3(0.700709, 0.284943, 0.014348), float3(0.543890, 0.427414, 0.028696),
    float3(0.387071, 0.569885, 0.043044), float3(0.230251, 0.712356, 0.057392),
    float3(0.073432, 0.854828, 0.071741), float3(0.000000, 0.997299, 0.002701),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000), float3(0.000000, 1.000000, 0.000000),
    float3(0.000000, 1.000000, 0.000000),
};