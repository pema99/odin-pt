module material;

import bsdf;
import raytracing;
import common;
import spectrum;

public struct MaterialInfo
{
    public float3 albedo;
    public float3 emission;
    public float metallic;
    public float roughness;
    public float index_of_refraction;
    public float dispersion;
    public float iridescence_factor;
    public float iridescence_ior;
    public float iridescence_thickness;
    public uint albedo_texture_index;
    public uint emission_texture_index;
    public uint metallic_texture_index;
    public uint roughness_texture_index;
    public uint normal_texture_index;
    public BSDFType bsdf_type;
};
StructuredBuffer<MaterialInfo> materials;
ParameterBlock<Texture2D[]> textures;
SamplerState texture_sampler;

public MaterialInfo get_material(inout HitAttributes attributes)
{
    MaterialInfo mat = materials[attributes.material_index];
    if (mat.albedo_texture_index != MAX_UINT) {
        float3 albedo_tex = textures[mat.albedo_texture_index].SampleLevel(texture_sampler, attributes.uv, 0).rgb;
        mat.albedo *= albedo_tex;
    }
    if (mat.emission_texture_index != MAX_UINT) {
        float3 emission_tex = textures[mat.emission_texture_index].SampleLevel(texture_sampler, attributes.uv, 0).rgb;
        mat.emission *= emission_tex;
    }
    if (mat.metallic_texture_index != MAX_UINT) {
        float metallic_tex = textures[mat.metallic_texture_index].SampleLevel(texture_sampler, attributes.uv, 0).r;
        mat.metallic *= metallic_tex;
    }
    if (mat.roughness_texture_index != MAX_UINT) {
        float roughness_tex = textures[mat.roughness_texture_index].SampleLevel(texture_sampler, attributes.uv, 0).r;
        mat.roughness *= roughness_tex;
    }
    if (mat.normal_texture_index != MAX_UINT) {
        float3 normal_tex = textures[mat.normal_texture_index].SampleLevel(texture_sampler, attributes.uv, 0).rgb;
        normal_tex = normalize(normal_tex * 2.0 - 1.0);
        float3 bitangent = cross(attributes.normal, attributes.tangent);
        attributes.normal = normalize(
            attributes.tangent * normal_tex.x
            + bitangent * normal_tex.y
            + attributes.normal * normal_tex.z);
    }
    return mat;
}

public BSDF get_material_bsdf<let MODE : SpectralMode>(MaterialInfo mat, SpectrumBasis spectrum_basis, bool entering, float3 normal, float3 specular_normal)
{
    float4 albedo_spectrum = upsample_ergb<MODE>(mat.albedo, spectrum_basis);

    BSDF bsdf = make_disney_bsdf(
        albedo_spectrum,
        mat.metallic,
        mat.roughness,
        mat.index_of_refraction,
        mat.iridescence_factor,
        mat.iridescence_ior,
        mat.iridescence_thickness,
        normal,
        specular_normal,
        spectrum_basis.wavelengths);

    if (mat.bsdf_type == BSDFType::Glass)
    {
        float4 ior;
        if (MODE == SpectralMode.RGB)
            ior = mat.index_of_refraction;
        else
            ior = dispersive_ior(mat.index_of_refraction, mat.dispersion, spectrum_basis.wavelengths);
        bsdf = GlassBSDF(mat.roughness, albedo_spectrum, entering ? ior : rcp(ior), specular_normal);
    }
    return bsdf;
}