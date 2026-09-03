import bsdf;
import geometry;
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
    public float3 attenuation_color;
    public float attenuation_distance;
    public uint albedo_texture_index;
    public uint emission_texture_index;
    public uint metallic_texture_index;
    public uint roughness_texture_index;
    public uint normal_texture_index;
    public BSDFType bsdf_type;
};

public struct HitInfo
{
    public float3 position;
    public float3 normal;
    public float3 geometric_normal;
    public uint triangle_count;
    public float triangle_area;
};

StructuredBuffer<MaterialInfo> materials;
ParameterBlock<Texture2D[]> textures;
SamplerState texture_sampler;

public void get_hit_info(Hit hit, float3 ray_origin, float3 ray_direction, out HitInfo hit_info, out MaterialInfo material)
{
    GeometryInfo geo = instance_to_pool[hit.instance_id];
    material = materials[geo.material_index];
    uint base = geo.index_offset + hit.primitive_index * 3;
    uint i0 = geo.vertex_offset + indices[base + 0];
    uint i1 = geo.vertex_offset + indices[base + 1];
    uint i2 = geo.vertex_offset + indices[base + 2];
    float3 barycentrics = float3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y);

    float3x3 to_world = (float3x3)hit.object_to_world;
    float3 p0 = positions[i0];
    float3 p1 = positions[i1];
    float3 p2 = positions[i2];
    float4 normal_and_area = triangle_normal_and_area(mul(to_world, p1 - p0), mul(to_world, p2 - p0));
    hit_info.geometric_normal = normal_and_area.xyz;
    if (determinant(to_world) < 0.0)
        hit_info.geometric_normal = -hit_info.geometric_normal;
    hit_info.position = hit.distance * ray_direction + ray_origin;

    float3x3 cofactors = float3x3(cross(to_world[1], to_world[2]), cross(to_world[2], to_world[0]), cross(to_world[0], to_world[1]));
    hit_info.normal = normalize(mul(cofactors, normals[i0] * barycentrics.x + normals[i1] * barycentrics.y + normals[i2] * barycentrics.z));

    // flip normals if they are pointing the wrong way
    if (dot(hit_info.geometric_normal, ray_direction) > 0.0)
        hit_info.geometric_normal = -hit_info.geometric_normal;
    if (dot(hit_info.normal, hit_info.geometric_normal) < 0.0)
        hit_info.normal = -hit_info.normal;

    // needed for MIS
    hit_info.triangle_count = geo.index_count / 3;
    hit_info.triangle_area = normal_and_area.w;

    bool has_albedo_texture = material.albedo_texture_index != MAX_UINT;
    bool has_emission_texture = material.emission_texture_index != MAX_UINT;
    bool has_metallic_texture = material.metallic_texture_index != MAX_UINT;
    bool has_roughness_texture = material.roughness_texture_index != MAX_UINT;
    bool has_normal_texture = material.normal_texture_index != MAX_UINT;
    if (!(has_albedo_texture || has_emission_texture || has_metallic_texture || has_roughness_texture || has_normal_texture))
        return;

    float2 uv = uvs[i0] * barycentrics.x + uvs[i1] * barycentrics.y + uvs[i2] * barycentrics.z;
    if (has_albedo_texture) {
        float3 albedo_tex = textures[material.albedo_texture_index].SampleLevel(texture_sampler, uv, 0).rgb;
        material.albedo *= albedo_tex;
    }
    if (has_emission_texture) {
        float3 emission_tex = textures[material.emission_texture_index].SampleLevel(texture_sampler, uv, 0).rgb;
        material.emission *= emission_tex;
    }
    if (has_metallic_texture) {
        float metallic_tex = textures[material.metallic_texture_index].SampleLevel(texture_sampler, uv, 0).r;
        material.metallic *= metallic_tex;
    }
    if (has_roughness_texture) {
        float roughness_tex = textures[material.roughness_texture_index].SampleLevel(texture_sampler, uv, 0).r;
        material.roughness *= roughness_tex;
    }
    if (has_normal_texture) {
        float3 normal_tex = textures[material.normal_texture_index].SampleLevel(texture_sampler, uv, 0).rgb;
        normal_tex = normalize(normal_tex * 2.0 - 1.0);
        float3 tangent = mul(to_world, tangents[i0] * barycentrics.x + tangents[i1] * barycentrics.y + tangents[i2] * barycentrics.z);
        tangent = normalize(tangent - hit_info.normal * dot(hit_info.normal, tangent));
        float3 bitangent = cross(hit_info.normal, tangent);
        hit_info.normal = normalize(
            tangent * normal_tex.x
            + bitangent * normal_tex.y
            + hit_info.normal * normal_tex.z);
    }
}

public float3 get_material_emission_fast(uint material_index, uint i0, uint i1, uint i2, float3 barycentrics)
{
    MaterialInfo mat = materials[material_index];
    if (mat.emission_texture_index == MAX_UINT)
        return mat.emission;
    float2 uv = uvs[i0] * barycentrics.x + uvs[i1] * barycentrics.y + uvs[i2] * barycentrics.z;
    return textures[mat.emission_texture_index].SampleLevel(texture_sampler, uv, 0).rgb;
}

public float4 get_material_absorption<let MODE : SpectralMode>(MaterialInfo mat, SpectrumBasis spectrum_basis)
{
    if (mat.attenuation_distance <= 0.0)
        return 0.0;

    float4 transmittance;
    if (MODE == SpectralMode.RGB)
        transmittance = float4(mat.attenuation_color, 1.0);
    else
        transmittance = upsample_ergb<MODE>(mat.attenuation_color, spectrum_basis);

    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_volume/README.md#attenuation
    return -log(transmittance) / mat.attenuation_distance;
}

public BSDF get_material_bsdf<let MODE : SpectralMode>(MaterialInfo mat, SpectrumBasis spectrum_basis, bool entering, float3 normal, float3 specular_normal)
{
    float4 albedo_spectrum = upsample_ergb<MODE>(mat.albedo, spectrum_basis);

    BSDF bsdf = make_disney_bsdf(
        albedo_spectrum,
        mat.metallic,
        mat.roughness,
        mat.index_of_refraction,
        MODE == SpectralMode.RGB ? 0.0 : mat.iridescence_factor,
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
        bsdf = make_glass_bsdf(
            mat.roughness,
            albedo_spectrum,
            ior,
            entering,
            MODE == SpectralMode.RGB ? 0.0 : mat.iridescence_factor,
            mat.iridescence_ior,
            mat.iridescence_thickness,
            specular_normal,
            spectrum_basis.wavelengths);
    }
    return bsdf;
}