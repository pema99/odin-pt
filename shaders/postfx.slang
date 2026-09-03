RWTexture2D<float4> input;
RWTexture2D<float4> output;
uniform Tonemapper tonemapper;
uniform float exposure;
uniform uint2 screen_size;

// Narkowicz ACES https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
float3 aces_narkowicz(float3 x)
{
    x *= 0.6; // overexposed without this!!
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Hill ACES https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
float3 aces_hill(float3 x)
{
    x *= 1.8;

    float3x3 aces_input = float3x3(
        float3(0.59719, 0.35458, 0.04823),
        float3(0.07600, 0.90834, 0.01566),
        float3(0.02840, 0.13383, 0.83777));
    float3x3 aces_output = float3x3(
        float3(1.60475, -0.53108, -0.07367),
        float3(-0.10208, 1.10813, -0.00605),
        float3(-0.00327, -0.07276, 1.07602));

    float3 color = mul(aces_input, x);

    float3 a = color * (color + 0.0245786) - 0.000090537;
    float3 b = color * (0.983729 * color + 0.4329510) + 0.238081;
    color = a / b;

    color = mul(aces_output, color);

    return saturate(color);
}

float3 reinhard(float3 x)
{
    return x / (x + 1.0);
}

float3 neutral_curve(float3 x, float a, float b, float c, float d, float e, float f)
{
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

// from unity
float3 neutral_tonemap(float3 x)
{
    // Tonemap
    float a = 0.2;
    float b = 0.29;
    float c = 0.24;
    float d = 0.272;
    float e = 0.02;
    float f = 0.3;
    float white_level = 5.3;
    float white_clip = 1.0;

    float3 white_scale = 1.0 / neutral_curve(white_level, a, b, c, d, e, f);
    float3 result = neutral_curve(x * white_scale, a, b, c, d, e, f);
    result *= white_scale;

    // Post-curve white point adjustment
    result /= white_clip;

    return result;
}

float3 uncharted_partial(float3 x)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// 64.github.io/tonemapping
float3 uncharted(float3 v)
{
    float exposure_bias = 2.0;
    float3 curr = uncharted_partial(v * exposure_bias);

    float3 W = 11.2;
    float3 white_scale = 1.0 / uncharted_partial(W);
    return curr * white_scale;
}

enum Tonemapper
{
    None,
    Reinhard,
    AcesNarkowicz,
    AcesHill,
    Neutral,
    Uncharted,
};

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint2 id: SV_DispatchThreadID)
{
    if (any(id.xy >= screen_size))
        return;

    float4 value = input[id.xy];
    value.rgb *= exp2(exposure);
    switch (tonemapper)
    {
        case Tonemapper.Reinhard:
            value.rgb = reinhard(value.rgb);
            break;
        case Tonemapper.AcesNarkowicz:
            value.rgb = aces_narkowicz(value.rgb);
            break;
        case Tonemapper.AcesHill:
            value.rgb = aces_hill(value.rgb);
            break;
        case Tonemapper.Neutral:
            value.rgb = neutral_tonemap(value.rgb);
            break;
        case Tonemapper.Uncharted:
            value.rgb = uncharted(value.rgb);
            break;
        default: 
            break; // none
    }
    output[id.xy] = value;
}
