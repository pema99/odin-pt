module rng;

// Once:
public static const uint RNG_DIM_JITTER = 0;
// Per bounce:
public static const uint RNG_DIMS_PER_BOUNCE = 2;
public static const uint RNG_DIM_SCATTER = 1;
public static const uint RNG_DIM_RUSSIAN_ROULETTE = 2;

// From https://github.com/skeeto/hash-prospector/issues/19
public uint low_bias_hash(uint value)
{
    value ^= value >> 16;
    value *= 0x21f0aaad;
    value ^= value >> 15;
    value *= 0xd35a2d97;
    value ^= value >> 15;
    return value;
}

public uint pixel_hash(uint2 pixelCoord)
{
    return low_bias_hash((pixelCoord.x & 0xFFFF) | (pixelCoord.y << 16));
}

// https://andrew-helmer.github.io/tree-shuffling/
uint shuffle_index(uint x, uint seed)
{
    x = reversebits(x);
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    x = reversebits(x);
    return x;
}

// Somewhat based on https://jcgt.org/published/0011/01/04/
public float2 kronecker_2d(uint seed, uint sampleIndex, uint dimension)
{
    sampleIndex = shuffle_index(sampleIndex, low_bias_hash(dimension));
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

public float2 kronecker_2d(uint2 pixelCoord, uint sampleIndex, uint dimension)
{
    return kronecker_2d(pixel_hash(pixelCoord), sampleIndex, dimension);
}
