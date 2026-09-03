import common;

public void ortho_basis_from_vector(float3 n, out float3 b1, out float3 b2)
{
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = float3(b, sign + n.y * n.y * a, -n.y);
}

public float3x3 ortho_basis_from_vector(float3 n)
{
    float3 t, b;
    ortho_basis_from_vector(n, t, b);
    return transpose(float3x3(t, b, n));
}

public float3 cosine_hemisphere(float2 u, float3 normal)
{
    float a = sqrt(u.x);
    float b = 2.0f * PI * u.y;
    float3 localDir = float3(a * cos(b), a * sin(b), sqrt(1.0f - u.x));
    float3x3 basis = ortho_basis_from_vector(normal);
    return mul(basis, localDir);
}

// A Low-Distortion Map Between Triangle and Square, Heitz, 2019
// https://eheitzresearch.wordpress.com/749-2/
public void square_to_triangle(inout float x, inout float y)
{
    if (y > x) {
        x *= 0.5;
        y -= x;
    } else {
        y *= 0.5;
        x -= y;
    }
}

public float balance_heuristic(float density0, float density1)
{
    return density0 / (density0 + density1);
}

public float power_heuristic(float density0, float density1)
{
    return density0 * density0 / (density0 * density0 + density1 * density1);
}