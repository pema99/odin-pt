module sampling;

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