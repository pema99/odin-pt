public static const float PI = 3.14159265358979323846;
public static const uint MAX_UINT = 0xFFFFFFFF;

public float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

public float max3(float3 c)
{
    return max(c.x, max(c.y, c.z));
}

public float max4(float4 c)
{
    return max(c.x, max(c.y, max(c.z, c.w)));
}

public float4 triangle_normal_and_area(float3 v1v0, float3 v2v0)
{
    float3 cross_product = cross(v1v0, v2v0);
    float cross_length = length(cross_product);
    return float4(cross_product * rcp(cross_length), 0.5 * cross_length);
}

public float4 triangle_normal_and_area(float3 v0, float3 v1, float3 v2)
{
    return triangle_normal_and_area(v1 - v0, v2 - v0);
}