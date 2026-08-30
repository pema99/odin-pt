module common;

public static const float PI = 3.14159265358979323846;
public static const uint MAX_UINT = 0xFFFFFFFF;

public float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}