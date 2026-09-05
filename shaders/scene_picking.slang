import scene;
import geometry;
import common;

uniform float2 mouse_coord;
uniform uint2 screen_size;
uniform float fov;
cbuffer Camera
{
    float3 cam_pos;
    float3 cam_right;
    float3 cam_up;
    float3 cam_forward;
};

struct PickObject
{
    uint instance_id;
};

RWStructuredBuffer<PickObject> pick_object;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint2 id: SV_DispatchThreadID)
{
    float3 ray_origin = cam_pos;
    float3 ray_direction = primary_ray_direction(mouse_coord, screen_size, cam_right, cam_up, cam_forward, fov);

    Hit hit = trace_ray(ray_origin, ray_direction);

    PickObject pick = (PickObject)0;
    pick.instance_id = MAX_UINT;

    if (hit.is_valid())
    {
        pick.instance_id = hit.instance_id;
    }

    pick_object[0] = pick;
}