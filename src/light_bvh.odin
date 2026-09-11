package main

import "core:math/linalg"
import "core:math"
import "gpu"

// TODO: Compress this to be smaller
Light_Bounds :: struct {
	aabb: AABB,
	direction: [3]f32,
	cos_cone_angle: f32,
	power: f32,
	double_sided: b32,
}

// https://pbr-book.org/4ed/Geometry_and_Transformations/Spherical_Geometry
union_cone :: proc(direction_a: [3]f32, cos_cone_angle_a: f32, direction_b: [3]f32, cos_cone_angle_b: f32) -> (direction: [3]f32, cos_cone_angle: f32)
{
	// One cone empty
	if math.is_inf(cos_cone_angle_a) {
		return direction_b, cos_cone_angle_b
	}
	if math.is_inf(cos_cone_angle_b) {
		return direction_a, cos_cone_angle_a
	}

	// One cone contain inside other
	cone_angle_a := math.acos(cos_cone_angle_a)
	cone_angle_b := math.acos(cos_cone_angle_b)
	angle_between := linalg.angle_between(direction_a, direction_b)
	if min(angle_between + cone_angle_b, math.PI) <= cone_angle_a {
		return direction_a, cos_cone_angle_a
	}
	if min(angle_between + cone_angle_a, math.PI) <= cone_angle_b {
		return direction_b, cos_cone_angle_b
	}

	// Get merged cone angle
	cone_angle := (cone_angle_a + cone_angle_b + angle_between) * 0.5
	if cone_angle >= math.PI {
		return {0,0,1}, -1 // entire sphere
	}

	// Get merged cone direction.
	// Rotate dir_a inside the plane containing the both cone dirs, towards dir_b,
	// by rotation_angle (to sit at the center of the arc spanning both cones far edges)
	rotation_axis := linalg.cross(direction_a, direction_b)
	if linalg.length2(rotation_axis) == 0 {
		return {0,0,1}, -1 // entire sphere
	}
	rotation_axis = linalg.normalize(rotation_axis)
	rotation_angle := cone_angle - cone_angle_a
	direction = linalg.normalize(
		direction_a * math.cos(rotation_angle) +
		linalg.cross(rotation_axis, direction_a) * math.sin(rotation_angle))
	cos_cone_angle = math.cos(cone_angle)
	return direction, cos_cone_angle
}

union_light_bounds :: proc(a: Light_Bounds, b: Light_Bounds) -> Light_Bounds
{
	if a.power == 0.0 do return b
	if b.power == 0.0 do return a

	direction, cos_cone_angle := union_cone(a.direction, a.cos_cone_angle, b.direction, b.cos_cone_angle)

	return Light_Bounds {
		aabb = aabb_union(a.aabb, b.aabb),
		direction = direction,
		cos_cone_angle = cos_cone_angle,
		power = a.power + b.power,
		double_sided = a.double_sided || b.double_sided,
	}
}

Light_BVH_Node :: struct {
	bounds: Light_Bounds,
	data: bit_field u32 {
		child_or_light_index: u32 | 31,
		is_leaf: b32 | 1,
	}
}

light_bvh_leaf_new :: proc(light_index: u32, bounds: Light_Bounds) -> Light_BVH_Node {
	return Light_BVH_Node {
		bounds = bounds,
		data = {
			child_or_light_index = light_index,
			is_leaf = true
		}
	}
}

light_bvh_inner_new :: proc(second_child_index: u32, bounds: Light_Bounds) -> Light_BVH_Node {
	return Light_BVH_Node {
		bounds = bounds,
		data = {
			child_or_light_index = second_child_index,
			is_leaf = false
		}
	}
}

Emissive_Triangle :: struct {
	instance_index: u32,
	primitive_index: u32,
}

Light_BVH :: struct {
	lights: GPU_List(Emissive_Triangle),
	nodes: GPU_List(Light_BVH_Node),
	light_to_bit_trail: GPU_List(u32),
	instance_to_light: GPU_List(u32),
}

Indexed_Light :: struct {
	index: u32,
	bounds: Light_Bounds,
}

// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/lightsamplers.h#L383
// Originally from "Importance Sampling of Many Lights with Adaptive Tree Splitting", Conty Estevez and Kulla, 2018
// Tries to minimize, on both sides of the split:
// - how spread out lights are (cluster surface area)
// - how much their normals point apart
// - how badly the split axis fits the clusters shape
// weighted by the power in the cluster
light_bvh_split_cost :: proc(split_bounds: Light_Bounds, light_bounds: AABB, dim: int) -> f32 {
    theta_o := math.acos(split_bounds.cos_cone_angle)
    theta_w := min(theta_o + math.PI / 2.0, math.PI);
    sin_theta_o := math.sqrt(1 - (split_bounds.cos_cone_angle * split_bounds.cos_cone_angle));
    m_omega := 2 * math.PI * (1 - split_bounds.cos_cone_angle) +
                   math.PI / 2 *
                            (2 * theta_w * sin_theta_o - math.cos(theta_o - 2 * theta_w) -
                             2 * theta_o * sin_theta_o + split_bounds.cos_cone_angle);
    diagonal := aabb_diagonal(light_bounds)
    kr := max(diagonal.x, max(diagonal.y, diagonal.z)) / diagonal[dim];
    return split_bounds.power * m_omega * kr * aabb_surface_area(split_bounds.aabb);
}

light_bvh_build :: proc(bvh: ^Light_BVH, build_lights: []Indexed_Light, bit_trail: u32, start: int, end: int, depth: int) -> Indexed_Light {
	// leaf
	if end - start == 1 {
		node_index := bvh.nodes.length
		light := build_lights[start]
		gpu_list_add(&bvh.nodes, light_bvh_leaf_new(light.index, light.bounds))
		bvh.light_to_bit_trail.array[light.index] = bit_trail
		return {node_index, light.bounds}
	}

	// calc bounds lights and bounds of light centroids
	bounds := aabb_empty()
	centroid_bounds := aabb_empty()
	for i := start; i < end; i += 1 {
		light := build_lights[i]
		bounds = aabb_union(bounds, light.bounds.aabb)
		centroid_bounds = aabb_encapsulate(centroid_bounds, aabb_centroid(light.bounds.aabb))
	}

	// find best split
	num_buckets :: 16
	min_split_cost := math.INF_F32
	min_split_bucket := -1
	min_split_dim := -1
	for dim := 0; dim < 3; dim += 1 {
		if centroid_bounds.min[dim] == centroid_bounds.max[dim] do continue // 0 size

		// bounds for each bucket
		buckets := [num_buckets]Light_Bounds{}
		for i := start; i < end; i += 1 {
			centroid := aabb_centroid(build_lights[i].bounds.aabb)
			axis_offset := aabb_offset(centroid_bounds, centroid)[dim]
			bucket := int(min(num_buckets * axis_offset, num_buckets - 1))
			buckets[bucket] = union_light_bounds(buckets[bucket], build_lights[i].bounds)
		}

		// calc costs
		costs := [num_buckets-1]f32{}
		for i := 0; i < num_buckets - 1; i += 1 {
			left_bounds, right_bounds: Light_Bounds
			for j := 0; j <= i; j += 1 {
				left_bounds = union_light_bounds(left_bounds, buckets[j])
			}
			for j := i + 1; j < num_buckets; j += 1 {
				right_bounds = union_light_bounds(right_bounds, buckets[j])
			}
			costs[i] = light_bvh_split_cost(left_bounds, bounds, dim) + light_bvh_split_cost(right_bounds, bounds, dim)
		}

		// take best split
		for i := 1; i < num_buckets - 1; i += 1 {
			if costs[i] > 0 && costs[i] < min_split_cost {
				min_split_cost = costs[i]
				min_split_bucket = i
				min_split_dim = dim
			}
		}
	}

	// apply the split
	midpoint: int
	if min_split_dim < 0 {
		midpoint = (start + end) / 2
	} else {
		midpoint = start
		// partition by the split, get midpoint
		for i := start; i < end; i += 1 {
			centroid := aabb_centroid(build_lights[i].bounds.aabb)
			axis_offset := aabb_offset(centroid_bounds, centroid)[min_split_dim]
			bucket := int(min(num_buckets * axis_offset, num_buckets - 1))
			if bucket <= min_split_bucket {
				build_lights[i], build_lights[midpoint] = build_lights[midpoint], build_lights[i]
				midpoint += 1
			}
		}
		// fallback
		if midpoint == start || midpoint == end {
			midpoint = (start + end) / 2
		}
	}

	// reserve this node, then recurse so the first child is directly after it
	node_index := bvh.nodes.length
	gpu_list_add(&bvh.nodes, Light_BVH_Node{})
	left_child := light_bvh_build(bvh, build_lights, bit_trail, start, midpoint, depth + 1)
	right_child := light_bvh_build(bvh, build_lights, bit_trail | (u32(1) << u32(depth)), midpoint, end, depth + 1)

	// union them into this node
	node_bounds := union_light_bounds(left_child.bounds, right_child.bounds)
	bvh.nodes.array[node_index] = light_bvh_inner_new(right_child.index, node_bounds)
	return {node_index, node_bounds}
}

light_bvh_new :: proc(gp: ^Geometry_Pool, mp: ^Material_Pool) -> Light_BVH {
	lights := gpu_list_new(Emissive_Triangle)
	nodes := gpu_list_new(Light_BVH_Node)
	light_to_bit_trail := gpu_list_new(u32)
	instance_to_light := gpu_list_new(u32)

	no_lights := make([]u32, gp.instance_to_pool.length)
	defer delete(no_lights)
	for i in 0..<len(no_lights) do no_lights[i] = max(u32)
	gpu_list_add_range(&instance_to_light, no_lights)

	// add emissive triangles
	for instance_index in gp.emissive_instance_indices.array[:gp.emissive_instance_indices.length] {
		info := gp.instance_to_pool.array[instance_index]
		instance_to_light.array[instance_index] = lights.length
		triangles := make([]Emissive_Triangle, info.index_count / 3)
		defer delete(triangles)
		for triangle_index := u32(0); triangle_index < info.index_count / 3; triangle_index += 1 {
			triangles[triangle_index] = Emissive_Triangle {
				instance_index = instance_index,
				primitive_index = triangle_index
			}
		}
		gpu_list_add_range(&lights, triangles)
	}

	// compute bounds
	build_lights := make([]Indexed_Light, lights.length)
	defer delete(build_lights)
	for light_index := 0; light_index < int(lights.length); light_index += 1 {
		tri := lights.array[light_index]
		instance := gp.instance_to_pool.array[tri.instance_index]
		material_index := instance.material_index
		mat := mp.materials.array[material_index]

		base := instance.index_offset + tri.primitive_index * 3
		i0 := instance.vertex_offset + gp.indices.array[base + 0]
		i1 := instance.vertex_offset + gp.indices.array[base + 1]
		i2 := instance.vertex_offset + gp.indices.array[base + 2]

		transform := gp.transforms.array[tri.instance_index]
		v0 := gp.vertices.array[i0]
		v1 := gp.vertices.array[i1]
		v2 := gp.vertices.array[i2]
		m0 := transform * [4]f32{v0.x, v0.y, v0.z, 1}
		m1 := transform * [4]f32{v1.x, v1.y, v1.z, 1}
		m2 := transform * [4]f32{v2.x, v2.y, v2.z, 1}
		p0 := [3]f32{m0[0, 0], m0[1, 0], m0[2, 0]}
		p1 := [3]f32{m1[0, 0], m1[1, 0], m1[2, 0]}
		p2 := [3]f32{m2[0, 0], m2[1, 0], m2[2, 0]}

		max_emission := max(mat.emission.x, max(mat.emission.y, mat.emission.z))
		cross := linalg.cross(p1 - p0, p2 - p0)
		tri_area := 0.5 * linalg.length(cross)
		power := max_emission * math.PI * tri_area

		aabb := AABB {
			min = linalg.min(p0, linalg.min(p1, p2)),
			max = linalg.max(p0, linalg.max(p1, p2))
		}

		direction := linalg.normalize0(cross)

		build_lights[light_index] = Indexed_Light {
			index = u32(light_index),
			bounds = Light_Bounds {
				aabb = aabb,
				direction = direction,
				cos_cone_angle = 1.0,
				power = power,
				double_sided = mat.double_sided,
			}
		}
	}

	// Resize bit trail to fit
	empty_bits := make([]u32, lights.length)
	defer delete(empty_bits)
	gpu_list_add_range(&light_to_bit_trail, empty_bits)

	// Build bvh
	bvh := Light_BVH {
		lights = lights,
		nodes = nodes,
		light_to_bit_trail = light_to_bit_trail,
		instance_to_light = instance_to_light,
	}
	if lights.length > 0 {
		light_bvh_build(&bvh, build_lights, 0, 0, int(lights.length), 0)
	}
	return bvh
}

light_bvh_commit :: proc(bvh: ^Light_BVH, cmd: ^gpu.Cmd) {
	gpu_list_commit(&bvh.lights, cmd)
	gpu_list_commit(&bvh.nodes, cmd)
	gpu_list_commit(&bvh.light_to_bit_trail, cmd)
	gpu_list_commit(&bvh.instance_to_light, cmd)
}

light_bvh_delete :: proc(bvh: ^Light_BVH) {
	gpu_list_delete(&bvh.lights)
	gpu_list_delete(&bvh.nodes)
	gpu_list_delete(&bvh.light_to_bit_trail)
	gpu_list_delete(&bvh.instance_to_light)
}
