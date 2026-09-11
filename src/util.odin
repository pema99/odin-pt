package main

import "gpu"

import "core:math"
import "core:math/linalg"

// === GPU list ===
GPU_List :: struct($T: typeid) {
    array: []T,
    buffer: gpu.Buffer,
    length: u32
}

gpu_list_new :: proc($T: typeid) -> GPU_List(T) {
    return GPU_List(T) {
        array = make([]T, 16),
        buffer = gpu.create_buffer(16 * size_of(T), writable = true),
        length = 0,
    }
}

gpu_list_delete :: proc(list: ^GPU_List($T)) {
    delete(list.array)
    gpu.destroy_buffer(list.buffer)
}

gpu_list_add :: proc(list: ^GPU_List($T), item: T) {
    if list.length >= u32(len(list.array)) {
        new_array := make([]T, len(list.array) * 2)
        copy(new_array, list.array)
        delete(list.array)
        list.array = new_array
        gpu.destroy_buffer(list.buffer)
        list.buffer = gpu.create_buffer(len(list.array) * size_of(T), writable = true)
    }
    list.array[list.length] = item
    list.length += 1
}

gpu_list_add_range :: proc(list: ^GPU_List($T), items: []T) {
    needed := list.length + u32(len(items))
    capacity := u32(len(list.array))

    if needed > capacity {
        for u32(capacity) < needed {
            capacity *= 2
        }
        new_array := make([]T, capacity)
        copy(new_array, list.array[:list.length])
        delete(list.array)
        list.array = new_array
        gpu.destroy_buffer(list.buffer)
        list.buffer = gpu.create_buffer(uint(capacity * size_of(T)), writable = true)
    }

    copy(list.array[list.length:], items)
    list.length = needed
}

gpu_list_remove :: proc(list: ^GPU_List($T), index: u32) {
    // TODO
}

gpu_list_commit :: proc(list: ^GPU_List($T), cmd: ^gpu.Cmd) {
    gpu.upload_buffer(cmd, list.buffer, list.array[:list.length])
}

// === AABB ===
AABB :: struct {
	min, max: [3]f32
}

aabb_empty :: proc() -> AABB {
	return {min = math.INF_F32, max = math.NEG_INF_F32}
}

aabb_union :: proc(a, b: AABB) -> AABB {
	return {linalg.min(a.min, b.min), linalg.max(a.max, b.max)}
}

aabb_encapsulate:: proc(a: AABB, p: [3]f32) -> AABB {
	return {linalg.min(a.min, p), linalg.max(a.max, p)}
}

aabb_centroid :: proc(b: AABB) -> [3]f32 {
	return (b.min + b.max) * 0.5
}

// position of p within the bounds in [0; 1]
aabb_offset :: proc(b: AABB, p: [3]f32) -> [3]f32 {
	o := p - b.min
	for i in 0..<3 {
		if b.max[i] > b.min[i] do o[i] /= b.max[i] - b.min[i]
	}
	return o
}

aabb_diagonal :: proc(b: AABB) -> [3]f32 {
	return b.max - b.min
}

aabb_surface_area :: proc(b: AABB) -> f32 {
	d := b.max - b.min
	return 2 * (d.x*d.y + d.x*d.z + d.y*d.z)
}
