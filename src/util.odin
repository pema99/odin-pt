package main

import "gpu"

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

gpu_list_remove :: proc(list: ^GPU_List($T), index: u32) {
    // TODO
}

gpu_list_commit :: proc(list: ^GPU_List($T), cmd: ^gpu.Cmd) {
    gpu.upload_buffer(cmd, list.buffer, list.array[:list.length])
}