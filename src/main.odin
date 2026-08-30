package main

import "core:strings"
import "core:os"
import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg"

import "vendor:glfw"

import "gpu"

import imgui "lib:imgui"

// Constants
WIDTH : u32 : 1280
HEIGHT : u32 : 720

// Globals
max_bounces : i32 = 20
fov : f32 = 60.0

Camera :: struct {
	pos:   [3]f32,
	yaw:   f32,
	pitch: f32,
}

// TODO:
// - Region for copytexture/copybuffer commands
// - Samplers, mipmaps
// - Async readback?
// - Denoising
// - More shading models
// - NaNs
// - Get rid of "bend towards view dir" normal hack
// - Normal maps
// - Skybox
// - NEE + MIS
// - Many lights
// - Russian roulette
main :: proc() {
	context.logger = log.create_console_logger()
	defer log.destroy_console_logger(context.logger)

	gpu.init(
        title = "Odin Path Tracer", 
        width = WIDTH, height = HEIGHT,
        resizable = true,
        use_imgui = true,
        vsync = false,
        validation = ODIN_DEBUG)
	defer gpu.cleanup()

    shader_path := "shaders/megakernel.slang"
    last_shader_write, _ := os.last_write_time_by_name(shader_path)
	trace := gpu.compile_shader(shader_path)
	defer gpu.destroy_shader(trace)

	output := gpu.create_texture(WIDTH, HEIGHT, .R32G32B32A32_SFLOAT, writable = true)
	defer gpu.destroy_texture(output)

	cmd := gpu.create_cmd()
	defer gpu.destroy_cmd(cmd)

    scene_path := "assets/splash.glb"
    last_scene_write, _ := os.last_write_time_by_name(scene_path)
    load_cmd := gpu.create_cmd()
    scene := scene_load(strings.unsafe_string_to_cstring(scene_path), load_cmd)
    defer scene_delete(&scene)
	gpu.execute_cmd(load_cmd)
	gpu.destroy_cmd(load_cmd)

	cam := Camera{pos = {3, 2.5, 3}, yaw = -0.55, pitch = -0.35}
	last_mouse: [2]f64
	last_mouse.x, last_mouse.y = glfw.GetCursorPos(gpu.get_window())
	last_time := glfw.GetTime()

	kernel_size := gpu.get_kernel_size(trace, "main")
	num_groups := ([2]u32{WIDTH, HEIGHT} + kernel_size.xy - 1) / kernel_size.xy

    sample_count: u32 = 0
	for frame: u32 = 0; gpu.is_running(); frame += 1 {
		gpu.start_frame()

        now := glfw.GetTime()
        delta_time := min(f32(now - last_time), 0.1)
        last_time = now

        reset := update_camera(&cam, &last_mouse, delta_time)

		reset |= do_gui()
        if reset {
            sample_count = 0
        }

		gpu.reset_cmd(cmd)
		gpu.begin_profile(cmd, "frame")

		gpu.set_cbuffer(cmd, trace, "main", "Camera", &cam)
		gpu.set_uniform(cmd, trace, "main", "screen_size", [2]u32{output.width, output.height})
		gpu.set_uniform(cmd, trace, "main", "frame", frame)
        gpu.set_uniform(cmd, trace, "main", "sample_count", sample_count)
        gpu.set_uniform(cmd, trace, "main", "max_bounces", max_bounces)
        gpu.set_uniform(cmd, trace, "main", "fov", fov)
		gpu.set_tlas(cmd, trace, "main", "scene", scene.tlas)
        gpu.set_buffer(cmd, trace, "main", "instance_to_pool", scene.geometry_pool.instance_to_pool.buffer)
		gpu.set_buffer(cmd, trace, "main", "positions", scene.geometry_pool.vertices.buffer)
        gpu.set_buffer(cmd, trace, "main", "normals", scene.geometry_pool.normals.buffer)
        gpu.set_buffer(cmd, trace, "main", "tangents", scene.geometry_pool.tangents.buffer)
        gpu.set_buffer(cmd, trace, "main", "uvs", scene.geometry_pool.uvs.buffer)
		gpu.set_buffer(cmd, trace, "main", "indices", scene.geometry_pool.indices.buffer)
        gpu.set_buffer(cmd, trace, "main", "materials", scene.material_pool.materials.buffer)
		gpu.set_texture_array(cmd, trace, "main", "textures", scene.material_pool.textures)
        gpu.set_texture(cmd, trace, "main", "image", output)
		gpu.dispatch(cmd, trace, "main", num_groups.x, num_groups.y)

		gpu.end_profile(cmd)
		gpu.execute_cmd(cmd)
		gpu.end_frame(output)

        sample_count += 1

        // Handle resize
        framebuffer_width, framebuffer_height := glfw.GetFramebufferSize(gpu.get_window())
        if framebuffer_width > 0 && framebuffer_height > 0 && (u32(framebuffer_width) != output.width || u32(framebuffer_height) != output.height) {
            gpu.destroy_texture(output)
            output = gpu.create_texture(u32(framebuffer_width), u32(framebuffer_height), .R32G32B32A32_SFLOAT, writable = true)
            num_groups = ([2]u32{u32(framebuffer_width), u32(framebuffer_height)} + kernel_size.xy - 1) / kernel_size.xy
            sample_count = 0
        }

        // Reload scene if anything changed
        curr_scene_write, _ := os.last_write_time_by_name(scene_path)
        if curr_scene_write != last_scene_write {
            last_scene_write = curr_scene_write
            scene_delete(&scene)
            reload_cmd := gpu.create_cmd()
            scene = scene_load(strings.unsafe_string_to_cstring(scene_path), reload_cmd)
            gpu.execute_cmd(reload_cmd)
            gpu.destroy_cmd(reload_cmd)
        }
        curr_shader_write, _ := os.last_write_time_by_name(shader_path)
        if curr_shader_write != last_shader_write {
            last_shader_write = curr_shader_write
            gpu.destroy_shader(trace)
            trace = gpu.compile_shader(shader_path)
        }
	}
}

do_gui :: proc() -> bool {
    imgui.Begin("Settings")

    // Main UI
    changed := imgui.SliderInt("Max Bounces", &max_bounces, 1, 40)
    changed |= imgui.SliderFloat("FOV", &fov, 10.0, 120.0)

    // Stats
    frame_time := gpu.get_profile_time("frame")
    imgui.Text(fmt.ctprintf("%.2fms frametime", frame_time))

    imgui.End()

    return changed
}

update_camera :: proc(c: ^Camera, last_mouse: ^[2]f64, delta_time: f32) -> bool {
	reset := false
    io := imgui.GetIO()
	window := gpu.get_window()
	x, y := glfw.GetCursorPos(window)
	if !io.WantCaptureMouse && glfw.GetMouseButton(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS {
		c.yaw += f32(x - last_mouse.x) * 0.003
		c.pitch = clamp(c.pitch - f32(y - last_mouse.y) * 0.003, -1.5, 1.5)
        reset = true
	}
	last_mouse^ = {x, y}
	if io.WantCaptureKeyboard {
        return reset
    }
    
	forward := [3]f32{math.cos(c.pitch) * math.sin(c.yaw), math.sin(c.pitch), -math.cos(c.pitch) * math.cos(c.yaw)}
	right := linalg.normalize(linalg.cross(forward, [3]f32{0, 1, 0}))
	move: [3]f32
	if glfw.GetKey(window, glfw.KEY_W) == glfw.PRESS do move += forward
	if glfw.GetKey(window, glfw.KEY_S) == glfw.PRESS do move -= forward
	if glfw.GetKey(window, glfw.KEY_D) == glfw.PRESS do move += right
	if glfw.GetKey(window, glfw.KEY_A) == glfw.PRESS do move -= right
	if glfw.GetKey(window, glfw.KEY_E) == glfw.PRESS do move.y += 1
	if glfw.GetKey(window, glfw.KEY_Q) == glfw.PRESS do move.y -= 1
	speed: f32 = glfw.GetKey(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS ? 12.0 : 2
	if move != {} {
        c.pos += linalg.normalize(move) * speed * delta_time
        reset = true
    }
    return reset
}