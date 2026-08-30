package main

import "core:strings"
import "core:os"
import "core:fmt"
import "core:log"
import "core:time"
import "core:math"
import "core:math/linalg"

import "vendor:glfw"
import imgui "lib:imgui"

import "gpu"

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

// Constants
WIDTH : u32 : 1280
HEIGHT : u32 : 720

App_State :: struct {
    // Settings
    max_bounces: i32,
    fov: f32,

    // GPU state
    cmd: ^gpu.Cmd,
    trace: gpu.Shader,
    output: gpu.Texture,
    kernel_size: [3]u32,
    num_groups: [2]u32,

    // Frame state
    cam: Camera,
    last_mouse: [2]f64,
    last_time: f64,
    frame: u32,
    sample_count: u32,

    // Select scene, and hot reload
    last_shader_write: time.Time,
    shader_path: string,
    last_scene_write: time.Time,
    scene_index: i32,
    scene_names: [dynamic]cstring,
    scene: Scene,
}
app: App_State

app_init :: proc() -> App_State {
    context.logger = log.create_console_logger()

	gpu.init(
        title = "Odin Path Tracer", 
        width = WIDTH, height = HEIGHT,
        resizable = true,
        use_imgui = true,
        vsync = false,
        validation = ODIN_DEBUG)

    state := App_State{
        max_bounces = 20,
        fov = 60.0,
        scene_index = 0,
        scene_names = [dynamic]cstring{},
        shader_path = "shaders/megakernel.slang",
        cam = Camera{pos = {3, 2.5, 3}, yaw = -0.55, pitch = -0.35},
    }

    f, _ := os.open("assets")
    defer os.close(f)
    fis, _ := os.read_dir(f, -1, context.allocator)
    defer os.file_info_slice_delete(fis, context.allocator)
    for i := 0; i < len(fis); i += 1 {
        if !strings.ends_with(fis[i].name, ".glb") {
            continue
        }
        if strings.starts_with(fis[i].name, "splash") {
            state.scene_index = i32(len(state.scene_names))
        }
        append(&state.scene_names, strings.clone_to_cstring(fis[i].name))
    }

    if len(state.scene_names) == 0 {
        panic("no .glb files found in assets/")
    }

    app_load_scene(&state, state.scene_index)
    if state.scene.tlas.handle == 0 {
        panic("failed to load initial scene")
    }

    state.cmd = gpu.create_cmd()
    state.last_shader_write, _ = os.last_write_time_by_name(state.shader_path)
    trace, trace_ok := gpu.compile_shader(state.shader_path)
    if !trace_ok {
        panic("failed to compile shaders/megakernel.slang")
    }
    state.trace = trace
    state.output = gpu.create_texture(WIDTH, HEIGHT, .R32G32B32A32_SFLOAT, writable = true)
    state.kernel_size = gpu.get_kernel_size(state.trace, "main")
    state.num_groups = ([2]u32{WIDTH, HEIGHT} + state.kernel_size.xy - 1) / state.kernel_size.xy

    state.last_mouse.x, state.last_mouse.y = glfw.GetCursorPos(gpu.get_window())
    state.last_time = glfw.GetTime()

    return state
}

app_load_scene :: proc(state: ^App_State, index: i32) {
    state.scene_index = index
    scene_path := fmt.tprintf("assets/%s", state.scene_names[index])
    state.last_scene_write, _ = os.last_write_time_by_name(scene_path)

    load_cmd := gpu.create_cmd()
    defer gpu.destroy_cmd(load_cmd)
    scene, ok := scene_load(strings.unsafe_string_to_cstring(scene_path), load_cmd)
    if !ok {
        log.errorf("failed to load %s", scene_path)
        return
    }
    gpu.execute_cmd(load_cmd)

    if state.scene.tlas.handle != 0 {
        scene_delete(&state.scene)
    }
    state.scene = scene
    state.sample_count = 0
}

app_tick :: proc(state: ^App_State) {
    defer free_all(context.temp_allocator)

    app_do_frame(state)

    // Handle resize
    framebuffer_width, framebuffer_height := glfw.GetFramebufferSize(gpu.get_window())
    if framebuffer_width > 0 && framebuffer_height > 0 && (u32(framebuffer_width) != state.output.width || u32(framebuffer_height) != state.output.height) {
        gpu.destroy_texture(state.output)
        state.output = gpu.create_texture(u32(framebuffer_width), u32(framebuffer_height), .R32G32B32A32_SFLOAT, writable = true)
        state.num_groups = ([2]u32{u32(framebuffer_width), u32(framebuffer_height)} + state.kernel_size.xy - 1) / state.kernel_size.xy
        state.sample_count = 0
    }

    // Reload scene if anything changed
    scene_path := fmt.tprintf("assets/%s", state.scene_names[state.scene_index])
    last_scene_write, _ := os.last_write_time_by_name(scene_path)
    if last_scene_write != state.last_scene_write {
        app_load_scene(state, state.scene_index)
    }
    curr_shader_write, _ := os.last_write_time_by_name(state.shader_path)
    if curr_shader_write != state.last_shader_write {
        state.last_shader_write = curr_shader_write
        if new_trace, ok := gpu.compile_shader(state.shader_path); ok {
            gpu.destroy_shader(state.trace)
            state.trace = new_trace
            state.kernel_size = gpu.get_kernel_size(state.trace, "main")
            state.num_groups = ([2]u32{state.output.width, state.output.height} + state.kernel_size.xy - 1) / state.kernel_size.xy
            state.sample_count = 0
        }
    }
}

app_do_frame :: proc(state: ^App_State) {
    now := glfw.GetTime()
    delta_time := min(f32(now - state.last_time), 0.1)
    state.last_time = now
    reset := app_update_camera(&state.cam, &state.last_mouse, delta_time)

    cmd := state.cmd
    trace := state.trace

    gpu.start_frame()

        reset |= app_do_gui(state)
        if reset {
            state.sample_count = 0
        }
        
        scene := state.scene
        gpu.reset_cmd(cmd)
        gpu.begin_profile(cmd, "frame")

        gpu.set_cbuffer(cmd, trace, "main", "Camera", &state.cam)
        gpu.set_uniform(cmd, trace, "main", "screen_size", [2]u32{state.output.width, state.output.height})
        gpu.set_uniform(cmd, trace, "main", "frame", state.frame)
        gpu.set_uniform(cmd, trace, "main", "sample_count", state.sample_count)
        gpu.set_uniform(cmd, trace, "main", "max_bounces", state.max_bounces)
        gpu.set_uniform(cmd, trace, "main", "fov", state.fov)

        gpu.set_tlas(cmd, trace, "main", "scene", scene.tlas)
        gpu.set_buffer(cmd, trace, "main", "instance_to_pool", scene.geometry_pool.instance_to_pool.buffer)
        gpu.set_buffer(cmd, trace, "main", "positions", scene.geometry_pool.vertices.buffer)
        gpu.set_buffer(cmd, trace, "main", "normals", scene.geometry_pool.normals.buffer)
        gpu.set_buffer(cmd, trace, "main", "tangents", scene.geometry_pool.tangents.buffer)
        gpu.set_buffer(cmd, trace, "main", "uvs", scene.geometry_pool.uvs.buffer)
        gpu.set_buffer(cmd, trace, "main", "indices", scene.geometry_pool.indices.buffer)
        gpu.set_buffer(cmd, trace, "main", "materials", scene.material_pool.materials.buffer)
        gpu.set_texture_array(cmd, trace, "main", "textures", scene.material_pool.textures)
        gpu.set_texture(cmd, trace, "main", "image", state.output)
        gpu.dispatch(cmd, trace, "main", state.num_groups.x, state.num_groups.y)

        gpu.end_profile(cmd)
        gpu.execute_cmd(cmd)
        
    gpu.end_frame(state.output)

    state.sample_count += 1
    state.frame += 1
}

app_do_gui :: proc(state: ^App_State) -> bool {
    imgui.Begin("Settings")

    // Main UI
    changed := imgui.SliderInt("Max Bounces", &state.max_bounces, 1, 40)
    changed |= imgui.SliderFloat("FOV", &state.fov, 10.0, 120.0)
    if imgui.ComboChar("Scene", &state.scene_index, raw_data(state.scene_names[:]), i32(len(state.scene_names))) {
        app_load_scene(state, state.scene_index)
        changed = true
    }

    // Stats
    frame_time := gpu.get_profile_time("frame")
    imgui.Text(fmt.ctprintf("%.2fms frametime", frame_time))

    imgui.End()

    return changed
}

app_delete :: proc(state: ^App_State) {
    scene_delete(&state.scene)
    gpu.destroy_texture(state.output)
    gpu.destroy_shader(state.trace)
    gpu.destroy_cmd(state.cmd)
    for i := 0; i < len(state.scene_names); i += 1 {
        delete(state.scene_names[i])
    }
    delete(state.scene_names)
    gpu.cleanup()
    log.destroy_console_logger(context.logger)
}

Camera :: struct {
	pos:   [3]f32,
	yaw:   f32,
	pitch: f32,
}

app_update_camera :: proc(c: ^Camera, last_mouse: ^[2]f64, delta_time: f32) -> bool {
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

main :: proc() {
    app = app_init()
    defer app_delete(&app)

    for gpu.is_running() {
        app_tick(&app)
    }
}