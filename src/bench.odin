package main

import "core:fmt"
import "core:time"

import "gpu"

@(private)
BENCH_SCENES :: []string {
	"cornell.glb",
	"splash.glb",
	"gem_cuts.glb",
	"dispersion_test.glb",
	"sponza.glb"
}

app_bench :: proc(state: ^App_State, samples: u32) {
    fmt.printf("%-22s %-9s %14s %14s\n", "scene", "mode", "gpu ms/sample", "wall ms/sample")
    for scene_name in BENCH_SCENES {
        index := i32(-1)
        for name, i in state.scene_names {
            if string(name) == scene_name {
                index = i32(i)
            }
        }
        if index < 0 {
            fmt.printf("%-22s not found in assets/\n", scene_name)
            continue
        }
        for spectral in ([]bool{false, true}) {
            state.spectral_mode = spectral ? .Spectral : .RGB
            state.kernel_size = gpu.get_kernel_size(state.trace, spectral_mode_kernel(state.spectral_mode))
            state.num_groups = ([2]u32{state.output.width, state.output.height} + state.kernel_size.xy - 1) / state.kernel_size.xy
            app_load_scene(state, index)
            gpu_total: f64
            gpu_frames: u32
            start := time.tick_now()
            for state.sample_count < samples && gpu.is_running() {
                app_tick(state)
                gpu.wait_finish()
                frame_time := gpu.get_profile_time("frame")
                if frame_time > 0 {
                    gpu_total += f64(frame_time)
                    gpu_frames += 1
                }
            }
            wall := time.duration_milliseconds(time.tick_since(start))
            fmt.printf("%-22s %-9s %14s %14s\n",
                scene_name, spectral ? "spectral" : "rgb",
                fmt.tprintf("%.4f", gpu_total / f64(max(gpu_frames, 1))),
                fmt.tprintf("%.4f", wall / f64(samples)))
        }
    }
}
