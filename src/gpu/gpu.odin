// A small and simple abstraction layer over Vulkan, enabling easy use of compute shaders,
// and a basic frame loop. No rasterization or other fancy features.

// This is written entirely be Claude. I take no credit for it.

package gpu

import "core:encoding/json"
import "core:fmt"
import "core:log"
import "core:mem"
import "core:slice"
import "core:mem/virtual"
import "core:path/filepath"
import "core:strings"
import "vendor:glfw"
import vk "vendor:vulkan"
import imgui "lib:imgui"
import im_glfw "lib:imgui/imgui_impl_glfw"
import im_vk "lib:imgui/imgui_impl_vulkan"

// Returns the GLFW window created by init. Use it for input.
get_window :: proc() -> glfw.WindowHandle {
	return window
}

// Creates the window, the Vulkan device and the swapchain. Call once before anything else.
// `resizable` allows the window to be resized, the swapchain follows and end_frame() scales the texture to fit.
// `vsync` caps presentation to the display refresh rate, turn it off to render as fast as the GPU allows.
// `validation` enables the Khronos validation layer, which reports to stdout.
init :: proc(title: string, width, height: u32, resizable, use_imgui, vsync, validation: bool) {
	if virtual.arena_init_growing(&scratch) != nil do panic("scratch arena")
	if !glfw.Init() {
		desc, _ := glfw.GetError()
		fmt.panicf("glfwInit failed: %s", desc)
	}
	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	glfw.WindowHint(glfw.RESIZABLE, b32(resizable))
	window = glfw.CreateWindow(i32(width), i32(height), strings.clone_to_cstring(title, virtual.arena_allocator(&scratch)), nil, nil)
	if window == nil {
		desc, _ := glfw.GetError()
		fmt.panicf("glfwCreateWindow failed: %s", desc)
	}
	vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))

	app_info := vk.ApplicationInfo {
		sType            = .APPLICATION_INFO,
		pApplicationName = "odin-pt",
		apiVersion       = vk.API_VERSION_1_4,
	}
	layers := [?]cstring{"VK_LAYER_KHRONOS_validation"}
	instance_extensions := make([dynamic]cstring, virtual.arena_allocator(&scratch))
	append(&instance_extensions, ..glfw.GetRequiredInstanceExtensions())
	append(&instance_extensions, vk.KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, vk.EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)
	available_count: u32
	vk_check(vk.EnumerateInstanceExtensionProperties(nil, &available_count, nil))
	available := make([]vk.ExtensionProperties, available_count, virtual.arena_allocator(&scratch))
	vk_check(vk.EnumerateInstanceExtensionProperties(nil, &available_count, raw_data(available)))
	for &ext in available {
		if cstring(&ext.extensionName[0]) == vk.EXT_DEBUG_UTILS_EXTENSION_NAME {
			append(&instance_extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
			debug_utils = true
		}
	}
	vk_check(vk.CreateInstance(&vk.InstanceCreateInfo {
		sType                   = .INSTANCE_CREATE_INFO,
		pApplicationInfo        = &app_info,
		enabledExtensionCount   = u32(len(instance_extensions)),
		ppEnabledExtensionNames = raw_data(instance_extensions),
		enabledLayerCount       = validation ? len(layers) : 0,
		ppEnabledLayerNames     = raw_data(&layers),
	}, nil, &instance))
	vk.load_proc_addresses_instance(instance)

	device_count: u32
	vk_check(vk.EnumeratePhysicalDevices(instance, &device_count, nil))
	assert(device_count > 0, "no Vulkan devices")
	devices := make([]vk.PhysicalDevice, device_count, virtual.arena_allocator(&scratch))
	vk_check(vk.EnumeratePhysicalDevices(instance, &device_count, raw_data(devices)))
	physical_device = devices[0]

	props: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(physical_device, &props)
	vk.GetPhysicalDeviceMemoryProperties(physical_device, &memory_properties)
	uniform_alignment = int(props.limits.minUniformBufferOffsetAlignment)
	timestamp_period = props.limits.timestampPeriod / 1e6
	props12 := vk.PhysicalDeviceVulkan12Properties{sType = .PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES}
	props2 := vk.PhysicalDeviceProperties2{sType = .PHYSICAL_DEVICE_PROPERTIES_2, pNext = &props12}
	vk.GetPhysicalDeviceProperties2(physical_device, &props2)
	texture_array_capacity = min(65536, props12.maxPerStageDescriptorUpdateAfterBindSampledImages, props12.maxDescriptorSetUpdateAfterBindSampledImages)

	family_count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nil)
	families := make([]vk.QueueFamilyProperties, family_count, virtual.arena_allocator(&scratch))
	vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, raw_data(families))
	for f, i in families {
		if .GRAPHICS in f.queueFlags && .COMPUTE in f.queueFlags {
			queue_family = u32(i)
			break
		}
	}

	f_swapchain := vk.PhysicalDeviceSwapchainMaintenance1FeaturesEXT {
		sType                 = .PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
		swapchainMaintenance1 = true,
	}
	f_ray_query := vk.PhysicalDeviceRayQueryFeaturesKHR {
		sType    = .PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
		pNext    = &f_swapchain,
		rayQuery = true,
	}
	f_accel := vk.PhysicalDeviceAccelerationStructureFeaturesKHR {
		sType                 = .PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
		pNext                 = &f_ray_query,
		accelerationStructure = true,
	}
	f14 := vk.PhysicalDeviceVulkan14Features {
		sType          = .PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
		pNext          = &f_accel,
		pushDescriptor = true,
	}
	f13 := vk.PhysicalDeviceVulkan13Features {
		sType            = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		pNext            = &f14,
		synchronization2 = true,
		dynamicRendering = b32(use_imgui),
	}
	f12 := vk.PhysicalDeviceVulkan12Features {
		sType                                        = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		pNext                                        = &f13,
		bufferDeviceAddress                          = true,
		scalarBlockLayout                            = true,
		runtimeDescriptorArray                       = true,
		shaderSampledImageArrayNonUniformIndexing    = true,
		descriptorBindingPartiallyBound              = true,
		descriptorBindingSampledImageUpdateAfterBind = true,
		descriptorBindingUpdateUnusedWhilePending    = true,
	}
	device_extensions := [?]cstring {
		vk.KHR_SWAPCHAIN_EXTENSION_NAME,
		vk.KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		vk.KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
		vk.KHR_RAY_QUERY_EXTENSION_NAME,
		vk.EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
	}
	queue_priority: f32 = 1.0
	queue_info := vk.DeviceQueueCreateInfo {
		sType            = .DEVICE_QUEUE_CREATE_INFO,
		queueFamilyIndex = queue_family,
		queueCount       = 1,
		pQueuePriorities = &queue_priority,
	}
	vk_check(vk.CreateDevice(physical_device, &vk.DeviceCreateInfo {
		sType                   = .DEVICE_CREATE_INFO,
		pNext                   = &f12,
		queueCreateInfoCount    = 1,
		pQueueCreateInfos       = &queue_info,
		enabledExtensionCount   = len(device_extensions),
		ppEnabledExtensionNames = raw_data(&device_extensions),
	}, nil, &device))
	vk.load_proc_addresses_device(device)
	vk.GetDeviceQueue(device, queue_family, 0, &queue)

	imgui_enabled = use_imgui
	vsync_enabled = vsync
	vk_check(glfw.CreateWindowSurface(instance, window, nil, &surface))
	create_swapchain()

	vk_check(vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo {
		sType            = .COMMAND_POOL_CREATE_INFO,
		queueFamilyIndex = queue_family,
		flags            = {.RESET_COMMAND_BUFFER},
	}, nil, &command_pool))
	present_cb = allocate_command_buffer()
	present_cb_fence = create_fence(signaled = true)
	present_fence = create_fence(signaled = true)
	vk_check(vk.CreateSemaphore(device, &vk.SemaphoreCreateInfo{sType = .SEMAPHORE_CREATE_INFO}, nil, &image_available))
	vk_check(vk.CreateSemaphore(device, &vk.SemaphoreCreateInfo{sType = .SEMAPHORE_CREATE_INFO}, nil, &render_finished))

	vk_check(vk.CreateSampler(device, &vk.SamplerCreateInfo {
		sType        = .SAMPLER_CREATE_INFO,
		magFilter    = .LINEAR,
		minFilter    = .LINEAR,
		mipmapMode   = .LINEAR,
		addressModeU = .REPEAT,
		addressModeV = .REPEAT,
		addressModeW = .REPEAT,
		maxLod       = vk.LOD_CLAMP_NONE,
	}, nil, &default_sampler))

	array_binding_flags := vk.DescriptorBindingFlags{.PARTIALLY_BOUND, .UPDATE_AFTER_BIND, .UPDATE_UNUSED_WHILE_PENDING}
	array_flags_info := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
		sType         = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		bindingCount  = 1,
		pBindingFlags = &array_binding_flags,
	}
	array_binding := vk.DescriptorSetLayoutBinding {
		binding         = 0,
		descriptorType  = .SAMPLED_IMAGE,
		descriptorCount = texture_array_capacity,
		stageFlags      = {.COMPUTE},
	}
	vk_check(vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		pNext        = &array_flags_info,
		flags        = {.UPDATE_AFTER_BIND_POOL},
		bindingCount = 1,
		pBindings    = &array_binding,
	}, nil, &texture_array_layout))

	if !imgui_enabled do return
	imgui.CreateContext(nil)
	im_glfw.InitForVulkan(window, true)
	im_vk.LoadFunctions(vk.API_VERSION_1_4, proc "c" (name: cstring, user_data: rawptr) -> vk.ProcVoidFunction {
		return vk.GetInstanceProcAddr((cast(^vk.Instance)user_data)^, name)
	}, &instance)
	swapchain_format := vk.Format.B8G8R8A8_SRGB
	imgui_init := im_vk.InitInfo {
		ApiVersion = vk.API_VERSION_1_4,
		Instance = instance,
		PhysicalDevice = physical_device,
		Device = device,
		QueueFamily = queue_family,
		Queue = queue,
		DescriptorPoolSize = 16,
		MinImageCount = max(swapchain_min_images, 2),
		ImageCount = u32(len(swapchain_images)),
		UseDynamicRendering = true,
		PipelineInfoMain = {
			PipelineRenderingCreateInfo = {
				sType = .PIPELINE_RENDERING_CREATE_INFO,
				colorAttachmentCount = 1,
				pColorAttachmentFormats = &swapchain_format,
			},
		},
	}
	im_vk.Init(&imgui_init)
}

// Waits for the GPU and destroys everything created by init.
cleanup :: proc() {
	vk_check(vk.DeviceWaitIdle(device))
	if imgui_enabled {
		im_vk.Shutdown()
		im_glfw.Shutdown()
		imgui.DestroyContext(nil)
	}
	if slang_session != nil do spDestroySession(slang_session)
	vk.DestroySampler(device, default_sampler, nil)
	vk.DestroyDescriptorSetLayout(device, texture_array_layout, nil)
	vk.DestroySemaphore(device, image_available, nil)
	vk.DestroySemaphore(device, render_finished, nil)
	vk.DestroyFence(device, present_cb_fence, nil)
	vk.DestroyFence(device, present_fence, nil)
	for s in submissions {
		vk.DestroyFence(device, s.fence, nil)
		vk.DestroyQueryPool(device, s.query_pool, nil)
		free_buffer(s.staging)
		delete(s.scopes)
		flush_garbage(&s.garbage)
		delete(s.garbage)
		free(s)
	}
	for _, k in profile_keys do delete(k)
	delete(profile_keys)
	delete(profile_collect)
	delete(profile_times)
	delete(submissions)
	delete(pending)
	flush_garbage(&orphan_garbage)
	delete(orphan_garbage)
	delete(pending_images)
	delete(free_submissions)
	vk.DestroyCommandPool(device, command_pool, nil)
	vk.DestroySwapchainKHR(device, swapchain, nil)
	delete(swapchain_images)
	for v in swapchain_views do vk.DestroyImageView(device, v, nil)
	delete(swapchain_views)
	vk.DestroySurfaceKHR(instance, surface, nil)
	vk.DestroyDevice(device, nil)
	vk.DestroyInstance(instance, nil)
	glfw.DestroyWindow(window)
	glfw.Terminate()
	virtual.arena_destroy(&scratch)
}

// Pumps window events. Returns false once the window has been closed.
is_running :: proc() -> bool {
	glfw.PollEvents()
	return !glfw.WindowShouldClose(window)
}

// Starts a frame. Resets the library's scratch memory, so call it once per frame before recording anything.
start_frame :: proc() {
	virtual.arena_free_all(&scratch)
	frame_counter += 1
	if imgui_enabled {
		im_vk.NewFrame()
		im_glfw.NewFrame()
		imgui.NewFrame()
	}
}

wait_finish :: proc() {
	wait_idle()
	append(&free_submissions, acquire_submission())
}

// Ends the frame: blits a texture of any size and format onto the window and presents it.
// Expects linear colour, the swapchain is sRGB.
end_frame :: proc(t: Texture) {
	fb_w, fb_h := glfw.GetFramebufferSize(window)
	if fb_w == 0 || fb_h == 0 {
		if imgui_enabled do imgui.EndFrame()
		return
	}

	fences := [2]vk.Fence{present_cb_fence, present_fence}
	vk_check(vk.WaitForFences(device, len(fences), raw_data(&fences), true, max(u64)))
	vk_check(vk.ResetFences(device, len(fences), raw_data(&fences)))

	if swapchain_extent != {u32(fb_w), u32(fb_h)} do create_swapchain()
	image_index: u32
	res := vk.AcquireNextImageKHR(device, swapchain, max(u64), image_available, 0, &image_index)
	if res == .ERROR_OUT_OF_DATE_KHR {
		create_swapchain()
		res = vk.AcquireNextImageKHR(device, swapchain, max(u64), image_available, 0, &image_index)
	}
	if res != .SUCCESS && res != .SUBOPTIMAL_KHR do fmt.panicf("vkAcquireNextImageKHR: %v", res)
	image := swapchain_images[image_index]

	vk_check(vk.ResetCommandBuffer(present_cb, {}))
	vk_check(vk.BeginCommandBuffer(present_cb, &vk.CommandBufferBeginInfo{sType = .COMMAND_BUFFER_BEGIN_INFO, flags = {.ONE_TIME_SUBMIT}}))
	for pending_image in pending_images do transition_image(present_cb, pending_image, .UNDEFINED, .GENERAL)
	clear(&pending_images)
	full_barrier(present_cb)
	transition_image(present_cb, image, .UNDEFINED, .TRANSFER_DST_OPTIMAL)
	blit := vk.ImageBlit {
		srcSubresource = {aspectMask = {.COLOR}, layerCount = 1},
		srcOffsets     = {{0, 0, 0}, {i32(t.width), i32(t.height), 1}},
		dstSubresource = {aspectMask = {.COLOR}, layerCount = 1},
		dstOffsets     = {{0, 0, 0}, {i32(swapchain_extent.width), i32(swapchain_extent.height), 1}},
	}
	vk.CmdBlitImage(present_cb, t.image, .GENERAL, image, .TRANSFER_DST_OPTIMAL, 1, &blit, .LINEAR)
	if imgui_enabled {
		imgui.Render()
		transition_image(present_cb, image, .TRANSFER_DST_OPTIMAL, .COLOR_ATTACHMENT_OPTIMAL)
		color_attachment := vk.RenderingAttachmentInfo {
			sType       = .RENDERING_ATTACHMENT_INFO,
			imageView   = swapchain_views[image_index],
			imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
			loadOp      = .LOAD,
			storeOp     = .STORE,
		}
		vk.CmdBeginRendering(present_cb, &vk.RenderingInfo {
			sType                = .RENDERING_INFO,
			renderArea           = {extent = swapchain_extent},
			layerCount           = 1,
			colorAttachmentCount = 1,
			pColorAttachments    = &color_attachment,
		})
		im_vk.RenderDrawData(imgui.GetDrawData(), present_cb)
		vk.CmdEndRendering(present_cb)
		transition_image(present_cb, image, .COLOR_ATTACHMENT_OPTIMAL, .PRESENT_SRC_KHR)
	} else {
		transition_image(present_cb, image, .TRANSFER_DST_OPTIMAL, .PRESENT_SRC_KHR)
	}
	vk_check(vk.EndCommandBuffer(present_cb))

	wait_info := vk.SemaphoreSubmitInfo{sType = .SEMAPHORE_SUBMIT_INFO, semaphore = image_available, stageMask = {.ALL_COMMANDS}}
	signal_info := vk.SemaphoreSubmitInfo{sType = .SEMAPHORE_SUBMIT_INFO, semaphore = render_finished, stageMask = {.ALL_COMMANDS}}
	cb_info := vk.CommandBufferSubmitInfo{sType = .COMMAND_BUFFER_SUBMIT_INFO, commandBuffer = present_cb}
	vk_check(vk.QueueSubmit2(queue, 1, &vk.SubmitInfo2 {
		sType                    = .SUBMIT_INFO_2,
		waitSemaphoreInfoCount   = 1,
		pWaitSemaphoreInfos      = &wait_info,
		signalSemaphoreInfoCount = 1,
		pSignalSemaphoreInfos    = &signal_info,
		commandBufferInfoCount   = 1,
		pCommandBufferInfos      = &cb_info,
	}, present_cb_fence))

	present_fence_info := vk.SwapchainPresentFenceInfoEXT{sType = .SWAPCHAIN_PRESENT_FENCE_INFO_EXT, swapchainCount = 1, pFences = &present_fence}
	res = vk.QueuePresentKHR(queue, &vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		pNext              = &present_fence_info,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &render_finished,
		swapchainCount     = 1,
		pSwapchains        = &swapchain,
		pImageIndices      = &image_index,
	})
	if res != .SUCCESS && res != .SUBOPTIMAL_KHR && res != .ERROR_OUT_OF_DATE_KHR do fmt.panicf("vkQueuePresentKHR: %v", res)
}

// A storage buffer in device memory. Non-writable buffers can also be bound as a ConstantBuffer.
Buffer :: struct {
	handle:  vk.Buffer,
	memory:  vk.DeviceMemory,
	size:    uint,
	writable: bool,
	address: vk.DeviceAddress,
	mapped:  rawptr,
}

// A 2D texture in device memory, always in GENERAL layout. Non-writable textures are sample-only.
Texture :: struct {
	image:  vk.Image,
	view:   vk.ImageView,
	memory: vk.DeviceMemory,
	width:  u32,
	height: u32,
	format: vk.Format,
	writable: bool,
}

// Creates a device-local buffer of `size` bytes.
create_buffer :: proc(size: uint, writable := false) -> Buffer {
	usage := vk.BufferUsageFlags{.STORAGE_BUFFER, .TRANSFER_SRC, .TRANSFER_DST, .SHADER_DEVICE_ADDRESS, .ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .INDIRECT_BUFFER}
	if !writable do usage += {.UNIFORM_BUFFER}
	b := create_buffer_ex(int(size), usage, host_visible = false)
	b.writable = writable
	return b
}

// Waits for the GPU, then frees the buffer.
destroy_buffer :: proc(b: Buffer) {
	wait_idle()
	free_buffer(b)
}

// Copies the whole buffer back to the CPU, blocking until the GPU has finished writing it.
// The result has b.size / size_of(T) elements and is owned by the caller.
readback_buffer :: proc(b: Buffer, $T: typeid, allocator := context.allocator) -> []T {
	staging := create_buffer_ex(int(b.size), {.TRANSFER_DST}, host_visible = true)
	defer free_buffer(staging)

	sub := acquire_submission()
	cb := sub.cb
	vk_check(vk.ResetCommandBuffer(cb, {}))
	vk_check(vk.BeginCommandBuffer(cb, &vk.CommandBufferBeginInfo{sType = .COMMAND_BUFFER_BEGIN_INFO, flags = {.ONE_TIME_SUBMIT}}))
	full_barrier(cb)
	region := vk.BufferCopy{size = vk.DeviceSize(b.size)}
	vk.CmdCopyBuffer(cb, b.handle, staging.handle, 1, &region)
	vk_check(vk.EndCommandBuffer(cb))

	cb_info := vk.CommandBufferSubmitInfo{sType = .COMMAND_BUFFER_SUBMIT_INFO, commandBuffer = cb}
	vk_check(vk.QueueSubmit2(queue, 1, &vk.SubmitInfo2 {
		sType                  = .SUBMIT_INFO_2,
		commandBufferInfoCount = 1,
		pCommandBufferInfos    = &cb_info,
	}, sub.fence))
	vk_check(vk.WaitForFences(device, 1, &sub.fence, true, max(u64)))
	vk_check(vk.ResetFences(device, 1, &sub.fence))
	append(&free_submissions, sub)

	result := make([]T, b.size / size_of(T), allocator)
	mem.copy(raw_data(result), staging.mapped, int(b.size))
	return result
}

// Creates a 2D texture with one mip level.
create_texture :: proc(width, height: u32, format: vk.Format, writable := false) -> Texture {
	t := Texture{width = width, height = height, format = format, writable = writable}
	usage := vk.ImageUsageFlags{.SAMPLED, .TRANSFER_SRC, .TRANSFER_DST}
	if writable do usage += {.STORAGE}
	vk_check(vk.CreateImage(device, &vk.ImageCreateInfo {
		sType         = .IMAGE_CREATE_INFO,
		imageType     = .D2,
		format        = format,
		extent        = {width, height, 1},
		mipLevels     = 1,
		arrayLayers   = 1,
		samples       = {._1},
		tiling        = .OPTIMAL,
		usage         = usage,
		sharingMode   = .EXCLUSIVE,
		initialLayout = .UNDEFINED,
	}, nil, &t.image))
	reqs: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(device, t.image, &reqs)
	t.memory, _ = allocate(reqs, host_visible = false, device_address = false)
	vk_check(vk.BindImageMemory(device, t.image, t.memory, 0))
	vk_check(vk.CreateImageView(device, &vk.ImageViewCreateInfo {
		sType            = .IMAGE_VIEW_CREATE_INFO,
		image            = t.image,
		viewType         = .D2,
		format           = format,
		subresourceRange = {aspectMask = {.COLOR}, levelCount = 1, layerCount = 1},
	}, nil, &t.view))

	append(&pending_images, t.image)
	return t
}

// Waits for the GPU, then frees the texture.
destroy_texture :: proc(t: Texture) {
	wait_idle()
	for image, i in pending_images {
		if image == t.image {
			unordered_remove(&pending_images, i)
			break
		}
	}
	vk.DestroyImageView(device, t.view, nil)
	vk.DestroyImage(device, t.image, nil)
	vk.FreeMemory(device, t.memory, nil)
}

// A bindless texture array. Declare it in the shader as ParameterBlock<Texture2D[]> with no binding attributes.
Texture_Array :: struct {
	pool: vk.DescriptorPool,
	set:  vk.DescriptorSet,
}

// Creates an empty texture array with 65536 slots (fewer if the device limit is lower).
// Slots the shader never reads can stay empty.
create_texture_array :: proc() -> (ta: Texture_Array) {
	pool_size := vk.DescriptorPoolSize{type = .SAMPLED_IMAGE, descriptorCount = texture_array_capacity}
	vk_check(vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo {
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		flags         = {.UPDATE_AFTER_BIND},
		maxSets       = 1,
		poolSizeCount = 1,
		pPoolSizes    = &pool_size,
	}, nil, &ta.pool))
	layout := texture_array_layout
	vk_check(vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool     = ta.pool,
		descriptorSetCount = 1,
		pSetLayouts        = &layout,
	}, &ta.set))
	return
}

// Records a write of the texture into a slot.
// Writes apply per execute_cmd, so the last write to a slot wins for the whole list.
// On invalid use, prints the error and returns false.
texture_array_write :: proc(cmd: ^Cmd, ta: Texture_Array, slot: u32, t: Texture) -> bool {
	if slot >= texture_array_capacity {
		log.errorf("texture_array_write: slot %d out of range (capacity %d)", slot, texture_array_capacity)
		return false
	}
	append(&cmd.commands, Cmd_Write_Texture_Array{ta.set, slot, t.view})
	return true
}

// Waits for the GPU, then destroys the texture array.
destroy_texture_array :: proc(ta: Texture_Array) {
	wait_idle()
	vk.DestroyDescriptorPool(device, ta.pool, nil)
}

// A compute shader with one pipeline per kernel and shared reflection data.
Shader :: struct {
	name:       string,
	kernels:    []Kernel,
	layout:     vk.PipelineLayout,
	set_layout: vk.DescriptorSetLayout,
	params:     []Shader_Param,
	cbuffers:   []Shader_Cbuffer,
}

@(private)
Kernel :: struct {
	name:       string,
	pipeline:   vk.Pipeline,
	group_size: [3]u32,
	used:       u64,
}

// Builds compute pipelines from a SPIR-V blob and the slangc reflection JSON for it (`slangc -reflection-json`).
// Every [shader("compute")] entry point becomes a kernel. `debug_name` is only used in error messages.
// On invalid input, prints the reason and returns ok = false.
load_shader :: proc(spv: []byte, reflection_json: []byte, debug_name := "shader") -> (s: Shader, ok: bool) #optional_ok {
	root, err := json.parse(reflection_json)
	if err != nil {
		log.errorf("load_shader %s: reflection JSON: %v", debug_name, err)
		return
	}
	defer json.destroy_value(root)
	obj, is_object := root.(json.Object)
	if !is_object {
		log.errorf("load_shader %s: reflection JSON is not an object", debug_name)
		return
	}
	r, r_ok := reflect_json(obj, debug_name)
	if !r_ok {
		free_reflection(&r)
		return
	}
	s, ok = create_shader(spv, &r, debug_name)
	if !ok do free_reflection(&r)
	return
}

// Compiles a Slang source file with the Slang library (slang.dll) and builds a pipeline per kernel.
// Every [shader("compute")] entry point becomes a kernel.
// Compile errors and invalid input print the diagnostics and return ok = false; warnings are printed.
compile_shader :: proc(path: string) -> (s: Shader, ok: bool) #optional_ok {
	if slang_session == nil do slang_session = spCreateSession(nil)
	request := spCreateCompileRequest(slang_session)
	defer spDestroyCompileRequest(request)
	args := [?]cstring {
		strings.clone_to_cstring(path, virtual.arena_allocator(&scratch)),
		"-target", "spirv",
		"-profile", "spirv_1_6+spvRayQueryKHR",
		"-fvk-use-entrypoint-name",
		"-fvk-use-scalar-layout",
		"-matrix-layout-column-major",
	}
	if spProcessCommandLineArguments(request, raw_data(&args), len(args)) < 0 {
		log.errorf("compile_shader %s: %s", path, spGetDiagnosticOutput(request))
		return
	}
	if spCompile(request) < 0 {
		log.errorf("compile_shader %s:\n%s", path, spGetDiagnosticOutput(request))
		return
	}
	if diagnostics := spGetDiagnosticOutput(request); diagnostics != nil && len(diagnostics) > 0 do log.warnf("%s", diagnostics)
	size: uint
	code := spGetCompileRequestCode(request, &size)
	if code == nil {
		log.errorf("compile_shader %s: no code generated (missing [shader(\"compute\")]?)", path)
		return
	}
	r, r_ok := reflect_slang(spGetReflection(request), path)
	if !r_ok {
		free_reflection(&r)
		return
	}
	s, ok = create_shader(slice.bytes_from_ptr(code, int(size)), &r, filepath.stem(path))
	if !ok do free_reflection(&r)
	return
}

// Waits for the GPU, then destroys the shader.
destroy_shader :: proc(s: Shader) {
	wait_idle()
	for k in s.kernels {
		vk.DestroyPipeline(device, k.pipeline, nil)
		delete(k.name)
	}
	delete(s.kernels)
	vk.DestroyPipelineLayout(device, s.layout, nil)
	vk.DestroyDescriptorSetLayout(device, s.set_layout, nil)
	for p in s.params do if p.cbuffer < 0 || p.name != "$Globals" do delete(p.name)
	for cb in s.cbuffers {
		for m in cb.members do delete(m.name)
		delete(cb.members)
	}
	delete(s.params)
	delete(s.cbuffers)
	delete(s.name)
}

// A command list. Purely CPU-side until execute_cmd(). Bindings stick per kernel across dispatches.
Cmd :: struct {
	commands: [dynamic]Command,
	blob:     [dynamic]byte,

	buffers:  map[Binding_Key]Buffer,
	textures: map[Binding_Key]Texture,
	tlases:   map[Binding_Key]Tlas,
	arrays:   map[Binding_Key]vk.DescriptorSet,
	uniforms: map[Binding_Key][]byte,
	cbuffers: map[Binding_Key][]byte,
	garbage:  [dynamic]Garbage,
}

// Creates an empty command list.
create_cmd :: proc() -> ^Cmd {
	return new(Cmd)
}

// Frees a command list.
destroy_cmd :: proc(cmd: ^Cmd) {
	append(&orphan_garbage, ..cmd.garbage[:])
	delete(cmd.garbage)
	delete(cmd.commands)
	delete(cmd.blob)
	delete(cmd.buffers)
	delete(cmd.textures)
	delete(cmd.tlases)
	delete(cmd.arrays)
	delete(cmd.uniforms)
	delete(cmd.cbuffers)
	free(cmd)
}

// Clears the list. Never blocks.
reset_cmd :: proc(cmd: ^Cmd) {
	append(&orphan_garbage, ..cmd.garbage[:])
	clear(&cmd.garbage)
	clear(&cmd.commands)
	clear(&cmd.blob)
}

// Sets the buffer for a kernel's parameter with this name.
// On invalid use, prints the error and returns false.
set_buffer :: proc(cmd: ^Cmd, s: Shader, kernel, name: string, b: Buffer) -> bool {
	kernel_index := find_kernel(s, kernel) or_return
	k := s.kernels[kernel_index]
	p := find_param(s, k, name) or_return
	#partial switch p.type {
	case .STORAGE_BUFFER:
		if p.writable && !b.writable {
			log.errorf("set_buffer %s.%s: %q is RW in the shader but the buffer is not writable", s.name, kernel, name)
			return false
		}
	case .UNIFORM_BUFFER:
		if b.writable {
			log.errorf("set_buffer %s.%s: only non-writable buffers can be bound as constant buffer %q", s.name, kernel, name)
			return false
		}
	case:
		log.errorf("set_buffer %s.%s: %q is not a buffer", s.name, kernel, name)
		return false
	}
	append(&cmd.commands, Cmd_Set_Buffer{k.pipeline, push_name(cmd, name), b})
	return true
}

// Sets the texture for a kernel's parameter with this name.
// On invalid use, prints the error and returns false.
set_texture :: proc(cmd: ^Cmd, s: Shader, kernel, name: string, t: Texture) -> bool {
	kernel_index := find_kernel(s, kernel) or_return
	k := s.kernels[kernel_index]
	p := find_param(s, k, name) or_return
	if p.type != .STORAGE_IMAGE && p.type != .SAMPLED_IMAGE {
		log.errorf("set_texture %s.%s: %q is not a texture", s.name, kernel, name)
		return false
	}
	if p.type == .STORAGE_IMAGE && !t.writable {
		log.errorf("set_texture %s.%s: %q is RW in the shader but the texture is not writable", s.name, kernel, name)
		return false
	}
	if p.space != 0 {
		log.errorf("set_texture %s.%s: %q is a texture array, use set_texture_array", s.name, kernel, name)
		return false
	}
	append(&cmd.commands, Cmd_Set_Texture{k.pipeline, push_name(cmd, name), t})
	return true
}

// Sets the TLAS for a kernel's RaytracingAccelerationStructure parameter with this name.
// On invalid use, prints the error and returns false.
set_tlas :: proc(cmd: ^Cmd, s: Shader, kernel, name: string, t: Tlas) -> bool {
	kernel_index := find_kernel(s, kernel) or_return
	k := s.kernels[kernel_index]
	p := find_param(s, k, name) or_return
	if p.type != .ACCELERATION_STRUCTURE_KHR {
		log.errorf("set_tlas %s.%s: %q is not an acceleration structure", s.name, kernel, name)
		return false
	}
	append(&cmd.commands, Cmd_Set_Tlas{k.pipeline, push_name(cmd, name), t})
	return true
}

// Sets the texture array for a kernel's ParameterBlock<Texture2D[]> parameter with this name.
// On invalid use, prints the error and returns false.
set_texture_array :: proc(cmd: ^Cmd, s: Shader, kernel, name: string, ta: Texture_Array) -> bool {
	kernel_index := find_kernel(s, kernel) or_return
	k := s.kernels[kernel_index]
	p := find_param(s, k, name) or_return
	if p.space == 0 {
		log.errorf("set_texture_array %s.%s: %q is not a texture array", s.name, kernel, name)
		return false
	}
	append(&cmd.commands, Cmd_Set_Texture_Array{k.pipeline, push_name(cmd, name), ta.set})
	return true
}

// Sets one constant buffer member of a kernel by name, e.g. "time", "cam_pos", "params.sub".
// The value is copied and size-checked now. On invalid use, prints the error and returns false.
set_uniform :: proc(cmd: ^Cmd, s: Shader, kernel, name: string, value: $T) -> bool {
	v := value
	kernel_index := find_kernel(s, kernel) or_return
	k := s.kernels[kernel_index]
	m := find_member(s, k, name) or_return
	if size_of(T) != m.size {
		log.errorf("set_uniform %s.%s: %q is %d bytes, but the shader member is %d bytes", s.name, kernel, name, size_of(T), m.size)
		return false
	}
	append(&cmd.commands, Cmd_Set_Uniform{k.pipeline, push_name(cmd, name), push_blob(cmd, mem.ptr_to_bytes(&v))})
	return true
}

// Sets a whole constant buffer of a kernel by block name: "Camera", "params" or "$Globals".
// Members set with set_uniform override it. On invalid use, prints the error and returns false.
set_cbuffer :: proc(cmd: ^Cmd, s: Shader, kernel, name: string, data: ^$T) -> bool {
	kernel_index := find_kernel(s, kernel) or_return
	k := s.kernels[kernel_index]
	cb := find_cbuffer(s, k, name) or_return
	if size_of(T) > cb.size {
		log.errorf("set_cbuffer %s.%s: %q is %d bytes, but the shader block is %d bytes", s.name, kernel, name, size_of(T), cb.size)
		return false
	}
	append(&cmd.commands, Cmd_Set_Cbuffer{k.pipeline, push_name(cmd, name), push_blob(cmd, mem.ptr_to_bytes(data))})
	return true
}

// Dispatches a kernel with this many thread groups.
// On invalid use, prints the error and returns false.
dispatch :: proc(cmd: ^Cmd, s: Shader, kernel: string, groups_x: u32, groups_y: u32 = 1, groups_z: u32 = 1) -> bool {
	kernel_index := find_kernel(s, kernel) or_return
	append(&cmd.commands, Cmd_Dispatch{s, kernel_index, groups_x, groups_y, groups_z})
	return true
}

// Dispatches a kernel with thread group counts read on the GPU from a buffer at a byte offset: three consecutive u32s (x, y, z).
// On invalid use, prints the error and returns false.
dispatch_indirect :: proc(cmd: ^Cmd, s: Shader, kernel: string, b: Buffer, offset: uint = 0) -> bool {
	kernel_index := find_kernel(s, kernel) or_return
	if offset < 0 || offset % 4 != 0 {
		log.errorf("dispatch_indirect %s.%s: offset %d must be a non-negative multiple of 4", s.name, kernel, offset)
		return false
	}
	if offset + 3 * size_of(u32) > b.size {
		log.errorf("dispatch_indirect %s.%s: offset %d leaves no room for 3 u32s in a %d byte buffer", s.name, kernel, offset, b.size)
		return false
	}
	append(&cmd.commands, Cmd_Dispatch_Indirect{s, kernel_index, b, offset})
	return true
}

// Returns the thread group size of a kernel, as declared with [numthreads(x, y, z)].
// On an unknown kernel, prints the error and returns ok = false.
get_kernel_size :: proc(s: Shader, kernel: string) -> (size: [3]u32, ok: bool) #optional_ok {
	kernel_index := find_kernel(s, kernel) or_return
	return s.kernels[kernel_index].group_size, true
}

@(private)
find_kernel :: proc(s: Shader, kernel: string) -> (int, bool) {
	for k, i in s.kernels do if k.name == kernel do return i, true
	log.errorf("%s: no kernel named %q", s.name, kernel)
	return 0, false
}

@(private)
kernel_uses :: proc(k: Kernel, param: int) -> bool {
	return k.used & (u64(1) << u32(param)) != 0
}

@(private)
find_param :: proc(s: Shader, k: Kernel, name: string) -> (Shader_Param, bool) {
	for p, i in s.params {
		if p.name != name do continue
		if !kernel_uses(k, i) {
			log.errorf("%s: kernel %q does not use %q", s.name, k.name, name)
			return {}, false
		}
		return p, true
	}
	log.errorf("%s: no parameter named %q", s.name, name)
	return {}, false
}

@(private)
find_member :: proc(s: Shader, k: Kernel, name: string) -> (Shader_Member, bool) {
	for p, i in s.params {
		if p.cbuffer < 0 do continue
		for m in s.cbuffers[p.cbuffer].members {
			if m.name != name do continue
			if !kernel_uses(k, i) {
				log.errorf("%s: kernel %q does not use %q", s.name, k.name, name)
				return {}, false
			}
			return m, true
		}
	}
	log.errorf("%s: no constant buffer member named %q", s.name, name)
	return {}, false
}

@(private)
find_cbuffer :: proc(s: Shader, k: Kernel, name: string) -> (Shader_Cbuffer, bool) {
	for p, i in s.params {
		if p.cbuffer < 0 || s.cbuffers[p.cbuffer].name != name do continue
		if !kernel_uses(k, i) {
			log.errorf("%s: kernel %q does not use %q", s.name, k.name, name)
			return {}, false
		}
		return s.cbuffers[p.cbuffer], true
	}
	log.errorf("%s: no constant buffer named %q", s.name, name)
	return {}, false
}

// Copies CPU data into a buffer. The data is copied into the list at call time.
// If the data does not fit, prints the error and returns false.
upload_buffer :: proc(cmd: ^Cmd, dst: Buffer, data: []$T) -> bool {
	bytes := slice.to_bytes(data)
	if uint(len(bytes)) > dst.size {
		log.errorf("upload_buffer: %d bytes into a %d byte buffer", len(bytes), dst.size)
		return false
	}
	append(&cmd.commands, Cmd_Upload_Buffer{dst, push_blob(cmd, bytes)})
	return true
}

// Copies tightly packed pixels into a texture. Must be exactly width*height texels.
upload_texture :: proc(cmd: ^Cmd, dst: Texture, pixels: []$T) {
	append(&cmd.commands, Cmd_Upload_Texture{dst, push_blob(cmd, slice.to_bytes(pixels))})
}

// Copies a whole buffer into another of the same size.
// On a size mismatch, prints the error and returns false.
copy_buffer :: proc(cmd: ^Cmd, src, dst: Buffer) -> bool {
	if src.size != dst.size {
		log.errorf("copy_buffer: sizes differ (%d vs %d)", src.size, dst.size)
		return false
	}
	append(&cmd.commands, Cmd_Copy_Buffer{src, dst})
	return true
}

// Copies a whole texture into another of the same size.
// On a size mismatch, prints the error and returns false.
copy_texture :: proc(cmd: ^Cmd, src, dst: Texture) -> bool {
	if src.width != dst.width || src.height != dst.height {
		log.errorf("copy_texture: sizes differ (%dx%d vs %dx%d)", src.width, src.height, dst.width, dst.height)
		return false
	}
	append(&cmd.commands, Cmd_Copy_Texture{src, dst})
	return true
}

// Starts a named GPU timing scope. Scopes may nest and a name may be used several times per frame.
begin_profile :: proc(cmd: ^Cmd, name: string) {
	append(&cmd.commands, Cmd_Begin_Profile{push_name(cmd, name)})
}

// Ends the innermost open profiling scope.
end_profile :: proc(cmd: ^Cmd) {
	append(&cmd.commands, Cmd_End_Profile{})
}

// GPU time in milliseconds spent inside the named scope during the most recently completed frame,
// summed if the scope ran more than once. 0 until a frame using the scope has finished on the GPU.
get_profile_time :: proc(name: string) -> f32 {
	return profile_times[name]
}

// Records the list into a Vulkan command buffer and submits it. Returns immediately.
// Submissions run in order on one queue, and a full barrier before each dispatch or copy makes earlier writes visible.
// If a dispatch is missing a binding, prints the error and returns false without submitting anything.
execute_cmd :: proc(cmd: ^Cmd) -> bool {
	sub := acquire_submission()

	need := 0
	for c in cmd.commands {
		#partial switch v in c {
		case Cmd_Upload_Buffer:
			need += align_up(v.data.len, 16)
		case Cmd_Upload_Texture:
			need += align_up(v.data.len, 16)
		case Cmd_Dispatch:
			for cb in v.shader.cbuffers do need += align_up(cb.size, uniform_alignment)
		case Cmd_Dispatch_Indirect:
			for cb in v.shader.cbuffers do need += align_up(cb.size, uniform_alignment)
		}
	}
	ensure_staging(sub, need)
	sub.staging_offset = 0

	clear(&cmd.buffers)
	clear(&cmd.textures)
	clear(&cmd.tlases)
	clear(&cmd.arrays)
	clear(&cmd.uniforms)
	clear(&cmd.cbuffers)

	cb := sub.cb
	vk_check(vk.ResetCommandBuffer(cb, {}))
	vk_check(vk.BeginCommandBuffer(cb, &vk.CommandBufferBeginInfo{sType = .COMMAND_BUFFER_BEGIN_INFO, flags = {.ONE_TIME_SUBMIT}}))
	vk.CmdResetQueryPool(cb, sub.query_pool, 0, MAX_PROFILE_QUERIES)
	for image in pending_images do transition_image(cb, image, .UNDEFINED, .GENERAL)
	sub.query_count = 0
	sub.frame = frame_counter
	clear(&sub.scopes)
	open_scopes := make([dynamic]Profile_Scope, virtual.arena_allocator(&scratch))
	for c in cmd.commands {
		switch v in c {
		case Cmd_Begin_Profile:
			if sub.query_count >= MAX_PROFILE_QUERIES {
				log.errorf("begin_profile %q: more than %d profiling scopes in one execute_cmd", span_string(cmd, v.name), MAX_PROFILE_QUERIES / 2)
				vk_check(vk.ResetCommandBuffer(cb, {}))
				append(&free_submissions, sub)
				return false
			}
			vk.CmdWriteTimestamp2(cb, {.ALL_COMMANDS}, sub.query_pool, sub.query_count)
			append(&open_scopes, Profile_Scope{name = profile_key(span_string(cmd, v.name)), begin = sub.query_count})
			sub.query_count += 1
		case Cmd_End_Profile:
			if len(open_scopes) == 0 {
				log.errorf("end_profile without a matching begin_profile")
				vk_check(vk.ResetCommandBuffer(cb, {}))
				append(&free_submissions, sub)
				return false
			}
			scope := pop(&open_scopes)
			scope.end = sub.query_count
			vk.CmdWriteTimestamp2(cb, {.ALL_COMMANDS}, sub.query_pool, sub.query_count)
			sub.query_count += 1
			append(&sub.scopes, scope)
		case Cmd_Set_Buffer:
			cmd.buffers[{v.pipeline, span_string(cmd, v.name)}] = v.buffer
		case Cmd_Set_Texture:
			cmd.textures[{v.pipeline, span_string(cmd, v.name)}] = v.texture
		case Cmd_Set_Tlas:
			cmd.tlases[{v.pipeline, span_string(cmd, v.name)}] = v.tlas
		case Cmd_Set_Uniform:
			cmd.uniforms[{v.pipeline, span_string(cmd, v.name)}] = span_bytes(cmd, v.data)
		case Cmd_Set_Cbuffer:
			cmd.cbuffers[{v.pipeline, span_string(cmd, v.name)}] = span_bytes(cmd, v.data)
		case Cmd_Set_Texture_Array:
			cmd.arrays[{v.pipeline, span_string(cmd, v.name)}] = v.set
		case Cmd_Write_Texture_Array:
			image_info := vk.DescriptorImageInfo{imageView = v.view, imageLayout = .GENERAL}
			write := vk.WriteDescriptorSet {
				sType           = .WRITE_DESCRIPTOR_SET,
				dstSet          = v.set,
				dstBinding      = 0,
				dstArrayElement = v.slot,
				descriptorCount = 1,
				descriptorType  = .SAMPLED_IMAGE,
				pImageInfo      = &image_info,
			}
			vk.UpdateDescriptorSets(device, 1, &write, 0, nil)
		case Cmd_Dispatch:
			if !record_bindings(cmd, sub, v.shader, v.kernel) {
				vk_check(vk.ResetCommandBuffer(cb, {}))
				append(&free_submissions, sub)
				return false
			}
			begin_label(cb, v.shader, v.kernel)
			vk.CmdDispatch(cb, v.x, v.y, v.z)
			end_label(cb)
		case Cmd_Dispatch_Indirect:
			if !record_bindings(cmd, sub, v.shader, v.kernel) {
				vk_check(vk.ResetCommandBuffer(cb, {}))
				append(&free_submissions, sub)
				return false
			}
			begin_label(cb, v.shader, v.kernel)
			vk.CmdDispatchIndirect(cb, v.buffer.handle, vk.DeviceSize(v.offset))
			end_label(cb)
		case Cmd_Build_As:
			full_barrier(cb)
			geometry := v.geometry
			build := vk.AccelerationStructureBuildGeometryInfoKHR {
				sType                    = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
				type                     = v.type,
				flags                    = v.flags,
				mode                     = v.mode,
				srcAccelerationStructure = v.src,
				dstAccelerationStructure = v.dst,
				geometryCount            = 1,
				pGeometries              = &geometry,
				scratchData              = {deviceAddress = v.scratch},
			}
			range := vk.AccelerationStructureBuildRangeInfoKHR{primitiveCount = v.primitive_count}
			range_ptr: [^]vk.AccelerationStructureBuildRangeInfoKHR = &range
			vk.CmdBuildAccelerationStructuresKHR(cb, 1, &build, &range_ptr)
		case Cmd_Upload_Buffer:
			offset := staging_push(sub, span_bytes(cmd, v.data), 16)
			full_barrier(cb)
			region := vk.BufferCopy{srcOffset = vk.DeviceSize(offset), size = vk.DeviceSize(v.data.len)}
			vk.CmdCopyBuffer(cb, sub.staging.handle, v.dst.handle, 1, &region)
		case Cmd_Upload_Texture:
			offset := staging_push(sub, span_bytes(cmd, v.data), 16)
			full_barrier(cb)
			region := vk.BufferImageCopy {
				bufferOffset     = vk.DeviceSize(offset),
				imageSubresource = {aspectMask = {.COLOR}, layerCount = 1},
				imageExtent      = {u32(v.dst.width), u32(v.dst.height), 1},
			}
			vk.CmdCopyBufferToImage(cb, sub.staging.handle, v.dst.image, .GENERAL, 1, &region)
		case Cmd_Copy_Buffer:
			full_barrier(cb)
			region := vk.BufferCopy{size = vk.DeviceSize(v.src.size)}
			vk.CmdCopyBuffer(cb, v.src.handle, v.dst.handle, 1, &region)
		case Cmd_Copy_Texture:
			full_barrier(cb)
			region := vk.ImageCopy {
				srcSubresource = {aspectMask = {.COLOR}, layerCount = 1},
				dstSubresource = {aspectMask = {.COLOR}, layerCount = 1},
				extent         = {u32(v.src.width), u32(v.src.height), 1},
			}
			vk.CmdCopyImage(cb, v.src.image, .GENERAL, v.dst.image, .GENERAL, 1, &region)
		}
	}
	if len(open_scopes) > 0 do log.errorf("begin_profile %q was never ended", open_scopes[len(open_scopes) - 1].name)
	vk_check(vk.EndCommandBuffer(cb))

	cb_info := vk.CommandBufferSubmitInfo{sType = .COMMAND_BUFFER_SUBMIT_INFO, commandBuffer = cb}
	vk_check(vk.QueueSubmit2(queue, 1, &vk.SubmitInfo2 {
		sType                  = .SUBMIT_INFO_2,
		commandBufferInfoCount = 1,
		pCommandBufferInfos    = &cb_info,
	}, sub.fence))
	clear(&pending_images)
	append(&sub.garbage, ..cmd.garbage[:])
	clear(&cmd.garbage)
	append(&sub.garbage, ..orphan_garbage[:])
	clear(&orphan_garbage)
	append(&pending, sub)
	return true
}

// A bottom-level acceleration structure built from one triangle mesh.
Blas :: struct {
	handle:  vk.AccelerationStructureKHR,
	buffer:  Buffer,
	address: vk.DeviceAddress,
}

// A top-level acceleration structure over Blas instances.
Tlas :: struct {
	handle:         vk.AccelerationStructureKHR,
	buffer:         Buffer,
	instance_count: u32,
}

// One Blas placed in a Tlas. transform is a row-major 3x4, all zeros means identity.
// id is what InstanceID() returns in the shader. double_sided exempts the instance from back-face culling.
Instance :: struct {
	blas:         Blas,
	transform:    matrix[3, 4]f32,
	id:           u32,
	double_sided: bool,
}

// Creates an empty BLAS. Build it with build_blas.
create_blas :: proc() -> Blas {
	return {}
}

// Records a build of the BLAS from positions and triangle indices.
// Reuses the existing BLAS memory when it is big enough.
build_blas :: proc(cmd: ^Cmd, b: ^Blas, positions: [][3]f32, indices: []u32) {
	input_usage := vk.BufferUsageFlags{.ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .SHADER_DEVICE_ADDRESS}
	vb := create_host_buffer(slice.to_bytes(positions), input_usage)
	ib := create_host_buffer(slice.to_bytes(indices), input_usage)
	append(&cmd.garbage, Garbage{buffer = vb})
	append(&cmd.garbage, Garbage{buffer = ib})

	geometry := vk.AccelerationStructureGeometryKHR {
		sType        = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		geometryType = .TRIANGLES,
		flags        = {.OPAQUE},
	}
	geometry.geometry.triangles = {
		sType        = .ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		vertexFormat = .R32G32B32_SFLOAT,
		vertexData   = {deviceAddress = vb.address},
		vertexStride = size_of([3]f32),
		maxVertex    = u32(len(positions) - 1),
		indexType    = .UINT32,
		indexData    = {deviceAddress = ib.address},
	}
	record_as_build(cmd, .BOTTOM_LEVEL, &b.handle, &b.buffer, &geometry, u32(len(indices) / 3), false)
	b.address = vk.GetAccelerationStructureDeviceAddressKHR(device, &vk.AccelerationStructureDeviceAddressInfoKHR {
		sType                 = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		accelerationStructure = b.handle,
	})
}

// Waits for the GPU, then destroys the BLAS.
destroy_blas :: proc(b: Blas) {
	if b.handle == 0 do return
	wait_idle()
	vk.DestroyAccelerationStructureKHR(device, b.handle, nil)
	destroy_buffer(b.buffer)
}

// Creates an empty TLAS. Build it with build_tlas.
create_tlas :: proc() -> Tlas {
	return {}
}

// Records a build of the TLAS from instances.
// With refit set, updates the existing TLAS in place instead of rebuilding.
build_tlas :: proc(cmd: ^Cmd, t: ^Tlas, instances: []Instance, refit := false) {
	vk_instances := make([]vk.AccelerationStructureInstanceKHR, len(instances), virtual.arena_allocator(&scratch))
	for inst, i in instances {
		m := inst.transform
		if m == {} do m = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
		v := &vk_instances[i]
		for r in 0 ..< 3 do for c in 0 ..< 4 do v.transform.mat[r][c] = m[r, c]
		v.instanceCustomIndex = inst.id
		v.mask = 0xFF
		if inst.double_sided do v.flags = vk.GeometryInstanceFlagKHR(1 << u32(vk.GeometryInstanceFlagKHR.TRIANGLE_FACING_CULL_DISABLE))
		v.accelerationStructureReference = u64(inst.blas.address)
	}
	input_usage := vk.BufferUsageFlags{.ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR, .SHADER_DEVICE_ADDRESS}
	instance_buffer := create_host_buffer(slice.to_bytes(vk_instances), input_usage)
	append(&cmd.garbage, Garbage{buffer = instance_buffer})

	geometry := vk.AccelerationStructureGeometryKHR {
		sType        = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		geometryType = .INSTANCES,
	}
	geometry.geometry.instances = {
		sType = .ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
		data  = {deviceAddress = instance_buffer.address},
	}
	do_refit := refit && t.handle != 0 && t.instance_count == u32(len(instances))
	record_as_build(cmd, .TOP_LEVEL, &t.handle, &t.buffer, &geometry, u32(len(instances)), do_refit)
	t.instance_count = u32(len(instances))
}

// Waits for the GPU, then destroys the TLAS.
destroy_tlas :: proc(t: Tlas) {
	if t.handle == 0 do return
	wait_idle()
	vk.DestroyAccelerationStructureKHR(device, t.handle, nil)
	destroy_buffer(t.buffer)
}

@(private) window: glfw.WindowHandle
@(private) imgui_enabled: bool
@(private) vsync_enabled: bool
@(private) pending_images: [dynamic]vk.Image
@(private) orphan_garbage: [dynamic]Garbage
@(private) scratch: virtual.Arena
@(private) instance: vk.Instance

@(private) physical_device: vk.PhysicalDevice

@(private) device: vk.Device

@(private) queue: vk.Queue

@(private) queue_family: u32

@(private) surface: vk.SurfaceKHR

@(private) swapchain: vk.SwapchainKHR

@(private) swapchain_images: []vk.Image
@(private) swapchain_views: []vk.ImageView
@(private) swapchain_min_images: u32

@(private) swapchain_extent: vk.Extent2D

@(private) command_pool: vk.CommandPool

@(private) default_sampler: vk.Sampler

@(private) texture_array_layout: vk.DescriptorSetLayout

@(private) texture_array_capacity: u32

@(private) uniform_alignment: int

@(private) memory_properties: vk.PhysicalDeviceMemoryProperties

@(private)
Submission :: struct {
	cb:             vk.CommandBuffer,
	fence:          vk.Fence,
	staging:        Buffer,
	staging_offset: int,
	query_pool:     vk.QueryPool,
	query_count:    u32,
	scopes:         [dynamic]Profile_Scope,
	garbage:        [dynamic]Garbage,
	frame:          u64,
}

@(private) MAX_PROFILE_QUERIES :: 256

@(private)
Profile_Scope :: struct {
	name:       string,
	begin, end: u32,
}

@(private) timestamp_period: f32
@(private) frame_counter: u64
@(private) profile_keys: map[string]string
@(private) profile_collect: map[string]f32
@(private) profile_collect_frame: u64
@(private) profile_times: map[string]f32

@(private) submissions: [dynamic]^Submission
@(private) pending: [dynamic]^Submission
@(private) free_submissions: [dynamic]^Submission


@(private) present_cb: vk.CommandBuffer

@(private) present_cb_fence: vk.Fence

@(private) present_fence: vk.Fence

@(private) image_available: vk.Semaphore

@(private) render_finished: vk.Semaphore

@(private)
Shader_Param :: struct {
	name:     string,
	binding:  u32,
	space:    u32,
	type:     vk.DescriptorType,
	writable: bool,
	cbuffer:  int,
}

@(private)
Shader_Cbuffer :: struct {
	name:    string,
	size:    int,
	members: []Shader_Member,
}

@(private)
Shader_Member :: struct {
	name:   string,
	offset: int,
	size:   int,
}

@(private)
present_mode :: proc() -> vk.PresentModeKHR {
	if vsync_enabled do return .FIFO
	count: u32
	vk_check(vk.GetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, nil))
	modes := make([]vk.PresentModeKHR, count, virtual.arena_allocator(&scratch))
	vk_check(vk.GetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &count, raw_data(modes)))
	for m in modes do if m == .MAILBOX do return .MAILBOX
	for m in modes do if m == .IMMEDIATE do return .IMMEDIATE
	return .FIFO
}

@(private)
create_swapchain :: proc() {
	old := swapchain
	caps: vk.SurfaceCapabilitiesKHR
	vk_check(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &caps))
	fb_w, fb_h := glfw.GetFramebufferSize(window)
	swapchain_min_images = caps.minImageCount
	swapchain_extent = {
		clamp(u32(fb_w), caps.minImageExtent.width, caps.maxImageExtent.width),
		clamp(u32(fb_h), caps.minImageExtent.height, caps.maxImageExtent.height),
	}
	vk_check(vk.CreateSwapchainKHR(device, &vk.SwapchainCreateInfoKHR {
		sType            = .SWAPCHAIN_CREATE_INFO_KHR,
		surface          = surface,
		minImageCount    = caps.minImageCount,
		imageFormat      = .B8G8R8A8_SRGB,
		imageColorSpace  = .SRGB_NONLINEAR,
		imageExtent      = swapchain_extent,
		imageArrayLayers = 1,
		imageUsage       = imgui_enabled ? {.TRANSFER_DST, .COLOR_ATTACHMENT} : {.TRANSFER_DST},
		preTransform     = {.IDENTITY},
		compositeAlpha   = {.OPAQUE},
		presentMode      = present_mode(),
		oldSwapchain     = old,
	}, nil, &swapchain))
	if old != 0 {
		vk.DestroySwapchainKHR(device, old, nil)
		delete(swapchain_images)
		for v in swapchain_views do vk.DestroyImageView(device, v, nil)
		delete(swapchain_views)
	}
	image_count: u32
	vk_check(vk.GetSwapchainImagesKHR(device, swapchain, &image_count, nil))
	swapchain_images = make([]vk.Image, image_count)
	vk_check(vk.GetSwapchainImagesKHR(device, swapchain, &image_count, raw_data(swapchain_images)))
	if imgui_enabled {
		swapchain_views = make([]vk.ImageView, image_count)
		for image, i in swapchain_images {
			vk_check(vk.CreateImageView(device, &vk.ImageViewCreateInfo {
				sType            = .IMAGE_VIEW_CREATE_INFO,
				image            = image,
				viewType         = .D2,
				format           = .B8G8R8A8_SRGB,
				subresourceRange = {aspectMask = {.COLOR}, levelCount = 1, layerCount = 1},
			}, nil, &swapchain_views[i]))
		}
	}
}

@(private) debug_utils: bool

@(private)
begin_label :: proc(cb: vk.CommandBuffer, s: Shader, kernel: int) {
	if !debug_utils do return
	label := vk.DebugUtilsLabelEXT {
		sType      = .DEBUG_UTILS_LABEL_EXT,
		pLabelName = fmt.caprintf("%s.%s", s.name, s.kernels[kernel].name, allocator = virtual.arena_allocator(&scratch)),
	}
	vk.CmdBeginDebugUtilsLabelEXT(cb, &label)
}

@(private)
end_label :: proc(cb: vk.CommandBuffer) {
	if debug_utils do vk.CmdEndDebugUtilsLabelEXT(cb)
}

@(private)
vk_check :: proc(res: vk.Result, loc := #caller_location) {
	if res != .SUCCESS do fmt.panicf("Vulkan call failed: %v", res, loc = loc)
}

@(private)
wait_idle :: proc() {
	vk_check(vk.DeviceWaitIdle(device))
}

@(private)
align_up :: proc(x, a: int) -> int {
	return (x + a - 1) / a * a
}

@(private)
create_fence :: proc(signaled: bool) -> (f: vk.Fence) {
	flags: vk.FenceCreateFlags = signaled ? {.SIGNALED} : {}
	vk_check(vk.CreateFence(device, &vk.FenceCreateInfo{sType = .FENCE_CREATE_INFO, flags = flags}, nil, &f))
	return
}

@(private)
full_barrier :: proc(cb: vk.CommandBuffer) {
	barrier := vk.MemoryBarrier2 {
		sType         = .MEMORY_BARRIER_2,
		srcStageMask  = {.ALL_COMMANDS},
		srcAccessMask = {.MEMORY_READ, .MEMORY_WRITE},
		dstStageMask  = {.ALL_COMMANDS},
		dstAccessMask = {.MEMORY_READ, .MEMORY_WRITE},
	}
	vk.CmdPipelineBarrier2(cb, &vk.DependencyInfo{sType = .DEPENDENCY_INFO, memoryBarrierCount = 1, pMemoryBarriers = &barrier})
}

@(private)
transition_image :: proc(cb: vk.CommandBuffer, image: vk.Image, old_layout, new_layout: vk.ImageLayout) {
	barrier := vk.ImageMemoryBarrier2 {
		sType               = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask        = {.ALL_COMMANDS},
		srcAccessMask       = {.MEMORY_READ, .MEMORY_WRITE},
		dstStageMask        = {.ALL_COMMANDS},
		dstAccessMask       = {.MEMORY_READ, .MEMORY_WRITE},
		oldLayout           = old_layout,
		newLayout           = new_layout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image               = image,
		subresourceRange    = {aspectMask = {.COLOR}, levelCount = 1, layerCount = 1},
	}
	vk.CmdPipelineBarrier2(cb, &vk.DependencyInfo{sType = .DEPENDENCY_INFO, imageMemoryBarrierCount = 1, pImageMemoryBarriers = &barrier})
}

@(private)
allocate :: proc(reqs: vk.MemoryRequirements, host_visible, device_address: bool) -> (memory: vk.DeviceMemory, mapped: rawptr) {
	wanted: vk.MemoryPropertyFlags = host_visible ? {.HOST_VISIBLE, .HOST_COHERENT} : {.DEVICE_LOCAL}
	type_index := max(u32)
	for i in 0 ..< memory_properties.memoryTypeCount {
		if reqs.memoryTypeBits & (1 << i) != 0 && wanted <= memory_properties.memoryTypes[i].propertyFlags {
			type_index = i
			break
		}
	}
	if type_index == max(u32) do fmt.panicf("no memory type with %v", wanted)

	flags_info := vk.MemoryAllocateFlagsInfo{sType = .MEMORY_ALLOCATE_FLAGS_INFO, flags = {.DEVICE_ADDRESS}}
	info := vk.MemoryAllocateInfo{sType = .MEMORY_ALLOCATE_INFO, allocationSize = reqs.size, memoryTypeIndex = type_index}
	if device_address do info.pNext = &flags_info
	vk_check(vk.AllocateMemory(device, &info, nil, &memory))
	if host_visible do vk_check(vk.MapMemory(device, memory, 0, vk.DeviceSize(vk.WHOLE_SIZE), {}, &mapped))
	return
}

@(private)
create_buffer_ex :: proc(size: int, usage: vk.BufferUsageFlags, host_visible: bool) -> Buffer {
	b := Buffer{size = uint(size)}
	vk_check(vk.CreateBuffer(device, &vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = vk.DeviceSize(size),
		usage       = usage,
		sharingMode = .EXCLUSIVE,
	}, nil, &b.handle))
	reqs: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(device, b.handle, &reqs)
	device_address := .SHADER_DEVICE_ADDRESS in usage
	b.memory, b.mapped = allocate(reqs, host_visible, device_address)
	vk_check(vk.BindBufferMemory(device, b.handle, b.memory, 0))
	if device_address {
		b.address = vk.GetBufferDeviceAddress(device, &vk.BufferDeviceAddressInfo{sType = .BUFFER_DEVICE_ADDRESS_INFO, buffer = b.handle})
	}
	return b
}

@(private)
free_buffer :: proc(b: Buffer) {
	vk.DestroyBuffer(device, b.handle, nil)
	vk.FreeMemory(device, b.memory, nil)
}

@(private)
create_host_buffer :: proc(data: []byte, usage: vk.BufferUsageFlags) -> Buffer {
	b := create_buffer_ex(len(data), usage, host_visible = true)
	mem.copy(b.mapped, raw_data(data), len(data))
	return b
}

@(private)
ensure_staging :: proc(f: ^Submission, need: int) {
	if f.staging.size >= uint(need) do return
	if f.staging.handle != 0 do free_buffer(f.staging)
	size := max(need, int(f.staging.size) * 2, 16 * 1024 * 1024)
	f.staging = create_buffer_ex(size, {.TRANSFER_SRC, .UNIFORM_BUFFER}, host_visible = true)
}

@(private)
allocate_command_buffer :: proc() -> (cb: vk.CommandBuffer) {
	vk_check(vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = command_pool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}, &cb))
	return
}

@(private)
acquire_submission :: proc() -> ^Submission {
	for len(pending) > 0 && vk.GetFenceStatus(device, pending[0].fence) == .SUCCESS {
		s := pending[0]
		ordered_remove(&pending, 0)
		vk_check(vk.ResetFences(device, 1, &s.fence))
		collect_profile(s)
		flush_garbage(&s.garbage)
		append(&free_submissions, s)
	}
	if len(free_submissions) > 0 do return pop(&free_submissions)
	s := new(Submission)
	s.cb = allocate_command_buffer()
	s.fence = create_fence(signaled = false)
	vk_check(vk.CreateQueryPool(device, &vk.QueryPoolCreateInfo {
		sType      = .QUERY_POOL_CREATE_INFO,
		queryType  = .TIMESTAMP,
		queryCount = MAX_PROFILE_QUERIES,
	}, nil, &s.query_pool))
	ensure_staging(s, 1)
	append(&submissions, s)
	return s
}

@(private)
profile_key :: proc(name: string) -> string {
	if k, ok := profile_keys[name]; ok do return k
	k := strings.clone(name)
	profile_keys[k] = k
	return k
}

@(private)
collect_profile :: proc(s: ^Submission) {
	if len(s.scopes) == 0 do return
	results: [MAX_PROFILE_QUERIES]u64
	vk_check(vk.GetQueryPoolResults(device, s.query_pool, 0, s.query_count, size_of(u64) * int(s.query_count), &results[0], size_of(u64), {._64, .WAIT}))
	if s.frame != profile_collect_frame {
		clear(&profile_times)
		for k, v in profile_collect do profile_times[k] = v
		clear(&profile_collect)
		profile_collect_frame = s.frame
	}
	for scope in s.scopes {
		profile_collect[scope.name] = profile_collect[scope.name] + f32(results[scope.end] - results[scope.begin]) * timestamp_period
	}
}

@(private)
staging_push :: proc(f: ^Submission, data: []byte, alignment: int) -> int {
	offset := align_up(f.staging_offset, alignment)
	assert(uint(offset + len(data)) <= f.staging.size, "staging arena overflow (execute_cmd() sizes it up front, so this is a bug)")
	mem.copy(rawptr(uintptr(f.staging.mapped) + uintptr(offset)), raw_data(data), len(data))
	f.staging_offset = offset + len(data)
	return offset
}

@(private)
Shader_Entry :: struct {
	name:       string,
	group_size: [3]u32,
}

@(private)
Shader_Reflection :: struct {
	params:       [dynamic]Shader_Param,
	cbuffers:     [dynamic]Shader_Cbuffer,
	globals:      [dynamic]Shader_Member,
	globals_size: int,
	entries:      [dynamic]Shader_Entry,
}

@(private)
free_reflection :: proc(r: ^Shader_Reflection) {
	for p in r.params do delete(p.name)
	for cb in r.cbuffers {
		for m in cb.members do delete(m.name)
		delete(cb.members)
	}
	for m in r.globals do delete(m.name)
	for e in r.entries do delete(e.name)
	delete(r.params)
	delete(r.cbuffers)
	delete(r.globals)
	delete(r.entries)
}

@(private)
create_shader :: proc(spv: []byte, r: ^Shader_Reflection, debug_name: string) -> (result: Shader, ok: bool) {
	words := slice.reinterpret([]u32, spv)
	if len(words) <= 5 || words[0] != 0x0723_0203 {
		log.errorf("%s: invalid SPIR-V", debug_name)
		return
	}
	param_count := len(r.params) + (len(r.globals) > 0 ? 1 : 0)
	if param_count > 64 {
		log.errorf("%s: too many shader parameters (%d, max 64)", debug_name, param_count)
		return
	}
	if len(r.globals) > 0 {
		append(&r.params, Shader_Param{name = "$Globals", binding = 0, type = .UNIFORM_BUFFER, cbuffer = len(r.cbuffers)})
		append(&r.cbuffers, Shader_Cbuffer{name = "$Globals", size = align_up(r.globals_size, 16), members = r.globals[:]})
	}
	s := Shader{name = strings.clone(debug_name)}
	s.params = r.params[:]
	s.cbuffers = r.cbuffers[:]

	bindings := make([]vk.DescriptorSetLayoutBinding, len(s.params), virtual.arena_allocator(&scratch))
	binding_count := 0
	max_space: u32 = 0
	for p in s.params {
		if p.space != 0 {
			max_space = max(max_space, p.space)
			continue
		}
		bindings[binding_count] = {binding = p.binding, descriptorType = p.type, descriptorCount = 1, stageFlags = {.COMPUTE}}
		binding_count += 1
	}
	vk_check(vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		flags        = {.PUSH_DESCRIPTOR},
		bindingCount = u32(binding_count),
		pBindings    = raw_data(bindings),
	}, nil, &s.set_layout))
	set_layouts := make([]vk.DescriptorSetLayout, max_space + 1, virtual.arena_allocator(&scratch))
	set_layouts[0] = s.set_layout
	for i in 1 ..= max_space do set_layouts[i] = texture_array_layout
	vk_check(vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo {
		sType          = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount = max_space + 1,
		pSetLayouts    = raw_data(set_layouts),
	}, nil, &s.layout))

	module: vk.ShaderModule
	vk_check(vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(spv),
		pCode    = cast(^u32)raw_data(spv),
	}, nil, &module))
	defer vk.DestroyShaderModule(device, module, nil)

	usage := spirv_usage(spv, s.params)
	s.kernels = make([]Kernel, len(r.entries))
	for e, i in r.entries {
		pipeline: vk.Pipeline
		vk_check(vk.CreateComputePipelines(device, 0, 1, &vk.ComputePipelineCreateInfo {
			sType  = .COMPUTE_PIPELINE_CREATE_INFO,
			stage  = {
				sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
				stage  = {.COMPUTE},
				module = module,
				pName  = strings.clone_to_cstring(e.name, virtual.arena_allocator(&scratch)),
			},
			layout = s.layout,
		}, nil, &pipeline))
		s.kernels[i] = Kernel{name = e.name, pipeline = pipeline, group_size = e.group_size, used = usage[e.name] or_else max(u64)}
		if debug_utils {
			vk.SetDebugUtilsObjectNameEXT(device, &vk.DebugUtilsObjectNameInfoEXT {
				sType        = .DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
				objectType   = .PIPELINE,
				objectHandle = u64(pipeline),
				pObjectName  = fmt.caprintf("%s.%s", debug_name, e.name, allocator = virtual.arena_allocator(&scratch)),
			})
		}
	}
	delete(r.entries)
	return s, true
}

@(private)
spirv_usage :: proc(spv: []byte, params: []Shader_Param) -> map[string]u64 {
	words := slice.reinterpret([]u32, spv)
	alloc := virtual.arena_allocator(&scratch)
	bindings := make(map[u32]u32, alloc)
	spaces := make(map[u32]u32, alloc)
	for i := 5; i < len(words); {
		size := int(words[i] >> 16)
		if size <= 0 do break
		if words[i] & 0xFFFF == 71 && words[i + 2] == 33 do bindings[words[i + 1]] = words[i + 3]
		if words[i] & 0xFFFF == 71 && words[i + 2] == 34 do spaces[words[i + 1]] = words[i + 3]
		i += size
	}
	usage := make(map[string]u64, alloc)
	for i := 5; i < len(words); {
		size := int(words[i] >> 16)
		if size <= 0 do break
		if words[i] & 0xFFFF == 15 {
			name_bytes := slice.bytes_from_ptr(&words[i + 3], (size - 3) * 4)
			name_len := 0
			for name_bytes[name_len] != 0 do name_len += 1
			mask: u64
			for j in i + 3 + (name_len + 4) / 4 ..< i + size {
				if binding, ok := bindings[words[j]]; ok {
					for p, pi in params do if p.binding == binding && p.space == (spaces[words[j]] or_else 0) do mask |= u64(1) << u32(pi)
				}
			}
			usage[string(name_bytes[:name_len])] = mask
		}
		i += size
	}
	return usage
}

@(private)
reflect_json :: proc(obj: json.Object, path: string) -> (r: Shader_Reflection, ok: bool) {
	for p in obj["parameters"].(json.Array) {
		po := p.(json.Object)
		name := strings.clone(po["name"].(json.String))
		binding := po["binding"].(json.Object)
		type := po["type"].(json.Object)
		switch kind := binding["kind"].(json.String); kind {
		case "pushConstantBuffer":
			log.errorf("%s: %q is a push constant block, use a cbuffer instead", path, name)
			return
		case "uniform":

			offset := json_int(binding["offset"])
			size := json_int(binding["size"])
			add_members(&r.globals, name, type, offset, size)
			r.globals_size = max(r.globals_size, offset + size)
		case "subElementRegisterSpace":
			if type["kind"].(json.String) != "parameterBlock" {
				log.errorf("%s: %q must be a ParameterBlock<Texture2D[]>", path, name)
				return
			}
			append(&r.params, Shader_Param{name = name, space = u32(json_int(binding["index"])), type = .SAMPLED_IMAGE, cbuffer = -1})
		case "descriptorTableSlot":
			if space, has := binding["space"]; has && json_int(space) != 0 {
				log.errorf("%s: %q uses descriptor set %d; only set 0 is supported", path, name, json_int(space))
				return
			}
			param := Shader_Param{name = name, binding = u32(json_int(binding["index"])), cbuffer = -1}
			switch tkind := type["kind"].(json.String); tkind {
			case "constantBuffer":
				param.type = .UNIFORM_BUFFER
				param.cbuffer = len(r.cbuffers)
				append(&r.cbuffers, cbuffer_from_type(name, type))
			case "samplerState":
				param.type = .SAMPLER
			case "resource":
				access, has_access := type["access"]
				param.writable = has_access && access.(json.String) == "readWrite"
				switch shape := type["baseShape"].(json.String); shape {
				case "structuredBuffer", "byteAddressBuffer":
					param.type = .STORAGE_BUFFER
				case "texture1D", "texture2D", "texture3D", "textureCube":
					param.type = param.writable ? .STORAGE_IMAGE : .SAMPLED_IMAGE
				case "accelerationStructure":
					param.type = .ACCELERATION_STRUCTURE_KHR
				case:
					log.errorf("%s: unsupported resource shape %q for %q", path, shape, name)
					return
				}
			case:
				log.errorf("%s: unsupported parameter type %q for %q", path, tkind, name)
				return
			}
			append(&r.params, param)
		case:
			log.errorf("%s: unsupported binding kind %q for %q", path, kind, name)
			return
		}
	}
	for e in obj["entryPoints"].(json.Array) {
		eo := e.(json.Object)
		if eo["stage"].(json.String) != "compute" {
			log.errorf("%s: only compute kernels are supported", path)
			return
		}
		tg := eo["threadGroupSize"].(json.Array)
		append(&r.entries, Shader_Entry {
			name       = strings.clone(eo["name"].(json.String)),
			group_size = {u32(json_int(tg[0])), u32(json_int(tg[1])), u32(json_int(tg[2]))},
		})
	}
	if len(r.entries) == 0 {
		log.errorf("%s: no entry points", path)
		return
	}
	return r, true
}

@(private) slang_session: rawptr

@(private) SLANG_PARAMETER_CATEGORY_UNIFORM :: 8
@(private) SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT :: 9
@(private) SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER :: 11
@(private) SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE :: 20
@(private) SLANG_TYPE_KIND_ARRAY :: 2
@(private) SLANG_TYPE_KIND_PARAMETER_BLOCK :: 11
@(private) SLANG_TYPE_KIND_STRUCT :: 1
@(private) SLANG_TYPE_KIND_CONSTANT_BUFFER :: 6
@(private) SLANG_TYPE_KIND_RESOURCE :: 7
@(private) SLANG_TYPE_KIND_SAMPLER_STATE :: 8
@(private) SLANG_RESOURCE_BASE_SHAPE_MASK :: 0x0F
@(private) SLANG_TEXTURE_1D :: 0x01
@(private) SLANG_TEXTURE_2D :: 0x02
@(private) SLANG_TEXTURE_3D :: 0x03
@(private) SLANG_TEXTURE_CUBE :: 0x04
@(private) SLANG_STRUCTURED_BUFFER :: 0x06
@(private) SLANG_BYTE_ADDRESS_BUFFER :: 0x07
@(private) SLANG_ACCELERATION_STRUCTURE :: 0x09
@(private) SLANG_RESOURCE_ACCESS_READ_WRITE :: 2
@(private) SLANG_STAGE_COMPUTE :: 6

when ODIN_OS == .Windows {
	foreign import slang "system:slang.lib"
} else {
	foreign import slang "system:slang"
}

@(private, default_calling_convention = "c")
foreign slang {
	spCreateSession :: proc(deprecated: cstring) -> rawptr ---
	spDestroySession :: proc(session: rawptr) ---
	spCreateCompileRequest :: proc(session: rawptr) -> rawptr ---
	spDestroyCompileRequest :: proc(request: rawptr) ---
	spProcessCommandLineArguments :: proc(request: rawptr, args: [^]cstring, count: i32) -> i32 ---
	spCompile :: proc(request: rawptr) -> i32 ---
	spGetDiagnosticOutput :: proc(request: rawptr) -> cstring ---
	spGetCompileRequestCode :: proc(request: rawptr, size: ^uint) -> rawptr ---
	spGetReflection :: proc(request: rawptr) -> rawptr ---
	spReflection_GetParameterCount :: proc(reflection: rawptr) -> u32 ---
	spReflection_GetParameterByIndex :: proc(reflection: rawptr, index: u32) -> rawptr ---
	spReflection_getEntryPointCount :: proc(reflection: rawptr) -> u64 ---
	spReflection_getEntryPointByIndex :: proc(reflection: rawptr, index: u64) -> rawptr ---
	spReflectionEntryPoint_getName :: proc(entry_point: rawptr) -> cstring ---
	spReflectionEntryPoint_getStage :: proc(entry_point: rawptr) -> i32 ---
	spReflectionEntryPoint_getComputeThreadGroupSize :: proc(entry_point: rawptr, axis_count: u64, sizes: [^]u64) ---
	spReflectionVariableLayout_GetVariable :: proc(layout: rawptr) -> rawptr ---
	spReflectionVariableLayout_GetTypeLayout :: proc(layout: rawptr) -> rawptr ---
	spReflectionVariableLayout_GetOffset :: proc(layout: rawptr, category: i32) -> uint ---
	spReflectionParameter_GetBindingIndex :: proc(layout: rawptr) -> u32 ---
	spReflectionParameter_GetBindingSpace :: proc(layout: rawptr) -> u32 ---
	spReflectionVariable_GetName :: proc(variable: rawptr) -> cstring ---
	spReflectionTypeLayout_GetType :: proc(layout: rawptr) -> rawptr ---
	spReflectionTypeLayout_getKind :: proc(layout: rawptr) -> i32 ---
	spReflectionTypeLayout_GetSize :: proc(layout: rawptr, category: i32) -> uint ---
	spReflectionTypeLayout_GetFieldCount :: proc(layout: rawptr) -> u32 ---
	spReflectionTypeLayout_GetFieldByIndex :: proc(layout: rawptr, index: u32) -> rawptr ---
	spReflectionTypeLayout_GetElementTypeLayout :: proc(layout: rawptr) -> rawptr ---
	spReflectionTypeLayout_GetParameterCategory :: proc(layout: rawptr) -> i32 ---
	spReflectionType_GetName :: proc(type: rawptr) -> cstring ---
	spReflectionType_GetResourceShape :: proc(type: rawptr) -> i32 ---
	spReflectionType_GetResourceAccess :: proc(type: rawptr) -> i32 ---
}

@(private)
reflect_slang :: proc(reflection: rawptr, path: string) -> (r: Shader_Reflection, ok: bool) {
	for i in 0 ..< spReflection_GetParameterCount(reflection) {
		layout := spReflection_GetParameterByIndex(reflection, i)
		type_layout := spReflectionVariableLayout_GetTypeLayout(layout)
		name := strings.clone(string(spReflectionVariable_GetName(spReflectionVariableLayout_GetVariable(layout))))
		switch category := spReflectionTypeLayout_GetParameterCategory(type_layout); category {
		case SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER:
			log.errorf("%s: %q is a push constant block, use a cbuffer instead", path, name)
			return
		case SLANG_PARAMETER_CATEGORY_UNIFORM:
			offset := int(spReflectionVariableLayout_GetOffset(layout, SLANG_PARAMETER_CATEGORY_UNIFORM))
			size := int(spReflectionTypeLayout_GetSize(type_layout, SLANG_PARAMETER_CATEGORY_UNIFORM))
			slang_add_members(&r.globals, name, type_layout, offset, size)
			r.globals_size = max(r.globals_size, offset + size)
		case SLANG_PARAMETER_CATEGORY_SUB_ELEMENT_REGISTER_SPACE:
			element := spReflectionTypeLayout_GetElementTypeLayout(type_layout)
			valid := false
			if spReflectionTypeLayout_getKind(type_layout) == SLANG_TYPE_KIND_PARAMETER_BLOCK && spReflectionTypeLayout_getKind(element) == SLANG_TYPE_KIND_ARRAY {
				item := spReflectionTypeLayout_GetType(spReflectionTypeLayout_GetElementTypeLayout(element))
				valid = spReflectionType_GetResourceShape(item) & SLANG_RESOURCE_BASE_SHAPE_MASK == SLANG_TEXTURE_2D && spReflectionType_GetResourceAccess(item) != SLANG_RESOURCE_ACCESS_READ_WRITE
			}
			if !valid {
				log.errorf("%s: %q must be a ParameterBlock<Texture2D[]>", path, name)
				return
			}
			append(&r.params, Shader_Param{name = name, space = spReflectionParameter_GetBindingIndex(layout), type = .SAMPLED_IMAGE, cbuffer = -1})
		case SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT:
			if space := spReflectionParameter_GetBindingSpace(layout); space != 0 {
				log.errorf("%s: %q uses descriptor set %d; only set 0 is supported", path, name, space)
				return
			}
			param := Shader_Param{name = name, binding = spReflectionParameter_GetBindingIndex(layout), cbuffer = -1}
			switch kind := spReflectionTypeLayout_getKind(type_layout); kind {
			case SLANG_TYPE_KIND_CONSTANT_BUFFER:
				param.type = .UNIFORM_BUFFER
				param.cbuffer = len(r.cbuffers)
				append(&r.cbuffers, slang_cbuffer(name, type_layout))
			case SLANG_TYPE_KIND_SAMPLER_STATE:
				param.type = .SAMPLER
			case SLANG_TYPE_KIND_RESOURCE:
				type := spReflectionTypeLayout_GetType(type_layout)
				param.writable = spReflectionType_GetResourceAccess(type) == SLANG_RESOURCE_ACCESS_READ_WRITE
				switch shape := spReflectionType_GetResourceShape(type) & SLANG_RESOURCE_BASE_SHAPE_MASK; shape {
				case SLANG_STRUCTURED_BUFFER, SLANG_BYTE_ADDRESS_BUFFER:
					param.type = .STORAGE_BUFFER
				case SLANG_TEXTURE_1D, SLANG_TEXTURE_2D, SLANG_TEXTURE_3D, SLANG_TEXTURE_CUBE:
					param.type = param.writable ? .STORAGE_IMAGE : .SAMPLED_IMAGE
				case SLANG_ACCELERATION_STRUCTURE:
					param.type = .ACCELERATION_STRUCTURE_KHR
				case:
					log.errorf("%s: unsupported resource shape %d for %q", path, shape, name)
					return
				}
			case:
				log.errorf("%s: unsupported parameter type kind %d for %q", path, kind, name)
				return
			}
			append(&r.params, param)
		case:
			log.errorf("%s: unsupported parameter category %d for %q", path, category, name)
			return
		}
	}
	count := spReflection_getEntryPointCount(reflection)
	if count == 0 {
		log.errorf("%s: no entry points", path)
		return
	}
	for i in 0 ..< count {
		entry := spReflection_getEntryPointByIndex(reflection, i)
		if spReflectionEntryPoint_getStage(entry) != SLANG_STAGE_COMPUTE {
			log.errorf("%s: only compute kernels are supported", path)
			return
		}
		sizes: [3]u64
		spReflectionEntryPoint_getComputeThreadGroupSize(entry, 3, raw_data(&sizes))
		append(&r.entries, Shader_Entry {
			name       = strings.clone(string(spReflectionEntryPoint_getName(entry))),
			group_size = {u32(sizes[0]), u32(sizes[1]), u32(sizes[2])},
		})
	}
	return r, true
}

@(private)
slang_cbuffer :: proc(name: string, type_layout: rawptr) -> Shader_Cbuffer {
	element := spReflectionTypeLayout_GetElementTypeLayout(type_layout)
	size := int(spReflectionTypeLayout_GetSize(element, SLANG_PARAMETER_CATEGORY_UNIFORM))
	members := make([dynamic]Shader_Member)
	type_name := spReflectionType_GetName(spReflectionTypeLayout_GetType(element))
	named := type_name != nil && len(type_name) > 0
	slang_add_struct_members(&members, named ? name : "", element, 0)
	return Shader_Cbuffer{name = name, size = size, members = members[:]}
}

@(private)
slang_add_struct_members :: proc(members: ^[dynamic]Shader_Member, prefix: string, type_layout: rawptr, base: int) {
	for i in 0 ..< spReflectionTypeLayout_GetFieldCount(type_layout) {
		field := spReflectionTypeLayout_GetFieldByIndex(type_layout, i)
		field_name := string(spReflectionVariable_GetName(spReflectionVariableLayout_GetVariable(field)))
		full := prefix == "" ? strings.clone(field_name) : fmt.aprintf("%s.%s", prefix, field_name)
		field_layout := spReflectionVariableLayout_GetTypeLayout(field)
		offset := int(spReflectionVariableLayout_GetOffset(field, SLANG_PARAMETER_CATEGORY_UNIFORM))
		size := int(spReflectionTypeLayout_GetSize(field_layout, SLANG_PARAMETER_CATEGORY_UNIFORM))
		slang_add_members(members, full, field_layout, base + offset, size)
	}
}

@(private)
slang_add_members :: proc(members: ^[dynamic]Shader_Member, name: string, type_layout: rawptr, offset, size: int) {
	if spReflectionTypeLayout_getKind(type_layout) == SLANG_TYPE_KIND_STRUCT {
		slang_add_struct_members(members, name, type_layout, offset)
		delete(name)
	} else {
		append(members, Shader_Member{name = name, offset = offset, size = size})
	}
}

@(private)
json_int :: proc(v: json.Value) -> int {
	#partial switch n in v {
	case json.Integer:
		return int(n)
	case json.Float:
		return int(n)
	}
	panic("reflection: expected a number")
}

@(private)
cbuffer_from_type :: proc(name: string, type: json.Object) -> Shader_Cbuffer {
	elem := type["elementType"].(json.Object)
	size := json_int(type["elementVarLayout"].(json.Object)["binding"].(json.Object)["size"])
	members := make([dynamic]Shader_Member)
	_, named := elem["name"]
	add_struct_members(&members, named ? name : "", elem, 0)
	return Shader_Cbuffer{name = name, size = size, members = members[:]}
}

@(private)
add_struct_members :: proc(members: ^[dynamic]Shader_Member, prefix: string, type: json.Object, base: int) {
	for f in type["fields"].(json.Array) {
		fo := f.(json.Object)
		fname := fo["name"].(json.String)
		full := prefix == "" ? strings.clone(fname) : fmt.aprintf("%s.%s", prefix, fname)
		fb := fo["binding"].(json.Object)
		add_members(members, full, fo["type"].(json.Object), base + json_int(fb["offset"]), json_int(fb["size"]))
	}
}

@(private)
add_members :: proc(members: ^[dynamic]Shader_Member, name: string, type: json.Object, offset, size: int) {
	if type["kind"].(json.String) == "struct" {
		add_struct_members(members, name, type, offset)
		delete(name)
	} else {
		append(members, Shader_Member{name = name, offset = offset, size = size})
	}
}

@(private)
Span :: struct {
	offset, len: int,
}

@(private)
Binding_Key :: struct {
	pipeline: vk.Pipeline,
	name:     string,
}

@(private) Cmd_Set_Buffer :: struct { pipeline: vk.Pipeline, name: Span, buffer: Buffer }

@(private) Cmd_Set_Texture :: struct { pipeline: vk.Pipeline, name: Span, texture: Texture }

@(private) Cmd_Set_Tlas :: struct { pipeline: vk.Pipeline, name: Span, tlas: Tlas }

@(private) Cmd_Set_Uniform :: struct { pipeline: vk.Pipeline, name: Span, data: Span }

@(private) Cmd_Set_Cbuffer :: struct { pipeline: vk.Pipeline, name: Span, data: Span }

@(private) Cmd_Set_Texture_Array :: struct { pipeline: vk.Pipeline, name: Span, set: vk.DescriptorSet }

@(private) Cmd_Write_Texture_Array :: struct { set: vk.DescriptorSet, slot: u32, view: vk.ImageView }

@(private) Cmd_Dispatch :: struct { shader: Shader, kernel: int, x, y, z: u32 }

@(private) Cmd_Dispatch_Indirect :: struct { shader: Shader, kernel: int, buffer: Buffer, offset: uint }

@(private) Cmd_Upload_Buffer :: struct { dst: Buffer, data: Span }

@(private) Cmd_Upload_Texture :: struct { dst: Texture, data: Span }

@(private) Cmd_Copy_Buffer :: struct { src, dst: Buffer }

@(private) Cmd_Copy_Texture :: struct { src, dst: Texture }
@(private) Cmd_Build_As :: struct { geometry: vk.AccelerationStructureGeometryKHR, dst, src: vk.AccelerationStructureKHR, mode: vk.BuildAccelerationStructureModeKHR, type: vk.AccelerationStructureTypeKHR, flags: vk.BuildAccelerationStructureFlagsKHR, scratch: vk.DeviceAddress, primitive_count: u32 }
@(private) Cmd_Begin_Profile :: struct { name: Span }
@(private) Cmd_End_Profile :: struct {}

@(private)
Command :: union {
	Cmd_Set_Buffer,
	Cmd_Set_Texture,
	Cmd_Set_Tlas,
	Cmd_Set_Uniform,
	Cmd_Set_Cbuffer,
	Cmd_Set_Texture_Array,
	Cmd_Write_Texture_Array,
	Cmd_Dispatch,
	Cmd_Dispatch_Indirect,
	Cmd_Upload_Buffer,
	Cmd_Upload_Texture,
	Cmd_Copy_Buffer,
	Cmd_Copy_Texture,
	Cmd_Build_As,
	Cmd_Begin_Profile,
	Cmd_End_Profile,
}

@(private)
push_blob :: proc(cmd: ^Cmd, data: []byte) -> Span {
	s := Span{len(cmd.blob), len(data)}
	append(&cmd.blob, ..data)
	return s
}

@(private)
push_name :: proc(cmd: ^Cmd, name: string) -> Span {
	return push_blob(cmd, transmute([]byte)name)
}

@(private)
span_bytes :: proc(cmd: ^Cmd, s: Span) -> []byte {
	return cmd.blob[s.offset:][:s.len]
}

@(private)
span_string :: proc(cmd: ^Cmd, s: Span) -> string {
	return string(span_bytes(cmd, s))
}

@(private)
record_bindings :: proc(cmd: ^Cmd, sub: ^Submission, s: Shader, kernel: int) -> bool {
	cb := sub.cb
	k := s.kernels[kernel]
	n := len(s.params)
	writes := make([]vk.WriteDescriptorSet, n, virtual.arena_allocator(&scratch))
	buffer_infos := make([]vk.DescriptorBufferInfo, n, virtual.arena_allocator(&scratch))
	image_infos := make([]vk.DescriptorImageInfo, n, virtual.arena_allocator(&scratch))
	as_infos := make([]vk.WriteDescriptorSetAccelerationStructureKHR, n, virtual.arena_allocator(&scratch))
	as_handles := make([]vk.AccelerationStructureKHR, n, virtual.arena_allocator(&scratch))

	array_sets := make([]vk.DescriptorSet, n, virtual.arena_allocator(&scratch))
	count := 0
	for p, i in s.params {
		if !kernel_uses(k, i) do continue
		if p.space != 0 {
			arr, arr_ok := cmd.arrays[{k.pipeline, p.name}]
			if !arr_ok {
				log.errorf("dispatch %s.%s: texture array %q is not bound", s.name, k.name, p.name)
				return false
			}
			array_sets[i] = arr
			continue
		}
		w := vk.WriteDescriptorSet{sType = .WRITE_DESCRIPTOR_SET, dstBinding = p.binding, descriptorCount = 1, descriptorType = p.type}
		#partial switch p.type {
		case .STORAGE_BUFFER:
			b, ok := cmd.buffers[{k.pipeline, p.name}]
			if !ok {
				log.errorf("dispatch %s.%s: buffer %q is not bound", s.name, k.name, p.name)
				return false
			}
			buffer_infos[i] = {buffer = b.handle, range = vk.DeviceSize(vk.WHOLE_SIZE)}
			w.pBufferInfo = &buffer_infos[i]
		case .UNIFORM_BUFFER:
			if b, ok := cmd.buffers[{k.pipeline, p.name}]; ok {
				buffer_infos[i] = {buffer = b.handle, range = vk.DeviceSize(vk.WHOLE_SIZE)}
			} else {
				bytes := build_cbuffer(cmd, k.pipeline, s.cbuffers[p.cbuffer])
				offset := staging_push(sub, bytes, uniform_alignment)
				buffer_infos[i] = {buffer = sub.staging.handle, offset = vk.DeviceSize(offset), range = vk.DeviceSize(len(bytes))}
			}
			w.pBufferInfo = &buffer_infos[i]
		case .STORAGE_IMAGE, .SAMPLED_IMAGE:
			t, ok := cmd.textures[{k.pipeline, p.name}]
			if !ok {
				log.errorf("dispatch %s.%s: texture %q is not bound", s.name, k.name, p.name)
				return false
			}
			image_infos[i] = {imageView = t.view, imageLayout = .GENERAL}
			w.pImageInfo = &image_infos[i]
		case .SAMPLER:
			image_infos[i] = {sampler = default_sampler}
			w.pImageInfo = &image_infos[i]
		case .ACCELERATION_STRUCTURE_KHR:
			t, ok := cmd.tlases[{k.pipeline, p.name}]
			if !ok {
				log.errorf("dispatch %s.%s: acceleration structure %q is not bound", s.name, k.name, p.name)
				return false
			}
			as_handles[i] = t.handle
			as_infos[i] = {sType = .WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, accelerationStructureCount = 1, pAccelerationStructures = &as_handles[i]}
			w.pNext = &as_infos[i]
		case:
			fmt.panicf("dispatch %s.%s: unsupported descriptor type %v", s.name, k.name, p.type)
		}
		writes[count] = w
		count += 1
	}

	full_barrier(cb)
	vk.CmdBindPipeline(cb, .COMPUTE, k.pipeline)
	for p, i in s.params do if p.space != 0 && kernel_uses(k, i) do vk.CmdBindDescriptorSets(cb, .COMPUTE, s.layout, p.space, 1, &array_sets[i], 0, nil)
	if count > 0 do vk.CmdPushDescriptorSet(cb, .COMPUTE, s.layout, 0, u32(count), raw_data(writes))
	return true
}

@(private)
build_cbuffer :: proc(cmd: ^Cmd, pipeline: vk.Pipeline, cbuf: Shader_Cbuffer) -> []byte {
	bytes := make([]byte, align_up(cbuf.size, 4), virtual.arena_allocator(&scratch))
	if block, ok := cmd.cbuffers[{pipeline, cbuf.name}]; ok {
		copy(bytes, block)
	}
	for m in cbuf.members {
		if v, ok := cmd.uniforms[{pipeline, m.name}]; ok {
			copy(bytes[m.offset:], v)
		}
	}
	return bytes
}

@(private)
Garbage :: struct {
	as_handle: vk.AccelerationStructureKHR,
	buffer:    Buffer,
}

@(private)
flush_garbage :: proc(garbage: ^[dynamic]Garbage) {
	for g in garbage {
		if g.as_handle != 0 do vk.DestroyAccelerationStructureKHR(device, g.as_handle, nil)
		if g.buffer.handle != 0 do free_buffer(g.buffer)
	}
	clear(garbage)
}

@(private)
record_as_build :: proc(
	cmd: ^Cmd,
	type: vk.AccelerationStructureTypeKHR,
	handle: ^vk.AccelerationStructureKHR,
	buffer: ^Buffer,
	geometry: ^vk.AccelerationStructureGeometryKHR,
	primitive_count: u32,
	refit: bool,
) {
	flags := vk.BuildAccelerationStructureFlagsKHR{.PREFER_FAST_TRACE}
	if type == .TOP_LEVEL do flags += {.ALLOW_UPDATE}
	mode: vk.BuildAccelerationStructureModeKHR = refit ? .UPDATE : .BUILD
	build := vk.AccelerationStructureBuildGeometryInfoKHR {
		sType         = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		type          = type,
		flags         = flags,
		mode          = mode,
		geometryCount = 1,
		pGeometries   = geometry,
	}
	primitive_count := primitive_count
	sizes := vk.AccelerationStructureBuildSizesInfoKHR{sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR}
	vk.GetAccelerationStructureBuildSizesKHR(device, .DEVICE, &build, &primitive_count, &sizes)

	if !refit && (handle^ == 0 || vk.DeviceSize(buffer.size) < sizes.accelerationStructureSize) {
		if handle^ != 0 do append(&cmd.garbage, Garbage{as_handle = handle^, buffer = buffer^})
		buffer^ = create_buffer_ex(int(sizes.accelerationStructureSize), {.ACCELERATION_STRUCTURE_STORAGE_KHR, .SHADER_DEVICE_ADDRESS}, host_visible = false)
		vk_check(vk.CreateAccelerationStructureKHR(device, &vk.AccelerationStructureCreateInfoKHR {
			sType  = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
			buffer = buffer.handle,
			size   = sizes.accelerationStructureSize,
			type   = type,
		}, nil, handle))
	}

	scratch_size := refit ? sizes.updateScratchSize : sizes.buildScratchSize
	scratch_buffer := create_buffer_ex(int(max(scratch_size, 1)), {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS}, host_visible = false)
	append(&cmd.garbage, Garbage{buffer = scratch_buffer})

	append(&cmd.commands, Cmd_Build_As {
		geometry        = geometry^,
		dst             = handle^,
		src             = refit ? handle^ : 0,
		mode            = mode,
		type            = type,
		flags           = flags,
		scratch         = scratch_buffer.address,
		primitive_count = primitive_count,
	})
}
