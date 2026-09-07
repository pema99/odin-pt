package main

import "core:slice"
import "core:encoding/json"
import "core:strings"
import vk "vendor:vulkan"
import "core:path/filepath"
import "core:math/linalg"
import "core:math"
import cgltf "vendor:cgltf"
import ai "lib:assimp"
import stbi "vendor:stb/image"
import "gpu"
import "core:thread"
import "core:sys/info"

// TODO: Removing support for meshes, materials, instances

Geometry_Info :: struct {
    index_offset: u32,
    index_count: u32,
    vertex_offset: u32,
    material_index: u32,
}

Geometry_Pool :: struct {
    mesh_to_pool: [dynamic]Geometry_Info, // mesh index -> pool index
    instance_to_pool: GPU_List(Geometry_Info), // instanceID -> geometry
    transforms: GPU_List(matrix[3, 4]f32),
    indices: GPU_List(u32),
    vertices: GPU_List([3]f32),
    normals: GPU_List([3]f32),
    tangents: GPU_List([3]f32),
    uvs: GPU_List([2]f32),
    emissive_instance_indices: GPU_List(u32), // indices into instance_to_pool
}

gp_new :: proc() -> Geometry_Pool {
    return Geometry_Pool {
        mesh_to_pool = make([dynamic]Geometry_Info),
        instance_to_pool = gpu_list_new(Geometry_Info),
        transforms = gpu_list_new(matrix[3, 4]f32),
        indices = gpu_list_new(u32),
        vertices = gpu_list_new([3]f32),
        normals = gpu_list_new([3]f32),
        tangents = gpu_list_new([3]f32),
        uvs = gpu_list_new([2]f32),
        emissive_instance_indices = gpu_list_new(u32)
    }
}

gp_delete :: proc(pool: ^Geometry_Pool) {
    delete(pool.mesh_to_pool)
    gpu_list_delete(&pool.instance_to_pool)
    gpu_list_delete(&pool.transforms)
    gpu_list_delete(&pool.indices)
    gpu_list_delete(&pool.vertices)
    gpu_list_delete(&pool.normals)
    gpu_list_delete(&pool.tangents)
    gpu_list_delete(&pool.uvs)
    gpu_list_delete(&pool.emissive_instance_indices)
}

gp_add_mesh :: proc(
    pool: ^Geometry_Pool, 
    vertices: [][3]f32, 
    normals: [][3]f32, 
    tangents: [][3]f32, 
    uvs: [][2]f32,
    indices: []u32, 
    material_index: u32) -> u32 {

    info := Geometry_Info {
        index_offset = pool.indices.length,
        index_count = u32(len(indices)),
        vertex_offset = pool.vertices.length,
        material_index = material_index
    }
    append(&pool.mesh_to_pool, info)
    gpu_list_add_range(&pool.indices, indices)
    gpu_list_add_range(&pool.vertices, vertices)
    gpu_list_add_range(&pool.normals, normals)
    gpu_list_add_range(&pool.tangents, tangents)
    gpu_list_add_range(&pool.uvs, uvs)
    return u32(len(pool.mesh_to_pool)) - 1
}

gp_remove_mesh :: proc() {
    // TODO
}

gp_add_instance :: proc(pool: ^Geometry_Pool, mesh_index: u32, transform: matrix[3, 4]f32, emissive: bool) -> u32 {
    instance_info := pool.mesh_to_pool[mesh_index]
    gpu_list_add(&pool.instance_to_pool, instance_info)
    gpu_list_add(&pool.transforms, transform)
    if emissive {
        gpu_list_add(&pool.emissive_instance_indices, pool.instance_to_pool.length - 1)
    }
    return pool.instance_to_pool.length - 1
}

gp_remove_instance :: proc() {
    // TODO
}

gp_commit :: proc(pool: ^Geometry_Pool, cmd: ^gpu.Cmd) {
    gpu_list_commit(&pool.instance_to_pool, cmd)
    gpu_list_commit(&pool.transforms, cmd)
    gpu_list_commit(&pool.indices, cmd)
    gpu_list_commit(&pool.vertices, cmd)
    gpu_list_commit(&pool.normals, cmd)
    gpu_list_commit(&pool.tangents, cmd)
    gpu_list_commit(&pool.uvs, cmd)
    gpu_list_commit(&pool.emissive_instance_indices, cmd)
}

Material_BSDF :: enum u32 {
    Disney = 0,
    Glass = 1,
    Lambert = 2,
}

Material_Info :: struct {
    albedo: [3]f32,
    emission: [3]f32,
    metallic: f32,
    roughness: f32,
    index_of_refraction: f32,
    extinction: f32,
    dispersion: f32,
    iridescence_factor: f32,
    iridescence_ior: f32,
    iridescence_thickness: f32,
    attenuation_color: [3]f32,
    attenuation_distance: f32,
    albedo_texture_index: u32,
    emission_texture_index: u32,
    metallic_texture_index: u32,
    roughness_texture_index: u32,
    normal_texture_index: u32,
    double_sided: b32,
    bsdf_type: Material_BSDF, 
}

Material_Pool :: struct {
    materials: GPU_List(Material_Info),
    textures: gpu.Texture_Array,
    texture_list: [dynamic]gpu.Texture,
}

mp_new :: proc() -> Material_Pool {
    return Material_Pool {
        materials = gpu_list_new(Material_Info),
        textures = gpu.create_texture_array(),
        texture_list = make([dynamic]gpu.Texture),
    }
}

mp_delete :: proc(pool: ^Material_Pool) {
    gpu_list_delete(&pool.materials)
    for texture in pool.texture_list {
        gpu.destroy_texture(texture)
    }
    delete(pool.texture_list)
    gpu.destroy_texture_array(pool.textures)
}

mp_add_texture :: proc(pool: ^Material_Pool, cmd: ^gpu.Cmd, texture: gpu.Texture) -> u32 {
    index := u32(len(pool.texture_list))
    gpu.texture_array_write(cmd, pool.textures, index, texture)
    append(&pool.texture_list, texture)
    return index
}

mp_remove_texture :: proc(pool: ^Material_Pool, index: u32) {
    // TODO
}

mp_add_material :: proc(pool: ^Material_Pool, material: Material_Info) -> u32 {
    gpu_list_add(&pool.materials, material)
    return pool.materials.length - 1
}

mp_remove_material :: proc(pool: ^Material_Pool, index: u32) {
    // TODO
}

mp_commit :: proc(pool: ^Material_Pool, cmd: ^gpu.Cmd) {
    gpu_list_commit(&pool.materials, cmd)
}

Scene :: struct {
    blases: [dynamic]gpu.Blas,
    instances: [dynamic]gpu.Instance,
    tlas: gpu.Tlas,
    geometry_pool: Geometry_Pool,
    material_pool: Material_Pool,
    camera: Scene_Camera,
    has_camera: bool,
}

scene_load_node :: proc(scene: ^Scene, node: ^ai.Node, transform: ai.Matrix4x4) {
    new_transform := transform
    ai.MultiplyMatrix4(&new_transform, &node.mTransformation)

    for mesh_index: u32 = 0; mesh_index < node.mNumMeshes; mesh_index += 1 {
        blas_index := node.mMeshes[mesh_index]
        transform: matrix[3, 4]f32 = {
            new_transform.a1, new_transform.a2, new_transform.a3, new_transform.a4,
            new_transform.b1, new_transform.b2, new_transform.b3, new_transform.b4,
            new_transform.c1, new_transform.c2, new_transform.c3, new_transform.c4,
        }

        material_index := scene.geometry_pool.mesh_to_pool[blas_index].material_index
        material := scene.material_pool.materials.array[material_index]
        emissive := material.emission != {0, 0, 0}
        
        instance := gpu.Instance {
            blas = scene.blases[blas_index],
            transform = transform,
            id = gp_add_instance(&scene.geometry_pool, blas_index, transform, emissive),
            double_sided = bool(material.double_sided) || material.bsdf_type == .Glass,
        }
        append(&scene.instances, instance)
    }

    for i in 0..<node.mNumChildren {
        scene_load_node(scene, node.mChildren[i], new_transform)
    }
}

Decoded_Texture :: struct {
    source: ^ai.Texture,
    pixels: [^]u8,
    width: i32,
    height: i32,
}

decode_embedded_textures_parallel :: proc(ai_scene: ^ai.Scene) -> map[^ai.Texture]Decoded_Texture {
    results := make([]Decoded_Texture, ai_scene.mNumTextures)
    defer delete(results)

    _, logical_cores, _ := info.cpu_core_count()
    pool: thread.Pool
    thread.pool_init(&pool, context.allocator, max(logical_cores, 1))
    for texture_index: u32 = 0; texture_index < ai_scene.mNumTextures; texture_index += 1 {
        results[texture_index].source = ai_scene.mTextures[texture_index]
        if ai_scene.mTextures[texture_index].mHeight == 0 {
            thread.pool_add_task(&pool, context.allocator, proc(task: thread.Task) {
                decoded := (^Decoded_Texture)(task.data)
                channels: i32
                decoded.pixels = stbi.load_from_memory(cast([^]u8)decoded.source.pcData, i32(decoded.source.mWidth), &decoded.width, &decoded.height, &channels, 4)
            }, &results[texture_index])
        }
    }
    thread.pool_start(&pool)
    thread.pool_finish(&pool)
    thread.pool_destroy(&pool)

    decoded := make(map[^ai.Texture]Decoded_Texture, len(results))
    for result in results {
        decoded[result.source] = result
    }
    return decoded
}

ai_texture_load :: proc(cmd: ^gpu.Cmd, scene: ^Scene, ai_scene: ^ai.Scene, decoded_textures: map[^ai.Texture]Decoded_Texture, path: cstring, material: ^ai.Material, type: ai.TextureType) -> u32 {
    albedo_texture_index := max(u32)
    tex_path: ai.String
    if ai.GetMaterialTexture(material, type, 0, &tex_path, nil, nil, nil, nil, nil, nil) == .SUCCESS {
        tex_name := cstring(cast([^]u8)&tex_path.data[0])
        width, height, channels: i32
        pixels: [^]u8
        embedded := ai.GetEmbeddedTexture(ai_scene, tex_name)
        if embedded != nil {
            if decoded, found := decoded_textures[embedded]; found {
                pixels = decoded.pixels
                width = decoded.width
                height = decoded.height
            }
        } else {
            dir := filepath.dir(string(path))
            full, _ := filepath.join({dir, string(tex_name)})
            full_c := strings.clone_to_cstring(full)
            pixels = stbi.load(full_c, &width, &height, &channels, 4)
            delete(dir)
            delete(full)
            delete(full_c)
        }
        if pixels != nil {
            format := vk.Format.R8G8B8A8_UNORM
            if type == .DIFFUSE || type == .EMISSIVE {
                format = .R8G8B8A8_SRGB
            }
            texture := gpu.create_texture(u32(width), u32(height), format)
            gpu.upload_texture(cmd, texture, pixels[:width * height * 4])
            if embedded == nil {
                stbi.image_free(pixels)
            }
            albedo_texture_index = mp_add_texture(&scene.material_pool, cmd, texture)
        }
    }
    return albedo_texture_index
}

Gltf_Extra_Data :: struct {
    extinction: f32,
    dispersion: f32,
    iridescence_factor: f32,
    iridescence_ior: f32,
    iridescence_thickness: f32,
    attenuation_color: [3]f32,
    attenuation_distance: f32,
}

// assimp does not support all extensions, so read them with cgltf instead...
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_dispersion
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_iridescence
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_volume
gltf_read_extra_data :: proc(path: cstring, allocator := context.allocator) -> (result: map[string]Gltf_Extra_Data, unnamed: map[u32]Gltf_Extra_Data) {
    options: cgltf.options
    data, parse_result := cgltf.parse_file(options, path)
    if parse_result != .success {
        return
    }
    defer cgltf.free(data)

    result = make(map[string]Gltf_Extra_Data, allocator)
    unnamed = make(map[u32]Gltf_Extra_Data, allocator)
    for material, index in data.materials {
        extra: Gltf_Extra_Data
        if material.extras.data != nil {
            value, err := json.parse_string(string(cstring(material.extras.data)))
            if err == nil {
                if object, is_object := value.(json.Object); is_object {
                    if entry, has := object["extinction"]; has {
                        if number, is_number := entry.(json.Float); is_number do extra.extinction = f32(number)
                    }
                }
            }
            json.destroy_value(value)
        }
        if material.has_dispersion {
            extra.dispersion = material.dispersion.dispersion
        }
        if material.has_iridescence {
            extra.iridescence_factor = material.iridescence.iridescence_factor
            extra.iridescence_ior = material.iridescence.iridescence_ior
            extra.iridescence_thickness = material.iridescence.iridescence_thickness_max
        }
        if material.has_volume {
            extra.attenuation_color = material.volume.attenuation_color
            extra.attenuation_distance = material.volume.attenuation_distance
        }
        if material.name == nil {
            unnamed[u32(index)] = extra
        } else {
            result[strings.clone(string(material.name), allocator)] = extra
        }
    }
    return
}

Scene_Camera :: struct {
    position: [3]f32,
    forward: [3]f32,
    yfov: f32,
}

ai_read_camera :: proc(ai_scene: ^ai.Scene) -> (result: Scene_Camera, ok: bool) {
    if ai_scene.mNumCameras == 0 {
        return
    }
    camera := ai_scene.mCameras[0]
    root := ai_scene.mRootNode
    for child_index: u32 = 0; child_index < root.mNumChildren; child_index += 1 {
        node := root.mChildren[child_index]
        if cstring(rawptr(&node.mName.data[0])) != cstring(rawptr(&camera.mName.data[0])) {
            continue
        }
        m := root.mTransformation
        ai.MultiplyMatrix4(&m, &node.mTransformation)
        p := camera.mPosition
        l := camera.mLookAt
        result.position = {
            m.a1 * p.x + m.a2 * p.y + m.a3 * p.z + m.a4,
            m.b1 * p.x + m.b2 * p.y + m.b3 * p.z + m.b4,
            m.c1 * p.x + m.c2 * p.y + m.c3 * p.z + m.c4,
        }
        result.forward = linalg.normalize([3]f32{
            m.a1 * l.x + m.a2 * l.y + m.a3 * l.z,
            m.b1 * l.x + m.b2 * l.y + m.b3 * l.z,
            m.c1 * l.x + m.c2 * l.y + m.c3 * l.z,
        })
        aspect := camera.mAspect == 0 ? 1 : camera.mAspect
        result.yfov = 2 * math.atan(math.tan(camera.mHorizontalFOV / 2) / aspect)
        return result, true
    }
    return
}

scene_load :: proc(path: cstring, cmd: ^gpu.Cmd) -> (s: Scene, ok: bool) #optional_ok {
    ai_scene := ai.ImportFile(path, {
        .Triangulate,
        .FlipUVs,
        .CalcTangentSpace,
    });
    if ai_scene == nil {
        return {}, false
    }
    defer ai.ReleaseImport(ai_scene)

    scene: Scene
    scene.blases = make([dynamic]gpu.Blas, ai_scene.mNumMeshes)
    scene.instances = make([dynamic]gpu.Instance)
    scene.geometry_pool = gp_new()
    scene.material_pool = mp_new()

    decoded_textures := decode_embedded_textures_parallel(ai_scene)
    defer {
        for _, decoded in decoded_textures do stbi.image_free(decoded.pixels)
        delete(decoded_textures)
    }

    for mesh_index: u32 = 0; mesh_index < ai_scene.mNumMeshes; mesh_index += 1 {
        mesh := ai_scene.mMeshes[mesh_index]
        verts := slice.reinterpret([][3]f32, mesh.mVertices[:mesh.mNumVertices])
        normals := slice.reinterpret([][3]f32, mesh.mNormals[:mesh.mNumVertices])

        tangents := make([][3]f32, mesh.mNumVertices)
        defer delete(tangents)
        if mesh.mTangents != nil {
            for vert_index: u32 = 0; vert_index < mesh.mNumVertices; vert_index += 1 {
                tangents[vert_index] = {mesh.mTangents[vert_index].x, mesh.mTangents[vert_index].y, mesh.mTangents[vert_index].z}
            }
        }

        uvs := make([][2]f32, mesh.mNumVertices)
        defer delete(uvs)
        if mesh.mTextureCoords[0] != nil {
            for vert_index: u32 = 0; vert_index < mesh.mNumVertices; vert_index += 1 {
                uvs[vert_index][0] = mesh.mTextureCoords[0][vert_index].x
                uvs[vert_index][1] = mesh.mTextureCoords[0][vert_index].y
            }
        }

        faces := make([]u32, mesh.mNumFaces * 3)
        defer delete(faces)
        for face_index: u32 = 0; face_index < mesh.mNumFaces; face_index += 1 {
            face := mesh.mFaces[face_index]
            faces[face_index * 3 + 0] = face.mIndices[0]
            faces[face_index * 3 + 1] = face.mIndices[1]
            faces[face_index * 3 + 2] = face.mIndices[2]
        }
        
        gpu.build_blas(cmd, &scene.blases[mesh_index], verts, faces)
        gp_add_mesh(&scene.geometry_pool, verts, normals, tangents, uvs, faces, mesh.mMaterialIndex)
    }

    extra_data, unnamed_extra_data := gltf_read_extra_data(path)
    defer {
        for name in extra_data do delete(name)
        delete(extra_data)
        delete(unnamed_extra_data)
    }

    for material_index: u32 = 0; material_index < ai_scene.mNumMaterials; material_index += 1 {
        material := ai_scene.mMaterials[material_index]
        albedo := [3]f32{1, 1, 1}
        emission := [3]f32{0, 0, 0}
        metallic := f32(0)
        roughness := f32(1)
        index_of_refraction := f32(1.5)
        extinction := f32(0.0)
        dispersion := f32(0.0)
        iridescence_factor := f32(0.0)
        iridescence_ior := f32(1.3)
        iridescence_thickness := f32(400.0)
        attenuation_color := [3]f32{1, 1, 1}
        attenuation_distance := f32(0)
        double_sided := b32(false)
        bsdf_type := Material_BSDF.Disney
        
        ai_albedo: ai.Color4D
        if ai.GetMaterialColor(material, ai.MATKEY_COLOR_DIFFUSE, 0, 0, &ai_albedo) == .SUCCESS {
            albedo = [3]f32{ai_albedo.r, ai_albedo.g, ai_albedo.b}
        }
        ai_emission: ai.Color4D
        if ai.GetMaterialColor(material, ai.MATKEY_COLOR_EMISSIVE, 0, 0, &ai_emission) == .SUCCESS {
            emission = [3]f32{ai_emission.r, ai_emission.g, ai_emission.b}
        }
        emissive_intensity := f32(1)
        if ai.GetMaterialFloat(material, ai.MATKEY_EMISSIVE_INTENSITY, 0, 0, &emissive_intensity) == .SUCCESS {
            emission *= emissive_intensity
        }
        ai.GetMaterialFloat(material, ai.MATKEY_ROUGHNESS_FACTOR, 0, 0, &roughness)
        ai.GetMaterialFloat(material, ai.MATKEY_METALLIC_FACTOR, 0, 0, &metallic)
        ai.GetMaterialFloat(material, ai.MATKEY_REFRACTI, 0, 0, &index_of_refraction)
        transmission: f32
        if ai.GetMaterialFloat(material, ai.MATKEY_TRANSMISSION_FACTOR, 0, 0, &transmission) == .SUCCESS && transmission > 0 {
            bsdf_type = Material_BSDF.Glass
        }
        two_sided: i32
        if ai.GetMaterialInteger(material, ai.MATKEY_TWOSIDED, 0, 0, &two_sided) == .SUCCESS {
            double_sided = b32(two_sided != 0)
        }

        material_name: ai.String
        ai.GetMaterialString(material, ai.MATKEY_NAME, 0, 0, &material_name)
        extra, has_extra := extra_data[string(cstring(rawptr(&material_name.data[0])))]
        if !has_extra {
            extra, has_extra = unnamed_extra_data[material_index]
        }
        if has_extra {
            extinction = extra.extinction
            dispersion = extra.dispersion
            iridescence_factor = extra.iridescence_factor
            iridescence_ior = extra.iridescence_ior
            iridescence_thickness = extra.iridescence_thickness
            attenuation_color = extra.attenuation_color
            attenuation_distance = extra.attenuation_distance
        }

        mp_add_material(&scene.material_pool, Material_Info {
            albedo = albedo,
            emission = emission,
            metallic = metallic,
            roughness = roughness,
            index_of_refraction = index_of_refraction,
            extinction = extinction,
            dispersion = dispersion,
            iridescence_factor = iridescence_factor,
            iridescence_ior = iridescence_ior,
            iridescence_thickness = iridescence_thickness,
            attenuation_color = attenuation_color,
            attenuation_distance = attenuation_distance,
            double_sided = double_sided,
            albedo_texture_index = ai_texture_load(cmd, &scene, ai_scene, decoded_textures, path, material, .DIFFUSE),
            emission_texture_index = ai_texture_load(cmd, &scene, ai_scene, decoded_textures, path, material, .EMISSIVE),
            metallic_texture_index = ai_texture_load(cmd, &scene, ai_scene, decoded_textures, path, material, .METALNESS),
            roughness_texture_index = ai_texture_load(cmd, &scene, ai_scene, decoded_textures, path, material, .DIFFUSE_ROUGHNESS),
            normal_texture_index = ai_texture_load(cmd, &scene, ai_scene, decoded_textures, path, material, .NORMALS),
            bsdf_type = bsdf_type,
        })
    }

    transform: ai.Matrix4x4
    ai.IdentityMatrix4(&transform)
    scene_load_node(&scene, ai_scene.mRootNode, transform)
    scene.camera, scene.has_camera = ai_read_camera(ai_scene)

    gpu.build_tlas(cmd, &scene.tlas, scene.instances[:])

    gp_commit(&scene.geometry_pool, cmd)
    mp_commit(&scene.material_pool, cmd)

    return scene, true
}

scene_delete :: proc(scene: ^Scene)  {
    for blas in scene.blases {
        gpu.destroy_blas(blas)
    }
    gpu.destroy_tlas(scene.tlas)
    delete(scene.blases)
    delete(scene.instances)
    gp_delete(&scene.geometry_pool)
    mp_delete(&scene.material_pool)
    scene^ = {}
}
