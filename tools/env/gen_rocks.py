"""Blender: CC0 photoscanned boulders -> game-ready low-poly rocks with baked albedo + tangent-space normal maps.

  Blender -b --factory-startup -P gen_rocks.py -- [rock_a,rock_b,rock_c]

Source: Poly Haven glTF (2k) in tools/env/_work/dl/<asset>/ (see fetch_cc0.py).
Outputs: public/assets/env/rock_x.glb (material `rock`, TANGENT attribute included) and baked PNGs in
tools/env/_work/rocks/ (converted to WebP by finish_rocks.py).
"""
import math
import os
import sys
import numpy as np
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "_work", "dl")
WORK = os.path.join(HERE, "_work", "rocks")
PREV = os.path.join(HERE, "_work", "preview")
ENV = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
os.makedirs(WORK, exist_ok=True)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
JOBS = {
    # id: (source asset, target triangles, sink fraction of height, uniform scale)
    "rock_a": ("boulder_01", 1150, 0.12, 1.25),
    "rock_b": ("rock_07", 1150, 0.10, 10.0),
    "rock_c": ("namaqualand_boulder_04", 1150, 0.06, 1.0),
}
IDS = argv[0].split(",") if argv else list(JOBS.keys())

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8


def wipe():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for b in list(coll):
            if b.users == 0:
                coll.remove(b)


def only(ob, others=()):
    bpy.ops.object.select_all(action="DESELECT")
    for o in others:
        o.select_set(True)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def tri_count(me):
    return sum(len(p.vertices) - 2 for p in me.polygons)


def world_coords(ob):
    co = np.zeros(len(ob.data.vertices) * 3, dtype=np.float32)
    ob.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def process(rid):
    asset, target, sink, scale = JOBS[rid]
    wipe()
    src = os.path.join(DL, asset, f"{asset}_2k.gltf")
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    only(meshes[0], meshes)
    if len(meshes) > 1:
        bpy.ops.object.join()
    hi = bpy.context.view_layer.objects.active
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    hi.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(scene.objects):
        if o != hi:
            bpy.data.objects.remove(o, do_unlink=True)
    co = world_coords(hi)
    mn, mx = co.min(0), co.max(0)
    height = mx[2] - mn[2]
    hi.location = (-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -(mn[2] + sink * height))
    only(hi)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    hi.name = rid + "_hi"
    print(f"{rid}: source tris={tri_count(hi.data)} size={np.round(mx - mn, 2)}")

    # ---- low poly
    lo = hi.copy()
    lo.data = hi.data.copy()
    lo.name = rid
    lo.data.name = rid
    scene.collection.objects.link(lo)
    only(lo)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0005)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"{rid}: welded verts={len(lo.data.vertices)} tris={tri_count(lo.data)}")
    dec = lo.modifiers.new("dec", "DECIMATE")
    dec.ratio = min(1.0, (target * 0.97) / max(1, tri_count(lo.data)))
    bpy.ops.object.modifier_apply(modifier="dec")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.bisect(plane_co=(0, 0, 0), plane_no=(0, 0, 1), clear_inner=True, use_fill=False)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")
    while lo.data.uv_layers:
        lo.data.uv_layers.remove(lo.data.uv_layers[0])
    lo.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.012)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.shade_smooth()
    for attr in list(lo.data.color_attributes):
        lo.data.color_attributes.remove(attr)

    # ---- bake targets
    img_d = bpy.data.images.new(rid + "_d", 1024, 1024, alpha=False)
    img_n = bpy.data.images.new(rid + "_n", 1024, 1024, alpha=False)
    img_n.colorspace_settings.name = "Non-Color"
    mat = bpy.data.materials.new("rock")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Metallic"].default_value = 0.0
    tex_d = nt.nodes.new("ShaderNodeTexImage")
    tex_d.image = img_d
    tex_n = nt.nodes.new("ShaderNodeTexImage")
    tex_n.image = img_n
    lo.data.materials.clear()
    lo.data.materials.append(mat)

    size = float(max(mx - mn))
    bk = scene.render.bake
    bk.use_selected_to_active = True
    bk.cage_extrusion = 0.035 * size
    bk.max_ray_distance = 0.09 * size
    bk.margin = 12
    bk.use_clear = True
    only(lo, [hi])
    nt.nodes.active = tex_d
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_selected_to_active=True, cage_extrusion=bk.cage_extrusion,
                        max_ray_distance=bk.max_ray_distance, margin=12)
    nt.nodes.active = tex_n
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_selected_to_active=True, cage_extrusion=bk.cage_extrusion,
                        max_ray_distance=bk.max_ray_distance, margin=12)
    for img, suffix in ((img_d, "d"), (img_n, "n")):
        img.filepath_raw = os.path.join(WORK, f"{rid}_{suffix}.png")
        img.file_format = "PNG"
        img.save()

    # ---- export
    co = world_coords(lo)
    print(f"{rid}: low tris={tri_count(lo.data)} verts={len(lo.data.vertices)} bounds={np.round(co.min(0), 2)}..{np.round(co.max(0), 2)}")
    only(lo)
    bpy.ops.export_scene.gltf(filepath=os.path.join(ENV, rid + ".glb"), export_format="GLB", use_selection=True, export_apply=True,
                              export_yup=True, export_image_format="NONE", export_tangents=True, export_normals=True,
                              export_texcoords=True, export_cameras=False, export_lights=False, export_animations=False,
                              export_vertex_color="NONE")

    # ---- preview (low poly with its baked maps under sun + sky)
    nt.links.new(tex_d.outputs["Color"], bsdf.inputs["Base Color"])
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(tex_n.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    hi.hide_render = True
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.42, 0.58, 0.95, 1)
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 4.5
    sun.rotation_euler = (math.radians(55), 0, math.radians(40))
    scene.collection.objects.link(sun)
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    g = bpy.context.active_object
    gm = bpy.data.materials.new("g")
    gm.use_nodes = True
    gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.1, 0.12, 0.06, 1)
    g.data.materials.append(gm)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    from mathutils import Vector
    eye = Vector((size * 1.5, -size * 1.7, size * 0.55))
    cam.location = eye
    cam.rotation_euler = (Vector((0, 0, height * 0.35)) - eye).to_track_quat("-Z", "Y").to_euler()
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0.6
    scene.render.filepath = os.path.join(PREV, f"{rid}_preview.png")
    bpy.ops.render.render(write_still=True)
    print("ROCK DONE", rid)


for rid in IDS:
    process(rid)
