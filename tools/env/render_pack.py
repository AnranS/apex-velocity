"""Blender: consumer-style previews of props / rocks / ground cover (GLB + manifest textures, AO from COLOR_0).

  Blender -b --factory-startup -P render_pack.py -- [--ids a,b,c] [--samples 96]
"""
import math
import os
import sys
import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_props import MATS, PROPS  # noqa: E402

ENV = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
PREV = os.path.join(HERE, "_work", "preview")
TMP = os.path.join(HERE, "_work", "tmp")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


EXTRA = ["rock_a", "rock_b", "rock_c", "bush_a", "grass_tuft"]
IDS = arg("--ids", ",".join(list(PROPS.keys()) + EXTRA)).split(",")
SAMPLES = int(arg("--samples", "96"))
TEX = {
    "ad": (os.path.join(TMP, "test_ad.png"), None, False),
    "flag": (os.path.join(TMP, "test_flag.png"), None, False),
    "crowd": (os.path.join(ENV, "crowd_atlas.webp"), None, True),
    "leaves": (os.path.join(ENV, "foliage_atlas.webp"), None, True),
    "grass": (os.path.join(ENV, "grass_atlas.webp"), None, True),
}

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "METAL"
prefs.get_devices()
for d in prefs.devices:
    d.use = True
scene.cycles.device = "GPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "AgX"
scene.view_settings.exposure = 0.6
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.42, 0.58, 0.95, 1)
scene.world = world
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 4.5
sun.data.angle = math.radians(1.5)
sun.rotation_euler = Vector((-0.45, -0.6, 0.66)).to_track_quat("Z", "Y").to_euler()
scene.collection.objects.link(sun)
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, 0))
ground = bpy.context.active_object
gm = bpy.data.materials.new("ground")
gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.11, 0.115, 0.11, 1)
gm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
ground.data.materials.append(gm)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.clip_end = 2000
scene.collection.objects.link(cam)
scene.camera = cam


def preview_material(name, rid):
    m = bpy.data.materials.new("pv_" + name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    out = nt.nodes["Material Output"]
    base = name.split(".")[0]
    gd = MATS.get(base)
    col = (0.8, 0.8, 0.8, 1.0)
    if gd:
        col = tuple(gd["pbrMetallicRoughness"]["baseColorFactor"])
        bsdf.inputs["Roughness"].default_value = gd["pbrMetallicRoughness"]["roughnessFactor"]
        bsdf.inputs["Metallic"].default_value = gd["pbrMetallicRoughness"]["metallicFactor"]
        if "emissiveFactor" in gd and max(gd["emissiveFactor"]) > 0:
            bsdf.inputs["Emission Color"].default_value = (*gd["emissiveFactor"], 1.0)
            bsdf.inputs["Emission Strength"].default_value = 6.0
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Color"
    mul = nt.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    tex_info = TEX.get(base)
    if base == "rock":
        tex_info = (os.path.join(ENV, f"{rid}_d.webp"), os.path.join(ENV, f"{rid}_n.webp"), False)
    if tex_info:
        path, npath, cutout = tex_info
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(path, check_existing=True)
        nt.links.new(tex.outputs["Color"], mul.inputs["Color1"])
        if npath:
            ntex = nt.nodes.new("ShaderNodeTexImage")
            ntex.image = bpy.data.images.load(npath, check_existing=True)
            ntex.image.colorspace_settings.name = "Non-Color"
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nt.links.new(ntex.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
        if cutout:
            cut = nt.nodes.new("ShaderNodeMath")
            cut.operation = "GREATER_THAN"
            cut.inputs[1].default_value = 0.5
            nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
            nt.links.new(cut.outputs["Value"], bsdf.inputs["Alpha"])
    else:
        mul.inputs["Color1"].default_value = col
    if base in ("leaves", "grass", "rock"):
        nt.links.new(mul.outputs["Color"] if False else (nt.nodes[tex.name].outputs["Color"]), bsdf.inputs["Base Color"])
    else:
        nt.links.new(vc.outputs["Color"], mul.inputs["Color2"])
        nt.links.new(mul.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def bounds(objs):
    mn, mx = np.full(3, 1e9), np.full(3, -1e9)
    for ob in objs:
        co = np.zeros(len(ob.data.vertices) * 3, dtype=np.float32)
        ob.data.vertices.foreach_get("co", co)
        wm = np.array(ob.matrix_world)
        w = co.reshape(-1, 3) @ wm[:3, :3].T + wm[:3, 3]
        mn, mx = np.minimum(mn, w.min(0)), np.maximum(mx, w.max(0))
    return mn, mx


for rid in IDS:
    path = os.path.join(ENV, rid + ".glb")
    if not os.path.exists(path):
        print("missing", rid)
        continue
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    cache = {}
    for ob in objs:
        for slot in ob.material_slots:
            nm = slot.material.name
            if nm not in cache:
                cache[nm] = preview_material(nm, rid)
            slot.material = cache[nm]
    mn, mx = bounds(objs)
    c = (mn + mx) / 2
    size = float(np.linalg.norm(mx - mn))
    eye = Vector((c[0] + size * 0.62, mn[1] - size * 0.95, max(1.6, c[2] + size * 0.18)))
    tgt = Vector((c[0], c[1], c[2]))
    if "--top" in argv:      # close look at the top few metres (lamp heads, platforms)
        tgt = Vector((c[0], c[1], mx[2] - 1.6))
        eye = tgt + Vector((3.2, -7.5, -1.2))
    cam.location = eye
    cam.data.lens = 35
    cam.rotation_euler = (tgt - eye).to_track_quat("-Z", "Y").to_euler()
    scene.render.resolution_x = 960
    scene.render.resolution_y = 640
    scene.render.filepath = os.path.join(PREV, f"pack_{rid}{'_top' if '--top' in argv else ''}.png")
    bpy.ops.render.render(write_still=True)
    print("PACK", rid, "size", np.round(mx - mn, 2))
    for o in [o for o in bpy.data.objects if o not in before]:
        bpy.data.objects.remove(o, do_unlink=True)
print("PACK DONE")
