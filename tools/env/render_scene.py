"""Blender: one trackside composition using the whole pack (scale / orientation / look sanity check).

  Blender -b --factory-startup -P render_scene.py -- [--samples 128] [--cam start|forest]
Output: tools/env/_work/preview/scene_<cam>.png
"""
import json
import math
import os
import sys
import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_props import MATS  # noqa: E402

ENV = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
PREV = os.path.join(HERE, "_work", "preview")
TMP = os.path.join(HERE, "_work", "tmp")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


SAMPLES = int(arg("--samples", "128"))
CAM = arg("--cam", "start")
manifest = json.load(open(os.path.join(ENV, "manifest.json")))
ASSETS = {a["id"]: a for a in manifest["assets"]}

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
scene.cycles.transparent_max_bounces = 24
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "AgX"
scene.view_settings.exposure = 0.9
world = bpy.data.worlds.new("w")
world.use_nodes = True
wn = world.node_tree
bgn = wn.nodes["Background"]
sky = wn.nodes.new("ShaderNodeTexSky")
try:
    kinds = [e.identifier for e in sky.bl_rna.properties["sky_type"].enum_items]
    for k in ("MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
        if k in kinds:
            sky.sky_type = k
            break
    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(24)
        sky.sun_rotation = math.radians(250)
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = False
    wn.links.new(sky.outputs["Color"], bgn.inputs["Color"])
    bgn.inputs["Strength"].default_value = 0.22
    print("SKY", sky.sky_type, kinds)
except Exception as e:  # pragma: no cover
    print("sky fallback", e)
    bgn.inputs["Color"].default_value = (0.42, 0.58, 0.95, 1)
scene.world = world
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 6.5
sun.data.color = (1.0, 0.8, 0.58)
sun.data.angle = math.radians(1.2)
# golden hour: low sun from behind-left of the camera
sun.rotation_euler = Vector((-0.75, -0.5, 0.42)).to_track_quat("Z", "Y").to_euler()
scene.collection.objects.link(sun)


def flat_mat(name, col, rough=0.9):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1)
    b.inputs["Roughness"].default_value = rough
    return m


def plane(name, x0, x1, z0, z1, y, mat):
    """Rectangle given in glTF coords."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([(x0, -z0, y), (x1, -z0, y), (x1, -z1, y), (x0, -z1, y)], [], [(0, 3, 2, 1)])
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    return ob


plane("ground", -400, 400, -400, 400, 0.0, flat_mat("grassland", (0.07, 0.095, 0.035)))
plane("road", -400, 400, -8, 8, 0.02, flat_mat("asphalt", (0.035, 0.036, 0.04), 0.8))
plane("pitlane", -60, 60, -19, -8, 0.015, flat_mat("pitlane", (0.06, 0.06, 0.065), 0.85))
plane("apron", -40, 40, 8, 21, 0.012, flat_mat("apron", (0.16, 0.16, 0.15), 0.9))
white = flat_mat("line", (0.8, 0.8, 0.8), 0.6)
for z in (-7.7, 7.7):
    plane("edge", -400, 400, z - 0.1, z + 0.1, 0.03, white)
plane("startline", -0.3, 0.3, -7.6, 7.6, 0.03, white)


def img(path, non_color=False):
    im = bpy.data.images.load(path, check_existing=True)
    if non_color:
        im.colorspace_settings.name = "Non-Color"
    return im


def unflipped_normal(nt):
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = "MULTIPLY_ADD"
    m.inputs[1].default_value = -2.0
    m.inputs[2].default_value = 1.0
    nt.links.new(geo.outputs["Backfacing"], m.inputs[0])
    vm = nt.nodes.new("ShaderNodeVectorMath")
    vm.operation = "SCALE"
    nt.links.new(geo.outputs["Normal"], vm.inputs[0])
    nt.links.new(m.outputs["Value"], vm.inputs["Scale"])
    return vm.outputs["Vector"]


_cache = {}


def material_for(asset, mat_name):
    key = (asset["id"] if mat_name in ("bark", "rock") else "", mat_name)
    if key in _cache:
        return _cache[key]
    m = bpy.data.materials.new(f"{key[0]}_{mat_name}")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex_def = asset["textures"].get(mat_name)
    gd = MATS.get(mat_name)
    if gd:
        bsdf.inputs["Roughness"].default_value = gd["pbrMetallicRoughness"]["roughnessFactor"]
        bsdf.inputs["Metallic"].default_value = gd["pbrMetallicRoughness"]["metallicFactor"]
        if max(gd.get("emissiveFactor", [0])) > 0:
            bsdf.inputs["Emission Color"].default_value = (*gd["emissiveFactor"], 1.0)
            bsdf.inputs["Emission Strength"].default_value = 4.0
    preview_tex = {"ad": os.path.join(TMP, "test_ad.png"), "flag": os.path.join(TMP, "test_flag.png")}
    foliage = mat_name in ("leaves", "grass")
    if tex_def or mat_name in preview_tex:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img(os.path.join(ENV, tex_def["albedo"]) if tex_def else preview_tex[mat_name])
        colour = tex.outputs["Color"]
        if foliage or mat_name == "crowd":
            cut = nt.nodes.new("ShaderNodeMath")
            cut.operation = "GREATER_THAN"
            cut.inputs[1].default_value = 0.5
            nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
            nt.links.new(cut.outputs["Value"], bsdf.inputs["Alpha"])
        if foliage:
            nt.links.new(unflipped_normal(nt), bsdf.inputs["Normal"])
            bsdf.inputs["Roughness"].default_value = 0.7
            if "Subsurface Weight" in bsdf.inputs:
                pass
        elif tex_def and tex_def.get("normal"):
            ntex = nt.nodes.new("ShaderNodeTexImage")
            ntex.image = img(os.path.join(ENV, tex_def["normal"]), True)
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nt.links.new(ntex.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = tuple(gd["pbrMetallicRoughness"]["baseColorFactor"]) if gd else (0.8, 0.8, 0.8, 1)
        colour = rgb.outputs[0]
    if asset["kind"] == "prop":      # AO from COLOR_0
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = "Color"
        mul = nt.nodes.new("ShaderNodeMixRGB")
        mul.blend_type = "MULTIPLY"
        mul.inputs["Fac"].default_value = 1.0
        nt.links.new(colour, mul.inputs["Color1"])
        nt.links.new(vc.outputs["Color"], mul.inputs["Color2"])
        colour = mul.outputs["Color"]
    nt.links.new(colour, bsdf.inputs["Base Color"])
    _cache[key] = m
    return m


_templates = {}


def template(aid, lod=0):
    if (aid, lod) in _templates:
        return _templates[(aid, lod)]
    asset = ASSETS[aid]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(ENV, asset["lods"][lod]["file"]))
    objs = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in objs if o.type == "MESH"]
    for ob in meshes:
        for slot in ob.material_slots:
            slot.material = material_for(asset, slot.material.name.split(".")[0])
        ob.location = (0, 0, -900)
    for o in objs:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    _templates[(aid, lod)] = meshes
    return meshes


def place(aid, x, z, yaw=0.0, scale=1.0, lod=0, y=0.0):
    for ob in template(aid, lod):
        c = ob.copy()
        scene.collection.objects.link(c)
        c.location = (x, -z, y)
        c.rotation_mode = "XYZ"
        c.rotation_euler = (0.0, 0.0, math.radians(yaw))
        c.scale = (scale, scale, scale)


rng = np.random.default_rng(7)
place("gantry", 0, 0, -90)
for x in (-24, 0, 24):
    place("grandstand", x, 27, 180)
for x in (-30, -18, -6, 6, 18, 30):
    place("pit_building", x, -24, 0)
for x in (52, 62, 72):
    place("billboard", x, 12.5, 180)
place("billboard", 56, -12.5, 0)
for x, z in ((-46, 15.5), (46, 15.5), (-46, -15.5), (46, -15.5)):
    place("light_tower", x, z, 180 if z > 0 else 0)
place("marshal_post", 40, 11.5, 180)
place("camera_tower", -31, 12.0, 180)
for x in (-9, -6, -3, 3, 6, 9):
    place("flag_pole", x, -9.6, 0)
for i in range(34):
    place("tire_stack", 84 + i * 0.7, 9.4 + 0.0 * i, rng.uniform(0, 360))
for i in range(7):
    place("cone", -26 + i * 2.5, -8.6, rng.uniform(0, 360))
for (x, z, s, yaw, rid) in ((66, 16, 1.0, 20, "rock_c"), (70, 18.5, 0.8, 200, "rock_a"), (92, 14, 1.3, 80, "rock_b"), (-58, 13, 1.1, 140, "rock_a"),
                           (-62, 15.5, 0.7, 10, "rock_c")):
    place(rid, x, z, yaw, s)
for i in range(26):
    place("bush_a", rng.uniform(48, 110), rng.uniform(14, 30), rng.uniform(0, 360), rng.uniform(0.7, 1.3))
for i in range(260):
    place("grass_tuft", rng.uniform(-70, -36), rng.uniform(8.6, 13.5), rng.uniform(0, 360), rng.uniform(0.7, 1.4))
ids = ["pine_a", "pine_b", "broadleaf_a", "broadleaf_b"]
w = [0.5, 0.18, 0.2, 0.12]
for i in range(150):
    side = 1 if rng.uniform() < 0.6 else -1
    x = rng.uniform(-190, 220)
    z = side * rng.uniform(44, 130)
    if abs(x) > 48 and rng.uniform() < 0.6:
        z = side * rng.uniform(17, 60)
    tid = ids[int(rng.choice(4, p=w))]
    dist = math.hypot(x + 46, z)
    place(tid, x, z, rng.uniform(0, 360), rng.uniform(0.85, 1.2), lod=0 if dist < 110 else 1)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.clip_end = 3000
scene.collection.objects.link(cam)
scene.camera = cam


def look(eye_g, tgt_g, lens):
    eye = Vector((eye_g[0], -eye_g[2], eye_g[1]))
    tgt = Vector((tgt_g[0], -tgt_g[2], tgt_g[1]))
    cam.location = eye
    cam.rotation_euler = (tgt - eye).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens


if CAM == "start":
    look((-52, 1.5, 2.5), (8, 5.0, 0.5), 24)
elif CAM == "forest":
    look((38, 1.4, -3), (120, 5.0, 8), 26)
elif CAM == "aerial":
    look((-70, 38, -70), (5, 3, 0), 30)
scene.render.resolution_x = 1600
scene.render.resolution_y = 900
scene.render.filepath = os.path.join(PREV, f"scene_{CAM}.png")
bpy.ops.render.render(write_still=True)
print("SCENE DONE", CAM)
