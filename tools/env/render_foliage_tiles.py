"""Blender: render every foliage / grass cluster top-down into per-tile pass images.

  Blender -b --factory-startup -P render_foliage_tiles.py -- [--set foliage|grass] [--only name,name] [--scale 1.0]

Outputs tools/env/_work/atlas/<tile>_{albedo,normal,ao,height}.png
"""
import os
import sys
import numpy as np
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foliage_src as fs  # noqa: E402

OUT = os.path.join(HERE, "_work", "atlas")
os.makedirs(OUT, exist_ok=True)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    if name in argv:
        return argv[argv.index(name) + 1]
    return default


SET = arg("--set", "foliage")
ONLY = arg("--only")
SCALE = float(arg("--scale", "1.0"))

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 0
scene.render.film_transparent = True
scene.render.dither_intensity = 0.0
scene.render.use_file_extension = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "8"
scene.render.image_settings.compression = 30
scene.render.resolution_percentage = 100
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
scene.world = world

cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.clip_start = 0.1
cam_data.clip_end = 50.0
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m, nt, em


def mat_albedo():
    m, nt, em = new_mat("p_albedo")
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    nt.links.new(vc.outputs["Color"], em.inputs["Color"])
    return m


def mat_normal():
    m, nt, em = new_mat("p_normal")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    vm = nt.nodes.new("ShaderNodeVectorMath")
    vm.operation = "MULTIPLY_ADD"
    vm.inputs[1].default_value = (0.5, 0.5, 0.5)
    vm.inputs[2].default_value = (0.5, 0.5, 0.5)
    nt.links.new(geo.outputs["Normal"], vm.inputs[0])
    nt.links.new(vm.outputs["Vector"], em.inputs["Color"])
    return m


def mat_ao(dist):
    m, nt, em = new_mat("p_ao")
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 16
    ao.inputs["Distance"].default_value = dist
    nt.links.new(ao.outputs["AO"], em.inputs["Color"])
    return m


def mat_height(zmin, zmax):
    m, nt, em = new_mat("p_height")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = zmin
    mr.inputs["From Max"].default_value = zmax
    nt.links.new(geo.outputs["Position"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], em.inputs["Color"])
    return m


def make_object(name, v, c, t, smooth):
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.astype(np.float32).tolist(), [], t.astype(np.int32).tolist())
    me.polygons.foreach_set("use_smooth", smooth.astype(bool))
    me.update()
    attr = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    rgba = np.concatenate([np.clip(c, 0, 1), np.ones((len(c), 1))], 1).astype(np.float32)
    attr.data.foreach_set("color", rgba.ravel())
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    return ob


def render(path, view):
    scene.view_settings.view_transform = view
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


table = fs.tiles() if SET == "foliage" else fs.grass_tiles()
names = ONLY.split(",") if ONLY else list(table.keys())
for name in names:
    builder, rect, pix = table[name]
    v, c, t, s = builder()
    x0, x1, y0, y1 = rect
    w = int(round(pix[2] * SCALE))
    h = int(round(pix[3] * SCALE))
    ob = make_object(name, v, c, t, s)
    cam.location = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, 10.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam_data.ortho_scale = max(x1 - x0, y1 - y0)
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    zmin, zmax = float(v[:, 2].min()), float(v[:, 2].max())
    passes = [
        ("albedo", mat_albedo(), "Standard"),
        ("normal", mat_normal(), "Raw"),
        ("ao", mat_ao(0.22 if SET == "foliage" else 0.12), "Raw"),
        ("height", mat_height(zmin, zmax), "Raw"),
    ]
    for pname, mat, view in passes:
        ob.data.materials.clear()
        ob.data.materials.append(mat)
        render(os.path.join(OUT, f"{name}_{pname}.png"), view)
        bpy.data.materials.remove(mat)
    print(f"TILE {name}: verts={len(v)} tris={len(t)} z=[{zmin:.3f},{zmax:.3f}] -> {w}x{h}")
    bpy.data.objects.remove(ob, do_unlink=True)
print("DONE tiles")
