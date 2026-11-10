"""Blender: preview renders + impostor passes for vegetation GLBs.

  Blender -b --factory-startup -P render_trees.py -- --mode single   --ids pine_a,pine_b [--lod 0]
  Blender -b --factory-startup -P render_trees.py -- --mode impostor --ids pine_a,...
  Blender -b --factory-startup -P render_trees.py -- --mode stand
  Blender -b --factory-startup -P render_trees.py -- --mode debug    --ids pine_a   (vertex colour channels)

Previews -> tools/env/_work/preview, impostor passes -> tools/env/_work/imp
"""
import json
import math
import os
import sys
import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
PREV = os.path.join(HERE, "_work", "preview")
IMP = os.path.join(HERE, "_work", "imp")
os.makedirs(PREV, exist_ok=True)
os.makedirs(IMP, exist_ok=True)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


MODE = arg("--mode", "single")
IDS = (arg("--ids", "pine_a,pine_b,broadleaf_a,broadleaf_b")).split(",")
LOD = int(arg("--lod", "0"))
SAMPLES = int(arg("--samples", "96"))
TAG = arg("--tag", "")

BARK_TEX = {"pine_a": "bark_pine.webp", "pine_b": "bark_pine.webp", "broadleaf_a": "bark_oak.webp", "broadleaf_b": "bark_birch.webp"}
BARK_COL = {"pine_a": (0.105, 0.075, 0.055), "pine_b": (0.115, 0.075, 0.055), "broadleaf_a": (0.11, 0.09, 0.07),
            "broadleaf_b": (0.40, 0.385, 0.35)}

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
try:
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
except Exception as e:  # pragma: no cover
    print("GPU setup failed, CPU fallback:", e)
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 4
scene.cycles.transparent_max_bounces = 24
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.resolution_percentage = 100
scene.view_settings.look = "None"
world = bpy.data.worlds.new("w")
world.use_nodes = True
scene.world = world
bg = world.node_tree.nodes["Background"]


def set_world(color, strength):
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength


def load_image(name, non_color=False):
    path = os.path.join(ENV, name)
    if not os.path.exists(path):
        return None
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def unflipped_normal(nt):
    """Shading normal that ignores Cycles' backface flip (the game lights both faces with the puffed normal)."""
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


def mat_leaves(atlas="foliage_atlas.webp", translucent=0.3, name="leaves_preview"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = load_image(atlas)
    nrm = unflipped_normal(nt)
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    nt.links.new(tex.outputs["Color"], diff.inputs["Color"])
    nt.links.new(nrm, diff.inputs["Normal"])
    shader = diff.outputs["BSDF"]
    if translucent > 0:
        tr = nt.nodes.new("ShaderNodeBsdfTranslucent")
        boost = nt.nodes.new("ShaderNodeMixRGB")
        boost.blend_type = "MULTIPLY"
        boost.inputs["Fac"].default_value = 1.0
        boost.inputs["Color2"].default_value = (1.5, 1.6, 0.9, 1.0)
        nt.links.new(tex.outputs["Color"], boost.inputs["Color1"])
        nt.links.new(boost.outputs["Color"], tr.inputs["Color"])
        nt.links.new(nrm, tr.inputs["Normal"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = translucent
        nt.links.new(diff.outputs["BSDF"], mix.inputs[1])
        nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
        shader = mix.outputs["Shader"]
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    cut = nt.nodes.new("ShaderNodeMath")
    cut.operation = "GREATER_THAN"
    cut.inputs[1].default_value = 0.5
    nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
    amix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(cut.outputs["Value"], amix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], amix.inputs[1])
    nt.links.new(shader, amix.inputs[2])
    nt.links.new(amix.outputs["Shader"], out.inputs["Surface"])
    return m


def mat_bark(tree_id, name=None):
    m = bpy.data.materials.new(name or f"bark_{tree_id}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    img = load_image(BARK_TEX.get(tree_id, ""))
    if img is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        nt.links.new(tex.outputs["Color"], diff.inputs["Color"])
    else:
        diff.inputs["Color"].default_value = (*BARK_COL.get(tree_id, (0.12, 0.09, 0.07)), 1.0)
    nt.links.new(diff.outputs["BSDF"], out.inputs["Surface"])
    return m


def mat_unlit(image_name, cutout, fallback=(0.1, 0.08, 0.06)):
    m = bpy.data.materials.new("unlit_" + image_name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    img = load_image(image_name)
    shader = em.outputs["Emission"]
    if img is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        if cutout:
            transp = nt.nodes.new("ShaderNodeBsdfTransparent")
            cut = nt.nodes.new("ShaderNodeMath")
            cut.operation = "GREATER_THAN"
            cut.inputs[1].default_value = 0.5
            nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
            amix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(cut.outputs["Value"], amix.inputs["Fac"])
            nt.links.new(transp.outputs["BSDF"], amix.inputs[1])
            nt.links.new(shader, amix.inputs[2])
            shader = amix.outputs["Shader"]
    else:
        em.inputs["Color"].default_value = (*fallback, 1.0)
    nt.links.new(shader, out.inputs["Surface"])
    return m


def mat_data(kind, atlas="foliage_atlas.webp", cutout=True):
    """Emission material exporting data: kind = normal | vcol_r | vcol_g | vcol_b."""
    m = bpy.data.materials.new(f"data_{kind}_{int(cutout)}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    if kind == "normal":
        vm = nt.nodes.new("ShaderNodeVectorMath")
        vm.operation = "MULTIPLY_ADD"
        vm.inputs[1].default_value = (0.5, 0.5, 0.5)
        vm.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(unflipped_normal(nt), vm.inputs[0])
        nt.links.new(vm.outputs["Vector"], em.inputs["Color"])
    else:
        vc = nt.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = "Color"
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(vc.outputs["Color"], sep.inputs["Color"])
        ch = {"vcol_r": "Red", "vcol_g": "Green", "vcol_b": "Blue"}[kind]
        nt.links.new(sep.outputs[ch], em.inputs["Color"])
    shader = em.outputs["Emission"]
    if cutout:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = load_image(atlas)
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        cut = nt.nodes.new("ShaderNodeMath")
        cut.operation = "GREATER_THAN"
        cut.inputs[1].default_value = 0.5
        nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
        amix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(cut.outputs["Value"], amix.inputs["Fac"])
        nt.links.new(transp.outputs["BSDF"], amix.inputs[1])
        nt.links.new(shader, amix.inputs[2])
        shader = amix.outputs["Shader"]
    nt.links.new(shader, out.inputs["Surface"])
    return m


ROLES = {}


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    for ob in objs:
        ROLES[ob.name] = [(sl.material.name.split(".")[0] if sl.material else "") for sl in ob.material_slots]
    return objs


def assign(objs, mats):
    """mats: dict role (= original glTF material name) -> material."""
    for ob in objs:
        for slot, role in zip(ob.material_slots, ROLES[ob.name]):
            if role in mats:
                slot.material = mats[role]


def bounds_gltf(objs):
    """Returns (max radial extent from the Y axis, height) in glTF axes."""
    r, h = 0.0, 0.0
    for ob in objs:
        co = np.zeros(len(ob.data.vertices) * 3, dtype=np.float32)
        ob.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        wm = np.array(ob.matrix_world)
        w = co @ wm[:3, :3].T + wm[:3, 3]
        r = max(r, float(np.hypot(w[:, 0], w[:, 1]).max()))
        h = max(h, float(w[:, 2].max()))
    return r, h


def add_camera(ortho=False):
    cd = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cd)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cd.clip_start = 0.1
    cd.clip_end = 2000.0
    if ortho:
        cd.type = "ORTHO"
    return cam


def look_at(cam, eye, target):
    cam.location = eye
    d = Vector(target) - Vector(eye)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_sun(elev_deg=34.0, azim_deg=-62.0, strength=5.0, color=(1.0, 0.93, 0.82)):
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy = strength
    ld.color = color
    ld.angle = math.radians(1.5)
    ob = bpy.data.objects.new("sun", ld)
    scene.collection.objects.link(ob)
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    dirv = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))  # toward the sun
    ob.rotation_euler = dirv.to_track_quat("Z", "Y").to_euler()
    return ob


def add_ground(size=600.0, color=(0.085, 0.11, 0.05)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    g = bpy.context.active_object
    m = bpy.data.materials.new("ground")
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    g.data.materials.append(m)
    return g


def render(path, w, h, view="AgX", transparent=False, exposure=0.9):
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.film_transparent = transparent
    scene.view_settings.view_transform = view
    scene.view_settings.exposure = exposure
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("RENDERED", path)


def remove(objs):
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)


# ----------------------------------------------------------------------------------------- modes

def mode_single():
    set_world((0.42, 0.58, 0.95), 1.0)
    add_sun()
    add_ground()
    cam = add_camera()
    leaves = mat_leaves()
    for tid in IDS:
        objs = import_glb(os.path.join(ENV, f"{tid}_lod{LOD}.glb"))
        assign(objs, {"leaves": leaves, "bark": mat_bark(tid)})
        r, h = bounds_gltf(objs)
        cam.data.lens = 32
        # whole tree from eye level
        look_at(cam, (h * 0.55, -h * 1.45, 1.7), (0, 0, h * 0.5))
        render(os.path.join(PREV, f"{tid}_lod{LOD}_full{TAG}.png"), 720, 960)
        # 5 m from the trunk, looking up into the crown
        cam.data.lens = 22
        look_at(cam, (3.2, -4.6, 1.6), (0.3, 0.5, h * 0.62))
        render(os.path.join(PREV, f"{tid}_lod{LOD}_close{TAG}.png"), 960, 720)
        # 150 m away at game-like FOV (cropped later)
        cam.data.lens = 26
        look_at(cam, (60, -137, 1.7), (0, 0, h * 0.5))
        render(os.path.join(PREV, f"{tid}_lod{LOD}_far{TAG}.png"), 960, 540)
        remove(objs)


def mode_debug():
    set_world((0, 0, 0), 0.0)
    cam = add_camera()
    scene.cycles.use_denoising = False
    scene.cycles.samples = 16
    for tid in IDS:
        objs = import_glb(os.path.join(ENV, f"{tid}_lod{LOD}.glb"))
        print("COLOR ATTRS", tid, [list(o.data.color_attributes.keys()) for o in objs])
        r, h = bounds_gltf(objs)
        cam.data.lens = 32
        look_at(cam, (h * 0.55, -h * 1.45, 1.7), (0, 0, h * 0.5))
        for kind in ("vcol_r", "vcol_g", "vcol_b", "normal"):
            assign(objs, {"leaves": mat_data(kind, cutout=True), "bark": mat_data(kind, cutout=False)})
            render(os.path.join(PREV, f"{tid}_lod{LOD}_dbg_{kind}.png"), 360, 480, view="Raw", exposure=0.0)
        remove(objs)


def mode_impostor():
    cam = add_camera(ortho=True)
    views = 8
    leaves_amb = mat_leaves(translucent=0.0, name="leaves_amb")
    nrm_leaf = mat_data("normal", cutout=True)
    nrm_bark = mat_data("normal", cutout=False)
    unlit_leaf = mat_unlit("foliage_atlas.webp", True)
    info = {}
    for tid in IDS:
        objs = import_glb(os.path.join(ENV, f"{tid}_lod0.glb"))
        bark = mat_bark(tid)
        r, h = bounds_gltf(objs)
        world_w = r * 2.0 + 0.1
        world_h = h + 0.1
        px_per_m = min(256.0 / world_w, 512.0 / world_h)
        vw = int(math.ceil(world_w * px_per_m / 4.0) * 4)
        vh = int(math.ceil(world_h * px_per_m / 4.0) * 4)
        world_w = vw / px_per_m
        world_h = vh / px_per_m
        cam.data.ortho_scale = max(world_w, world_h)
        info[tid] = {"views": views, "viewWidth": vw, "viewHeight": vh, "worldWidth": round(world_w, 3),
                     "worldHeight": round(world_h, 3)}
        for k in range(views):
            phi = k * 2.0 * math.pi / views
            eye = (math.sin(phi) * 80.0, -math.cos(phi) * 80.0, world_h * 0.5)
            look_at(cam, eye, (0.0, 0.0, world_h * 0.5))
            # colour: albedo under uniform ambient light (soft AO, no sun direction)
            set_world((1.0, 1.0, 1.0), 1.0)
            scene.cycles.samples = SAMPLES
            scene.cycles.use_denoising = True
            assign(objs, {"leaves": leaves_amb, "bark": bark})
            render(os.path.join(IMP, f"{tid}_c{k}.png"), vw, vh, view="Standard", transparent=True, exposure=0.0)
            # unlit albedo + normals (world space, converted to view space by compose_impostors.py)
            set_world((0, 0, 0), 0.0)
            scene.cycles.samples = 32
            scene.cycles.use_denoising = False
            assign(objs, {"leaves": unlit_leaf, "bark": mat_unlit(BARK_TEX.get(tid, ""), False)})
            render(os.path.join(IMP, f"{tid}_a{k}.png"), vw, vh, view="Standard", transparent=True, exposure=0.0)
            assign(objs, {"leaves": nrm_leaf, "bark": nrm_bark})
            render(os.path.join(IMP, f"{tid}_n{k}.png"), vw, vh, view="Raw", transparent=True, exposure=0.0)
        remove(objs)
        with open(os.path.join(IMP, f"{tid}_info.json"), "w") as f:
            json.dump(info[tid], f, indent=1)


def mat_impostor(tid):
    m = bpy.data.materials.new("imp_" + tid)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = load_image(f"{tid}_imp.webp")
    ntex = nt.nodes.new("ShaderNodeTexImage")
    ntex.image = load_image(f"{tid}_imp_n.webp", non_color=True)
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    nt.links.new(tex.outputs["Color"], diff.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], diff.inputs["Normal"])
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    cut = nt.nodes.new("ShaderNodeMath")
    cut.operation = "GREATER_THAN"
    cut.inputs[1].default_value = 0.5
    nt.links.new(tex.outputs["Alpha"], cut.inputs[0])
    amix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(cut.outputs["Value"], amix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], amix.inputs[1])
    nt.links.new(diff.outputs["BSDF"], amix.inputs[2])
    nt.links.new(amix.outputs["Shader"], out.inputs["Surface"])
    return m


def billboard(tid, info, mat, pos, cam_pos, view, scale=1.0):
    w, h = info["worldWidth"] * scale, info["worldHeight"] * scale
    me = bpy.data.meshes.new("bb")
    me.from_pydata([(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, 0, h), (-w / 2, 0, h)], [], [(0, 1, 2, 3)])
    uvl = me.uv_layers.new(name="UVMap")
    n = info["views"]
    u0, u1 = view / n, (view + 1) / n
    for i, uv in enumerate([(u0, 0), (u1, 0), (u1, 1), (u0, 1)]):
        uvl.data[i].uv = uv
    me.materials.append(mat)
    ob = bpy.data.objects.new("bb", me)
    scene.collection.objects.link(ob)
    ob.location = pos
    d = Vector(cam_pos) - Vector(pos)
    ob.rotation_euler = (0, 0, math.atan2(d.y, d.x) + math.pi / 2)  # quad normal (-Y local) toward the camera
    return ob


def instance(objs, pos, yaw, scale):
    out = []
    for ob in objs:
        c = ob.copy()
        scene.collection.objects.link(c)
        c.location = pos
        c.rotation_mode = "XYZ"
        c.rotation_euler = (0.0, 0.0, yaw)
        c.scale = (scale, scale, scale)
        out.append(c)
    return out


def mode_stand():
    set_world((0.42, 0.58, 0.95), 1.0)
    add_sun()
    add_ground()
    cam = add_camera()
    rng = np.random.default_rng(12)
    leaves = mat_leaves()
    ids = ["pine_a", "pine_b", "broadleaf_a", "broadleaf_b"]
    weights = [0.45, 0.2, 0.2, 0.15]
    src = {}
    for tid in ids:
        for lod in (0, 1):
            objs = import_glb(os.path.join(ENV, f"{tid}_lod{lod}.glb"))
            assign(objs, {"leaves": leaves, "bark": mat_bark(tid)})
            for o in objs:
                o.location = (0, 0, -500)   # park the templates under the ground
            src[(tid, lod)] = objs
    infos = {tid: json.load(open(os.path.join(IMP, f"{tid}_info.json"))) for tid in ids}
    imats = {tid: mat_impostor(tid) for tid in ids}
    cam_pos = (0.0, -6.0, 1.7)
    # --- forest edge: LOD0 near, LOD1 mid, impostors far
    made = []
    for i in range(44):
        dist = 16.0 + (i / 43.0) ** 1.4 * 200.0 + rng.uniform(-4, 4)
        half = math.radians(27)
        ang = rng.uniform(-half, half)
        pos = (cam_pos[0] + math.sin(ang) * dist, cam_pos[1] + math.cos(ang) * dist, 0.0)
        tid = ids[int(rng.choice(len(ids), p=weights))]
        sc = rng.uniform(0.85, 1.2)
        if dist < 62:
            made += instance(src[(tid, 0)], pos, rng.uniform(0, 6.28), sc)
        elif dist < 115:
            made += instance(src[(tid, 1)], pos, rng.uniform(0, 6.28), sc)
        else:
            made.append(billboard(tid, infos[tid], imats[tid], pos, cam_pos, int(rng.integers(0, 8)), sc))
    cam.data.lens = 28
    look_at(cam, cam_pos, (0.0, 60.0, 9.0))
    render(os.path.join(PREV, f"stand_forest{TAG}.png"), 1280, 720)
    remove(made)
    # --- LOD comparison: per species LOD0 | LOD1 | impostor side by side at 45 m
    made = []
    x = -52.0
    for tid in ids:
        for kind in range(3):
            pos = (x, 45.0, 0.0)
            if kind < 2:
                made += instance(src[(tid, kind)], pos, 0.0, 1.0)
            else:
                d = (cam_pos[0] - pos[0], cam_pos[1] - pos[1])
                phi = math.atan2(d[0], -d[1])   # azimuth of the camera seen from the tree, glTF convention
                view = int(round((phi % (2 * math.pi)) / (2 * math.pi) * 8)) % 8
                made.append(billboard(tid, infos[tid], imats[tid], pos, cam_pos, view, 1.0))
            x += 9.5 if tid in ("pine_a", "broadleaf_b") else 13.0
        x += 3.0
    cam.data.lens = 21
    look_at(cam, (0.0, -52.0, 1.7), (0.0, 45.0, 8.5))
    render(os.path.join(PREV, f"stand_lodcompare{TAG}.png"), 1600, 520)


if MODE == "single":
    mode_single()
elif MODE == "debug":
    mode_debug()
elif MODE == "impostor":
    mode_impostor()
elif MODE == "stand":
    mode_stand()
print("DONE", MODE)
