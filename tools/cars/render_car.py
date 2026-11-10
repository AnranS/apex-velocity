"""Import an exported car GLB and render studio previews / the UI thumbnail with Cycles.

  blender -b -P tools/cars/render_car.py -- --glb public/assets/cars/aurion.glb \
      --out tools/cars/_work --prefix aurion --views q_front,q_rear,side --res 960x540

Rendering the *exported* file (not the build scene) doubles as a visual check of the GLB:
normals, hierarchy, wheel placement and vertex colours are exactly what the game will load.
"""
import bpy, sys, os, math, argparse
from mathutils import Vector


def parse():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--glb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', default='car')
    ap.add_argument('--views', default='q_front,q_rear,side,front,rear,top')
    ap.add_argument('--res', default='960x540')
    ap.add_argument('--samples', type=int, default=48)
    ap.add_argument('--paint', default='0.50,0.035,0.03')
    ap.add_argument('--transparent', action='store_true')
    ap.add_argument('--format', default='PNG')
    ap.add_argument('--ao', action='store_true', help='debug: show baked vertex AO as emission')
    ap.add_argument('--clay', action='store_true', help='debug: neutral glossy clay on everything')
    ap.add_argument('--brake', action='store_true')
    ap.add_argument('--wire', action='store_true', help='debug: show triangle edges')
    return ap.parse_args(argv)


def bl(p):  # GLB -> Blender coords
    return Vector((p[0], -p[2], p[1]))


VIEWS = {
    # name: (camera pos GLB, target GLB, focal mm)
    'q_front':  ((3.3, 1.25, 5.0), (0.0, 0.52, 0.15), 42),
    'q_front2': ((-3.9, 0.85, 4.4), (0.0, 0.5, 0.1), 42),
    'q_rear':   ((-3.5, 1.45, -5.0), (0.0, 0.55, -0.15), 42),
    'q_rear2':  ((4.0, 0.9, -4.5), (0.0, 0.5, -0.1), 42),
    'side':     ((10.0, 0.8, 0.0), (0.0, 0.58, 0.0), 62),
    'front':    ((0.0, 0.75, 10.0), (0.0, 0.58, 0.0), 110),
    'rear':     ((0.0, 0.85, -10.0), (0.0, 0.58, 0.0), 110),
    'top':      ((0.0, 12.0, 0.0), (0.0, 0.0, 0.0), 76),
    'high':     ((4.2, 3.6, 4.6), (0.0, 0.45, 0.0), 45),
    'low':      ((2.4, 0.42, 5.4), (0.0, 0.55, 0.2), 40),
    'wheel':    ((2.6, 0.55, 2.9), (0.9, 0.36, 1.25), 75),
    'wheel_r':  ((2.7, 0.6, -2.9), (0.9, 0.36, -1.3), 75),
    'lamp_f':   ((1.7, 1.0, 3.9), (0.45, 0.55, 2.0), 60),
    'lamp_r':   ((-1.6, 1.1, -3.9), (-0.3, 0.7, -2.1), 60),
    'cabin':    ((2.6, 1.7, 1.4), (0.0, 0.95, -0.1), 60),
    'side_int': ((3.2, 1.0, -1.9), (0.9, 0.6, -0.75), 60),
    'rear_low': ((-1.8, 0.5, -5.2), (0.0, 0.5, -0.3), 45),
    'wing':     ((-1.5, 1.5, -3.6), (-0.3, 0.92, -2.0), 70),
    'dash':     ((3.0, 1.1, -0.9), (0.92, 0.64, -0.3), 170),
    'bpillar':  ((2.4, 2.2, 0.4), (-0.5, 1.0, -0.5), 90),
    'thumb':    ((3.3, 1.25, 5.0), (0.0, 0.48, 0.0), 40),
}


def set_material(m, args, paint):
    name = m.name.split('.')[0]
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')

    def principled(col, metal, rough, coat=0.0, coat_rough=0.03, emis=None, estr=0.0):
        b = nt.nodes.new('ShaderNodeBsdfPrincipled')
        b.inputs['Base Color'].default_value = (*col, 1.0)
        b.inputs['Metallic'].default_value = metal
        b.inputs['Roughness'].default_value = rough
        b.inputs['Coat Weight'].default_value = coat
        b.inputs['Coat Roughness'].default_value = coat_rough
        if emis is not None:
            b.inputs['Emission Color'].default_value = (*emis, 1.0)
            b.inputs['Emission Strength'].default_value = estr
        nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
        return b

    if args.wire:
        base = {'paint': (0.75, 0.2, 0.15), 'glass': (0.3, 0.5, 0.8), 'trim_black': (0.25, 0.25, 0.25),
                'carbon': (0.35, 0.35, 0.5), 'underbody': (0.2, 0.3, 0.2), 'interior': (0.4, 0.3, 0.2),
                'light_head': (1, 1, 0.8), 'light_tail': (1, 0.3, 0.3), 'light_brake': (0.8, 0.1, 0.4)}.get(name, (0.6, 0.6, 0.6))
        b = nt.nodes.new('ShaderNodeBsdfDiffuse')
        w = nt.nodes.new('ShaderNodeWireframe')
        w.use_pixel_size = True
        w.inputs['Size'].default_value = 0.7
        mixc = nt.nodes.new('ShaderNodeMixRGB')
        mixc.inputs[1].default_value = (*base, 1)
        mixc.inputs[2].default_value = (0, 0, 0, 1)
        nt.links.new(w.outputs['Fac'], mixc.inputs['Fac'])
        nt.links.new(mixc.outputs['Color'], b.inputs['Color'])
        nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
        return
    if args.ao:
        e = nt.nodes.new('ShaderNodeEmission')
        a = nt.nodes.new('ShaderNodeVertexColor')
        nt.links.new(a.outputs['Color'], e.inputs['Color'])
        nt.links.new(e.outputs['Emission'], out.inputs['Surface'])
        return
    if args.clay and name not in ('glass',):
        principled((0.30, 0.31, 0.33), 0.0, 0.25, coat=1.0, coat_rough=0.05)
        return
    if name == 'paint':
        principled(paint, 0.72, 0.36, coat=1.0, coat_rough=0.02)
    elif name == 'paint_accent':
        principled((0.015, 0.015, 0.017), 0.4, 0.3, coat=1.0, coat_rough=0.04)
    elif name == 'glass':
        tr = nt.nodes.new('ShaderNodeBsdfTransparent')
        tr.inputs['Color'].default_value = (0.42, 0.47, 0.50, 1.0)
        gl = nt.nodes.new('ShaderNodeBsdfGlossy')
        gl.inputs['Roughness'].default_value = 0.0
        gl.inputs['Color'].default_value = (1, 1, 1, 1)
        fr = nt.nodes.new('ShaderNodeFresnel')
        fr.inputs['IOR'].default_value = 1.5
        mix = nt.nodes.new('ShaderNodeMixShader')
        nt.links.new(fr.outputs['Fac'], mix.inputs['Fac'])
        nt.links.new(tr.outputs['BSDF'], mix.inputs[1])
        nt.links.new(gl.outputs['BSDF'], mix.inputs[2])
        nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    elif name == 'trim_black':
        principled((0.012, 0.012, 0.013), 0.0, 0.5)
    elif name == 'carbon':
        principled((0.022, 0.022, 0.026), 0.2, 0.32, coat=0.8, coat_rough=0.08)
    elif name == 'metal':
        principled((0.8, 0.8, 0.82), 1.0, 0.15)
    elif name == 'light_head':
        principled((1, 1, 1), 0.0, 0.2, emis=(1.0, 0.97, 0.92), estr=9.0)
    elif name == 'light_tail':
        principled((0.5, 0.01, 0.01), 0.0, 0.25, emis=(1.0, 0.02, 0.015), estr=6.0)
    elif name == 'light_brake':
        principled((0.30, 0.008, 0.008), 0.0, 0.25, emis=(1.0, 0.02, 0.01), estr=(14.0 if args.brake else 0.35))
    elif name == 'interior':
        principled((0.03, 0.03, 0.034), 0.0, 0.7)
    elif name == 'underbody':
        principled((0.008, 0.008, 0.008), 0.0, 0.85)
    elif name == 'tire':
        principled((0.012, 0.012, 0.013), 0.0, 0.78)
    elif name == 'rim':
        principled((0.30, 0.31, 0.33), 1.0, 0.24)
    elif name == 'brake_disc':
        principled((0.42, 0.41, 0.40), 1.0, 0.42)
    elif name == 'caliper':
        principled((0.85, 0.42, 0.02), 0.2, 0.35, coat=1.0)
    else:
        principled((0.5, 0.0, 0.5), 0.0, 0.5)


def area(name, loc, target, size, energy, color=(1, 1, 1)):
    l = bpy.data.lights.new(name, 'AREA')
    l.shape = 'RECTANGLE'
    l.size, l.size_y = size
    l.energy = energy
    l.color = color
    o = bpy.data.objects.new(name, l)
    bpy.context.scene.collection.objects.link(o)
    o.location = bl(loc)
    d = bl(target) - o.location
    o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return o


def main():
    args = parse()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(args.glb))
    paint = tuple(float(v) for v in args.paint.split(','))
    for m in bpy.data.materials:
        set_material(m, args, paint)

    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    try:
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == 'METAL')
        sc.cycles.device = 'GPU'
    except Exception as e:
        print('GPU setup failed, CPU render:', e)
    sc.cycles.samples = args.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.glossy_bounces = 4
    sc.cycles.transmission_bounces = 6
    sc.cycles.transparent_max_bounces = 8
    w, h = (int(v) for v in args.res.split('x'))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = args.transparent
    sc.render.image_settings.file_format = args.format
    sc.render.image_settings.color_mode = 'RGBA' if args.transparent else 'RGB'
    if args.format == 'WEBP':
        sc.render.image_settings.quality = 92
    try:
        sc.view_settings.view_transform = 'AgX'
        sc.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass
    sc.view_settings.exposure = 0.0

    # world
    world = bpy.data.worlds.new('studio')
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    if args.ao:
        bg.inputs['Color'].default_value = (0.02, 0.02, 0.02, 1)
        bg.inputs['Strength'].default_value = 1.0
    else:
        bg.inputs['Color'].default_value = (0.11, 0.125, 0.15, 1)
        bg.inputs['Strength'].default_value = 0.55
    sc.world = world

    if not args.ao:
        area('key_top', (0.0, 5.5, 0.5), (0, 0.5, 0), (7.0, 10.0), 560)
        area('strip_l', (5.2, 2.4, 0.0), (0, 0.6, 0), (9.0, 0.7), 190)
        area('strip_r', (-5.2, 2.4, 0.0), (0, 0.6, 0), (9.0, 0.7), 190)
        area('strip_f', (0.0, 1.9, 7.0), (0, 0.6, 0), (7.0, 0.7), 100)
        area('strip_b', (0.0, 2.4, -7.0), (0, 0.6, 0), (7.0, 0.7), 100)
        area('rim_warm', (-4.5, 3.2, -5.5), (0, 0.6, 0), (3.0, 3.0), 150, (1.0, 0.86, 0.72))

    # ground
    bpy.ops.mesh.primitive_plane_add(size=140, location=(0, 0, 0))
    ground = bpy.context.active_object
    gm = bpy.data.materials.new('_ground')
    gm.use_nodes = True
    gb = gm.node_tree.nodes['Principled BSDF']
    gb.inputs['Base Color'].default_value = (0.035, 0.037, 0.04, 1)
    gb.inputs['Roughness'].default_value = 0.22
    ground.data.materials.append(gm)
    if args.transparent:
        ground.is_shadow_catcher = True
    if args.ao:
        ground.hide_render = True

    cam_data = bpy.data.cameras.new('cam')
    cam = bpy.data.objects.new('cam', cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam_data.sensor_width = 36
    cam_data.clip_start = 0.05
    cam_data.clip_end = 500

    os.makedirs(args.out, exist_ok=True)
    ext = {'PNG': 'png', 'WEBP': 'webp', 'JPEG': 'jpg'}[args.format]
    for v in args.views.split(','):
        pos, tgt, lens = VIEWS[v]
        cam.location = bl(pos)
        d = bl(tgt) - cam.location
        if v == 'top':
            cam.rotation_euler = (0.0, 0.0, -math.pi / 2)   # nose to the right of the image
        else:
            cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        cam_data.lens = lens
        path = os.path.join(os.path.abspath(args.out), f'{args.prefix}_{v}.{ext}')
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print('RENDERED', path)


main()
