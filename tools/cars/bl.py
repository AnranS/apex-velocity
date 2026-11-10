"""Blender-side helpers: numpy <-> bpy mesh conversion, decimation, materials, GLB export.

All geometry in this toolchain lives in game/GLB space (+Y up, +Z nose, +X left).
Blender space is Z up with the nose toward -Y:  (x, y, z)_bl = (X, -Z, Y)_glb.
"""
import bpy
import numpy as np

# name -> (base colour, metallic, roughness, emission colour, emission strength, alpha)
MATERIALS = {
    'paint':        ((0.55, 0.03, 0.03), 0.70, 0.35, None, 0.0, 1.0),
    'paint_accent': ((0.02, 0.02, 0.02), 0.50, 0.40, None, 0.0, 1.0),
    'glass':        ((0.02, 0.03, 0.035), 0.0, 0.03, None, 0.0, 0.45),
    'trim_black':   ((0.012, 0.012, 0.013), 0.0, 0.55, None, 0.0, 1.0),
    'carbon':       ((0.03, 0.03, 0.035), 0.3, 0.35, None, 0.0, 1.0),
    'metal':        ((0.75, 0.75, 0.78), 1.0, 0.18, None, 0.0, 1.0),
    'light_head':   ((1.0, 1.0, 1.0), 0.0, 0.2, (1.0, 0.97, 0.9), 6.0, 1.0),
    'light_tail':   ((0.6, 0.02, 0.02), 0.0, 0.3, (1.0, 0.03, 0.02), 4.0, 1.0),
    'light_brake':  ((0.35, 0.01, 0.01), 0.0, 0.3, (1.0, 0.02, 0.01), 1.0, 1.0),
    'interior':     ((0.035, 0.035, 0.04), 0.0, 0.7, None, 0.0, 1.0),
    'underbody':    ((0.01, 0.01, 0.01), 0.0, 0.8, None, 0.0, 1.0),
    'tire':         ((0.015, 0.015, 0.016), 0.0, 0.85, None, 0.0, 1.0),
    'rim':          ((0.42, 0.43, 0.45), 1.0, 0.28, None, 0.0, 1.0),
    'brake_disc':   ((0.35, 0.34, 0.33), 1.0, 0.45, None, 0.0, 1.0),
    'caliper':      ((0.75, 0.05, 0.03), 0.3, 0.4, None, 0.0, 1.0),
}
MAT_ORDER = list(MATERIALS.keys())
MAT_ID = {n: i for i, n in enumerate(MAT_ORDER)}


def to_bl(P):
    P = np.asarray(P, dtype=np.float64)
    return np.stack([P[:, 0], -P[:, 2], P[:, 1]], axis=1)


def from_bl(P):
    P = np.asarray(P, dtype=np.float64)
    return np.stack([P[:, 0], P[:, 2], -P[:, 1]], axis=1)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_material(name):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    col, metal, rough, emis, estr, alpha = MATERIALS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*col, 1.0)
    bsdf.inputs['Metallic'].default_value = metal
    bsdf.inputs['Roughness'].default_value = rough
    if emis is not None:
        bsdf.inputs['Emission Color'].default_value = (*emis, 1.0)
        bsdf.inputs['Emission Strength'].default_value = estr
    if alpha < 1.0:
        bsdf.inputs['Alpha'].default_value = alpha
        try:
            m.surface_render_method = 'BLENDED'
        except Exception:
            pass
    if name == 'paint':
        bsdf.inputs['Coat Weight'].default_value = 1.0
        bsdf.inputs['Coat Roughness'].default_value = 0.03
    return m


def new_mesh(name, V, F, normals=None, ao=None, mat_ids=None):
    """Create a bpy mesh from GLB-space numpy arrays. F is (M,3) or (M,4).
    mat_ids are global material ids (index into MAT_ORDER); only used materials get slots."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    me = bpy.data.meshes.new(name)
    nv, nf, k = len(V), len(F), F.shape[1]
    me.vertices.add(nv)
    me.vertices.foreach_set('co', to_bl(V).ravel())
    me.loops.add(nf * k)
    me.loops.foreach_set('vertex_index', F.ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set('loop_start', np.arange(0, nf * k, k, dtype=np.int32))
    if mat_ids is not None:
        mat_ids = np.asarray(mat_ids)
        used = np.unique(mat_ids)
        remap = np.zeros(len(MAT_ORDER), dtype=np.int32)
        for slot, gid in enumerate(used):
            me.materials.append(get_material(MAT_ORDER[int(gid)]))
            remap[gid] = slot
        me.polygons.foreach_set('material_index', remap[mat_ids].astype(np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set('use_smooth', np.ones(nf, dtype=bool))
    if normals is not None:
        me.normals_split_custom_set_from_vertices(to_bl(normals).tolist())
    if ao is not None:
        attr = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
        c = np.ones((nv, 4), dtype=np.float32)
        c[:, 0] = c[:, 1] = c[:, 2] = np.asarray(ao, dtype=np.float32)
        attr.data.foreach_set('color', c.ravel())
        me.color_attributes.active_color = attr
        try:
            me.color_attributes.render_color_index = 0
        except Exception:
            pass
    return me


def new_object(name, mesh=None, parent=None, location=None, rot_z=0.0):
    """location is given in GLB space."""
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    if location is not None:
        ob.location = to_bl(np.array([location]))[0]
    if rot_z:
        ob.rotation_euler = (0.0, 0.0, rot_z)
    if parent is not None:
        ob.parent = parent
    return ob


def decimate(V, F, ratio=None, target_tris=None, symmetry=False, only_verts=None):
    """Quadric-collapse decimation through Blender's modifier. Returns (V2, T2) triangles.
    only_verts: optional index array - restrict collapsing to these vertices (the modifier leaves
    vertices outside its vertex group alone), the rest of the mesh is kept as is."""
    F = np.asarray(F)
    ntri = len(F) * (2 if F.shape[1] == 4 else 1)
    if target_tris is not None:
        ratio = min(1.0, float(target_tris) / max(ntri, 1))
    me = new_mesh('_dec', V, F)
    ob = bpy.data.objects.new('_dec', me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new('dec', 'DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    if symmetry:
        mod.use_symmetry = True
        mod.symmetry_axis = 'X'
    if only_verts is not None:
        vg = ob.vertex_groups.new(name='sel')
        vg.add([int(i) for i in only_verts], 1.0, 'REPLACE')
        mod.vertex_group = 'sel'
        mod.vertex_group_factor = 1.0
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    m2 = ev.to_mesh()
    nv = len(m2.vertices)
    co = np.empty(nv * 3, dtype=np.float64)
    m2.vertices.foreach_get('co', co)
    m2.calc_loop_triangles()
    nt = len(m2.loop_triangles)
    tri = np.empty(nt * 3, dtype=np.int32)
    m2.loop_triangles.foreach_get('vertices', tri)
    V2 = from_bl(co.reshape(-1, 3))
    T2 = tri.reshape(-1, 3).copy()
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    return V2, T2


def decimate_zones(V, F, zones, n_rest):
    """Decimate a mesh zone by zone with separate triangle budgets (pure quadric error metric
    inside each zone). zones: list of (mask_fn(P)->bool, budget); earlier zones win where they
    overlap; n_rest is the budget for everything else."""
    F = np.asarray(F)
    if F.shape[1] == 4:
        import sdf as _sdf
        F = _sdf.quads_to_tris(V, F)
    T = F
    for i in range(len(zones) + 1):
        taken = np.zeros(len(V), dtype=bool)
        zmask = None
        for j, (fn, budget) in enumerate(zones):
            m = fn(V) & ~taken
            taken |= m
            if j == i:
                zmask, n = m, budget
        if i == len(zones):
            zmask, n = ~taken, n_rest
        in_zone = int(zmask[T].all(axis=1).sum())
        if in_zone <= n:
            continue
        V, T = decimate(V, T, target_tris=len(T) - in_zone + n, only_verts=np.flatnonzero(zmask))
    return V, T


def export_glb(path, objects):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.gltf(
        filepath=path, export_format='GLB', use_selection=True, export_apply=False,
        export_yup=True, export_normals=True, export_tangents=False, export_texcoords=True,
        export_materials='EXPORT', export_vertex_color='ACTIVE', export_all_vertex_colors=False,
        export_active_vertex_color_when_no_material=True,
        export_cameras=False, export_lights=False, export_animations=False, export_extras=False,
        export_skins=False, export_morph=False)
