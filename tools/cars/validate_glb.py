"""Validate the exported car assets against the game contract (pure python + numpy).

  python3 tools/cars/validate_glb.py            # checks public/assets/cars/{aurion,valken,zephyr}*
"""
import sys, os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'public', 'assets', 'cars'))
MATERIALS = {'paint', 'paint_accent', 'glass', 'trim_black', 'carbon', 'metal', 'light_head', 'light_tail',
             'light_brake', 'interior', 'underbody', 'tire', 'rim', 'brake_disc', 'caliper'}
WHEELS = ['fl', 'fr', 'rl', 'rr']
COMP = {5120: 'i1', 5121: 'u1', 5122: 'i2', 5123: 'u2', 5125: 'u4', 5126: 'f4'}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


class Glb:
    def __init__(self, path):
        self.data = open(path, 'rb').read()
        assert self.data[:4] == b'glTF', 'not a GLB'
        jlen = struct.unpack('<I', self.data[12:16])[0]
        self.j = json.loads(self.data[20:20 + jlen])
        self.bin0 = 20 + jlen + 8

    def acc(self, i):
        a = self.j['accessors'][i]
        bv = self.j['bufferViews'][a['bufferView']]
        off = self.bin0 + bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        k = NCOMP[a['type']]
        stride = bv.get('byteStride')
        dt = np.dtype('<' + COMP[a['componentType']])
        if stride and stride != dt.itemsize * k:
            raw = np.frombuffer(self.data, dtype=np.uint8, count=stride * a['count'], offset=off).reshape(a['count'], stride)
            arr = raw[:, :dt.itemsize * k].copy().view(dt).reshape(a['count'], k)
        else:
            arr = np.frombuffer(self.data, dtype=dt, count=a['count'] * k, offset=off).reshape(a['count'], k)
        if a.get('normalized'):
            arr = arr.astype(np.float64) / np.iinfo(dt).max
        return arr

    def node(self, name):
        for i, n in enumerate(self.j['nodes']):
            if n.get('name') == name:
                return i, n
        return None, None

    def mesh_stats(self, node):
        """triangles, material names, positions (N,3), has COLOR_0/NORMAL for a mesh node."""
        mesh = self.j['meshes'][node['mesh']]
        tris = 0
        mats = []
        P = []
        ok_col = ok_nrm = True
        col_min = 1.0
        for prim in mesh['primitives']:
            assert prim.get('mode', 4) == 4, 'non-triangle primitive'
            tris += self.j['accessors'][prim['indices']]['count'] // 3
            mats.append(self.j['materials'][prim['material']]['name'])
            P.append(self.acc(prim['attributes']['POSITION']))
            ok_col &= 'COLOR_0' in prim['attributes']
            ok_nrm &= 'NORMAL' in prim['attributes']
            if 'COLOR_0' in prim['attributes']:
                col_min = min(col_min, float(self.acc(prim['attributes']['COLOR_0'])[:, :3].min()))
            n = self.acc(prim['attributes']['NORMAL'])
            ln = np.linalg.norm(n, axis=1)
            assert np.all(np.abs(ln - 1.0) < 0.02), 'non-unit normals'
        return tris, mats, np.vstack(P), ok_col, ok_nrm, col_min


def quat_rot(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def check(car_id):
    errs = []
    info = {}

    def expect(cond, msg):
        if not cond:
            errs.append(msg)

    meta_path = os.path.join(OUT, f'{car_id}.json')
    expect(os.path.exists(meta_path), 'missing metadata json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for key in ('id', 'displayName', 'length', 'width', 'height', 'wheelbase', 'axleFrontZ', 'axleRearZ', 'trackFront',
                'trackRear', 'wheelRadiusFront', 'wheelRadiusRear', 'wheelWidthFront', 'wheelWidthRear', 'bodyBox',
                'points', 'triangles'):
        expect(key in meta, f'json missing {key}')
    for key in ('headlightL', 'headlightR', 'taillightL', 'taillightR', 'exhaustL', 'exhaustR', 'camHood',
                'camCockpit', 'camBumper', 'wingCenter'):
        expect(key in meta.get('points', {}), f'json points missing {key}')

    for lod, fname, max_total, max_body, max_wheel, max_mb in ((0, f'{car_id}.glb', 160000, 120000, 10000, 5.2),
                                                               (1, f'{car_id}_lod1.glb', 18000, 18000, 10000, 0.8)):
        path = os.path.join(OUT, fname)
        if not os.path.exists(path):
            errs.append(f'missing {fname}')
            continue
        g = Glb(path)
        size_mb = os.path.getsize(path) / 1e6
        expect(size_mb <= max_mb, f'{fname}: {size_mb:.2f} MB > {max_mb}')
        for k in ('cameras', 'animations', 'skins'):
            expect(not g.j.get(k), f'{fname}: contains {k}')
        expect('KHR_lights_punctual' not in json.dumps(g.j.get('extensions', {})), f'{fname}: contains lights')
        used_mats = set()
        total = 0
        # body
        bi, body = g.node('body')
        expect(body is not None and 'mesh' in (body or {}), f'{fname}: no body mesh node')
        if body is not None:
            expect('rotation' not in body and 'scale' not in body and 'translation' not in body, f'{fname}: body has a transform')
            tris, mats, P, okc, okn, cmin = g.mesh_stats(body)
            total += tris
            used_mats |= set(mats)
            expect(tris <= max_body, f'{fname}: body {tris} tris > {max_body}')
            expect(okc, f'{fname}: body missing COLOR_0')
            expect(okn, f'{fname}: body missing NORMAL')
            bmin, bmax = P.min(axis=0), P.max(axis=0)
            info[f'lod{lod}_body'] = tris
            if lod == 0 and meta:
                expect(np.allclose(bmin, meta['bodyBox']['min'], atol=0.01) and np.allclose(bmax, meta['bodyBox']['max'], atol=0.01),
                       f'{fname}: bodyBox mismatch {bmin} {bmax}')
                expect(abs((bmax[2] - bmin[2]) - meta['length']) < 0.02, 'length mismatch')
                expect(abs(bmax[1] - meta['height']) < 0.02, 'height mismatch')
            expect(bmin[1] > 0.04, f'{fname}: body touches the ground ({bmin[1]:.3f})')
            expect(abs(bmin[0] + bmax[0]) < 0.03, f'{fname}: body not centred in x')
        # wheels
        for w in WHEELS:
            wi, wn = g.node(f'wheel_{w}')
            if wn is None:
                errs.append(f'{fname}: missing wheel_{w}')
                continue
            expect('mesh' not in wn, f'{fname}: wheel_{w} is not an empty')
            rot = wn.get('rotation', [0, 0, 0, 1])
            expect(np.allclose(np.abs(rot), [0, 0, 0, 1], atol=1e-5), f'{fname}: wheel_{w} has rotation {rot}')
            expect(np.allclose(wn.get('scale', [1, 1, 1]), 1, atol=1e-6), f'{fname}: wheel_{w} has scale')
            tr = np.array(wn.get('translation', [0, 0, 0]), dtype=float)
            front = w[0] == 'f'
            left = w[1] == 'l'
            if meta:
                ex = (1 if left else -1) * 0.5 * (meta['trackFront'] if front else meta['trackRear'])
                ey = meta['wheelRadiusFront'] if front else meta['wheelRadiusRear']
                ez = meta['axleFrontZ'] if front else meta['axleRearZ']
                expect(np.allclose(tr, [ex, ey, ez], atol=1e-4), f'{fname}: wheel_{w} at {tr} expected {[ex, ey, ez]}')
            kids = [g.j['nodes'][c].get('name') for c in wn.get('children', [])]
            expect(sorted(kids) == sorted([f'wheel_{w}_mesh', f'caliper_{w}']), f'{fname}: wheel_{w} children {kids}')
            wt = 0
            for c in wn.get('children', []):
                cn = g.j['nodes'][c]
                if 'mesh' not in cn:
                    errs.append(f'{fname}: {cn.get("name")} has no mesh')
                    continue
                tris, mats, P, okc, okn, cmin = g.mesh_stats(cn)
                wt += tris
                used_mats |= set(mats)
                expect(okc, f'{fname}: {cn["name"]} missing COLOR_0')
                expect(np.allclose(cn.get('scale', [1, 1, 1]), 1, atol=1e-6), f'{fname}: {cn["name"]} has scale')
                expect(np.allclose(cn.get('translation', [0, 0, 0]), 0, atol=1e-6), f'{fname}: {cn["name"]} is offset')
                q = cn.get('rotation', [0, 0, 0, 1])
                Pw = np.array([quat_rot(q, p) for p in P[::max(1, len(P) // 4000)]]) + tr
                if cn['name'].endswith('_mesh'):
                    expect(abs(Pw[:, 1].min()) < 0.006, f'{fname}: {cn["name"]} tyre bottom at y={Pw[:, 1].min():.4f}')
                    # outer face must point away from the car centre
                    side = 1 if left else -1
                    expect(side * (Pw[:, 0].mean() - tr[0]) > -0.05, f'{fname}: {cn["name"]} mirrored wrongly')
                    # rotation axis = X through the node origin
                    r = np.sqrt((Pw[:, 1] - tr[1]) ** 2 + (Pw[:, 2] - tr[2]) ** 2)
                    expect(abs(r.max() - ey) < 0.004 if meta else True, f'{fname}: {cn["name"]} radius {r.max():.4f}')
            total += wt
            expect(wt <= max_wheel, f'{fname}: wheel_{w} {wt} tris > {max_wheel}')
            info[f'lod{lod}_wheel_{w}'] = wt
        expect(used_mats <= MATERIALS, f'{fname}: unknown materials {used_mats - MATERIALS}')
        expect(total <= max_total, f'{fname}: total {total} tris > {max_total}')
        info[f'lod{lod}_total'] = total
        info[f'lod{lod}_mb'] = round(size_mb, 2)
        info[f'lod{lod}_materials'] = sorted(used_mats)
        if meta and 'triangles' in meta:
            expect(meta['triangles'].get(f'lod{lod}') == total, f'{fname}: json triangle count {meta["triangles"].get(f"lod{lod}")} != {total}')
    thumb = os.path.join(OUT, f'{car_id}_thumb.webp')
    expect(os.path.exists(thumb), 'missing thumbnail')
    if os.path.exists(thumb):
        try:
            from PIL import Image
            im = Image.open(thumb)
            expect(im.size == (960, 540), f'thumb size {im.size}')
            expect(im.mode == 'RGBA', f'thumb mode {im.mode}')
        except ImportError:
            pass
    return errs, info


if __name__ == '__main__':
    cars = sys.argv[1:] or ['aurion', 'valken', 'zephyr']
    bad = 0
    for c in cars:
        errs, info = check(c)
        print(f'== {c}: {"OK" if not errs else "FAIL"}')
        for k, v in info.items():
            print(f'   {k}: {v}')
        for e in errs:
            print('   !!', e)
        bad += len(errs)
    sys.exit(1 if bad else 0)
