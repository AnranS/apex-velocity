"""Tiny hard-surface modelling kit for trackside props (numpy; AO needs Blender's mathutils BVH).

Coordinates are glTF: +Y up, metres, "front" = +Z.  Faces are flat-shaded quads/triangles (or smooth for
cylinders / lathes); every face-corner becomes its own vertex.  finalize() bakes per-vertex ambient occlusion
(including a ground plane at y = 0) into COLOR_0 as grayscale and returns glb.Part lists per material.
"""
import math
import numpy as np
from glb import Part

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def v3(*a):
    return np.array(a[0] if len(a) == 1 else a, dtype=np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Kit:
    def __init__(self, max_edge=2.0):
        self.polys = []          # (material, P[k,3], N[k,3], UV[k,2])
        self.max_edge = max_edge
        self.node = "main"
        self.nodes = {"main": []}

    def set_node(self, name):
        self.node = name
        self.nodes.setdefault(name, [])

    # ------------------------------------------------------------------ low level
    def _emit(self, mat, P, N, UV):
        self.nodes[self.node].append((mat, np.asarray(P, dtype=np.float64), np.asarray(N, dtype=np.float64),
                                      np.asarray(UV, dtype=np.float64)))

    @staticmethod
    def _planar_uv(P, n):
        a = int(np.argmax(np.abs(n)))
        if a == 1:
            return np.stack([P[:, 0], P[:, 2]], 1)
        if a == 0:
            return np.stack([P[:, 2], -P[:, 1]], 1)
        return np.stack([P[:, 0], -P[:, 1]], 1)

    def quad(self, mat, p0, p1, p2, p3, uv=None, out=None, normals=None, max_edge=None):
        """Quad p0..p3 (any winding if `out` hint given, else CCW seen from the front). Subdivided so that no
        edge exceeds max_edge (keeps vertex AO meaningful on large faces)."""
        P = np.array([p0, p1, p2, p3], dtype=np.float64)
        n = np.cross(P[1] - P[0], P[3] - P[0])
        if np.linalg.norm(n) < 1e-12:
            n = np.cross(P[2] - P[1], P[3] - P[1])
        n = unit(n)
        UVq = None if uv is None else np.array(uv, dtype=np.float64)
        Nq = None if normals is None else np.array(normals, dtype=np.float64)
        if out is not None and np.dot(n, out) < 0:
            P = P[[0, 3, 2, 1]]
            n = -n
            if UVq is not None:
                UVq = UVq[[0, 3, 2, 1]]
            if Nq is not None:
                Nq = Nq[[0, 3, 2, 1]]
        me = self.max_edge if max_edge is None else max_edge
        nu = max(1, int(math.ceil(max(np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[2] - P[3])) / me)))
        nv = max(1, int(math.ceil(max(np.linalg.norm(P[3] - P[0]), np.linalg.norm(P[2] - P[1])) / me)))

        def bil(A, s, t):
            return (A[0] * (1 - s) + A[1] * s) * (1 - t) + (A[3] * (1 - s) + A[2] * s) * t

        for i in range(nu):
            for j in range(nv):
                s0, s1, t0, t1 = i / nu, (i + 1) / nu, j / nv, (j + 1) / nv
                Pq = np.array([bil(P, s0, t0), bil(P, s1, t0), bil(P, s1, t1), bil(P, s0, t1)])
                if Nq is not None:
                    Nn = np.array([unit(bil(Nq, s0, t0)), unit(bil(Nq, s1, t0)), unit(bil(Nq, s1, t1)), unit(bil(Nq, s0, t1))])
                else:
                    Nn = np.tile(n, (4, 1))
                if UVq is not None:
                    U = np.array([bil(UVq, s0, t0), bil(UVq, s1, t0), bil(UVq, s1, t1), bil(UVq, s0, t1)])
                else:
                    U = self._planar_uv(Pq, n)
                self._emit(mat, Pq, Nn, U)

    def tri(self, mat, p0, p1, p2, out=None, normals=None, uv=None):
        P = np.array([p0, p1, p2], dtype=np.float64)
        n = unit(np.cross(P[1] - P[0], P[2] - P[0]))
        Nq = None if normals is None else np.array(normals, dtype=np.float64)
        if out is not None and np.dot(n, out) < 0:
            P = P[[0, 2, 1]]
            n = -n
            if Nq is not None:
                Nq = Nq[[0, 2, 1]]
        self._emit(mat, P, np.tile(n, (3, 1)) if Nq is None else Nq, self._planar_uv(P, n) if uv is None else uv)

    # ------------------------------------------------------------------ solids
    def box(self, mat, c, size, faces="xXyYzZ", R=None, mats=None, max_edge=None):
        """Axis-aligned (or rotated by R about its centre) box. faces: x/X = -X/+X ... ; mats: dict face -> material."""
        c = v3(c)
        h = v3(size) * 0.5
        R = np.eye(3) if R is None else R
        ax = [R[:, 0], R[:, 1], R[:, 2]]
        table = {"X": (0, 1, 1, 2), "x": (0, -1, 2, 1), "Y": (1, 1, 2, 0), "y": (1, -1, 0, 2), "Z": (2, 1, 0, 1), "z": (2, -1, 1, 0)}
        for f in faces:
            a, s, u, v = table[f]
            o = c + ax[a] * (s * h[a])
            du, dv = ax[u] * h[u], ax[v] * h[v]
            m = mat if not mats or f not in mats else mats[f]
            self.quad(m, o - du - dv, o + du - dv, o + du + dv, o - du + dv, max_edge=max_edge)

    def beam(self, mat, a, b, w, h=None, up=Y, caps=False, max_edge=None):
        """Rectangular-section member from a to b (w across, h along `up`)."""
        a, b = v3(a), v3(b)
        h = w if h is None else h
        d = unit(b - a)
        side = np.cross(up, d)
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(X, d)
        side = unit(side)
        upv = np.cross(d, side)
        offs = [side * (w / 2) + upv * (h / 2), -side * (w / 2) + upv * (h / 2), -side * (w / 2) - upv * (h / 2),
                side * (w / 2) - upv * (h / 2)]
        for i in range(4):
            o0, o1 = offs[i], offs[(i + 1) % 4]
            self.quad(mat, a + o0, a + o1, b + o1, b + o0, out=o0 + o1, max_edge=max_edge)
        if caps:
            self.quad(mat, a + offs[0], a + offs[1], a + offs[2], a + offs[3], out=-d)
            self.quad(mat, b + offs[0], b + offs[1], b + offs[2], b + offs[3], out=d)

    def cyl(self, mat, a, b, r0, r1=None, sides=8, cap_a=None, cap_b=None, smooth=True, segs=1):
        """Cylinder / frustum from a (radius r0) to b (radius r1). cap_a / cap_b: material name or None."""
        a, b = v3(a), v3(b)
        r1 = r0 if r1 is None else r1
        d = unit(b - a)
        ref = X if abs(d[0]) < 0.9 else Y
        e1 = unit(np.cross(d, ref))
        e2 = np.cross(d, e1)
        ang = np.linspace(0, 2 * math.pi, sides + 1)
        ring = [e1 * math.cos(t) + e2 * math.sin(t) for t in ang]
        slope = (r0 - r1) / max(np.linalg.norm(b - a), 1e-9)
        for k in range(segs):
            f0, f1 = k / segs, (k + 1) / segs
            pa, pb = a + (b - a) * f0, a + (b - a) * f1
            ra, rb = r0 + (r1 - r0) * f0, r0 + (r1 - r0) * f1
            for i in range(sides):
                n0, n1 = unit(ring[i] + d * slope), unit(ring[i + 1] + d * slope)
                P = [pa + ring[i] * ra, pa + ring[i + 1] * ra, pb + ring[i + 1] * rb, pb + ring[i] * rb]
                N = [n0, n1, n1, n0] if smooth else None
                self.quad(mat, *P, out=ring[i] + ring[i + 1], normals=N, max_edge=1e9)
        for cap, p, r, o in ((cap_a, a, r0, -d), (cap_b, b, r1, d)):
            if cap and r > 1e-6:
                for i in range(sides):
                    self.tri(cap, p, p + ring[i] * r, p + ring[i + 1] * r, out=o)

    def lathe(self, mat, profile, sides=12, center=(0, 0, 0), closed=False, mats=None):
        """Revolve a (radius, y) profile around the Y axis through `center`. Smooth normals across the profile."""
        c = v3(center)
        prof = [np.array(p, dtype=np.float64) for p in profile]
        m = len(prof)
        segs = m if closed else m - 1
        cen = np.mean(prof, axis=0)
        # 2D normals per profile vertex (averaged from neighbouring segments, pointing away from the section centroid)
        seg_n = []
        for i in range(segs):
            p0, p1 = prof[i], prof[(i + 1) % m]
            t = p1 - p0
            n = unit(np.array([t[1], -t[0]]))
            mid = (p0 + p1) / 2
            if closed and np.dot(n, mid - cen) < 0:
                n = -n
            if not closed and n[0] < 0:
                n = -n
            seg_n.append(n)
        vert_n = []
        for i in range(m):
            if closed:
                n = seg_n[i - 1] + seg_n[i]
            else:
                n = seg_n[min(i, segs - 1)] + seg_n[max(i - 1, 0)]
            vert_n.append(unit(n))
        ang = np.linspace(0, 2 * math.pi, sides + 1)
        for i in range(segs):
            i1 = (i + 1) % m
            mname = mat if not mats else mats[i]
            for k in range(sides):
                c0, s0, c1, s1 = math.cos(ang[k]), math.sin(ang[k]), math.cos(ang[k + 1]), math.sin(ang[k + 1])

                def pt(p, cc, ss):
                    return c + np.array([p[0] * cc, p[1], p[0] * ss])

                def nr(n, cc, ss):
                    return np.array([n[0] * cc, n[1], n[0] * ss])

                P = [pt(prof[i], c0, s0), pt(prof[i], c1, s1), pt(prof[i1], c1, s1), pt(prof[i1], c0, s0)]
                N = [nr(vert_n[i], c0, s0), nr(vert_n[i], c1, s1), nr(vert_n[i1], c1, s1), nr(vert_n[i1], c0, s0)]
                if np.linalg.norm(P[0] - P[1]) < 1e-9:      # degenerate at the axis
                    self.tri(mname, P[0], P[2], P[3], out=nr(seg_n[i], (c0 + c1) / 2, (s0 + s1) / 2), normals=[N[0], N[2], N[3]])
                elif np.linalg.norm(P[2] - P[3]) < 1e-9:
                    self.tri(mname, P[0], P[1], P[2], out=nr(seg_n[i], (c0 + c1) / 2, (s0 + s1) / 2), normals=[N[0], N[1], N[2]])
                else:
                    self.quad(mname, *P, out=nr(seg_n[i], (c0 + c1) / 2, (s0 + s1) / 2), normals=N, max_edge=1e9)

    def truss(self, mat, a, b, depth, panels, chord=0.14, web=0.09, up=Y, verticals=True):
        """Planar Warren truss between a and b: top chord at +up*depth, bottom chord on the a-b line."""
        a, b = v3(a), v3(b)
        upv = unit(up) * depth
        self.beam(mat, a, b, chord)
        self.beam(mat, a + upv, b + upv, chord)
        for i in range(panels):
            p0 = a + (b - a) * (i / panels)
            p1 = a + (b - a) * ((i + 1) / panels)
            if i % 2 == 0:
                self.beam(mat, p0, p1 + upv, web)
            else:
                self.beam(mat, p0 + upv, p1, web)
            if verticals:
                self.beam(mat, p0, p0 + upv, web)
        if verticals:
            self.beam(mat, b, b + upv, web)

    # ------------------------------------------------------------------ output
    def triangle_count(self):
        return sum(len(P) - 2 for polys in self.nodes.values() for (_m, P, _n, _u) in polys)

    def finalize(self, ao=True, rays=64, max_dist=4.0, strength=0.92, ground=True, ao_nodes=None):
        """Returns {node: [Part,...]} with COLOR_0 = baked AO (grayscale)."""
        tris_all = []
        for polys in self.nodes.values():
            for (_m, P, _n, _u) in polys:
                for k in range(1, len(P) - 1):
                    tris_all.append((P[0], P[k], P[k + 1]))
        bvh = None
        if ao:
            from mathutils.bvhtree import BVHTree
            verts, faces = [], []
            for t in tris_all:
                i = len(verts)
                verts += [tuple(t[0]), tuple(t[1]), tuple(t[2])]
                faces.append((i, i + 1, i + 2))
            if ground:
                allp = np.array(verts)
                ext = float(np.abs(allp[:, [0, 2]]).max()) + 30.0
                i = len(verts)
                verts += [(-ext, 0.0, -ext), (ext, 0.0, -ext), (ext, 0.0, ext), (-ext, 0.0, ext)]
                faces += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
            bvh = BVHTree.FromPolygons(verts, faces, all_triangles=True, epsilon=0.0)
        # cosine-weighted hemisphere directions (local frame, z = normal)
        k = np.arange(rays) + 0.5
        phi = k * 2.399963229728653
        cos_t = np.sqrt(1.0 - k / rays)
        sin_t = np.sqrt(1.0 - cos_t ** 2)
        hemi = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], 1)
        cache = {}

        def vertex_ao(p, n):
            key = (round(p[0], 3), round(p[1], 3), round(p[2], 3), round(n[0], 2), round(n[1], 2), round(n[2], 2))
            if key in cache:
                return cache[key]
            ref = X if abs(n[0]) < 0.9 else Y
            t = unit(np.cross(n, ref))
            b = np.cross(n, t)
            rot = (hash(key) % 628) / 100.0
            ct, st = math.cos(rot), math.sin(rot)
            t2 = t * ct + b * st
            b2 = b * ct - t * st
            dirs = hemi[:, 0:1] * t2 + hemi[:, 1:2] * b2 + hemi[:, 2:3] * n
            o = (p[0] + n[0] * 0.012, p[1] + n[1] * 0.012, p[2] + n[2] * 0.012)
            occ = 0.0
            for d in dirs:
                hit = bvh.ray_cast(o, (d[0], d[1], d[2]), max_dist)
                if hit[0] is not None:
                    occ += 1.0 - (hit[3] / max_dist) ** 1.5
            val = max(0.12, 1.0 - strength * occ / rays)
            cache[key] = val
            return val

        out = {}
        for node, polys in self.nodes.items():
            by_mat = {}
            for (m, P, N, U) in polys:
                acc = by_mat.setdefault(m, {"p": [], "n": [], "u": [], "c": [], "i": [], "k": 0})
                base = acc["k"]
                for q in range(len(P)):
                    acc["p"].append(P[q])
                    acc["n"].append(N[q])
                    acc["u"].append(U[q])
                    a = vertex_ao(P[q], N[q]) if (bvh is not None and (ao_nodes is None or node in ao_nodes)) else 1.0
                    acc["c"].append((a, a, a))
                for q in range(1, len(P) - 1):
                    acc["i"] += [base, base + q, base + q + 1]
                acc["k"] += len(P)
            out[node] = [Part(m, np.array(a["p"]), np.array(a["n"]), np.array(a["u"]), np.array(a["c"]), np.array(a["i"]))
                         for m, a in by_mat.items()]
        return out
