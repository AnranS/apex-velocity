"""Car construction library: parametric implicit car body (hull + canopy + cuts),
material painting by implicit regions, parts, wheels, interior, AO, assembly.

Geometry is authored in game/GLB space: +X = car left, +Y = up, +Z = nose, metres,
origin on the ground under the wheelbase midpoint.
"""
import time
import numpy as np
import sdf
from sdf import smin, smax, sstep, Profile, gauss, window
import meshops
import bl

MID = bl.MAT_ID


def log(*a):
    print('[car]', *a, flush=True)


# =============================================================================== skin painter
class Skin:
    """Triangle mesh lying on the zero set of `fn`, with per-triangle materials.
    Regions are cut exactly into the mesh (see meshops.split_by_field)."""

    def __init__(self, V, T, fn, mat='paint', guard_cap=0.06):
        self.guard_cap = guard_cap
        self.V = np.asarray(V, dtype=np.float64)
        self.T = np.asarray(T, dtype=np.int32)
        self.fn = fn
        self.M = np.full(len(self.T), MID[mat], dtype=np.int32)
        self.recesses = []
        self.locked = np.zeros(len(self.V), dtype=bool)   # vertices lying on material boundaries

    def _mask(self, only):
        if only is None:
            return np.ones(len(self.T), dtype=bool)
        ids = [MID[o] for o in only]
        return np.isin(self.M, ids)

    def _locked(self):
        if len(self.locked) < len(self.V):
            self.locked = np.concatenate([self.locked, np.zeros(len(self.V) - len(self.locked), dtype=bool)])
        return self.locked

    def _project(self, idx):
        if self.fn is not None and len(idx):
            self.V[idx] = sdf.project(self.fn, self.V[idx], iters=2, max_step=0.01)

    def _bisect(self, hit, max_len):
        """Bisect the longest edge of every selected triangle whose longest edge exceeds max_len
        (longest-edge refinement keeps triangles well shaped). Returns False when nothing to do."""
        Th = self.T[hit]
        if len(Th) == 0:
            return False
        P = self.V[Th]
        L = np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1),
                      np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                      np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], axis=1)
        k = np.argmax(L, axis=1)
        need = L[np.arange(len(Th)), k] > max_len
        if not need.any():
            return False
        a = Th[need, k[need]]
        b = Th[need, (k[need] + 1) % 3]
        N = len(self.V)
        key = np.unique(np.minimum(a, b).astype(np.int64) * N + np.maximum(a, b).astype(np.int64))
        V, T, parent = meshops.subdivide(self.V, self.T, key // N, key % N, 0.5)
        if self.fn is not None:
            V[N:] = sdf.project(self.fn, V[N:], iters=2, max_step=0.01)
        self.V, self.T, self.M = V, T, self.M[parent]
        return True

    def refine_curvature(self, max_dev_deg=16.0, min_len=0.012, only=None, iters=4, max_tris=None):
        """Bisect triangles whose vertex normals disagree by more than max_dev (under-tessellated
        high-curvature areas show up as wobbly reflections on glossy paint). Worst first; stops at
        max_tris."""
        cosmax = np.cos(np.radians(max_dev_deg))
        for _ in range(iters):
            if max_tris is not None and len(self.T) >= max_tris:
                return False
            mask = self._mask(only)
            N = sdf.normals(self.fn, self.V)
            n0, n1, n2 = N[self.T[:, 0]], N[self.T[:, 1]], N[self.T[:, 2]]
            c = np.minimum(np.minimum((n0 * n1).sum(1), (n1 * n2).sum(1)), (n0 * n2).sum(1))
            P = self.V[self.T]
            Lmax = np.maximum(np.maximum(np.linalg.norm(P[:, 1] - P[:, 0], axis=1), np.linalg.norm(P[:, 2] - P[:, 1], axis=1)),
                              np.linalg.norm(P[:, 0] - P[:, 2], axis=1))
            hit = mask & (c < cosmax) & (Lmax > min_len)
            if not hit.any():
                return True
            if max_tris is not None:
                room = max(int((max_tris - len(self.T)) / 2.6), 0)
                idx = np.flatnonzero(hit)
                if len(idx) > room:
                    idx = idx[np.argsort(c[idx])[:room]]
                    hit = np.zeros(len(self.T), dtype=bool)
                    hit[idx] = True
                    if room == 0:
                        return False
            if not self._bisect(hit, min_len):
                return True
        return True

    def limit_edges(self, max_len, tri_filter=None, iters=14):
        """Cap edge length (QEM decimation leaves huge triangles on flat panels)."""
        for _ in range(iters):
            hit = np.ones(len(self.T), dtype=bool) if tri_filter is None else tri_filter(self)
            if not self._bisect(hit, max_len):
                break

    def _samples(self, field):
        """Field values at 7 points per triangle (corners, edge midpoints, centroid)."""
        d = field(self.V)
        P = self.V[self.T]
        mids = np.concatenate([0.5 * (P[:, 0] + P[:, 1]), 0.5 * (P[:, 1] + P[:, 2]), 0.5 * (P[:, 2] + P[:, 0]), P.mean(axis=1)], axis=0)
        dm = field(mids).reshape(4, -1).T
        return np.concatenate([d[self.T], dm], axis=1)

    def _bisect_edges(self, a, b):
        N = len(self.V)
        key = np.unique(np.minimum(a, b).astype(np.int64) * N + np.maximum(a, b).astype(np.int64))
        V, T, parent = meshops.subdivide(self.V, self.T, key // N, key % N, 0.5)
        if self.fn is not None:
            V[N:] = sdf.project(self.fn, V[N:], iters=2, max_step=0.01)
        self.V, self.T, self.M = V, T, self.M[parent]

    def guard(self, field, cap=0.06, only=None, iters=10):
        """Break up big triangles that sit within their own size of the curve field == 0, so that the
        field is close to linear inside every triangle the cut will touch (long QEM needles could
        otherwise be crossed twice by the curve and be missed)."""
        for _ in range(iters):
            mask = self._mask(only)
            if not mask.any():
                return
            ds = np.abs(self._samples(field)).min(axis=1)
            P = self.V[self.T]
            Lmax = np.maximum(np.maximum(np.linalg.norm(P[:, 1] - P[:, 0], axis=1), np.linalg.norm(P[:, 2] - P[:, 1], axis=1)),
                              np.linalg.norm(P[:, 0] - P[:, 2], axis=1))
            hit = mask & (Lmax > cap) & (ds < 0.6 * Lmax)
            if not self._bisect(hit, cap):
                break

    def refine(self, field, max_len, only=None, band=None, iters=16):
        """Refine so that the curve field == 0 is resolved with segments <= max_len.
        band=None: for every triangle crossed by the zero set, bisect its edge that runs *along* the
        curve (the one whose end points lie on the same side) - cheap: no isotropic refinement.
        band=(lo, hi): isotropic longest-edge refinement of triangles with a sample in lo < field < hi
        (thin strips narrower than the triangles)."""
        for _ in range(iters):
            mask = self._mask(only)
            if not mask.any():
                return
            ds = self._samples(field)
            if band is not None:
                hit = ((ds > band[0]) & (ds < band[1])).any(axis=1)
                if not self._bisect(hit & mask, max_len):
                    break
                continue
            crossed = (ds.min(axis=1) < 0) & (ds.max(axis=1) > 0) & mask
            if not crossed.any():
                break
            idx = np.flatnonzero(crossed)
            Tc = self.T[idx]
            neg = ds[idx, :3] < 0
            nn = neg.sum(axis=1)
            mixed = (nn == 1) | (nn == 2)
            lone = np.where(nn == 1, np.argmax(neg, axis=1), np.argmin(neg, axis=1))
            P = self.V[Tc]
            L = np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1),
                          np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                          np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], axis=1)
            # edge k joins vertex k and k+1; the edge opposite to `lone` is edge (lone+1)%3
            ek = np.where(mixed, (lone + 1) % 3, np.argmax(L, axis=1))
            r = np.arange(len(idx))
            need = L[r, ek] > max_len
            if not need.any():
                break
            self._bisect_edges(Tc[need, ek[need]], Tc[need, (ek[need] + 1) % 3])

    def cut(self, field, only=None, snap=0.22, where=None):
        """Cut along field == 0 (restricted to triangles whose material is in `only` and,
        optionally, whose centroid satisfies where(P) < 0)."""
        d = field(self.V)
        mask = self._mask(only)
        if where is not None:
            mask &= where(self.V[self.T].mean(axis=1)) < 0
        lk = self._locked()
        V, T, parent, touched = meshops.split_by_field(self.V, self.T, d, snap=snap, tri_mask=mask, locked=lk)
        self.V, self.T, self.M = V, T, self.M[parent]
        self._project(np.flatnonzero(touched))
        self.locked = np.concatenate([lk, np.zeros(len(V) - len(lk), dtype=bool)]) | touched

    def region(self, field, mat, only=None, refine=None, snap=0.22, thin=False, guard=True):
        """Assign `mat` to the part of the skin where field < 0 (exact boundary).
        thin=True for strips narrower than the local triangles (refines everything near/inside).
        guard: cap for triangles near the boundary (None for fields that run parallel to the surface)."""
        if guard and self.guard_cap:
            self.guard(field, self.guard_cap, only=only)
        if refine:
            self.refine(field, refine, only=only, band=((-1e9, 0.6 * refine) if thin else None))
        self.cut(field, only=only, snap=snap)
        if not self.guard_cap:
            # coarse meshes: split only needles the curve crosses twice (clearly on both sides) and cut again
            for _ in range(3):
                ds = self._samples(field)
                bad = (ds.min(axis=1) < -0.02) & (ds.max(axis=1) > 0.02) & self._mask(only)
                if not bad.any() or not self._bisect(bad, 0.05):
                    break
                self.cut(field, only=only, snap=snap)
        cen = self.V[self.T].mean(axis=1)
        inside = (field(cen) < 0) & self._mask(only)
        self.M[inside] = MID[mat]
        return int(inside.sum())

    def band(self, center, half_width, mat, only=None, extent=None, refine=None, where=None):
        """Thin strip |center| < half_width (optionally limited by extent < 0).
        `where` (field, < 0 = eligible) keeps the two long cuts local."""
        if where is None and extent is not None:
            where = lambda P: extent(P) - 0.12
        gfield = center if extent is None else (lambda P: np.maximum(np.abs(center(P)) - half_width, extent(P)))
        self.guard(gfield, self.guard_cap or 0.11, only=only)
        if refine:
            self.refine(center, refine, only=only, band=(-0.6 * refine, 0.6 * refine))
        self.cut(lambda P: center(P) - half_width, only=only, snap=0.12, where=where)
        self.cut(lambda P: center(P) + half_width, only=only, snap=0.12, where=where)
        cen = self.V[self.T].mean(axis=1)
        inside = (np.abs(center(cen)) < half_width) & self._mask(only)
        if extent is not None:
            # cut the strip ends
            idx = np.flatnonzero(inside)
            d = extent(self.V)
            tri_mask = np.zeros(len(self.T), dtype=bool)
            tri_mask[idx] = True
            lk = self._locked()
            V, T, parent, touched = meshops.split_by_field(self.V, self.T, d, snap=0.12, tri_mask=tri_mask, locked=lk)
            self.V, self.T, self.M = V, T, self.M[parent]
            self._project(np.flatnonzero(touched))
            self.locked = np.concatenate([lk, np.zeros(len(V) - len(lk), dtype=bool)]) | touched
            cen = self.V[self.T].mean(axis=1)
            inside = (np.abs(center(cen)) < half_width) & (extent(cen) < 0) & self._mask(only)
        self.M[inside] = MID[mat]
        return int(inside.sum())

    def recess(self, field, depth, ramp=0.012, mats=None, refine=0.022):
        """Sink the part of the skin where field < 0 by `depth` behind a chamfered wall of width `ramp`.
        An inner edge loop is cut at field == -ramp so the wall is a clean strip; the rim stays exactly
        on the surface. Applied in finish()."""
        inner = lambda P: field(P) + ramp
        if self.guard_cap:
            self.guard(inner, self.guard_cap, only=mats)
        if refine:
            self.refine(inner, refine, only=mats)
        self.cut(inner, only=mats, snap=0.12)
        self.recesses.append((field, depth, ramp, mats))

    def delete(self, tri_mask):
        self.T = self.T[~tri_mask]
        self.M = self.M[~tri_mask]

    def finish(self, normal_fn=None, flip=False):
        if normal_fn is None:
            normal_fn = self.fn
        N = sdf.normals(normal_fn, self.V)
        V = self.V.copy()
        T = self.T.copy()
        ao_scale = np.ones(len(V))
        for field, depth, ramp, mats in self.recesses:
            d = field(self.V)
            tri_ok = self._mask(mats)
            cd = field(self.V[self.T].mean(axis=1))
            inside = tri_ok & (cd < 0)
            wall = inside & (cd > -ramp)
            floor = inside & ~wall
            # floor: full depth, keeps the smooth surface normals
            fv = np.zeros(len(self.V), dtype=bool)
            fv[self.T[floor].ravel()] = True
            V[:len(self.V)][fv] -= N[:len(self.V)][fv] * depth
            ao_scale[:len(self.V)][fv] *= 0.82
            # wall: duplicated vertices with their own normals
            wt = np.flatnonzero(wall)
            if len(wt):
                wverts = np.unique(self.T[wt].ravel())
                new_id = np.full(len(self.V), -1, dtype=np.int64)
                new_id[wverts] = len(V) + np.arange(len(wverts))
                amt = depth * np.clip(-d[wverts] / ramp, 0.0, 1.0)
                # watertight seams: vertices shared with the floor sink fully, rim vertices not at all
                ov = np.zeros(len(self.V), dtype=bool)
                ov[self.T[~inside].ravel()] = True
                amt = np.where(fv[wverts], depth, np.where(ov[wverts], 0.0, amt))
                Vw = self.V[wverts] - N[wverts] * amt[:, None]
                Tw = new_id[self.T[wt]]
                V = np.vstack([V, Vw])
                T[wt] = Tw
                # wall normals from the wall strip itself
                Nw = np.zeros((len(V), 3))
                fn_ = meshops.tri_normals(V, T[wt])
                for k in range(3):
                    np.add.at(Nw, T[wt][:, k], fn_)
                Nw = Nw[len(V) - len(wverts):]
                ln = np.linalg.norm(Nw, axis=1)
                Nw = np.where(ln[:, None] > 1e-12, Nw / np.maximum(ln, 1e-12)[:, None], N[wverts])
                N = np.vstack([N, Nw])
                ao_scale = np.concatenate([ao_scale, np.full(len(wverts), 0.9)])
        self.ao_scale = ao_scale
        used = np.zeros(len(V), dtype=bool)
        used[T.ravel()] = True
        remap = np.full(len(V), -1, dtype=np.int64)
        remap[used] = np.arange(int(used.sum()))
        T = remap[T].astype(np.int32)
        V = V[used]; N = N[used]
        self.ao_scale = ao_scale[used]
        # make every triangle face the way its shading normals do (decimation / snapping can fold a few)
        gn = meshops.tri_normals(V, T)
        bad = (gn * (N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]])).sum(axis=1) < 0
        if bad.any():
            T[bad] = T[bad][:, ::-1]
        self.flipped_fixed = int(bad.sum())
        if flip:
            T = T[:, ::-1].copy()
            N = -N
        return dict(V=V, T=T, N=N, M=self.M.copy(), C=None, ao_scale=self.ao_scale)


def piece(V, T, N, mat, ao=None):
    return dict(V=np.asarray(V, float), T=np.asarray(T, np.int32), N=np.asarray(N, float),
                M=np.full(len(T), MID[mat], dtype=np.int32), C=ao)


def sdf_part(fn, bmin, bmax, h, mat, target_tris=None, symmetric=True, normals='grad', sharp_angle=35.0):
    """Mesh a small implicit part. normals: 'grad' (smooth, from the field) or 'auto' (crease split)."""
    V, Q = sdf.mesh_sdf(fn, bmin, bmax, h, symmetric=symmetric)
    if len(Q) == 0:
        raise RuntimeError('empty part')
    ntri = 2 * len(Q)
    if target_tris is not None and ntri > target_tris:
        V, T = bl.decimate(V, Q, target_tris=target_tris, symmetry=False)
        V = sdf.project(fn, V, iters=2, max_step=0.5 * h)
    else:
        T = sdf.quads_to_tris(V, Q)
    V, T = meshops.compact(V, T)[:2]
    if normals == 'grad':
        N = sdf.normals(fn, V)
    else:
        V, T, N = meshops.auto_smooth_split(V, T, sharp_angle)
    return piece(V, T, N, mat)


def transform_piece(p, offset=(0, 0, 0), mirror_x=False):
    V = p['V'].copy(); N = p['N'].copy(); T = p['T'].copy()
    if mirror_x:
        V[:, 0] *= -1; N[:, 0] *= -1; T = T[:, ::-1].copy()
    V += np.asarray(offset, float)[None, :]
    return dict(V=V, T=T, N=N, M=p['M'].copy(), C=None if p.get('C') is None else p['C'].copy())


def sdf_ao(fn, V, N, dists=(0.015, 0.04, 0.09, 0.18, 0.32), strength=2.1, floor=0.1, cone=True):
    """Cheap, noise-free ambient occlusion from an implicit scene: march along the normal and
    along four directions tilted 50 degrees away from it."""
    V = np.asarray(V, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    dirs = [N]
    wdir = [1.0]
    if cone:
        ref = np.where(np.abs(N[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
        t1 = np.cross(N, ref)
        t1 /= np.maximum(np.linalg.norm(t1, axis=1), 1e-12)[:, None]
        t2 = np.cross(N, t1)
        ca, sa = np.cos(np.radians(50.0)), np.sin(np.radians(50.0))
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dirs.append(N * ca + (t1 * a + t2 * b) * sa)
            wdir.append(0.6)
    total = np.zeros(len(V))
    for D, wd in zip(dirs, wdir):
        occ = np.zeros(len(V))
        wsum = 0.0
        w = 1.0
        for h in dists:
            P = V + D * h
            f = fn(P[:, 0], P[:, 1], P[:, 2])
            occ += w * np.clip((h * 0.92 - f) / h, 0.0, 1.0)
            wsum += w
            w *= 0.75
        total += wd * occ / wsum
    ao = 1.0 - strength * total / sum(wdir)
    return np.clip(ao, floor, 1.0)


# =============================================================================== wheels
def bspline_curve(ctrl, n):
    """Clamped uniform cubic B-spline through a 2D control polygon, n samples."""
    P = np.asarray(ctrl, dtype=np.float64)
    P = np.vstack([P[0], P[0], P, P[-1], P[-1]])
    segs = len(P) - 3
    ts = np.linspace(0, segs, n)
    out = []
    for t in ts:
        i = min(int(t), segs - 1)
        u = t - i
        b0 = (1 - u) ** 3 / 6.0
        b1 = (3 * u ** 3 - 6 * u ** 2 + 4) / 6.0
        b2 = (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6.0
        b3 = u ** 3 / 6.0
        out.append(b0 * P[i] + b1 * P[i + 1] + b2 * P[i + 2] + b3 * P[i + 3])
    return np.array(out)


def tire_profile(R, R_rim, w, grooves=3, lod=0):
    hw = 0.5 * w
    SW = R - R_rim
    side_ctrl = [(-hw + 0.016, R_rim - 0.004), (-hw + 0.002, R_rim + 0.004), (-hw - 0.006, R_rim + 0.016),
                 (-hw - 0.011, R_rim + 0.40 * SW), (-hw - 0.012, R_rim + 0.62 * SW),
                 (-hw - 0.004, R - 0.030), (-hw + 0.012, R - 0.010), (-hw + 0.040, R - 0.0025)]
    n_side = 9 if lod == 0 else 4
    side = bspline_curve(side_ctrl, n_side)
    tw = hw - 0.040
    r_t = lambda x: R - 0.0025 - 0.0 * x
    pts = [tuple(p) for p in side]
    if lod == 0:
        if grooves == 3:
            gx = [-0.48 * tw, 0.0, 0.48 * tw]
        else:
            gx = [-0.62 * tw, -0.21 * tw, 0.21 * tw, 0.62 * tw]
        gw, gd = 0.0065, 0.007
        for g in gx:
            a = (g - gw, R); b = (g - gw + 0.0015, R - gd); c = (g + gw - 0.0015, R - gd); d = (g + gw, R)
            pts += [a, a, b, b, c, c, d, d]
    else:
        pts += [(0.0, R)]
    pts += [(-x, r) for (x, r) in [tuple(p) for p in side][::-1]]
    return pts


def build_tire(R, R_rim, w, grooves=3, lod=0):
    segs = 52 if lod == 0 else 20
    prof = tire_profile(R, R_rim, w, grooves, lod)
    V, T, N = meshops.lathe(prof, segs, axis='x')
    r = np.sqrt(V[:, 1] ** 2 + V[:, 2] ** 2)
    ao = np.clip(0.55 + 0.45 * (r - R_rim) / max(R - R_rim, 1e-3) * 1.6, 0.5, 1.0)
    groove = r < (R - 0.004)
    onTread = np.abs(V[:, 0]) < (0.5 * w - 0.04)
    ao = np.where(groove & onTread, 0.45, ao)
    return piece(V, T, N, 'tire', ao)


def build_barrel(R_rim, w, lod=0):
    hw = 0.5 * w
    segs = 48 if lod == 0 else 20
    a = (hw - 0.014, R_rim + 0.004); b = (hw, R_rim + 0.007); c = (hw + 0.0015, R_rim - 0.010)
    d = (hw - 0.007, R_rim - 0.021); e = (-hw + 0.02, R_rim - 0.026); f = (-hw + 0.002, R_rim - 0.026)
    prof = [a, b, b, c, d, e, f] if lod == 0 else [a, b, b, d, f]
    V, T, N = meshops.lathe(prof, segs, axis='x')
    ao = np.clip(0.35 + 0.65 * (V[:, 0] + hw) / w, 0.3, 1.0)
    ao = np.where(V[:, 0] > hw - 0.02, 1.0, ao * 0.8)
    return piece(V, T, N, 'rim', ao)


def build_disc(R_disc, x_d, lod=0):
    segs = 36 if lod == 0 else 14
    Ri = R_disc - 0.078
    A = (x_d - 0.015, Ri); B = (x_d - 0.015, R_disc); C = (x_d + 0.015, R_disc); D = (x_d + 0.015, Ri)
    E = (x_d + 0.042, Ri - 0.014); Fp = (x_d + 0.046, 0.055)
    prof = [A, B, B, C, C, D, D, E, Fp] if lod == 0 else [B, C, C, D, D, Fp]
    V, T, N = meshops.lathe(prof, segs, axis='x')
    ao = np.full(len(V), 0.72)
    return piece(V, T, N, 'brake_disc', ao)


def rim_face_field(R_rim, w, style, concave):
    """Spokes + hub implicit field (wheel-local, +X outboard)."""
    hw = 0.5 * w
    x_rim = hw - 0.024
    x_hub = x_rim - concave
    r_hub = 0.078

    def xface(r):
        s = np.clip((r - r_hub * 0.6) / (R_rim - r_hub * 0.6), 0.0, 1.0)
        return x_hub + (x_rim - x_hub) * s ** 1.5

    def seg_d(u, v, a, b):
        ex, ey = b[0] - a[0], b[1] - a[1]
        wx, wy = u - a[0], v - a[1]
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        return np.sqrt((wx - ex * t) ** 2 + (wy - ey * t) ** 2), t

    def fn(X, Y, Z):
        r = np.sqrt(Y * Y + Z * Z)
        th = np.arctan2(Z, Y)
        if style == 'twin5':
            n = 5
        elif style == 'split5':
            n = 5
        elif style == 'multi10':
            n = 10
        else:
            n = 6
        sector = 2 * np.pi / n
        thf = np.mod(th + 0.5 * sector, sector) - 0.5 * sector
        u = r * np.cos(thf)
        v = np.abs(r * np.sin(thf))
        xf = xface(r)
        rr = np.clip((r - r_hub) / (R_rim - r_hub), 0.0, 1.0)
        depth = 0.046 - 0.016 * rr
        Rout = R_rim - 0.012
        if style == 'twin5':
            d, t = seg_d(u, v, (0.055, 0.016), (Rout, 0.066))
            g_t = d - (0.0135 - 0.002 * t)
        elif style == 'split5':
            d1, t1 = seg_d(u, v, (0.04, 0.0), (0.150, 0.0))
            d2, t2 = seg_d(u, v, (0.135, 0.004), (Rout, 0.082))
            g_t = smin(d1 - (0.034 - 0.008 * t1), d2 - (0.0175 - 0.003 * t2), 0.02)
        elif style == 'multi10':
            d, t = seg_d(u, v, (0.05, 0.0), (Rout, 0.0))
            g_t = d - (0.0125 - 0.003 * t)
        else:  # 'twin6'
            d, t = seg_d(u, v, (0.055, 0.014), (Rout, 0.052))
            g_t = d - (0.012 - 0.002 * t)
        spokes = smax(smax(g_t, X - xf, 0.007), (xf - depth) - X, 0.007)
        # keep spokes inside the barrel radius
        spokes = np.maximum(spokes, r - (R_rim - 0.016))
        hub = sdf.sd_cyl_x(X, Y, Z, 0.0, 0.0, r_hub, x_hub - 0.055, x_hub + 0.006, 0.008)
        f = smin(spokes, hub, 0.016)
        if style == 'multi10':
            nut = sdf.sd_cyl_x(X, Y, Z, 0.0, 0.0, 0.034, x_hub, x_hub + 0.030, 0.006)
            f = smin(f, nut, 0.004)
            ring = np.sqrt((np.sqrt(Y * Y + Z * Z) - 0.058) ** 2 + (X - (x_hub + 0.006)) ** 2) - 0.004
            f = smax(f, -ring, 0.002)
        else:
            cap = sdf.sd_cyl_x(X, Y, Z, 0.0, 0.0, 0.030, x_hub, x_hub + 0.012, 0.006)
            f = smin(f, cap, 0.006)
            # lug holes (5)
            s5 = 2 * np.pi / 5
            t5 = np.mod(th + 0.5 * s5 + 0.5 * s5, s5) - 0.5 * s5
            hu = r * np.cos(t5) - 0.054
            hv = r * np.sin(t5)
            hole = np.maximum(np.sqrt(hu * hu + hv * hv) - 0.0105, (x_hub - 0.012) - X)
            f = smax(f, -hole, 0.003)
        return f

    return fn, x_hub


def build_rim_face(R_rim, w, style, concave, lod=0):
    fn, x_hub = rim_face_field(R_rim, w, style, concave)
    hw = 0.5 * w
    h = 0.004 if lod == 0 else 0.008
    bmin = (-(R_rim + 0.01), -(R_rim + 0.01), -(R_rim + 0.01))
    bmax = (R_rim + 0.01, R_rim + 0.01, R_rim + 0.01)
    # grid in x only needs the rim width
    F, origin = sdf.eval_grid(fn, (-hw - 0.01, bmin[1], bmin[2]), (hw + 0.01, bmax[1], bmax[2]), h, symmetric=False)
    V, Q = sdf.surface_nets(F, origin, h)
    V = sdf.project(fn, V, iters=3, max_step=0.75 * h)
    target = 4300 if lod == 0 else 650
    V, T = bl.decimate(V, Q, target_tris=target)
    V = sdf.project(fn, V, iters=2, max_step=0.004)
    V, T = meshops.compact(V, T)[:2]
    N = sdf.normals(fn, V, eps=1e-3)
    ao = sdf_ao(fn, V, N, dists=(0.006, 0.014, 0.03, 0.06), strength=1.3, floor=0.25, cone=False)
    # darken toward the inboard side (inside the barrel)
    ao *= np.clip(0.55 + 0.45 * (V[:, 0] - (x_hub - 0.05)) / 0.05, 0.55, 1.0)
    return piece(V, T, N, 'rim', ao), x_hub


def build_caliper(R_disc, x_d, ang_deg, lod=0):
    """Wheel-local caliper for a LEFT wheel. ang: position angle measured from +Y toward +Z."""
    phi = np.radians(ang_deg)
    r_mid = R_disc - 0.036
    hr, ht, hx = 0.052, 0.088, 0.050

    def fn(X, Y, Z):
        r = np.sqrt(Y * Y + Z * Z)
        th = np.arctan2(Z, Y)
        dth = np.mod(th - phi + np.pi, 2 * np.pi) - np.pi
        g_r = np.abs(r - r_mid) - hr
        g_t = np.abs(dth) * r_mid - ht
        g_x = np.abs(X - x_d) - hx
        f = smax(smax(g_r, g_t, 0.022), g_x, 0.016)
        # slot for the disc on the inner radius side
        slot = np.maximum(np.abs(X - x_d) - 0.019, r - (R_disc + 0.004))
        f = smax(f, -slot, 0.004)
        return f

    cy, cz = r_mid * np.cos(phi), r_mid * np.sin(phi)
    ext = 0.13
    h = 0.005 if lod == 0 else 0.01
    V, Q = sdf.mesh_sdf(fn, (x_d - 0.07, cy - ext, cz - ext), (x_d + 0.07, cy + ext, cz + ext), h, symmetric=False)
    target = 320 if lod == 0 else 44
    V, T = bl.decimate(V, Q, target_tris=target)
    V = sdf.project(fn, V, iters=2, max_step=0.004)
    V, T = meshops.compact(V, T)[:2]
    N = sdf.normals(fn, V, eps=1e-3)
    return piece(V, T, N, 'caliper', np.full(len(V), 0.8))


def build_wheel(R, w, style, concave, grooves, lod=0):
    """Returns (wheel_piece (tyre+rim+disc), dict of handy numbers). Wheel-local, LEFT side."""
    R_rim = R - (0.082 if R < 0.345 else 0.088)
    tire = build_tire(R, R_rim, w, grooves, lod)
    barrel = build_barrel(R_rim, w, lod)
    face, x_hub = build_rim_face(R_rim, w, style, concave, lod)
    R_disc = R_rim - 0.048
    x_d = x_hub - 0.082
    disc = build_disc(R_disc, x_d, lod)
    wheel = meshops.merge_meshes([tire, barrel, face, disc])
    return wheel, dict(R_rim=R_rim, R_disc=R_disc, x_d=x_d, x_hub=x_hub)


# =============================================================================== car base
class CarBase:
    """Parametric implicit car. Subclasses set numbers/profiles in setup() and may override
    side_detail(), add_volumes(), cut_volumes(), paint(), parts(), interior()."""

    id = 'base'
    name = 'BASE'
    # wheels
    axle_f = 1.35; axle_r = -1.35
    track_f = 1.68; track_r = 1.66
    rf = 0.335; rr = 0.350
    wf = 0.25; wr = 0.31
    gap_f = 0.050; gap_r = 0.055          # tyre-to-arch gap
    rim_style = 'twin5'
    concave_f = 0.035; concave_r = 0.065
    # hull blends
    k_shoulder = 0.10; k_nose = 0.09; k_tail = 0.06; k_sill = 0.05
    k_canopy = 0.035; k_roofrail = 0.06
    k_well = 0.018
    fender_x0 = 0.40; fender_x1 = 0.72
    hood_crown = 0.04
    lean_up = 0.08; lean_pow = 2.2
    tuck = 0.055; tuck_pow = 2.0
    arch_lip = 0.010; arch_lip_off = 0.028; arch_lip_w = 0.024
    tumble = 0.47; belt_ref = 0.80; canopy_drop = 0.16
    half_w = 0.99
    z_nose = 2.34; z_tail = -2.28
    nose_plan = 0.34; nose_pow = 2.5; nose_vert = 0.9; nose_tip_y = 0.33
    tail_plan = 0.26; tail_pow = 3.0; tail_tuck = 0.10; tail_tuck_y = (0.52, 0.22)
    height = 1.19

    def __init__(self):
        self.setup()
        self.well_rf = self.rf + self.gap_f
        self.well_rr = self.rr + self.gap_r
        self.well_xin_f = 0.5 * self.track_f - 0.5 * self.wf - 0.085
        self.well_xin_r = 0.5 * self.track_r - 0.5 * self.wr - 0.075

    def setup(self):
        raise NotImplementedError

    # ---------------------------------------------------------------- hull pieces
    def top_height(self, aX, Z):
        Hc = self.p_hood(Z)
        Hf = self.p_fender(Z)
        x0 = self.fender_x0(Z) if callable(self.fender_x0) else self.fender_x0
        x1 = self.fender_x1(Z) if callable(self.fender_x1) else self.fender_x1
        t = sstep(x0, x1, aX)
        return Hc + (Hf - Hc) * t - self.hood_crown * aX * aX

    def arch_lips(self, Y, Z):
        out = 0.0
        if self.arch_lip <= 0:
            return out
        for zc, yc, R in ((self.axle_f, self.rf, self.well_rf), (self.axle_r, self.rr, self.well_rr)):
            rho = np.sqrt((Z - zc) ** 2 + (Y - yc) ** 2)
            out = out + self.arch_lip * gauss(rho, R + self.arch_lip_off, self.arch_lip_w)
        return out

    def side_detail(self, Y, Z):
        return 0.0

    def side_width(self, Y, Z):
        Wb = self.p_width(Z)
        yw = self.p_ywide(Z)
        crest = self.p_fender(Z)
        floor = self.p_floor(Z)
        up = np.clip((Y - yw) / np.maximum(crest - yw, 0.08), 0.0, 1.6)
        dn = np.clip((yw - Y) / np.maximum(yw - floor, 0.08), 0.0, 1.6)
        W = Wb - self.lean_up * up ** self.lean_pow - self.tuck * dn ** self.tuck_pow
        return W + self.arch_lips(Y, Z) + self.side_detail(Y, Z)

    def front_face(self, aX, Y):
        return self.z_nose - self.nose_plan * (aX / self.half_w) ** self.nose_pow - self.nose_vert * (Y - self.nose_tip_y) ** 2

    def rear_face(self, aX, Y):
        return (self.z_tail + self.tail_plan * (aX / self.half_w) ** self.tail_pow
                + self.tail_tuck * sstep(self.tail_tuck_y[0], self.tail_tuck_y[1], Y))

    def hull(self, X, Y, Z):
        aX = np.abs(X)
        g_top = Y - self.top_height(aX, Z)
        g_side = aX - self.side_width(Y, Z)
        f = smax(g_top, g_side, self.k_shoulder)
        f = smax(f, Z - self.front_face(aX, Y), self.k_nose)
        f = smax(f, self.rear_face(aX, Y) - Z, self.k_tail)
        f = smax(f, self.p_floor(Z) - Y, self.k_sill)
        return f

    def canopy_ts(self, X, Y, Z):
        """> 0 where the canopy top surface (windshield / roof) dominates, < 0 on the side glass."""
        aX = np.abs(X)
        g_top = Y - (self.p_roof(Z) - self.p_crown(Z) * aX * aX)
        g_side = aX - (self.p_cwidth(Z) - self.tumble * (Y - self.belt_ref))
        return g_top - g_side

    def greenhouse(self, P):
        """Field < 0 on the canopy surface standing proud of the hull."""
        X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        return np.maximum(0.004 - self.hull(X, Y, Z), self.canopy(X, Y, Z) - 0.02)

    def inlay(self, skin, fn, depth, fine, mat='trim_black', only=('paint',), refine=0.024, ramp=0.012):
        """Flush dark panel (lamp housing / vent / grille) with a clean outline, sunk by `depth`."""
        fld = lambda P: fn(P[:, 0], P[:, 1], P[:, 2])
        skin.region(fld, mat, only=list(only), refine=refine if fine else None)
        if fine and depth > 0:
            skin.recess(fld, depth, ramp=ramp, mats=[mat, 'light_head', 'light_tail', 'light_brake'], refine=refine)

    def canopy(self, X, Y, Z):
        aX = np.abs(X)
        g_top = Y - (self.p_roof(Z) - self.p_crown(Z) * aX * aX)
        g_side = aX - (self.p_cwidth(Z) - self.tumble * (Y - self.belt_ref))
        f = smax(g_top, g_side, self.k_roofrail)
        # keep the canopy solid inside the hull envelope (it is an infinite prism otherwise)
        f = smax(f, (self.belt_ref - self.canopy_drop) - Y, 0.02)
        f = smax(f, (self.rear_face(aX, Y) + 0.04) - Z, self.k_tail)
        f = smax(f, Z - (self.front_face(aX, Y) - 0.04), self.k_nose)
        return f

    def add_volumes(self, f, X, Y, Z):
        return f

    def body0(self, X, Y, Z):
        f = smin(self.hull(X, Y, Z), self.canopy(X, Y, Z), self.k_canopy)
        return self.add_volumes(f, X, Y, Z)

    def wells(self, X, Y, Z):
        aX = np.abs(X)
        wf = sdf.sd_cyl_x(aX, Y, Z, self.rf, self.axle_f, self.well_rf, self.well_xin_f, 3.0)
        wr = sdf.sd_cyl_x(aX, Y, Z, self.rr, self.axle_r, self.well_rr, self.well_xin_r, 3.0)
        return np.minimum(wf, wr)

    def cut_list(self):
        """List of cuts: dict(fn=field(X,Y,Z) negative inside the removed volume, k=blend,
        mat=material of the exposed interior, margin=how far the paint wraps into the lip,
        inset=depth (optional: carve a uniform-depth pocket inside the prism fn instead))."""
        return []

    def cut_volumes(self, f, f0, X, Y, Z):
        for c in self.cut_list():
            g = c['fn'](X, Y, Z)
            if c.get('inset'):
                f = self.inset(f, g, c['inset'], c.get('k', 0.012))
            else:
                f = smax(f, -g, c.get('k', 0.02))
        return f

    def body(self, X, Y, Z):
        f0 = self.body0(X, Y, Z)
        f = smax(f0, -self.wells(X, Y, Z), self.k_well)
        return self.cut_volumes(f, f0, X, Y, Z)

    @staticmethod
    def inset(f, region, depth, k=0.012):
        """Carve a pocket of uniform depth below the current surface inside region<0."""
        return smax(f, np.minimum(-region, f + depth), k)

    def bounds(self):
        return (-(self.half_w + 0.05), -0.02, self.z_tail - 0.06), (self.half_w + 0.05, self.height + 0.05, self.z_nose + 0.06)

    # ---------------------------------------------------------------- painting
    def paint(self, skin, lod):
        """Assign materials. Base: underbody + wheel wells + cut interiors."""
        self.paint_base(skin, lod)

    def paint_base(self, skin, lod):
        fl = lambda P: P[:, 1] - (self.p_floor(P[:, 2]) + 0.014)
        skin.region(fl, 'underbody', only=['paint'], guard=False)
        wl = lambda P: self.wells(P[:, 0], P[:, 1], P[:, 2]) - 0.007
        skin.region(wl, 'underbody', only=['paint'], refine=(0.03 if lod == 0 else None), guard=False)
        for c in self.cut_list():
            m = c.get('margin', 0.3 * c.get('k', 0.02))
            fld = c['fn']
            skin.region(lambda P, fld=fld, m=m: fld(P[:, 0], P[:, 1], P[:, 2]) - m, c.get('mat', 'trim_black'),
                        only=['paint'], refine=(c.get('refine', 0.03) if lod == 0 else None), guard=False)

    def hidden_mask(self, P):
        """True for skin points in low-priority areas (underbody, wheel wells, cut interiors)."""
        X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        h = Y < self.p_floor(Z) + 0.014
        h |= self.wells(X, Y, Z) < 0.004
        for c in self.cut_list():
            if not c.get('inset'):
                h |= c['fn'](X, Y, Z) < 0.003
        return h

    def parts(self, lod):
        return []

    def decals(self, dec, lod):
        pass

    def interior(self, lod):
        return []

    # ---------------------------------------------------------------- AO scene
    def ao_scene(self, extra=None):
        def fn(X, Y, Z):
            f = self.body(X, Y, Z)
            aX = np.abs(X)
            wf = sdf.sd_cyl_x(aX, Y, Z, self.rf, self.axle_f, self.rf, 0.5 * self.track_f - 0.5 * self.wf, 0.5 * self.track_f + 0.5 * self.wf)
            wr = sdf.sd_cyl_x(aX, Y, Z, self.rr, self.axle_r, self.rr, 0.5 * self.track_r - 0.5 * self.wr, 0.5 * self.track_r + 0.5 * self.wr)
            f = np.minimum(f, np.minimum(wf, wr))
            f = np.minimum(f, Y + 0.0)
            if extra is not None:
                f = np.minimum(f, extra(X, Y, Z))
            return f
        return fn

    # ---------------------------------------------------------------- meta
    def points(self):
        return {}


def build_body(car, lod=0, draft=False, ref=None):
    t0 = time.time()
    bmin, bmax = car.bounds()
    if draft:
        h, target = 0.016, None
    elif lod == 0:
        h, target = 0.010, getattr(car, 'skin_tris', 36000)
    else:
        h, target = 0.016, getattr(car, 'skin_tris_lod1', 5600)
    V, Q = sdf.mesh_sdf(car.body, bmin, bmax, h, symmetric=True)
    log(f'body nets: {len(V)} verts {len(Q)} quads h={h} ({time.time() - t0:.1f}s)')
    if target is not None and lod > 0:
        V, T = bl.decimate(V, Q, target_tris=target)
        V = sdf.project(car.body, V, iters=3, max_step=0.01)
        log(f'decimated -> {len(T)} tris ({time.time() - t0:.1f}s)')
    elif target is not None:
        n_hidden = 7000
        n_canopy = getattr(car, 'canopy_tris', 4500)
        canopy_zone = lambda P: car.hull(P[:, 0], P[:, 1], P[:, 2]) > 0.004
        V, T = bl.decimate_zones(V, Q, [(car.hidden_mask, n_hidden), (canopy_zone, n_canopy)], target)
        V = sdf.project(car.body, V, iters=3, max_step=0.01)
        log(f'decimated -> {len(T)} tris ({time.time() - t0:.1f}s)')
    else:
        T = sdf.quads_to_tris(V, Q)
    skin = Skin(V, T, car.body, guard_cap=(0.06 if lod == 0 else None))
    if not draft:
        vis = lambda sk: ~car.hidden_mask(sk.V[sk.T].mean(axis=1))
        skin.limit_edges(0.20 if lod == 0 else 0.30, tri_filter=vis)
        log(f'edge-limited -> {len(skin.T)} tris')
    car.paint(skin, lod)
    if not draft and lod == 0:
        n0 = len(skin.T)
        budget = getattr(car, 'skin_budget', 106000) if lod == 0 else getattr(car, 'skin_budget_lod1', 10800)
        reached = None
        for dev in ((20.0, 16.0, 13.0, 11.0, 9.5, 8.5, 7.5, 6.5) if lod == 0 else (40.0, 30.0, 24.0, 19.0)):
            if len(skin.T) >= budget:
                break
            skin.refine_curvature(dev, 0.014 if lod == 0 else 0.03, only=['paint'], max_tris=budget)
            reached = dev
        log(f'curvature-refined paint: +{len(skin.T) - n0} tris (down to {reached} deg, budget {budget})')
    out = skin.finish()
    log(f'skin painted: {len(out["T"])} tris ({time.time() - t0:.1f}s)')
    return out


def assemble(car, lod=0, draft=False, ref=None):
    """Build every piece and create the Blender objects following the GLB contract.
    ref: finished LOD0 skin piece (materials are transferred from it for lower LODs).
    Returns (objects, stats, merged body, skin piece)."""
    t0 = time.time()
    body_skin = build_body(car, lod, draft, ref)
    pieces = [body_skin]
    dec = Decals(car, body_skin)
    car.decals(dec, lod)
    pieces += dec.pieces
    log(f'decals: {sum(len(p["T"]) for p in dec.pieces)} tris in {len(dec.pieces)} pieces')
    car.part_fields = []
    pieces += car.parts(lod)
    extra = None
    if car.part_fields:
        fields = list(car.part_fields)

        def extra(X, Y, Z):
            f = fields[0](X, Y, Z)
            for g in fields[1:]:
                f = np.minimum(f, g(X, Y, Z))
            return f
    scene = car.ao_scene(extra)
    for p in pieces:
        if p.get('C') is None:
            p['C'] = sdf_ao(scene, p['V'], p['N'])
            if p.get('ao_scale') is not None:
                p['C'] = p['C'] * p['ao_scale']
            # glass / lamps stay clean
            mats_clean = np.isin(p['M'], [MID['glass'], MID['light_head'], MID['light_tail'], MID['light_brake']])
            if mats_clean.any():
                vclean = np.zeros(len(p['V']), dtype=bool)
                vclean[p['T'][mats_clean].ravel()] = True
                p['C'] = np.where(vclean, np.maximum(p['C'], 0.9), p['C'])
    n_ext = sum(len(p['T']) for p in pieces)
    interior = car.interior(lod)
    pieces += interior
    log(f'pieces: skin {len(body_skin["T"])} parts {n_ext - len(body_skin["T"])} interior {[len(p["T"]) for p in interior]}')
    body = meshops.merge_meshes(pieces)
    objs = []
    me = bl.new_mesh('body', body['V'], body['T'], normals=body['N'], ao=body['C'], mat_ids=body['M'])
    ob_body = bl.new_object('body', me)
    objs.append(ob_body)
    stats = {'body': len(body['T'])}

    wheel_defs = {
        'f': (car.rf, car.wf, car.concave_f, 3, -72.0),
        'r': (car.rr, car.wr, car.concave_r, 4, 72.0),
    }
    wheel_mesh = {}
    cal_mesh = {}
    for key, (R, w, conc, grooves, cal_ang) in wheel_defs.items():
        wheel, info = build_wheel(R, w, car.rim_style, conc, grooves, lod)
        wheel_mesh[key] = bl.new_mesh(f'wheel_{key}', wheel['V'], wheel['T'], normals=wheel['N'], ao=wheel['C'], mat_ids=wheel['M'])
        stats[f'wheel_{key}'] = len(wheel['T'])
        cal = build_caliper(info['R_disc'], info['x_d'], cal_ang, lod)
        cal_r = transform_piece(cal, mirror_x=True)
        cal_mesh[key + 'l'] = bl.new_mesh(f'caliper_{key}l', cal['V'], cal['T'], normals=cal['N'], ao=cal['C'], mat_ids=cal['M'])
        cal_mesh[key + 'r'] = bl.new_mesh(f'caliper_{key}r', cal_r['V'], cal_r['T'], normals=cal_r['N'], ao=cal_r['C'], mat_ids=cal_r['M'])
        stats[f'caliper_{key}'] = len(cal['T'])
    for name, key, side in (('fl', 'f', 1), ('fr', 'f', -1), ('rl', 'r', 1), ('rr', 'r', -1)):
        R = car.rf if key == 'f' else car.rr
        tr = car.track_f if key == 'f' else car.track_r
        z = car.axle_f if key == 'f' else car.axle_r
        e = bl.new_object(f'wheel_{name}', None, location=(side * 0.5 * tr, R, z))
        e.empty_display_size = 0.1
        m = bl.new_object(f'wheel_{name}_mesh', wheel_mesh[key], parent=e, rot_z=(0.0 if side > 0 else np.pi))
        c = bl.new_object(f'caliper_{name}', cal_mesh[key + ('l' if side > 0 else 'r')], parent=e)
        objs += [e, m, c]
    stats['total'] = stats['body'] + 2 * (stats['wheel_f'] + stats['wheel_r'] + stats['caliper_f'] + stats['caliper_r'])
    log(f'assembled lod{lod}: {stats} ({time.time() - t0:.1f}s)')
    return objs, stats, body, body_skin


# =============================================================================== generic parts
def part_splitter(car, y0, y1, ext=0.04, z_back=1.78, x_max=None, mat='carbon', tris=1000, lod=0):
    x_max = x_max or (car.half_w - 0.02)
    yb, hy = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
    y_ref = float(car.p_floor(car.z_nose)) + 0.03

    def fn(X, Y, Z):
        aX = np.abs(X)
        zf = car.front_face(aX, y_ref) + ext
        g = smax(Z - zf, z_back - Z, 0.012)
        g = smax(g, aX - x_max, 0.03)
        g = smax(g, np.abs(Y - yb) - hy, 0.008)
        return g

    p = sdf_part(fn, (-1.05, y0 - 0.02, z_back - 0.03), (1.05, y1 + 0.02, car.z_nose + ext + 0.03),
                 0.006 if lod == 0 else 0.012, mat, target_tris=tris if lod == 0 else tris // 5, normals='auto')
    return p, fn


def part_skirt(car, z0, z1, y0, y1, out=0.018, depth=0.14, mat='carbon', tris=500, lod=0):
    def fn(X, Y, Z):
        aX = np.abs(X)
        w = car.side_width(y1 + 0.0 * Z, Z) + out
        g = smax(aX - w, (w - depth) - aX, 0.01)
        g = smax(g, np.abs(Y - 0.5 * (y0 + y1)) - 0.5 * (y1 - y0), 0.008)
        g = smax(g, np.abs(Z - 0.5 * (z0 + z1)) - 0.5 * abs(z1 - z0), 0.02)
        return g

    p = sdf_part(fn, (-1.06, y0 - 0.02, min(z0, z1) - 0.03), (1.06, y1 + 0.02, max(z0, z1) + 0.03),
                 0.006 if lod == 0 else 0.012, mat, target_tris=tris if lod == 0 else tris // 4, normals='auto')
    return p, fn


def part_fins(car, xs, z0, z1, y_low0, y_low1, thick=0.014, mat='carbon', tris=900, lod=0, top_fn=None):
    """Vertical diffuser strakes hanging from the hull floor between z0 (front) and z1 (rear)."""
    def fn(X, Y, Z):
        aX = np.abs(X)
        d = np.abs(aX - xs[0])
        for xi in xs[1:]:
            d = np.minimum(d, np.abs(aX - xi))
        g = d - 0.5 * thick
        top = (top_fn(Z) if top_fn is not None else car.p_floor(Z)) + 0.012
        g = smax(g, Y - top, 0.004)
        ylow = y_low0 + (y_low1 - y_low0) * np.clip((z0 - Z) / (z0 - z1), 0.0, 1.0)
        g = smax(g, ylow - Y, 0.006)
        g = smax(g, np.abs(Z - 0.5 * (z0 + z1)) - 0.5 * abs(z0 - z1), 0.008)
        return g

    p = sdf_part(fn, (-(max(xs) + 0.05), min(y_low0, y_low1) - 0.02, min(z0, z1) - 0.03),
                 (max(xs) + 0.05, float((top_fn or car.p_floor)(min(z0, z1))) + 0.04, max(z0, z1) + 0.03),
                 0.005 if lod == 0 else 0.01, mat, target_tris=tris if lod == 0 else tris // 5, normals='auto')
    return p, fn


def part_wing(zc, yc, chord, span_half, thick=0.026, angle_deg=7.0, pylons=(0.42,), deck_y=0.9,
              endplate=0.05, pylon_len=0.06, mat='carbon', tris=1500, lod=0, swan=False, normals='grad'):
    a = np.radians(angle_deg)
    cy, cz = np.sin(a), -np.cos(a)       # chord direction (towards the rear, rising)
    ny, nz = np.cos(a), np.sin(a)
    hc, ht = 0.5 * chord, 0.5 * thick

    def fn(X, Y, Z):
        aX = np.abs(X)
        dy = Y - yc; dz = Z - zc
        v = dy * cy + dz * cz
        w = dy * ny + dz * nz
        # stadium section tapering to the trailing edge
        tt = ht * (1.0 - 0.45 * np.clip(v / hc, 0.0, 1.0))
        qv = np.abs(v) - (hc - ht)
        qw = np.abs(w) - (tt - ht * 0.85)
        blade = np.sqrt(np.maximum(qv, 0.0) ** 2 + np.maximum(qw, 0.0) ** 2) + np.minimum(np.maximum(qv, qw), 0.0) - ht * 0.85
        blade = smax(blade, aX - span_half, 0.008)
        f = blade
        if endplate > 0:
            ep = sdf.sd_box(aX, Y, Z, (span_half, yc - 0.3 * endplate, zc), (0.007, endplate, hc + 0.012), 0.005)
            f = smin(f, ep, 0.006)
        for xp in pylons:
            if swan:
                py = sdf.sd_box(aX, Y, Z, (xp, 0.5 * (deck_y + yc) + 0.035, zc + 0.02),
                                (0.007, 0.5 * (yc - deck_y) + 0.05, pylon_len), 0.005)
            else:
                py = sdf.sd_box(aX, Y, Z, (xp, 0.5 * (deck_y + yc), zc + 0.01),
                                (0.009, 0.5 * (yc - deck_y) + 0.012, pylon_len), 0.006)
            f = smin(f, py, 0.008)
        return f

    ext = span_half + 0.04
    p = sdf_part(fn, (-ext, deck_y - 0.04, zc - hc - 0.06), (ext, yc + max(endplate, 0.02) + 0.09, zc + hc + 0.09),
                 0.004 if lod == 0 else 0.01, mat, target_tris=tris if lod == 0 else tris // 5, normals=normals)
    return p, fn


def part_mirrors(pod_c, pod_r, stalk_a, stalk_b, stalk_r=0.016, lod=0, pod_mat='paint'):
    def pod(X, Y, Z):
        # rounded wedge: box with generous rounding, tapering towards the front (+Z)
        aX = np.abs(X)
        t = np.clip((Z - pod_c[2]) / pod_r[2], -1.0, 1.0)
        sc = 1.0 - 0.28 * np.maximum(t, 0.0)
        return sdf.sd_box(pod_c[0] + (aX - pod_c[0]) / sc, pod_c[1] + (Y - pod_c[1]) / sc, Z, pod_c, pod_r, min(pod_r) * 0.75) * 0.8

    def stalk(X, Y, Z):
        return sdf.sd_capsule(np.abs(X), Y, Z, stalk_a, stalk_b, stalk_r)

    m = max(pod_r) + 0.03
    h = 0.005 if lod == 0 else 0.012
    p1 = sdf_part(pod, (-(pod_c[0] + m), pod_c[1] - m, pod_c[2] - m), (pod_c[0] + m, pod_c[1] + m, pod_c[2] + m), h,
                  pod_mat, target_tris=440 if lod == 0 else 80)
    lo = np.minimum(stalk_a, stalk_b) - 0.04
    hi = np.maximum(stalk_a, stalk_b) + 0.04
    p2 = sdf_part(stalk, (-hi[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]), h, 'trim_black', target_tris=160 if lod == 0 else 40)
    fn = lambda X, Y, Z: np.minimum(pod(X, Y, Z), stalk(X, Y, Z))
    return [p1, p2], fn


def part_exhaust(x, y, z_tip, length=0.2, r_out=0.048, r_in=0.040, segs=24, lod=0):
    """Round tips pointing rearward (-Z). Mirrored to both sides when x != 0."""
    segs = segs if lod == 0 else 10
    z_in = z_tip + length
    prof = [(z_in - 0.02, 0.0), (z_in - 0.02, r_in), (z_tip, r_in), (z_tip, r_in), (z_tip, r_out), (z_tip, r_out), (z_in, r_out)]
    V, T, N = meshops.lathe(prof, segs, axis='z')
    ao = np.where(np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2) < r_in + 1e-4, np.clip((z_in - V[:, 2]) / length, 0.0, 1.0) ** 2 * 0.6 + 0.08, 1.0)
    V = V + np.array([x, y, 0.0])[None, :]
    p = piece(V, T, N, 'metal', ao)
    out = [p]
    if abs(x) > 1e-6:
        out.append(transform_piece(p, mirror_x=True))
    return out


def build_interior(car, z_front, z_rear, y_floor, seat_x, seat_z, glass_test, lod=0, wheel_side=1, offset=0.012):
    """Dark cabin tub (inner lining with window openings) + seats + dash + steering wheel."""
    pieces = []
    h = 0.02 if lod == 0 else 0.035

    def tub(X, Y, Z):
        f = car.body0(X, Y, Z) + offset
        f = np.maximum(f, y_floor - Y)
        f = smax(f, Z - z_front, 0.03)
        f = smax(f, z_rear - Z, 0.03)
        return f

    bmin, bmax = car.bounds()
    V, Q = sdf.mesh_sdf(tub, (bmin[0], y_floor - 0.04, z_rear - 0.05), (bmax[0], bmax[1], z_front + 0.05), h, symmetric=True)
    V, T = bl.decimate(V, Q, target_tris=900 if lod == 0 else 300, symmetry=False)
    V = sdf.project(tub, V, iters=2, max_step=0.01)
    skin = Skin(V, T, tub, 'interior')
    # window openings: evaluate the glass fields at the matching point of the outer skin
    # (level sets of body0 are parallel, so stepping along its gradient lands on the skin)
    def gfield(P):
        n = sdf.normals(car.body0, P)
        return glass_test(P + n * offset)
    skin.refine(gfield, 0.10 if lod == 0 else 0.16)
    skin.cut(gfield)
    cen = skin.V[skin.T].mean(axis=1)
    skin.delete(gfield(cen) < 0)
    tubp = skin.finish(flip=True)
    yy = tubp['V'][:, 1]
    tubp['C'] = np.clip(0.25 + 0.5 * (yy - y_floor) / max(car.height - y_floor, 0.1), 0.2, 0.7)
    tubp.pop('ao_scale', None)
    pieces.append(tubp)
    if lod > 0:
        return pieces

    sx = seat_x
    def seats(X, Y, Z):
        aX = np.abs(X)
        cush = sdf.sd_box(aX, Y, Z, (sx, y_floor + 0.11, seat_z + 0.05), (0.235, 0.07, 0.26), 0.05)
        # backrest leaning back ~20 deg
        a = np.radians(20.0)
        dy = Y - (y_floor + 0.16); dz = Z - (seat_z - 0.19)
        v = dy * np.cos(a) - dz * np.sin(a)       # along the backrest (up)
        w = dy * np.sin(a) + dz * np.cos(a)       # thickness direction
        back = sdf.sd_box(aX - sx, v, w, (0.0, 0.36, 0.0), (0.235, 0.40, 0.055), 0.05)
        bol = sdf.sd_box(np.abs(aX - sx), v, w, (0.215, 0.25, 0.06), (0.035, 0.26, 0.07), 0.03)
        head = sdf.sd_box(aX - sx, v, w, (0.0, 0.70, 0.0), (0.14, 0.11, 0.06), 0.05)
        f = smin(smin(cush, back, 0.04), smin(bol, head, 0.04), 0.04)
        return f

    def dash(X, Y, Z):
        aX = np.abs(X)
        top = sdf.sd_box(aX, Y, Z, (0.0, y_floor + 0.53, z_front - 0.20), (0.78, 0.10, 0.22), 0.07)
        tunnel = sdf.sd_box(aX, Y, Z, (0.0, y_floor + 0.14, 0.5 * (z_front + z_rear) + 0.1), (0.105, 0.15, 0.5 * (z_front - z_rear) - 0.12), 0.05)
        binn = sdf.sd_ellipsoid(X, Y, Z, (wheel_side * sx, y_floor + 0.63, z_front - 0.33), (0.19, 0.075, 0.16))
        return smin(smin(top, tunnel, 0.05), binn, 0.04)

    wc = np.array([wheel_side * sx, y_floor + 0.58, z_front - 0.50])
    ta = np.radians(24.0)
    axis = np.array([0.0, np.sin(ta), -np.cos(ta)])     # wheel plane normal: towards the driver, slightly up

    def swheel(X, Y, Z):
        px = X - wc[0]; py = Y - wc[1]; pz = Z - wc[2]
        ax = px * axis[0] + py * axis[1] + pz * axis[2]
        rx = px - ax * axis[0]; ry = py - ax * axis[1]; rz = pz - ax * axis[2]
        rad = np.sqrt(rx * rx + ry * ry + rz * rz)
        ring = np.sqrt((rad - 0.175) ** 2 + ax * ax) - 0.017
        hubd = np.sqrt(np.maximum(rad - 0.05, 0.0) ** 2 + np.maximum(np.abs(ax + 0.02) - 0.03, 0.0) ** 2) - 0.012
        # 3 spokes: horizontal pair + one down
        spoke_h = np.sqrt(ry * ry + rz * rz + ax * ax) - 0.014 + 1e3 * (np.abs(rx) > 0.175)
        down = np.sqrt(rx * rx + ax * ax) - 0.014 + 1e3 * ((py > 0) | (rad > 0.175))
        col = sdf.sd_capsule(X, Y, Z, wc - axis * 0.02, wc - axis * 0.30, 0.03)
        f = smin(smin(ring, hubd, 0.01), np.minimum(spoke_h, down), 0.01)
        return smin(f, col, 0.01)

    pieces.append(sdf_part(seats, (-(sx + 0.3), y_floor, seat_z - 0.55), (sx + 0.3, y_floor + 0.95, seat_z + 0.36), 0.012, 'interior', target_tris=1500))
    pieces.append(sdf_part(dash, (-0.82, y_floor - 0.02, z_rear + 0.05), (0.82, y_floor + 0.75, z_front + 0.05), 0.014, 'interior',
                           target_tris=700, symmetric=False))
    pieces.append(sdf_part(swheel, (wc[0] - 0.22, wc[1] - 0.24, wc[2] - 0.34), (wc[0] + 0.22, wc[1] + 0.24, wc[2] + 0.12), 0.006, 'interior',
                           target_tris=520, symmetric=False))
    for p in pieces[1:]:
        yy = p['V'][:, 1]
        p['C'] = np.clip(0.3 + 0.6 * (yy - y_floor) / 0.8, 0.25, 0.85)
    return pieces


# =============================================================================== metadata
def metadata(car, body, tris, skin=None):
    V = body['V']
    bmin = V.min(axis=0); bmax = V.max(axis=0)
    sv = skin['V'] if skin is not None else V
    skin_w = float(2.0 * np.abs(sv[:, 0]).max())
    roof = float(sv[:, 1].max())
    r3 = lambda a: [round(float(v), 4) for v in a]
    pts = car.points()
    meta = {
        'id': car.id,
        'displayName': car.name,
        'length': round(float(bmax[2] - bmin[2]), 3),
        'width': round(skin_w, 3),
        'widthWithMirrors': round(float(2.0 * max(abs(bmin[0]), abs(bmax[0]))), 3),
        'height': round(float(bmax[1]), 3),
        'roofHeight': round(roof, 3),
        'wheelbase': round(float(car.axle_f - car.axle_r), 3),
        'axleFrontZ': round(float(car.axle_f), 4),
        'axleRearZ': round(float(car.axle_r), 4),
        'trackFront': round(float(car.track_f), 4),
        'trackRear': round(float(car.track_r), 4),
        'wheelRadiusFront': round(float(car.rf), 4),
        'wheelRadiusRear': round(float(car.rr), 4),
        'wheelWidthFront': round(float(car.wf), 4),
        'wheelWidthRear': round(float(car.wr), 4),
        'rideHeight': round(float(car.p_floor(0.0)), 4),
        'bodyBox': {'min': r3(bmin), 'max': r3(bmax)},
        'points': {k: r3(v) for k, v in pts.items()},
        'triangles': tris,
        'defaultPaint': [round(float(v), 3) for v in getattr(car, 'paint_color', (0.55, 0.03, 0.03))],
    }
    return meta


# =============================================================================== decals
class Decals:
    """Thin features (panel seams, LED strips, lamp dots) as ribbons / discs draped a hair above the
    finished skin instead of being cut into it: costs a few hundred triangles instead of thousands
    and leaves the paint tessellation undisturbed."""

    def __init__(self, car, skin_piece):
        from mathutils.bvhtree import BVHTree
        self.car = car
        self.V = skin_piece['V']
        self.T = skin_piece['T']
        self.M = skin_piece['M']
        self.bvh = BVHTree.FromPolygons([tuple(v) for v in self.V.tolist()], [tuple(t) for t in self.T.tolist()])
        self.pieces = []

    # ---- ray helpers
    def cast(self, origin, direction, dist=10.0):
        from mathutils import Vector
        loc, nrm, idx, d = self.bvh.ray_cast(Vector(origin), Vector(direction), dist)
        if loc is None:
            return None
        return np.array(loc), int(idx)

    def project_2d(self, pts2d, view, side=1):
        """Project 2D points onto the skin. view: 'side' (pts = (z, y), ray along x from `side`),
        'top' (pts = (x, z), ray from above), 'rear' / 'front' (pts = (x, y))."""
        out = []
        for a, b in pts2d:
            if view == 'side':
                o, d = (1.6 * side, b, a), (-1.0 * side, 0.0, 0.0)
            elif view == 'top':
                o, d = (a, 2.0, b), (0.0, -1.0, 0.0)
            elif view == 'rear':
                o, d = (a, b, self.car.z_tail - 1.0), (0.0, 0.0, 1.0)
            else:
                o, d = (a, b, self.car.z_nose + 1.0), (0.0, 0.0, -1.0)
            out.append(self.cast(o, d))
        return out

    def normal_at(self, P):
        return sdf.normals(self.car.body, np.atleast_2d(P))

    def drape(self, Q, Nq, eps):
        """Snap points onto the skin along their normal and lift them by eps. Returns (P, ok)."""
        P = np.array(Q, dtype=np.float64)
        ok = np.ones(len(Q), dtype=bool)
        for i in range(len(Q)):
            h = self.cast(tuple(Q[i] + Nq[i] * 0.03), tuple(-Nq[i]), 0.08)
            if h is None:
                ok[i] = False
            else:
                P[i] = h[0] + Nq[i] * eps
        return P, ok

    # ---- builders
    def ribbon(self, hits, half_width, mat, eps=0.001, only=None, closed=False):
        """hits: list of (pos, tri_index) or None (from project_2d) along a path."""
        only_ids = None if only is None else [MID[o] for o in only]
        runs, cur = [], []
        for h in hits:
            good = h is not None and (only_ids is None or self.M[h[1]] in only_ids)
            if good:
                cur.append(h[0])
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        if closed and len(runs) == 1 and len(runs[0]) == len(hits):
            runs[0] = runs[0] + [runs[0][0]]
        for run in runs:
            if len(run) < 2:
                continue
            C = np.array(run)
            Nc = self.normal_at(C)
            tang = np.zeros_like(C)
            tang[1:-1] = C[2:] - C[:-2]
            tang[0] = C[1] - C[0]
            tang[-1] = C[-1] - C[-2]
            if closed and len(runs) == 1:
                tang[0] = tang[-1] = C[1] - C[-2]
            tang -= Nc * (tang * Nc).sum(axis=1)[:, None]
            tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-12)[:, None]
            B = np.cross(Nc, tang)
            L, okl = self.drape(C + B * half_width, Nc, eps)
            R, okr = self.drape(C - B * half_width, Nc, eps)
            n = len(C)
            V = np.vstack([L, R])
            Nn = np.vstack([Nc, Nc])
            tris = []
            for i in range(n - 1):
                if not (okl[i] and okr[i] and okl[i + 1] and okr[i + 1]):
                    continue
                tris.append((i, n + i, n + i + 1))
                tris.append((i, n + i + 1, i + 1))
            if not tris:
                continue
            T = np.array(tris, dtype=np.int32)
            gn = meshops.tri_normals(V, T)
            flip = (gn * Nn[T[:, 0]]).sum(axis=1) < 0
            T[flip] = T[flip][:, ::-1]
            self.pieces.append(piece(V, T, Nn, mat, np.ones(len(V))))

    def disc(self, centre, radius, mat, eps=0.0015, segs=14):
        c = np.asarray(centre, dtype=np.float64)
        n = self.normal_at(c)[0]
        a = np.cross(n, [0.0, 1.0, 0.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(n, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        ang = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        ring = c[None, :] + radius * (np.cos(ang)[:, None] * a[None, :] + np.sin(ang)[:, None] * b[None, :])
        Q = np.vstack([c[None, :], ring])
        Nq = np.repeat(n[None, :], len(Q), axis=0)
        P, ok = self.drape(Q, Nq, eps)
        if not ok.all():
            return
        T = np.array([(0, 1 + i, 1 + (i + 1) % segs) for i in range(segs)], dtype=np.int32)
        gn = meshops.tri_normals(P, T)
        flip = (gn * Nq[T[:, 0]]).sum(axis=1) < 0
        T[flip] = T[flip][:, ::-1]
        self.pieces.append(piece(P, T, Nq, mat, np.ones(len(P))))


def outline_points(poly, round_r=0.0, step=0.02, closed=True):
    """Sample the outline of a 2D polygon (optionally with rounded corners) every `step`."""
    P = np.asarray(poly, dtype=np.float64)
    n = len(P)
    pts = []
    for i in range(n):
        if not closed and i == n - 1:
            pts.append(P[i])
            break
        a = P[i]; b = P[(i + 1) % n]
        if round_r > 0 and closed:
            prev = P[i - 1]; nxt = P[(i + 2) % n]
            # trim the segment ends by the corner radius (corner arcs approximated by 3 points)
            d = (b - a) / np.linalg.norm(b - a)
            a2 = a + d * round_r; b2 = b - d * round_r
            seg_len = np.linalg.norm(b2 - a2)
            m = max(int(np.ceil(seg_len / step)), 1)
            for k in range(m + 1):
                pts.append(a2 + (b2 - a2) * (k / m))
            # corner arc at b towards the next segment
            d2 = (nxt - b) / np.linalg.norm(nxt - b)
            c0 = b2; c2 = b + d2 * round_r
            for t in (0.25, 0.5, 0.75):
                q = (1 - t) ** 2 * c0 + 2 * (1 - t) * t * b + t ** 2 * c2
                pts.append(q)
        else:
            seg_len = np.linalg.norm(b - a)
            m = max(int(np.ceil(seg_len / step)), 1)
            for k in range(m):
                pts.append(a + (b - a) * (k / m))
    return np.array(pts)


def polyline_points(pts, step=0.02):
    P = np.asarray(pts, dtype=np.float64)
    out = []
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        m = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        for k in range(m):
            out.append(a + (b - a) * (k / m))
    out.append(P[-1])
    return np.array(out)
