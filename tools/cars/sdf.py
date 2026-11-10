"""Tiny numpy implicit-surface (SDF) modelling kernel used to sculpt the car bodies.

Conventions: all functions take broadcastable arrays X, Y, Z in *game/GLB space*
(+X = car left, +Y = up, +Z = forward/nose, metres) and return a field that is
negative inside the solid.  Fields are distance-like (|grad| ~ 1) near the surface.
"""
import numpy as np


# ----------------------------------------------------------------------------- blends
def smin(a, b, k):
    """C2 (cubic) smooth minimum = smooth union with blend size k (metres)."""
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) * (1.0 / k)
    return np.minimum(a, b) - h * h * h * (k / 6.0)


def smax(a, b, k):
    """C2 smooth maximum = smooth intersection (use smax(a, -b, k) to subtract b)."""
    if k <= 0.0:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) * (1.0 / k)
    return np.maximum(a, b) + h * h * h * (k / 6.0)


def sstep(e0, e1, x):
    """Smootherstep (C2). Works for e0 > e1 too (falling edge)."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    return a + (b - a) * t


def gauss(x, c, w):
    u = (x - c) / w
    return np.exp(-u * u)


def window(x, lo, hi, soft):
    """Smooth 0..1 window that is 1 inside [lo, hi] with soft edges of width `soft`."""
    return sstep(lo - soft, lo + soft, x) * (1.0 - sstep(hi - soft, hi + soft, x))


# ----------------------------------------------------------------------------- profiles
class Profile:
    """Fair 1D function v(u) given by a control polygon [(u, v), ...].

    Clamped uniform cubic B-spline (C2, no overshoot, passes through the first and
    last control points) tabulated densely; evaluation is a table lookup so it is
    cheap on multi-million point grids.
    """

    def __init__(self, pts, samples=12000):
        pts = np.asarray(pts, dtype=np.float64)
        if pts[0, 0] > pts[-1, 0]:
            pts = pts[::-1]
        P = np.vstack([pts[0], pts[0], pts, pts[-1], pts[-1]])
        segs = len(P) - 3
        per = max(16, samples // segs)
        t = np.linspace(0.0, 1.0, per, endpoint=False)[:, None]
        b0 = (1 - t) ** 3 / 6.0
        b1 = (3 * t ** 3 - 6 * t ** 2 + 4) / 6.0
        b2 = (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6.0
        b3 = t ** 3 / 6.0
        out = []
        for i in range(segs):
            out.append(b0 * P[i] + b1 * P[i + 1] + b2 * P[i + 2] + b3 * P[i + 3])
        out.append(pts[-1][None, :])
        c = np.vstack(out)
        u = c[:, 0]
        # enforce monotonic u (control polygons are authored monotonic; guard anyway)
        u = np.maximum.accumulate(u)
        self.u = u
        self.v = c[:, 1]
        self.lo = float(u[0])
        self.hi = float(u[-1])

    def __call__(self, u):
        return np.interp(u, self.u, self.v)


def prof(*pts):
    return Profile(list(pts))


# ----------------------------------------------------------------------------- 3D primitives
def sd_box(X, Y, Z, c, half, r=0.0):
    qx = np.abs(X - c[0]) - (half[0] - r)
    qy = np.abs(Y - c[1]) - (half[1] - r)
    qz = np.abs(Z - c[2]) - (half[2] - r)
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
    return outside + inside - r


def sd_ellipsoid(X, Y, Z, c, rad):
    px = (X - c[0]); py = (Y - c[1]); pz = (Z - c[2])
    k0 = np.sqrt((px / rad[0]) ** 2 + (py / rad[1]) ** 2 + (pz / rad[2]) ** 2)
    k1 = np.sqrt((px / rad[0] ** 2) ** 2 + (py / rad[1] ** 2) ** 2 + (pz / rad[2] ** 2) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_sphere(X, Y, Z, c, r):
    return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r


def sd_cyl_x(X, Y, Z, cy, cz, R, x0, x1, r=0.0):
    """Capped cylinder with axis parallel to X spanning x0..x1 (optional edge rounding r)."""
    d = np.sqrt((Y - cy) ** 2 + (Z - cz) ** 2) - (R - r)
    a = np.abs(X - 0.5 * (x0 + x1)) - (0.5 * (x1 - x0) - r)
    return np.sqrt(np.maximum(d, 0.0) ** 2 + np.maximum(a, 0.0) ** 2) + np.minimum(np.maximum(d, a), 0.0) - r


def sd_cyl_y(X, Y, Z, cx, cz, R, y0, y1, r=0.0):
    d = np.sqrt((X - cx) ** 2 + (Z - cz) ** 2) - (R - r)
    a = np.abs(Y - 0.5 * (y0 + y1)) - (0.5 * (y1 - y0) - r)
    return np.sqrt(np.maximum(d, 0.0) ** 2 + np.maximum(a, 0.0) ** 2) + np.minimum(np.maximum(d, a), 0.0) - r


def sd_cyl_z(X, Y, Z, cx, cy, R, z0, z1, r=0.0):
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - (R - r)
    a = np.abs(Z - 0.5 * (z0 + z1)) - (0.5 * (z1 - z0) - r)
    return np.sqrt(np.maximum(d, 0.0) ** 2 + np.maximum(a, 0.0) ** 2) + np.minimum(np.maximum(d, a), 0.0) - r


def sd_capsule(X, Y, Z, a, b, r):
    pax = X - a[0]; pay = Y - a[1]; paz = Z - a[2]
    bax = b[0] - a[0]; bay = b[1] - a[1]; baz = b[2] - a[2]
    h = np.clip((pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz), 0.0, 1.0)
    return np.sqrt((pax - bax * h) ** 2 + (pay - bay * h) ** 2 + (paz - baz * h) ** 2) - r


def sd_cone_capsule(X, Y, Z, a, b, ra, rb):
    """Capsule with linearly varying radius (approximate distance)."""
    pax = X - a[0]; pay = Y - a[1]; paz = Z - a[2]
    bax = b[0] - a[0]; bay = b[1] - a[1]; baz = b[2] - a[2]
    h = np.clip((pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz), 0.0, 1.0)
    return np.sqrt((pax - bax * h) ** 2 + (pay - bay * h) ** 2 + (paz - baz * h) ** 2) - (ra + (rb - ra) * h)


def sd_plane(X, Y, Z, p, n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    return (X - p[0]) * n[0] + (Y - p[1]) * n[1] + (Z - p[2]) * n[2]


# ----------------------------------------------------------------------------- 2D helpers
def sd_polygon(px, py, poly):
    """Signed distance to a closed 2D polygon (negative inside). Vectorised over points."""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    px, py = np.broadcast_arrays(px, py)
    d = np.full(px.shape, np.inf)
    inside = np.zeros(px.shape, dtype=bool)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        dx, dy = wx - ex * t, wy - ey * t
        d = np.minimum(d, dx * dx + dy * dy)
        c1 = py >= ay
        c2 = py < by
        c3 = ex * wy > ey * wx
        inside ^= (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
    return np.where(inside, -1.0, 1.0) * np.sqrt(d)


def sd_polyline(px, py, pts):
    """Unsigned distance to an open 2D polyline."""
    pts = np.asarray(pts, dtype=np.float64)
    px, py = np.broadcast_arrays(px, py)
    d = np.full(px.shape, np.inf)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        ex, ey = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        dx, dy = wx - ex * t, wy - ey * t
        d = np.minimum(d, dx * dx + dy * dy)
    return np.sqrt(d)


def sd_rrect(px, py, cx, cy, hx, hy, r):
    qx = np.abs(px - cx) - (hx - r)
    qy = np.abs(py - cy) - (hy - r)
    return np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2) + np.minimum(np.maximum(qx, qy), 0.0) - r


def rot2(a, b, ang):
    """Rotate 2D coords (a, b) by -ang (i.e. express points in a frame rotated by +ang)."""
    c, s = np.cos(ang), np.sin(ang)
    return a * c + b * s, -a * s + b * c


# ----------------------------------------------------------------------------- sampling / meshing
def grad(fn, P, eps=7e-4):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    g = np.empty_like(P)
    g[:, 0] = (fn(x + eps, y, z) - fn(x - eps, y, z)) / (2 * eps)
    g[:, 1] = (fn(x, y + eps, z) - fn(x, y - eps, z)) / (2 * eps)
    g[:, 2] = (fn(x, y, z + eps) - fn(x, y, z - eps)) / (2 * eps)
    return g


def normals(fn, P, eps=7e-4):
    g = grad(fn, P, eps)
    l = np.linalg.norm(g, axis=1)
    return g / np.maximum(l, 1e-12)[:, None]


def project(fn, P, iters=3, eps=7e-4, max_step=0.02):
    """Newton-project points onto the zero level set of fn."""
    P = np.array(P, dtype=np.float64)
    for _ in range(iters):
        f = fn(P[:, 0], P[:, 1], P[:, 2])
        g = grad(fn, P, eps)
        g2 = np.maximum((g * g).sum(1), 1e-10)
        step = (f / g2)[:, None] * g
        sl = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, max_step / np.maximum(sl, 1e-12))[:, None]
        P -= step
    return P


def eval_grid(fn, bmin, bmax, h, symmetric=True, dtype=np.float32):
    """Sample fn on a regular grid. With symmetric=True the grid is mirrored about x=0
    (only x>=0 is evaluated). Returns (F, origin)."""
    ny = int(np.ceil((bmax[1] - bmin[1]) / h)) + 1
    nz = int(np.ceil((bmax[2] - bmin[2]) / h)) + 1
    ys = (bmin[1] + np.arange(ny) * h).astype(np.float64)
    zs = (bmin[2] + np.arange(nz) * h).astype(np.float64)
    if symmetric:
        nxh = int(np.ceil(bmax[0] / h)) + 1
        xs = np.arange(0, nxh + 1) * h
    else:
        nx = int(np.ceil((bmax[0] - bmin[0]) / h)) + 1
        xs = bmin[0] + np.arange(nx) * h
    X = xs[:, None, None]; Y = ys[None, :, None]; Z = zs[None, None, :]
    F = np.asarray(fn(X, Y, Z))
    F = np.ascontiguousarray(np.broadcast_to(F, (len(xs), ny, nz))).astype(dtype)
    if symmetric:
        F = np.concatenate([F[:0:-1], F], axis=0)
        origin = np.array([-nxh * h, bmin[1], bmin[2]])
    else:
        origin = np.array([bmin[0], bmin[1], bmin[2]])
    return F, origin


_CORNERS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def surface_nets(F, origin, h):
    """Naive surface nets. Returns (V float64 (N,3), Q int32 (M,4)) with outward winding
    (normals point from negative/inside to positive/outside)."""
    nx, ny, nz = F.shape
    inside = F < 0
    cnt = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.int8)
    for dx, dy, dz in _CORNERS:
        cnt += inside[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz]
    active = (cnt > 0) & (cnt < 8)
    ci, cj, ck = np.nonzero(active)
    n = len(ci)
    cell_id = np.full((nx - 1, ny - 1, nz - 1), -1, dtype=np.int32)
    cell_id[ci, cj, ck] = np.arange(n, dtype=np.int32)
    vals = np.stack([F[ci + dx, cj + dy, ck + dz] for dx, dy, dz in _CORNERS], axis=1).astype(np.float64)
    cpos = np.array(_CORNERS, dtype=np.float64)
    acc = np.zeros((n, 3)); w = np.zeros(n)
    for a, b in _EDGES:
        fa = vals[:, a]; fb = vals[:, b]
        m = (fa < 0) != (fb < 0)
        den = fa - fb
        t = np.where(m, fa / np.where(den == 0, 1.0, den), 0.0)
        p = cpos[a][None, :] + (cpos[b] - cpos[a])[None, :] * t[:, None]
        acc += p * m[:, None]
        w += m
    local = acc / np.maximum(w, 1.0)[:, None]
    V = (np.stack([ci, cj, ck], axis=1) + local) * h + origin[None, :]

    quads = []
    # x edges
    s = inside[:-1, 1:-1, 1:-1] != inside[1:, 1:-1, 1:-1]
    i, j, k = np.nonzero(s); j = j + 1; k = k + 1
    q = np.stack([cell_id[i, j - 1, k - 1], cell_id[i, j, k - 1], cell_id[i, j, k], cell_id[i, j - 1, k]], axis=1)
    fl = ~inside[i, j, k]
    q[fl] = q[fl][:, ::-1]
    quads.append(q)
    # y edges
    s = inside[1:-1, :-1, 1:-1] != inside[1:-1, 1:, 1:-1]
    i, j, k = np.nonzero(s); i = i + 1; k = k + 1
    q = np.stack([cell_id[i - 1, j, k - 1], cell_id[i - 1, j, k], cell_id[i, j, k], cell_id[i, j, k - 1]], axis=1)
    fl = ~inside[i, j, k]
    q[fl] = q[fl][:, ::-1]
    quads.append(q)
    # z edges
    s = inside[1:-1, 1:-1, :-1] != inside[1:-1, 1:-1, 1:]
    i, j, k = np.nonzero(s); i = i + 1; j = j + 1
    q = np.stack([cell_id[i - 1, j - 1, k], cell_id[i, j - 1, k], cell_id[i, j, k], cell_id[i - 1, j, k]], axis=1)
    fl = ~inside[i, j, k]
    q[fl] = q[fl][:, ::-1]
    quads.append(q)
    Q = np.concatenate(quads, axis=0).astype(np.int32)
    Q = Q[(Q >= 0).all(axis=1)]
    return V, Q


def quads_to_tris(V, Q):
    """Split quads along the shorter diagonal."""
    a, b, c, d = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    d1 = np.linalg.norm(V[a] - V[c], axis=1)
    d2 = np.linalg.norm(V[b] - V[d], axis=1)
    m = d1 <= d2
    t1 = np.where(m[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
    t2 = np.where(m[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
    return np.concatenate([t1, t2], axis=0).astype(np.int32)


def mesh_sdf(fn, bmin, bmax, h, symmetric=True, iters=3):
    """Convenience: grid sample + surface nets + Newton projection. Returns (V, Q)."""
    F, origin = eval_grid(fn, bmin, bmax, h, symmetric)
    V, Q = surface_nets(F, origin, h)
    if iters > 0 and len(V):
        V = project(fn, V, iters=iters, max_step=0.75 * h)
    return V, Q
