"""Shared mesh helpers for procedural vegetation (pure numpy, +Y up, metres)."""
import numpy as np
from glb import Part

UP = np.array([0.0, 1.0, 0.0])


def norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Acc:
    """Accumulates one primitive (positions / normals / uvs / colours / triangle indices)."""

    def __init__(self):
        self.pos, self.nrm, self.uv, self.col, self.idx = [], [], [], [], []
        self.n = 0

    def add(self, pos, nrm, uv, col, idx):
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        k = len(pos)
        self.pos.append(pos)
        self.nrm.append(np.asarray(nrm, dtype=np.float64).reshape(k, 3))
        self.uv.append(np.asarray(uv, dtype=np.float64).reshape(k, 2))
        self.col.append(np.asarray(col, dtype=np.float64).reshape(k, 3))
        self.idx.append(np.asarray(idx, dtype=np.int64).reshape(-1) + self.n)
        self.n += k

    @property
    def triangles(self):
        return sum(len(i) for i in self.idx) // 3

    def part(self, material):
        if not self.pos:
            return None
        return Part(material, np.concatenate(self.pos), norm(np.concatenate(self.nrm)), np.concatenate(self.uv),
                    np.clip(np.concatenate(self.col), 0, 1), np.concatenate(self.idx))


def frames(pts):
    """Parallel-transport frames along a polyline: (tangent, normal, binormal) with binormal = t x n."""
    pts = np.asarray(pts, dtype=np.float64)
    k = len(pts)
    tang = norm(np.gradient(pts, axis=0))
    t0 = tang[0]
    ref = np.array([1.0, 0.0, 0.0]) if abs(t0[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    n = norm(np.cross(t0, ref))
    ns = [n]
    for i in range(1, k):
        a, b = tang[i - 1], tang[i]
        axis = np.cross(a, b)
        s = np.linalg.norm(axis)
        c = float(np.dot(a, b))
        if s > 1e-9:
            axis = axis / s
            ang = np.arctan2(s, c)
            n = n * np.cos(ang) + np.cross(axis, n) * np.sin(ang) + axis * np.dot(axis, n) * (1 - np.cos(ang))
        n = norm(n - b * np.dot(n, b))
        ns.append(n)
    ns = np.array(ns)
    return tang, ns, np.cross(tang, ns)


def tube(acc, pts, radii, sides, wind, phase, ao, tex_scale=1.0):
    """Tapered tube around a polyline. UV: u wraps around (integer repeats), v runs along the length
    with ~square texels; colour = (wind, phase, ao)."""
    pts = np.asarray(pts, dtype=np.float64)
    k = len(pts)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (k,))
    _t, ns, bs = frames(pts)
    ang = np.linspace(0.0, 2.0 * np.pi, sides + 1)
    ca, sa = np.cos(ang), np.sin(ang)
    dirs = ns[:, None, :] * ca[None, :, None] + bs[:, None, :] * sa[None, :, None]
    pos = pts[:, None, :] + dirs * radii[:, None, None]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    circ = 2.0 * np.pi * max(float(radii.max()), 1e-3)
    u_rep = max(1, int(round(circ / tex_scale)))
    v_per_m = u_rep / circ
    s1 = sides + 1
    u = np.broadcast_to(np.linspace(0.0, u_rep, s1), (k, s1))
    v = np.broadcast_to((cum * v_per_m)[:, None], (k, s1))
    uv = np.stack([u, v], -1)
    wind = np.broadcast_to(np.asarray(wind, dtype=np.float64), (k,))
    ao = np.broadcast_to(np.asarray(ao, dtype=np.float64), (k,))
    col = np.stack([np.broadcast_to(wind[:, None], (k, s1)), np.full((k, s1), float(phase)),
                    np.broadcast_to(ao[:, None], (k, s1))], -1)
    i = np.arange(k - 1)[:, None]
    j = np.arange(sides)[None, :]
    a = i * s1 + j
    b = a + 1
    c = a + s1
    d = c + 1
    tris = np.stack([a, b, c, b, d, c], -1).reshape(-1, 3)
    acc.add(pos.reshape(-1, 3), dirs.reshape(-1, 3), uv.reshape(-1, 2), col.reshape(-1, 3), tris)


def bezier(p0, p1, p2, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def arc_point(base, D, N, L, bend, s):
    """Point / direction / normal at fraction s of a constant-curvature strip that bends toward -N."""
    a = bend * s
    if abs(bend) < 1e-5:
        return base + D * (L * s), D, N
    rho = L / bend
    p = base + D * (rho * np.sin(a)) - N * (rho * (1.0 - np.cos(a)))
    return p, D * np.cos(a) - N * np.sin(a), N * np.cos(a) + D * np.sin(a)


def make_card(base, D, S, L, cols, uvrect, nseg=1, bend=0.0):
    """A strip of quads: length along D (bending toward -N by `bend` radians in total), columns across.
    cols: list of (offset_along_S, offset_along_N, v_fraction).  N = S x D is the card's front normal.
    uvrect = (u0, v0, u1, v1): u runs base -> tip, v across.
    Returns pos, uv, t (0 base..1 tip), edge (0 centre..1 rim), local normal, triangles."""
    D = norm(D)
    S = norm(S - D * np.dot(S, D))
    N = np.cross(S, D)
    u0, v0, u1, v1 = uvrect
    nc = len(cols)
    pos, uv, ts, eds, nl = [], [], [], [], []
    for j in range(nseg + 1):
        t = j / nseg
        c, _dt, nt = arc_point(np.asarray(base, dtype=np.float64), D, N, L, bend, t)
        for (o_s, o_n, vf) in cols:
            pos.append(c + S * o_s + nt * o_n)
            uv.append((u0 + (u1 - u0) * t, v0 + (v1 - v0) * vf))
            ts.append(t)
            eds.append(abs(vf - 0.5) * 2.0)
            nl.append(nt)
    tris = []
    for j in range(nseg):
        for k in range(nc - 1):
            a = j * nc + k
            b = a + 1
            cc = a + nc
            d = cc + 1
            tris += [(a, cc, b), (b, cc, d)]
    return np.array(pos), np.array(uv), np.array(ts), np.array(eds), np.array(nl), np.array(tris)


def fib_dirs(n, rng):
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    th = i * 2.39996323 + rng.uniform(0.0, 2.0 * np.pi)
    return np.stack([r * np.cos(th), y, r * np.sin(th)], 1)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def trunk_axis_fn(H, amp, ph1, ph2, lean):
    """Gently curved, leaning trunk centreline: y (array) -> points."""
    def f(y):
        y = np.asarray(y, dtype=np.float64)
        u = y / H
        x = amp * np.sin(u * 2.6 + ph1) * u + lean[0] * u ** 1.5
        z = amp * np.cos(u * 2.2 + ph2) * u + lean[1] * u ** 1.5
        return np.stack([x, y, z], -1)
    return f
