"""High-detail foliage clusters (needle boughs, pine tufts, leaf sprays) as raw triangle soups.

Pure numpy; consumed by render_foliage_tiles.py (Blender) which renders each cluster
top-down (camera looks along -Z, image right = +X, image up = +Y) into atlas tiles.
All clusters lie roughly in the XY plane with their attachment point at the middle of
the LEFT tile edge (x = 0, y = 0) and grow toward +X.
"""
import numpy as np

Z = np.array([0.0, 0.0, 1.0])


def srgb(r, g, b):
    c = np.array([r, g, b], dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class MeshBuf:
    def __init__(self):
        self.v, self.c, self.t, self.s = [], [], [], []
        self.n = 0

    def add(self, verts, cols, tris, smooth=False):
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        cols = np.asarray(cols, dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.v.append(verts)
        self.c.append(cols)
        self.t.append(tris + self.n)
        self.s.append(np.full(len(tris), bool(smooth)))
        self.n += len(verts)

    def arrays(self):
        return (np.concatenate(self.v), np.concatenate(self.c), np.concatenate(self.t), np.concatenate(self.s))


# --------------------------------------------------------------------------- primitives

def add_needles(mb, bases, dirs, lens, widths, col_base, col_tip):
    side = np.cross(dirs, Z)
    sn = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(sn < 1e-6, np.array([1.0, 0.0, 0.0]), side / np.maximum(sn, 1e-9))
    hw = (widths * 0.5)[:, None]
    v0 = bases - side * hw
    v1 = bases + side * hw
    v2 = bases + dirs * lens[:, None]
    verts = np.stack([v0, v1, v2], 1).reshape(-1, 3)
    cols = np.stack([col_base, col_base, col_tip], 1).reshape(-1, 3)
    n = len(bases)
    mb.add(verts, cols, np.arange(n * 3).reshape(n, 3), smooth=False)


def add_ribbon(mb, pts, radii, col):
    pts = np.asarray(pts, dtype=np.float64)
    k = len(pts)
    if k < 2:
        return
    tan = np.gradient(pts, axis=0)
    side = np.cross(tan, Z)
    sn = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(sn < 1e-9, np.array([0.0, 1.0, 0.0]), side / np.maximum(sn, 1e-12))
    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (k,))[:, None]
    left = pts - side * r
    right = pts + side * r
    verts = np.concatenate([left, right])
    col = np.asarray(col, dtype=np.float64)
    if col.ndim == 1:
        cols = np.tile(col, (2 * k, 1))
    else:
        cols = np.concatenate([col, col])
    i = np.arange(k - 1)
    t1 = np.stack([i, i + k, i + k + 1], 1)
    t2 = np.stack([i, i + k + 1, i + 1], 1)
    mb.add(verts, cols, np.concatenate([t1, t2]), smooth=False)


def polyline(start, heading, length, n, curl=0.0, droop=0.0, wiggle=0.0, rng=None, z_rise=0.0):
    """Walk in the XY plane; heading (rad) changes linearly by `curl` over the length.
    z = start.z + z_rise*u*length - droop*u^1.6*length."""
    u = np.linspace(0.0, 1.0, n)
    ang = heading + curl * u
    if wiggle and rng is not None:
        ang = ang + wiggle * np.sin(u * rng.uniform(4.0, 9.0) + rng.uniform(0, 6.28))
    ds = length / (n - 1)
    dx = np.cos(ang) * ds
    dy = np.sin(ang) * ds
    x = start[0] + np.concatenate([[0.0], np.cumsum(dx[:-1])])
    y = start[1] + np.concatenate([[0.0], np.cumsum(dy[:-1])])
    z = start[2] + z_rise * u * length - droop * (u ** 1.6) * length
    return np.stack([x, y, z], 1), ang


def resample(pts, step, jitter=0.0, rng=None):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-9:
        return pts[:1], np.zeros(1), total
    q = np.arange(step * 0.5, total, step)
    if jitter and rng is not None:
        q = np.clip(q + rng.uniform(-jitter, jitter, len(q)) * step, 0, total)
    out = np.stack([np.interp(q, s, pts[:, k]) for k in range(3)], 1)
    return out, q / total, total


def headings_at(pts, frac):
    d = np.gradient(pts, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    u = np.linspace(0, 1, len(pts))
    return np.interp(frac, u, ang)


def pick_colors(rng, palette, n, jitter=0.06):
    pal = np.asarray(palette)
    idx = rng.integers(0, len(pal), n)
    w = rng.uniform(0, 1, (n, 1))
    idx2 = rng.integers(0, len(pal), n)
    c = pal[idx] * w + pal[idx2] * (1 - w)
    return c * (1.0 + rng.uniform(-jitter, jitter, (n, 1)))


# --------------------------------------------------------------------------- fir bough

FIR_PAL = [srgb(40, 60, 36), srgb(50, 72, 40), srgb(58, 82, 44), srgb(46, 70, 48), srgb(66, 88, 46)]
FIR_TIP = srgb(98, 124, 62)
FIR_TWIG = srgb(74, 56, 42)


def needles_on_twig(mb, rng, pts, pal, tip_col, n_len=(0.034, 0.048), n_w=0.0056, step=0.0065,
                    side_ang=(52, 76), tilt=(-12, 22), top=True, tipness=None, end_taper=0.18):
    """Pectinate needles along both sides of a twig, plus a top row."""
    p, frac, total = resample(pts, step, jitter=0.35, rng=rng)
    n = len(p)
    if n == 0:
        return
    head = headings_at(pts, frac)
    sgn = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    a = head + sgn * np.radians(rng.uniform(side_ang[0], side_ang[1], n))
    tl = np.radians(rng.uniform(tilt[0], tilt[1], n))
    dirs = _norm(np.stack([np.cos(a) * np.cos(tl), np.sin(a) * np.cos(tl), np.sin(tl)], 1))
    taper = np.clip((1.0 - frac) / max(end_taper, 1e-6), 0.45, 1.0)
    lens = rng.uniform(n_len[0], n_len[1], n) * taper
    cols = pick_colors(rng, pal, n)
    t = frac if tipness is None else np.clip(tipness[0] + (tipness[1] - tipness[0]) * frac, 0, 1)
    tw = np.clip((t - 0.55) / 0.45, 0, 1)[:, None] * 0.75
    cols = cols * (1 - tw) + tip_col * tw
    add_needles(mb, p, dirs, lens, np.full(n, n_w), cols * 0.82, cols * 1.12 + 0.004)
    if top:
        p2, frac2, _ = resample(pts, step * 1.9, jitter=0.4, rng=rng)
        m = len(p2)
        if m:
            head2 = headings_at(pts, frac2)
            a2 = head2 + np.radians(rng.uniform(-28, 28, m))
            tl2 = np.radians(rng.uniform(38, 66, m))
            d2 = _norm(np.stack([np.cos(a2) * np.cos(tl2), np.sin(a2) * np.cos(tl2), np.sin(tl2)], 1))
            c2 = pick_colors(rng, pal, m)
            t2 = frac2 if tipness is None else np.clip(tipness[0] + (tipness[1] - tipness[0]) * frac2, 0, 1)
            tw2 = np.clip((t2 - 0.55) / 0.45, 0, 1)[:, None] * 0.75
            c2 = c2 * (1 - tw2) + tip_col * tw2
            add_needles(mb, p2 + Z * 0.004, d2, rng.uniform(n_len[0], n_len[1], m) * 0.9, np.full(m, n_w * 1.1), c2 * 0.95, c2 * 1.2 + 0.004)


def build_fir_bough(seed, L=2.06, half_w=0.50, droop=0.10, n_scale=1.0):
    rng = np.random.default_rng(seed)
    mb = MeshBuf()
    main, _ = polyline(np.array([0.0, 0.0, 0.0]), rng.uniform(-0.03, 0.03), L, 48, curl=rng.uniform(-0.08, 0.08),
                       droop=droop * 0.5, wiggle=0.035, rng=rng, z_rise=0.03)
    main[:, 1] -= np.linspace(0, 1, len(main)) * main[-1, 1]  # keep the tip on the axis
    seg = np.linalg.norm(np.diff(main, axis=0), axis=1)
    s_main = np.concatenate([[0.0], np.cumsum(seg)])
    add_ribbon(mb, main, np.linspace(0.016, 0.004, len(main)), FIR_TWIG)
    needles_on_twig(mb, rng, main[2:], FIR_PAL, FIR_TIP, n_len=(0.046 * n_scale, 0.062 * n_scale), tipness=(0.2, 1.0))

    ls_max = half_w / np.sin(np.radians(60))

    def shape(u):
        return np.where(u < 0.28, 0.25 + 0.75 * (u / 0.28) ** 0.75, 0.10 + 0.90 * (1 - (u - 0.28) / 0.72) ** 1.15)

    s = 0.07
    k = 0
    while s < L - 0.04:
        u = s / L
        base = np.array([np.interp(s, s_main, main[:, 0]), np.interp(s, s_main, main[:, 1]), np.interp(s, s_main, main[:, 2])])
        hmain = headings_at(main, np.array([u]))[0]
        for sgn in (1.0, -1.0):
            ang = hmain + sgn * np.radians(64 - 26 * u + rng.uniform(-6, 6))
            ln = float(ls_max * shape(np.array([u]))[0] * rng.uniform(0.86, 1.08))
            if ln < 0.035:
                continue
            npts = max(6, int(ln / 0.03))
            tw, _ = polyline(base + Z * rng.uniform(-0.008, 0.012), ang, ln, npts, curl=-sgn * np.radians(rng.uniform(8, 20)),
                             droop=droop * rng.uniform(0.7, 1.4), wiggle=0.05, rng=rng)
            add_ribbon(mb, tw, np.linspace(0.0075, 0.0025, npts), FIR_TWIG * rng.uniform(0.85, 1.1))
            needles_on_twig(mb, rng, tw, FIR_PAL, FIR_TIP, n_len=(0.044 * n_scale, 0.060 * n_scale), tipness=(0.25 + 0.3 * u, 1.0))
            # secondary twigs
            segt = np.linalg.norm(np.diff(tw, axis=0), axis=1)
            st = np.concatenate([[0.0], np.cumsum(segt)])
            q = rng.uniform(0.035, 0.05)
            j = 0
            while q < ln - 0.03:
                v = q / ln
                b2 = np.array([np.interp(q, st, tw[:, 0]), np.interp(q, st, tw[:, 1]), np.interp(q, st, tw[:, 2])])
                h2 = headings_at(tw, np.array([v]))[0]
                sg2 = 1.0 if j % 2 == 0 else -1.0
                a2 = h2 + sg2 * np.radians(rng.uniform(42, 58))
                l2 = float(ln * 0.50 * (1 - v) ** 0.8 * rng.uniform(0.8, 1.15))
                if l2 > 0.028:
                    n2 = max(4, int(l2 / 0.025))
                    t2, _ = polyline(b2 + Z * rng.uniform(-0.006, 0.008), a2, l2, n2, curl=-sg2 * np.radians(rng.uniform(4, 14)),
                                     droop=droop * rng.uniform(0.5, 1.5), rng=rng)
                    add_ribbon(mb, t2, np.linspace(0.0045, 0.002, n2), FIR_TWIG * rng.uniform(0.85, 1.1))
                    needles_on_twig(mb, rng, t2, FIR_PAL, FIR_TIP, n_len=(0.038 * n_scale, 0.054 * n_scale),
                                    tipness=(0.45 + 0.25 * u, 1.0), top=(j % 2 == 0))
                q += rng.uniform(0.038, 0.056)
                j += 1
        # an upper layer of shorter, steeper branchlets fills the area near the axis
        if k % 2 == 0 and u < 0.9:
            for sgn in (1.0, -1.0):
                ang = hmain + sgn * np.radians(rng.uniform(30, 46))
                ln = float(ls_max * shape(np.array([u]))[0] * rng.uniform(0.45, 0.65))
                if ln < 0.06:
                    continue
                npts = max(6, int(ln / 0.03))
                tw, _ = polyline(base + Z * rng.uniform(0.02, 0.04), ang, ln, npts, curl=-sgn * np.radians(rng.uniform(0, 14)),
                                 droop=-0.03, wiggle=0.05, rng=rng)
                add_ribbon(mb, tw, np.linspace(0.006, 0.002, npts), FIR_TWIG * rng.uniform(0.85, 1.1))
                needles_on_twig(mb, rng, tw, FIR_PAL, FIR_TIP, n_len=(0.042 * n_scale, 0.058 * n_scale), tipness=(0.3 + 0.3 * u, 1.0))
        s += rng.uniform(0.056, 0.076) * (1.0 - 0.3 * u)
        k += 1
    return mb.arrays()


# --------------------------------------------------------------------------- pine tufts (long needles)

PINE_PAL = [srgb(56, 78, 38), srgb(68, 90, 42), srgb(80, 102, 48), srgb(60, 86, 50), srgb(92, 112, 54)]
PINE_TIP = srgb(118, 136, 66)
PINE_TWIG = srgb(96, 70, 52)


def add_tuft(mb, rng, c, d, n=190, length=(0.13, 0.20), cone=84.0, width=0.0056, pal=PINE_PAL, tip=PINE_TIP):
    d = _norm(np.asarray(d, dtype=np.float64))
    # orthonormal frame around d
    ref = Z if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _norm(np.cross(d, ref))
    e2 = np.cross(d, e1)
    cosmax = np.cos(np.radians(cone))
    cu = 1.0 - rng.uniform(0, 1, n) ** 1.25 * (1.0 - cosmax)
    su = np.sqrt(np.maximum(0.0, 1 - cu * cu))
    ph = rng.uniform(0, 2 * np.pi, n)
    dirs = cu[:, None] * d + (su * np.cos(ph))[:, None] * e1 + (su * np.sin(ph))[:, None] * e2
    dirs[:, 2] -= 0.10 * su  # gravity droop on outer needles
    dirs = _norm(dirs)
    bases = c - d * rng.uniform(0.0, 0.075, n)[:, None] + rng.normal(0, 0.004, (n, 3))
    lens = rng.uniform(length[0], length[1], n) * (0.75 + 0.25 * cu)
    cols = pick_colors(rng, pal, n)
    add_needles(mb, bases, dirs, lens, np.full(n, width), cols * 0.78, cols * 0.55 + tip * 0.55)


def build_pine_clump(seed, size=1.0, n_tips=9, elong=1.0):
    """A branch end: forked twig entering at the left edge centre, needle tufts at every tip."""
    rng = np.random.default_rng(seed)
    mb = MeshBuf()
    cx, cy = 0.53 * size * elong, 0.0
    stem_end = np.array([0.30 * size * elong, rng.uniform(-0.03, 0.03) * size, 0.0])
    stem, _ = polyline(np.array([0.0, 0.0, 0.0]), np.arctan2(stem_end[1], stem_end[0]), np.linalg.norm(stem_end), 10,
                       wiggle=0.05, rng=rng)
    add_ribbon(mb, stem, np.linspace(0.013, 0.010, len(stem)), PINE_TWIG)
    R = 0.35 * size
    tips = []
    # distribute tip targets on a disc (sunflower) around the clump centre
    for i in range(n_tips):
        rr = R * np.sqrt((i + 0.5) / n_tips) * rng.uniform(0.85, 1.1)
        aa = i * 2.39996 + rng.uniform(-0.3, 0.3)
        tips.append(np.array([cx + rr * np.cos(aa) * elong, cy + rr * np.sin(aa), rng.uniform(-0.04, 0.10)]))
    for i, tp in enumerate(tips):
        v = tp - stem[-1]
        ln = np.linalg.norm(v[:2])
        tw, _ = polyline(stem[-1], np.arctan2(v[1], v[0]), ln, max(5, int(ln / 0.04)), curl=rng.uniform(-0.35, 0.35),
                         wiggle=0.08, rng=rng, z_rise=v[2] / max(ln, 1e-6))
        add_ribbon(mb, tw, np.linspace(0.009, 0.005, len(tw)), PINE_TWIG * rng.uniform(0.85, 1.1))
        d = tw[-1] - tw[-2]
        d = _norm(d)
        # outward in plane, some lifted toward the camera
        lift = rng.uniform(0.0, 0.9) if i % 3 else rng.uniform(1.0, 2.4)
        d = _norm(d + Z * lift)
        add_tuft(mb, rng, tw[-1], d, n=int(rng.uniform(170, 220)))
        if rng.uniform() < 0.9:  # an inner tuft along the twig
            m = tw[len(tw) // 2]
            add_tuft(mb, rng, m + Z * 0.02, _norm(np.array([rng.normal(0, 0.5), rng.normal(0, 0.5), 1.4])), n=110, length=(0.13, 0.19))
    return mb.arrays()


def build_pine_branch(seed, L=1.9, half_w=0.44):
    rng = np.random.default_rng(seed)
    mb = MeshBuf()
    main, _ = polyline(np.array([0.0, 0.0, 0.0]), rng.uniform(-0.04, 0.04), L - 0.16, 30, curl=rng.uniform(-0.12, 0.12),
                       wiggle=0.06, rng=rng, z_rise=0.02)
    main[:, 1] -= np.linspace(0, 1, len(main)) * main[-1, 1]
    add_ribbon(mb, main, np.linspace(0.016, 0.007, len(main)), PINE_TWIG)
    seg = np.linalg.norm(np.diff(main, axis=0), axis=1)
    sm = np.concatenate([[0.0], np.cumsum(seg)])
    s = 0.16
    k = 0
    while s < sm[-1] - 0.05:
        u = s / sm[-1]
        base = np.array([np.interp(s, sm, main[:, c]) for c in range(3)])
        hm = headings_at(main, np.array([u]))[0]
        sgn = 1.0 if k % 2 == 0 else -1.0
        ln = half_w * rng.uniform(0.55, 0.92) * (1.0 - 0.45 * u)
        ang = hm + sgn * np.radians(rng.uniform(42, 66))
        tw, _ = polyline(base, ang, ln, max(5, int(ln / 0.04)), curl=-sgn * rng.uniform(0.1, 0.5), wiggle=0.08, rng=rng,
                         z_rise=rng.uniform(-0.05, 0.2))
        add_ribbon(mb, tw, np.linspace(0.011, 0.006, len(tw)), PINE_TWIG * rng.uniform(0.85, 1.1))
        d = _norm(tw[-1] - tw[-2])
        add_tuft(mb, rng, tw[-1], _norm(d + Z * rng.uniform(0.0, 0.8)), n=int(rng.uniform(170, 220)))
        if rng.uniform() < 0.85:
            add_tuft(mb, rng, base + Z * 0.02, _norm(np.array([np.cos(hm) * 0.5, np.sin(hm) * 0.5 - sgn * 0.6, 1.2])), n=120,
                     length=(0.13, 0.19))
        s += rng.uniform(0.085, 0.13)
        k += 1
    d = _norm(main[-1] - main[-2])
    add_tuft(mb, rng, main[-1], _norm(d + Z * 0.15), n=190, length=(0.16, 0.24))
    return mb.arrays()


# --------------------------------------------------------------------------- broadleaf sprays

LEAF_A_PAL = [srgb(44, 66, 30), srgb(54, 78, 34), srgb(64, 88, 38), srgb(50, 74, 40), srgb(76, 96, 42)]
LEAF_B_PAL = [srgb(70, 92, 40), srgb(82, 104, 44), srgb(94, 114, 50), srgb(78, 100, 52), srgb(106, 120, 54)]
LEAF_TWIG = srgb(84, 68, 52)
LEAF_TWIG_B = srgb(120, 112, 100)


def _rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])  # +pitch lifts the tip
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


_LEAF_T = np.array([0.0, 0.10, 0.27, 0.47, 0.67, 0.86, 1.0])


def add_leaf(mb, rng, base, yaw, pitch, roll, L, W, col, profile="ovate"):
    ts = _LEAF_T
    if profile == "ovate":
        w = W * np.sin(np.pi * ts ** 0.72) ** 0.9
    else:  # round / blunt
        w = W * np.sin(np.pi * ts ** 0.95) ** 0.55 * (1.0 - 0.35 * np.clip(ts - 0.8, 0, 1) / 0.2)
    w[0] = 0.0
    w[-1] = 0.0
    n = len(ts)
    mid = np.stack([ts * L, np.zeros(n), -0.16 * L * ts ** 2], 1)
    cup = rng.uniform(0.10, 0.34)
    e_l = mid + np.stack([np.zeros(n), w, cup * w], 1)
    e_r = mid + np.stack([np.zeros(n), -w, cup * w], 1)
    verts = np.concatenate([mid, e_l, e_r])
    tris = []
    for i in range(n - 1):
        m0, m1, l0, l1, r0, r1 = i, i + 1, n + i, n + i + 1, 2 * n + i, 2 * n + i + 1
        tris += [[m0, m1, l1], [m0, l1, l0], [m0, r0, r1], [m0, r1, m1]]
    rot = _rot(yaw, pitch, roll)
    verts = verts @ rot.T + base
    shade = (0.86 + 0.2 * ts)[:, None]
    cmid = np.clip(col * 1.12 + np.array([0.006, 0.006, 0.0]), 0, 1) * shade
    cedge = col * shade
    cols = np.concatenate([cmid, cedge, cedge])
    mb.add(verts, cols, np.array(tris), smooth=True)


def leaves_on_twig(mb, rng, pts, pal, L=(0.10, 0.14), W=(0.030, 0.040), spacing=(0.028, 0.040), profile="ovate",
                   skip_start=0.0, tilt=32.0, yellow=0.05):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    q = max(skip_start, 0.01)
    k = int(rng.integers(0, 2))
    while q <= total:
        u = q / total
        base = np.array([np.interp(q, s, pts[:, c]) for c in range(3)])
        h = headings_at(pts, np.array([u]))[0]
        last = q + spacing[1] > total
        sgn = 1.0 if k % 2 == 0 else -1.0
        yaw = h + (0.0 if last else sgn * np.radians(rng.uniform(38, 72))) + np.radians(rng.uniform(-10, 10))
        col = pick_colors(rng, pal, 1, jitter=0.10)[0]
        if rng.uniform() < yellow:
            col = col * 0.65 + srgb(124, 120, 56) * 0.35
        add_leaf(mb, rng, base + Z * rng.uniform(-0.012, 0.03), yaw, np.radians(rng.uniform(-tilt * 0.6, tilt)),
                 np.radians(rng.uniform(-tilt, tilt)), rng.uniform(L[0], L[1]) * (0.8 + 0.2 * (1 - u)), rng.uniform(W[0], W[1]),
                 col, profile)
        q += rng.uniform(spacing[0], spacing[1])
        k += 1


def build_leaf_spray(seed, size_x=1.0, size_y=1.0, species="a"):
    rng = np.random.default_rng(seed)
    mb = MeshBuf()
    if species == "a":
        pal, Lr, Wr, sp, prof, twc = LEAF_A_PAL, (0.110, 0.150), (0.033, 0.043), (0.021, 0.030), "ovate", LEAF_TWIG
    else:
        pal, Lr, Wr, sp, prof, twc = LEAF_B_PAL, (0.070, 0.094), (0.029, 0.036), (0.015, 0.021), "round", LEAF_TWIG_B
    Lm = size_x * 0.80
    half = size_y * 0.5
    main, _ = polyline(np.array([0.0, 0.0, 0.0]), rng.uniform(-0.06, 0.06), Lm, 26, curl=rng.uniform(-0.25, 0.25), wiggle=0.07, rng=rng)
    main[:, 1] -= np.linspace(0, 1, len(main)) ** 1.5 * main[-1, 1] * 0.7
    add_ribbon(mb, main, np.linspace(0.011, 0.003, len(main)), twc)
    leaves_on_twig(mb, rng, main, pal, Lr, Wr, sp, prof, skip_start=0.07 * size_x)
    seg = np.linalg.norm(np.diff(main, axis=0), axis=1)
    sm = np.concatenate([[0.0], np.cumsum(seg)])
    s = 0.09 * size_x
    k = int(rng.integers(0, 2))
    while s < sm[-1] - 0.08:
        u = s / sm[-1]
        base = np.array([np.interp(s, sm, main[:, c]) for c in range(3)])
        hm = headings_at(main, np.array([u]))[0]
        sgn = 1.0 if k % 2 == 0 else -1.0
        # side twig long enough to reach toward the tile edge, shorter near the tip (rounded spray outline)
        reach = half * 0.92 * np.sqrt(max(0.0, 1.0 - (2.0 * u - 0.85) ** 2 * 0.9)) if u > 0.4 else half * 0.92 * (0.45 + 0.55 * u / 0.4)
        ang = hm + sgn * np.radians(rng.uniform(44, 70))
        ln = max(0.08, reach / np.sin(abs(ang - hm)) * rng.uniform(0.78, 1.0))
        ln = min(ln, half * 1.25)
        tw, _ = polyline(base, ang, ln, max(6, int(ln / 0.04)), curl=-sgn * rng.uniform(0.15, 0.6), wiggle=0.1, rng=rng,
                         z_rise=rng.uniform(-0.06, 0.10))
        add_ribbon(mb, tw, np.linspace(0.006, 0.002, len(tw)), twc * rng.uniform(0.85, 1.1))
        leaves_on_twig(mb, rng, tw, pal, Lr, Wr, sp, prof, skip_start=0.02)
        # tertiary twigs
        segt = np.linalg.norm(np.diff(tw, axis=0), axis=1)
        st = np.concatenate([[0.0], np.cumsum(segt)])
        q = rng.uniform(0.07, 0.12)
        j = 0
        while q < ln - 0.06:
            v = q / ln
            b2 = np.array([np.interp(q, st, tw[:, c]) for c in range(3)])
            h2 = headings_at(tw, np.array([v]))[0]
            sg2 = 1.0 if j % 2 == 0 else -1.0
            l2 = ln * 0.45 * (1 - v) * rng.uniform(0.7, 1.1)
            if l2 > 0.06:
                t2, _ = polyline(b2, h2 + sg2 * np.radians(rng.uniform(40, 62)), l2, max(4, int(l2 / 0.04)),
                                 curl=-sg2 * rng.uniform(0.0, 0.5), rng=rng, z_rise=rng.uniform(-0.08, 0.12))
                add_ribbon(mb, t2, np.linspace(0.004, 0.0018, len(t2)), twc * rng.uniform(0.85, 1.1))
                leaves_on_twig(mb, rng, t2, pal, Lr, Wr, sp, prof, skip_start=0.015)
            q += rng.uniform(0.09, 0.15)
            j += 1
        s += rng.uniform(0.060, 0.090) * (size_x ** 0.5)
        k += 1
    return mb.arrays()


# --------------------------------------------------------------------------- grass (side view: x right, y up)

GRASS_PAL = [srgb(72, 96, 40), srgb(88, 112, 46), srgb(104, 122, 52), srgb(120, 128, 58), srgb(82, 104, 50)]
GRASS_DRY = srgb(150, 140, 84)


def build_grass_clump(seed, width=0.8, height=0.5, n_blades=150, dry=0.25):
    """Blades grow +Y from the bottom edge (y = 0); rendered with the camera looking along -Z."""
    rng = np.random.default_rng(seed)
    mb = MeshBuf()
    for i in range(n_blades):
        x0 = rng.normal(0.0, width * 0.19)
        x0 = float(np.clip(x0, -width * 0.42, width * 0.42))
        edge = 1.0 - (abs(x0) / (width * 0.5)) ** 2
        h = height * rng.uniform(0.45, 0.97) * (0.55 + 0.45 * edge)
        lean = rng.normal(0.0, 0.22) + x0 / width * 0.9
        bend = rng.uniform(0.1, 0.9) * np.sign(lean if abs(lean) > 0.05 else rng.normal())
        w0 = rng.uniform(0.007, 0.013)
        n = 7
        u = np.linspace(0, 1, n)
        ang = np.pi / 2 - lean - bend * u ** 1.5
        ds = h / (n - 1)
        x = x0 + np.concatenate([[0], np.cumsum(np.cos(ang[:-1]) * ds)])
        y = np.concatenate([[0], np.cumsum(np.sin(ang[:-1]) * ds)])
        z = rng.uniform(-0.15, 0.15) + u * rng.uniform(-0.05, 0.05)
        pts = np.stack([x, y, z], 1)
        col = pick_colors(rng, GRASS_PAL, 1, jitter=0.1)[0]
        if rng.uniform() < dry:
            col = col * 0.45 + GRASS_DRY * 0.55
        cols = col[None, :] * (0.55 + 0.6 * u)[:, None]
        tipmix = (u ** 3)[:, None] * 0.35
        cols = cols * (1 - tipmix) + GRASS_DRY * tipmix
        add_ribbon(mb, pts, w0 * (1 - u ** 1.7) + 0.0012, cols)
    return mb.arrays()


# --------------------------------------------------------------------------- tile table

def _fit(arr, rect, margin=0.035, grass=False):
    """Uniformly shrink a cluster about its attachment point so nothing is clipped by the tile border."""
    v, c, t, s = arr
    x0, x1, y0, y1 = rect
    m = margin * min(x1 - x0, y1 - y0)
    if grass:
        sc = min((x1 - m) / max(np.abs(v[:, 0]).max(), 1e-6), (y1 - m) / max(v[:, 1].max(), 1e-6), 1.0)
    else:
        sc = min((x1 - m) / max(v[:, 0].max(), 1e-6), (y1 - m) / max(np.abs(v[:, 1]).max(), 1e-6), 1.0)
    return v * sc, c, t, s


def _tile(builder, rect, pix, grass=False):
    return (lambda: _fit(builder(), rect, grass=grass), rect, pix)


# name -> (builder, world rect (x0, x1, y0, y1), pixel rect in the 2048x2048 atlas (px, py, w, h))
def tiles():
    return {
        "fir_a": _tile(lambda: build_fir_bough(11, L=2.06, half_w=0.50), (0.0, 2.2, -0.55, 0.55), (0, 0, 1024, 512)),
        "fir_b": _tile(lambda: build_fir_bough(23, L=2.08, half_w=0.44, droop=0.16), (0.0, 2.2, -0.55, 0.55), (0, 512, 1024, 512)),
        "pine_tuft_a": _tile(lambda: build_pine_clump(5, size=1.0, n_tips=21), (0.0, 1.0, -0.5, 0.5), (1024, 0, 512, 512)),
        "pine_tuft_b": _tile(lambda: build_pine_clump(17, size=1.0, n_tips=18, elong=1.05), (0.0, 1.0, -0.5, 0.5), (1536, 0, 512, 512)),
        "pine_branch": _tile(lambda: build_pine_branch(31), (0.0, 2.0, -0.5, 0.5), (1024, 512, 1024, 512)),
        "leaf_a1": _tile(lambda: build_leaf_spray(41, 1.0, 1.0, "a"), (0.0, 1.0, -0.5, 0.5), (0, 1024, 512, 512)),
        "leaf_a2": _tile(lambda: build_leaf_spray(47, 1.0, 1.0, "a"), (0.0, 1.0, -0.5, 0.5), (512, 1024, 512, 512)),
        "leaf_b1": _tile(lambda: build_leaf_spray(53, 0.8, 0.8, "b"), (0.0, 0.8, -0.4, 0.4), (1024, 1024, 512, 512)),
        "leaf_b2": _tile(lambda: build_leaf_spray(59, 0.8, 0.8, "b"), (0.0, 0.8, -0.4, 0.4), (1536, 1024, 512, 512)),
        "leaf_a_branch": _tile(lambda: build_leaf_spray(67, 1.9, 0.95, "a"), (0.0, 1.9, -0.475, 0.475), (0, 1536, 1024, 512)),
        "leaf_b_branch": _tile(lambda: build_leaf_spray(71, 1.5, 0.75, "b"), (0.0, 1.5, -0.375, 0.375), (1024, 1536, 1024, 512)),
    }


ATLAS_SIZE = (2048, 2048)


def tile_uv(name):
    """Atlas UV rect (u0, v0, u1, v1), glTF convention (v = 0 at the image top)."""
    px, py, w, h = tiles()[name][2]
    W, H = ATLAS_SIZE
    return (px / W, py / H, (px + w) / W, (py + h) / H)


# grass atlas: 4 clumps side by side (side view, blades grow up from the bottom edge)
def grass_tiles():
    return {
        "grass_0": _tile(lambda: build_grass_clump(3, 0.8, 0.5, 170, dry=0.2), (-0.4, 0.4, 0.0, 0.5), (0, 0, 256, 160), True),
        "grass_1": _tile(lambda: build_grass_clump(7, 0.8, 0.5, 130, dry=0.35), (-0.4, 0.4, 0.0, 0.5), (256, 0, 256, 160), True),
        "grass_2": _tile(lambda: build_grass_clump(13, 0.8, 0.42, 190, dry=0.12), (-0.4, 0.4, 0.0, 0.5), (512, 0, 256, 160), True),
        "grass_3": _tile(lambda: build_grass_clump(19, 0.8, 0.5, 110, dry=0.55), (-0.4, 0.4, 0.0, 0.5), (768, 0, 256, 160), True),
    }


GRASS_ATLAS_SIZE = (1024, 160)


def grass_tile_uv(name):
    px, py, w, h = grass_tiles()[name][2]
    W, H = GRASS_ATLAS_SIZE
    return (px / W, py / H, (px + w) / W, (py + h) / H)
