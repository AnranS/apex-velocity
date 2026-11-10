"""Procedural trees -> GLB (LOD0 + LOD1).   python3 gen_trees.py [ids...]

Materials: `bark` (tiling bark texture, UV repeats) and `leaves` (alpha-cutout cards on foliage_atlas).
COLOR_0: R = wind sway weight (0 trunk base .. 1 leaf tips), G = per-branch phase, B = ambient occlusion.
Foliage normals are "puffed" (outward from the crown), identical on both faces of a card.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foliage_src as fs  # noqa: E402
from glb import write_glb, material_def  # noqa: E402
from treelib import (UP, Acc, norm, lerp, smoothstep, tube, bezier, arc_point, make_card, fib_dirs,  # noqa: E402
                     random_rotation, trunk_axis_fn)

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))

MATS = {
    "bark": material_def("bark", color=(1.0, 1.0, 1.0, 1.0), roughness=0.95),
    "leaves": material_def("leaves", color=(1.0, 1.0, 1.0, 1.0), roughness=0.78, alpha_mode="MASK", alpha_cutoff=0.5,
                           double_sided=True),
}


# ============================================================================ conifer (fir / spruce)

def gen_fir(seed, lod, H=19.5, r_base=0.30, y0=2.5, Rmax=4.0, tiles=("fir_a", "fir_b"), dense=1.0):
    rs = np.random.default_rng(seed)             # structure (shared by both LODs)
    rng = np.random.default_rng(seed * 7 + 13 + lod)
    bark, leaves = Acc(), Acc()
    ph = rs.uniform(0, 6.28, 2)
    lean = rs.uniform(-0.3, 0.3, 2)
    axis = trunk_axis_fn(H, 0.2, ph[0], ph[1], lean)

    fr = np.array([0, 0.012, 0.032, 0.08, 0.17, 0.29, 0.42, 0.55, 0.68, 0.8, 0.9, 1.0]) if lod == 0 else \
        np.array([0, 0.03, 0.19, 0.47, 0.76, 1.0])
    ys = fr * H
    rad = np.maximum(r_base * (1 - ys / H) ** 0.8 * (1 + 0.55 * np.exp(-ys / 0.25)), 0.02)

    def wind_trunk(y):
        return 0.30 * (np.asarray(y, dtype=np.float64) / H) ** 2

    tube(bark, axis(ys), rad, 10 if lod == 0 else 6, wind_trunk(ys), 0.0,
         np.interp(ys, [0, y0, y0 + 3.0], [0.95, 0.72, 0.45]), tex_scale=1.0)

    def crown_r(y):
        t = np.clip((np.asarray(y, dtype=np.float64) - y0) / (H - y0), 0, 1)
        return Rmax * (1 - t) ** 0.8 * (0.72 + 0.28 * smoothstep(0.0, 0.18, t)) + 0.16

    def add(card, h_dir, s0, s1, phase, ybase, ao_mul=1.0):
        pos, uv, t, edge, nl, tris = card
        ax = axis(pos[:, 1])
        rh = pos - ax
        rh[:, 1] = 0.0
        rr = np.linalg.norm(rh, axis=1)
        rdir = norm(0.5 * h_dir[None, :] + 0.5 * rh / np.maximum(rr, 0.05)[:, None])
        n = norm(0.78 * rdir + 0.55 * UP[None, :] + 0.2 * nl)
        tt = np.clip((pos[:, 1] - y0) / (H - y0), 0, 1)
        rho = rr / crown_r(pos[:, 1])
        ao = (0.28 + 0.72 * smoothstep(0.12, 0.9, rho)) * (0.8 + 0.2 * tt) * ao_mul
        s = s0 + (s1 - s0) * t
        wind = np.clip(wind_trunk(ybase) + 0.62 * s ** 1.1 + 0.1 * edge, 0, 1)
        leaves.add(pos, n, uv, np.stack([wind, np.full(len(pos), phase), np.clip(ao, 0, 1)], 1), tris)

    def tent_cols(W, phi):
        return [(W / 2 * np.cos(phi), -W / 2 * np.sin(phi), 0.0), (0.0, 0.0, 0.5), (-W / 2 * np.cos(phi), -W / 2 * np.sin(phi), 1.0)]

    def flat_cols(W):
        return [(W / 2, 0.0, 0.0), (-W / 2, 0.0, 1.0)]

    def tile():
        return fs.tile_uv(tiles[int(rng.integers(0, len(tiles)))])

    # a few dead stubs under the live crown
    if lod == 0:
        for k in range(6):
            yb = rng.uniform(1.3, y0)
            az = rng.uniform(0, 6.28)
            h = np.array([np.cos(az), 0.0, np.sin(az)])
            ln = rng.uniform(0.5, 1.3)
            p0 = axis(yb)
            pts = np.array([p0, p0 + h * ln * 0.5 - UP * 0.05 * ln, p0 + h * ln - UP * 0.22 * ln])
            tube(bark, pts, [0.03, 0.02, 0.008], 3, float(wind_trunk(yb)), 0.0, 0.7)

    spacing_mul = 1.0 if lod == 0 else 1.5
    wmul = 1.0 if lod == 0 else 1.6
    y = y0
    wi = 0
    while y < H - 1.0:
        t = (y - y0) / (H - y0)
        nb = int(round(lerp(6.4, 4.6, t) * dense)) if lod == 0 else (6 if t < 0.5 else 5)
        az0 = wi * 2.39996 + rs.uniform(0, 0.6)
        for b in range(nb):
            if lod == 0 and rng.uniform() < 0.05:
                continue
            az = az0 + b * 2 * np.pi / nb + rng.uniform(-0.26, 0.26)
            yb = y + rng.uniform(-0.14, 0.14)
            Lb = float(crown_r(yb)) * rng.uniform(0.74, 1.12)
            pitch0 = np.radians(lerp(-10.0, 40.0, t ** 1.15) + rng.uniform(-7, 7))
            droop = np.radians(lerp(27.0, 7.0, t) * rng.uniform(0.75, 1.25))
            phase = rng.uniform()
            O = axis(yb)
            h = np.array([np.cos(az), 0.0, np.sin(az)])
            D0 = norm(h * np.cos(pitch0) + UP * np.sin(pitch0))
            S = np.cross(h, UP)
            N0 = np.cross(S, D0)
            base = O + D0 * 0.1
            Lc = Lb * 0.98
            phi = np.radians(rng.uniform(16, 30))
            long_branch = Lb > 1.9
            if lod == 0 and long_branch:
                # outer frond + a fan of side boughs along the limb
                s_main = rng.uniform(0.32, 0.4)
                p, dS, nS = arc_point(base, D0, N0, Lc, droop, s_main)
                Lm = Lc * (1 - s_main) * 1.05
                add(make_card(p, dS, np.cross(dS, nS), Lm, tent_cols(Lm * 0.5, phi), tile(), 3, bend=droop * (1 - s_main)),
                    h, s_main, 1.0, phase, yb)
                for sgn, s_lo, s_hi, lf in ((1.0, 0.08, 0.18, 0.56), (-1.0, 0.2, 0.32, 0.54), (1.0, 0.4, 0.5, 0.46), (-1.0, 0.54, 0.64, 0.4)):
                    s_att = rng.uniform(s_lo, s_hi)
                    p, dS, nS = arc_point(base, D0, N0, Lc, droop, s_att)
                    sS = np.cross(dS, nS)
                    psi = np.radians(rng.uniform(38, 58))
                    d2 = norm(dS * np.cos(psi) + sgn * sS * np.sin(psi) + nS * rng.uniform(-0.12, 0.18))
                    n2 = norm(nS - d2 * np.dot(nS, d2))
                    s2 = np.cross(d2, n2)
                    roll = np.radians(rng.uniform(-28, 28))
                    s2 = norm(s2 * np.cos(roll) + n2 * np.sin(roll))
                    L2 = Lb * lf * rng.uniform(0.88, 1.12)
                    add(make_card(p, d2, s2, L2, flat_cols(L2 * 0.5), tile(), 2, bend=droop * 0.6 + np.radians(8)),
                        h, s_att, 1.0, phase, yb)
            else:
                W = Lc * 0.5 * rng.uniform(0.92, 1.06) * wmul
                if lod == 0:
                    nseg = 2
                else:
                    nseg = 2 if long_branch else 1
                add(make_card(base - D0 * 0.12 * Lc, D0, S, Lc * 1.1, tent_cols(W, phi), tile(), nseg, bend=droop), h, 0.0, 1.0, phase, yb)
            # hanging curtain under the bough (keeps the tree opaque from the side)
            hfin = min(0.34 * Lb, 0.95) * (1.0 if lod == 0 else 1.45)
            add(make_card(base + D0 * 0.05 * Lc, D0, S, Lc * 0.93, [(0.0, 0.03, 0.5), (0.0, -hfin, 1.0)], tile(),
                          2 if lod == 0 else 1, bend=droop), h, 0.05, 0.98, phase, yb, ao_mul=0.82)
        # short inner boughs hide the trunk between whorls
        if lod == 0:
            for k in range(3):
                az = rng.uniform(0, 6.28)
                yb = y + rng.uniform(0.05, 0.35)
                h = np.array([np.cos(az), 0.0, np.sin(az)])
                pitch = np.radians(lerp(8.0, 50.0, t) + rng.uniform(-8, 12))
                D0 = norm(h * np.cos(pitch) + UP * np.sin(pitch))
                S = np.cross(h, UP)
                roll = np.radians(rng.uniform(-35, 35))
                S = norm(S * np.cos(roll) + np.cross(S, D0) * np.sin(roll))
                L2 = max(0.8, float(crown_r(yb)) * rng.uniform(0.5, 0.75))
                add(make_card(axis(yb) - D0 * 0.16 * L2, D0, S, L2, flat_cols(L2 * 0.5), tile(), 2, bend=np.radians(rng.uniform(5, 20))),
                    h, 0.0, 0.7, rng.uniform(), yb, ao_mul=0.9)
        y += lerp(0.60, 0.33, t) * spacing_mul
        wi += 1

    # crown tip: a few steep boughs + the leader
    top = axis(H - 1.0)
    for b in range(5 if lod == 0 else 3):
        az = b * 2 * np.pi / (5 if lod == 0 else 3) + rng.uniform(0, 1.0)
        h = np.array([np.cos(az), 0.0, np.sin(az)])
        pitch = np.radians(rng.uniform(52, 68))
        D0 = norm(h * np.cos(pitch) + UP * np.sin(pitch))
        S = np.cross(h, UP)
        Lc = rng.uniform(0.95, 1.25)
        add(make_card(top + UP * rng.uniform(-0.3, 0.2), D0, S, Lc, tent_cols(Lc * 0.5, np.radians(22)), fs.tile_uv(tiles[b % len(tiles)]),
                      2 if lod == 0 else 1, bend=np.radians(10)), h, 0.6, 1.0, rng.uniform(), H - 1.0)
    for k in range(2):
        h = np.array([np.cos(k * np.pi / 2 + 0.4), 0.0, np.sin(k * np.pi / 2 + 0.4)])
        Lc = 1.75
        card = make_card(axis(H - 1.45), UP, h, Lc, [(Lc * 0.24, 0.0, 0.0), (-Lc * 0.24, 0.0, 1.0)], fs.tile_uv(tiles[k % len(tiles)]), 1)
        add(card, h, 0.7, 1.0, 0.5, H - 1.0)
    return bark, leaves, {"height": H, "radius": Rmax}


# ============================================================================ crown trees (pine umbrella, broadleaf)

def gen_crown_tree(seed, lod, P):
    rs = np.random.default_rng(seed)
    rng = np.random.default_rng(seed * 11 + 5 + lod)
    bark, leaves = Acc(), Acc()
    H = P["H"]
    y_top = P["leader_top"]
    lean = np.array(P.get("lean", (0.0, 0.0))) * rs.choice([-1.0, 1.0], 2)
    ph = rs.uniform(0, 6.28, 2)
    axis = trunk_axis_fn(y_top, P.get("curve", 0.15), ph[0], ph[1], lean)
    cc = np.array(P["crown_c"], dtype=np.float64)
    ax_c = axis(min(cc[1], y_top))
    cc = cc + np.array([ax_c[0], 0.0, ax_c[2]])
    cr = np.array(P["crown_r"], dtype=np.float64)
    r_base = P["trunk_r"]
    fy0, fy1 = P["fork"]

    def trunk_r(y):
        # r_base = radius at breast height; nearly cylindrical bole, short root flare, quick taper above the fork
        y = np.asarray(y, dtype=np.float64)
        low = r_base * (1 - 0.14 * np.clip(y / fy0, 0, 1))
        up = np.clip((y - fy0) / (y_top - fy0), 0, 1)
        r = low * (1 - up) ** 1.25 + 0.03 * up
        return np.maximum(r * (1 + 0.55 * np.exp(-y / 0.22)), 0.025)

    def wind_trunk(y):
        return 0.26 * (np.asarray(y, dtype=np.float64) / H) ** 2

    if lod == 0:
        fr = np.array([0, 0.012, 0.035, 0.09, 0.2, 0.33, 0.46, 0.6, 0.74, 0.87, 1.0])
    else:
        fr = np.array([0, 0.03, 0.22, 0.5, 0.78, 1.0])
    ys = fr * y_top
    tube(bark, axis(ys), trunk_r(ys), 10 if lod == 0 else 6, wind_trunk(ys), 0.0,
         np.interp(ys, [0, fy0, fy0 + 1.5], [0.92, 0.76, 0.5]), tex_scale=P.get("bark_scale", 1.0))

    # ---- lobes (foliage masses)
    n_lobes = P["n_lobes"]
    under = P["under"]
    cand = fib_dirs(int(round(n_lobes / ((1 - under) / 2))), rs)
    cand = cand[cand[:, 1] > under][:n_lobes]
    rho = rs.uniform(P.get("shell", (0.64, 0.92))[0], P.get("shell", (0.64, 0.92))[1], len(cand))
    centres = cc + cand * cr * rho[:, None] + rs.normal(0, 0.22, (len(cand), 3))
    n_inner = int(round(P.get("inner", 0.2) * n_lobes))
    inner = cc + rs.normal(0, 1.0, (n_inner, 3)) * cr * 0.28 + np.array([0.0, cr[1] * 0.22, 0.0])
    centres = np.concatenate([centres, inner])
    lobe_r = rs.uniform(P["lobe_r"][0], P["lobe_r"][1], len(centres))
    lobe_phase = rs.uniform(0, 1, len(centres))
    flat = P.get("flat", 0.85)

    # ---- limbs: k-means over lobe centres
    K = P["n_limbs"]
    az = np.arctan2(centres[:, 2] - cc[2], centres[:, 0] - cc[0])
    order = np.argsort(az)
    cent = np.array([centres[order[int((i + 0.5) * len(order) / K)]] for i in range(K)])
    for _ in range(8):
        d = np.linalg.norm(centres[:, None, :] - cent[None, :, :], axis=2)
        lab = np.argmin(d, axis=1)
        for k in range(K):
            if np.any(lab == k):
                cent[k] = centres[lab == k].mean(axis=0)
    limb_pts = []
    for k in range(K):
        E = cc + (cent[k] - cc) * P.get("limb_reach", 0.62)
        axE = axis(min(E[1], y_top))
        hd = np.hypot(E[0] - axE[0], E[2] - axE[2])
        y_s = float(np.clip(E[1] - P.get("limb_rise", 1.0) * hd - 0.2, fy0, fy1))
        S0 = axis(y_s)
        ln = np.linalg.norm(E - S0)
        C = S0 + (E - S0) * 0.45 + UP * 0.22 * ln * P.get("limb_arc", 1.0)
        pts = bezier(S0, C, E, 7 if lod == 0 else 4)
        limb_pts.append(pts)
        u = np.linspace(0, 1, len(pts))
        r0 = float(trunk_r(y_s)) * 0.7
        tube(bark, pts, lerp(r0, 0.06, u ** 0.8), 6 if lod == 0 else 4, lerp(float(wind_trunk(y_s)), 0.45, u), 0.0,
             lerp(0.55, 0.42, u), tex_scale=P.get("bark_scale", 1.0))

    # ---- twigs to every lobe (LOD0 only; hidden inside foliage at distance)
    if lod == 0:
        for i, c in enumerate(centres):
            k = int(lab[i]) if i < len(lab) else 0
            pts = limb_pts[k]
            d = np.linalg.norm(pts[2:] - c[None, :], axis=1)
            j = int(np.argmin(d)) + 2
            st = pts[j]
            ln = np.linalg.norm(c - st)
            if ln < 0.35:
                continue
            ctrl = st + (c - st) * 0.5 + UP * 0.12 * ln + rng.normal(0, 0.08, 3) * ln
            tw = bezier(st, ctrl, c, 4)
            u = np.linspace(0, 1, 4)
            tube(bark, tw, lerp(0.05, 0.014, u), 3, lerp(0.45, 0.6, u), lobe_phase[i], 0.45, tex_scale=P.get("bark_scale", 1.0))

    # ---- foliage cards
    n_cards = P["cards"] if lod == 0 else P["cards1"]
    size_mul = 1.0 if lod == 0 else P.get("lod1_scale", 1.4)
    cr_mean = float(cr.mean())

    def add(card, c, rl, phase, wind0):
        pos, uv, t, edge, nl, tris = card
        out_c = (pos - cc) / cr
        rho_c = np.linalg.norm(out_c, axis=1)
        n = norm(0.5 * norm(pos - c) + 0.62 * norm(out_c) + 0.28 * UP[None, :] + 0.12 * nl)
        dn = norm(out_c)[:, 1]
        ao = (0.32 + 0.68 * smoothstep(0.35, 0.95, rho_c)) * lerp(0.72, 1.0, smoothstep(-0.55, 0.3, dn))
        wind = np.clip(wind0 + 0.18 * np.clip(rho_c, 0, 1) + 0.3 * t + 0.06 * edge, 0, 1)
        g = np.clip(phase + rng.uniform(-0.04, 0.04), 0, 1)
        leaves.add(pos, n, uv, np.stack([wind, np.full(len(pos), g), np.clip(ao, 0, 1)], 1), tris)

    tiles_s, tiles_l = P["tiles"], P["tiles_long"]
    for i, c in enumerate(centres):
        rl = lobe_r[i]
        rr = np.array([rl, rl * flat, rl])
        dirs = fib_dirs(n_cards, rng) @ random_rotation(rng).T
        for d in dirs:
            p = c + d * rr * rng.uniform(0.5, 0.95)
            out = norm((p - cc) / cr)
            if np.dot(d, out) < -0.55 and rng.uniform() < 0.7:
                p = c + (-d) * rr * rng.uniform(0.5, 0.95)   # mirror inward-facing cards to the outside
                d = -d
            Nc = norm(d + 0.45 * rng.normal(size=3) + 0.35 * out)
            axp = axis(min(p[1], y_top))
            hd = np.array([p[0] - axp[0], 0.0, p[2] - axp[2]])
            hd = norm(hd) if np.linalg.norm(hd) > 1e-3 else np.array([1.0, 0.0, 0.0])
            Dd = hd * 0.8 + rng.normal(size=3) * 0.6 - UP * P.get("droop", 0.3)
            Dd = Dd - Nc * np.dot(Dd, Nc)
            if np.linalg.norm(Dd) < 1e-3:
                Dd = np.cross(Nc, UP)
            Dd = norm(Dd)
            S = np.cross(Dd, Nc)
            if rng.uniform() < P["long_ratio"]:
                tile = tiles_l[int(rng.integers(0, len(tiles_l)))]
                L = rng.uniform(*P["size_long"]) * size_mul
                W = L * 0.5
            else:
                tile = tiles_s[int(rng.integers(0, len(tiles_s)))]
                L = rng.uniform(*P["size"]) * size_mul
                W = L
            cols = [(W / 2, 0.0, 0.0), (-W / 2, 0.0, 1.0)]
            nseg = 2 if lod == 0 else 1
            card = make_card(p - Dd * L * 0.5, Dd, S, L, cols, fs.tile_uv(tile), nseg, bend=np.radians(rng.uniform(8, 34)))
            add(card, c, rl, lobe_phase[i], P.get("wind0", 0.5))
        # two crossed vertical cards through the lobe centre: opaque from every side
        a0 = rng.uniform(0, np.pi)
        for k in range(2):
            hdir = np.array([np.cos(a0 + k * np.pi / 2), 0.0, np.sin(a0 + k * np.pi / 2)])
            L = rl * 1.7 * (1.0 if lod == 0 else 1.15)
            tile = tiles_s[int(rng.integers(0, len(tiles_s)))]
            card = make_card(c - hdir * L * 0.5, hdir, UP, L, [(L * flat * 0.5, 0.0, 0.0), (-L * flat * 0.5, 0.0, 1.0)],
                             fs.tile_uv(tile), 1)
            add(card, c, rl, lobe_phase[i], P.get("wind0", 0.5))
    return bark, leaves, {"height": H, "radius": float(max(cr[0], cr[2]))}


SPECIES = {
    "pine_a": lambda lod: gen_fir(101, lod, H=19.5, r_base=0.27, y0=2.5, Rmax=4.0),
    "pine_b": lambda lod: gen_crown_tree(202, lod, dict(
        H=15.6, leader_top=12.6, trunk_r=0.31, fork=(5.6, 10.2), lean=(0.7, 0.4), curve=0.32,
        crown_c=(0.0, 12.4, 0.0), crown_r=(5.4, 3.0, 5.4), under=-0.1, shell=(0.55, 0.93), inner=0.3, flat=0.6,
        n_lobes=26, lobe_r=(1.2, 1.7), n_limbs=6, limb_reach=0.72, limb_rise=0.85, limb_arc=0.8,
        cards=24, cards1=8, tiles=("pine_tuft_a", "pine_tuft_b"), tiles_long=("pine_branch",), long_ratio=0.35,
        size=(1.1, 1.5), size_long=(1.8, 2.4), droop=0.1, wind0=0.5, bark_scale=1.0)),
    "broadleaf_a": lambda lod: gen_crown_tree(303, lod, dict(
        H=13.0, leader_top=8.2, trunk_r=0.37, fork=(2.3, 5.0), lean=(0.25, 0.2), curve=0.16,
        crown_c=(0.0, 7.9, 0.0), crown_r=(5.2, 4.6, 5.2), under=-0.6, shell=(0.6, 0.93), inner=0.25, flat=0.85,
        n_lobes=34, lobe_r=(1.1, 1.6), n_limbs=5, limb_reach=0.6, limb_rise=0.9, limb_arc=1.0,
        cards=28, cards1=8, tiles=("leaf_a1", "leaf_a2"), tiles_long=("leaf_a_branch",), long_ratio=0.25,
        size=(0.95, 1.35), size_long=(1.7, 2.3), droop=0.3, wind0=0.5, bark_scale=1.0)),
    "broadleaf_b": lambda lod: gen_crown_tree(404, lod, dict(
        H=10.9, leader_top=8.3, trunk_r=0.165, fork=(2.2, 5.4), lean=(0.2, 0.15), curve=0.12,
        crown_c=(0.0, 6.6, 0.0), crown_r=(2.8, 4.2, 2.8), under=-0.75, shell=(0.55, 0.9), inner=0.3, flat=1.0,
        n_lobes=34, lobe_r=(0.7, 1.0), n_limbs=5, limb_reach=0.55, limb_rise=1.3, limb_arc=0.6,
        cards=20, cards1=6, tiles=("leaf_b1", "leaf_b2"), tiles_long=("leaf_b_branch",), long_ratio=0.3,
        size=(0.8, 1.1), size_long=(1.3, 1.7), droop=0.45, wind0=0.52, bark_scale=0.7)),
}


def build(tree_id):
    out = []
    for lod in (0, 1):
        bark, leaves, info = SPECIES[tree_id](lod)
        parts = [p for p in (bark.part("bark"), leaves.part("leaves")) if p is not None]
        name = f"{tree_id}_lod{lod}"
        path = os.path.join(OUT, name + ".glb")
        size = write_glb(path, [{"name": name, "parts": parts}], MATS)
        allp = np.concatenate([p.positions for p in parts])
        print(f"{name}: tris={sum(p.triangles for p in parts)} (bark {bark.triangles}, leaves {leaves.triangles}) "
              f"verts={len(allp)} bounds=({allp.min(0).round(2)} .. {allp.max(0).round(2)}) {size // 1024} KB")
        out.append(path)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ids = sys.argv[1:] or list(SPECIES.keys())
    for tid in ids:
        build(tid)
