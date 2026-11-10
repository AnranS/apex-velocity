"""Race facility props -> GLB (run inside Blender for the BVH used by the AO bake).

  Blender -b --factory-startup -P gen_props.py -- [ids,comma,separated]

All props: origin at ground centre, +Y up, metres, front = +Z, flat PBR colours on named materials,
COLOR_0 = baked ambient occlusion (grayscale).
"""
import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glb import write_glb, material_def  # noqa: E402
from proplib import Kit, X, Y, Z, v3, rot_x, rot_y  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def M(name, col, rough=0.8, metal=0.0, **kw):
    return material_def(name, color=(*col, 1.0), metallic=metal, roughness=rough, **kw)


MATS = {m["name"]: m for m in [
    M("concrete", (0.50, 0.49, 0.47), 0.92),
    M("steel", (0.44, 0.46, 0.49), 0.5, 0.45),
    M("steel_dark", (0.07, 0.075, 0.085), 0.55, 0.4),
    M("panel", (0.80, 0.81, 0.83), 0.5, 0.05),
    M("panel_dark", (0.05, 0.055, 0.065), 0.5, 0.05),
    M("seats", (0.62, 0.05, 0.04), 0.6),
    M("crowd", (1.0, 1.0, 1.0), 0.9, alpha_mode="MASK", alpha_cutoff=0.5, double_sided=True),
    M("glass", (0.04, 0.06, 0.08), 0.06),
    M("garage_door", (0.62, 0.63, 0.66), 0.55, 0.3),
    M("ad", (1.0, 1.0, 1.0), 0.6),
    M("lamp", (0.95, 0.95, 0.9), 0.3, emissive=(1.0, 0.95, 0.85)),
    M("tire", (0.015, 0.015, 0.016), 0.92),
    M("paint_band", (0.85, 0.85, 0.83), 0.7),
    M("cone_orange", (0.9, 0.17, 0.02), 0.6),
    M("cone_white", (0.9, 0.9, 0.88), 0.45),
    M("flag", (1.0, 1.0, 1.0), 0.8, double_sided=True),
    M("marshal_orange", (0.92, 0.33, 0.03), 0.6),
    M("wood", (0.36, 0.25, 0.15), 0.85),
    M("interior_dark", (0.012, 0.012, 0.014), 0.9),
] + [M(f"startlight_{i}", (0.10, 0.01, 0.01), 0.3, emissive=(0.0, 0.0, 0.0)) for i in range(5)]}


def rail(k, mat, a, b, height, posts_every=2.0, bars=(1.0, 0.55), post=0.05, bar=0.045):
    """Simple guard rail along the segment a-b standing on its points."""
    a, b = v3(a), v3(b)
    ln = np.linalg.norm(b - a)
    n = max(1, int(round(ln / posts_every)))
    for i in range(n + 1):
        p = a + (b - a) * (i / n)
        k.beam(mat, p, p + Y * height, post)
    for f in bars:
        k.beam(mat, a + Y * height * f, b + Y * height * f, bar)


# ============================================================================ grandstand (24 m module)

def grandstand():
    k = Kit(max_edge=2.0)
    x0, x1 = -12.0, 12.0
    rows, depth, rise = 12, 0.8, 0.42
    zf, y1 = 5.0, 1.7                # front wall plane, first-row floor height
    zr0 = zf - 1.2                   # front walkway depth 1.2
    h = [y1 + i * rise for i in range(rows)]
    zb = zr0 - rows * depth          # back edge of the top tread
    zw = zb - 0.5                    # back wall plane
    # front wall + walkway
    k.quad("concrete", (x0, 0, zf), (x1, 0, zf), (x1, y1, zf), (x0, y1, zf), out=Z)
    k.quad("concrete", (x0, y1, zf), (x1, y1, zf), (x1, y1, zr0), (x0, y1, zr0), out=Y)
    for xe, o in ((x0, -X), (x1, X)):
        k.quad("concrete", (xe, 0, zf), (xe, y1, zf), (xe, y1, zr0), (xe, 0, zr0), out=o)
    # stepped slab: treads, risers, end caps
    for i in range(rows):
        za, zc = zr0 - i * depth, zr0 - (i + 1) * depth
        k.quad("concrete", (x0, h[i], za), (x1, h[i], za), (x1, h[i], zc), (x0, h[i], zc), out=Y)
        if i > 0:
            k.quad("concrete", (x0, h[i - 1], za), (x1, h[i - 1], za), (x1, h[i], za), (x0, h[i], za), out=Z)
        for xe, o in ((x0, -X), (x1, X)):
            k.quad("concrete", (xe, 0, za), (xe, h[i], za), (xe, h[i], zc), (xe, 0, zc), out=o)
    top = h[-1]
    k.quad("concrete", (x0, top, zb), (x1, top, zb), (x1, top, zw), (x0, top, zw), out=Y)
    for xe, o in ((x0, -X), (x1, X)):
        k.quad("concrete", (xe, 0, zb), (xe, top, zb), (xe, top, zw), (xe, 0, zw), out=o)
    # seating blocks between two stair aisles
    aisles = [(-4.6, -3.4), (3.4, 4.6)]
    blocks = [(-11.7, -4.6), (-3.4, 3.4), (4.6, 11.7)]
    rng = np.random.default_rng(3)
    for i in range(rows):
        za = zr0 - i * depth
        for (xa, xb) in blocks:
            xc, bw = (xa + xb) / 2, xb - xa
            k.box("steel_dark", (xc, h[i] + 0.2, za - 0.58), (bw, 0.4, 0.06), faces="XxZz")
            k.box("seats", (xc, h[i] + 0.43, za - 0.47), (bw, 0.06, 0.42), faces="XxYZz")
            k.box("seats", (xc, h[i] + 0.66, za - 0.71), (bw, 0.42, 0.05), faces="XxYZz")
            # crowd card (alpha cutout spectators): u repeats every 16 seats (8 m), v top -> bottom
            u0 = rng.integers(0, 16) / 16.0
            ua, ub = u0 + (xa - xa) / 8.0, u0 + (xb - xa) / 8.0
            yb, yt = h[i] + 0.44, h[i] + 0.44 + 0.86
            k.quad("crowd", (xa, yb, za - 0.42), (xb, yb, za - 0.42), (xb, yt, za - 0.56), (xa, yt, za - 0.56),
                   uv=[(ua, 1.0), (ub, 1.0), (ub, 0.0), (ua, 0.0)], out=Z, max_edge=1e9)
        if i > 0:   # half steps in the aisles
            for (xa, xb) in aisles:
                k.box("concrete", ((xa + xb) / 2, h[i - 1] + rise / 4, za + depth / 4), (xb - xa, rise / 2, depth / 2), faces="XxYZ")
    # front rail on the walkway edge
    rail(k, "steel", (x0, y1, zf - 0.06), (x1, y1, zf - 0.06), 1.05)
    # back wall (outside cladding + inside face above the top row)
    y_back = 9.0
    k.quad("panel", (x0, 0, zw), (x1, 0, zw), (x1, y_back, zw), (x0, y_back, zw), out=-Z)
    k.quad("panel_dark", (x0, top, zw + 0.02), (x1, top, zw + 0.02), (x1, y_back, zw + 0.02), (x0, y_back, zw + 0.02), out=Z)
    # cantilever roof, rising toward the track
    zr_b, zr_f = zw - 0.4, zf + 1.3
    yr_b, yr_f = 9.0, 10.7
    t = 0.22
    k.quad("panel", (x0, yr_b + t, zr_b), (x1, yr_b + t, zr_b), (x1, yr_f + t, zr_f), (x0, yr_f + t, zr_f), out=Y)
    k.quad("panel", (x0, yr_b, zr_b), (x1, yr_b, zr_b), (x1, yr_f, zr_f), (x0, yr_f, zr_f), out=-Y)
    k.quad("panel_dark", (x0, yr_f, zr_f), (x1, yr_f, zr_f), (x1, yr_f + t, zr_f), (x0, yr_f + t, zr_f), out=Z)
    k.quad("panel_dark", (x0, yr_b, zr_b), (x1, yr_b, zr_b), (x1, yr_b + t, zr_b), (x0, yr_b + t, zr_b), out=-Z)
    for xe, o in ((x0, -X), (x1, X)):
        k.quad("panel_dark", (xe, yr_b, zr_b), (xe, yr_b + t, zr_b), (xe, yr_f + t, zr_f), (xe, yr_f, zr_f), out=o)

    def roof_y(z):
        return yr_b + (yr_f - yr_b) * (z - zr_b) / (zr_f - zr_b)

    # steel frames: rear column + tapering raking truss under the roof
    zc = zw - 0.05
    for xf in (-9.0, -3.0, 3.0, 9.0):
        k.box("steel", (xf, roof_y(zc) / 2, zc), (0.42, roof_y(zc), 0.42), faces="XxZz")
        n = 9
        zt = zr_f - 0.5
        tops = [v3(xf, roof_y(zc + (zt - zc) * i / n) - 0.08, zc + (zt - zc) * i / n) for i in range(n + 1)]
        bots = [tops[i] - Y * (1.5 * (1 - i / n) ** 1.3 + 0.28) for i in range(n + 1)]
        k.beam("steel", tops[0], tops[-1], 0.16)
        for i in range(n):
            k.beam("steel", bots[i], bots[i + 1], 0.14)
            k.beam("steel", bots[i], tops[i + 1], 0.09) if i % 2 == 0 else k.beam("steel", tops[i], bots[i + 1], 0.09)
            k.beam("steel", bots[i], tops[i], 0.08)
        k.beam("steel", bots[-1], tops[-1], 0.08)
        # stay from the column down to the slab top keeps the cantilever believable
        k.beam("steel", v3(xf, top + 0.1, zc + 0.6), bots[2], 0.12)
    for i in range(6):   # purlins
        z = zr_b + 0.9 + (zr_f - zr_b - 1.6) * i / 5
        k.beam("steel", (x0, roof_y(z) - 0.1, z), (x1, roof_y(z) - 0.1, z), 0.1, 0.16)
    return k


# ============================================================================ pit building (12 m module)

def pit_building():
    k = Kit(max_edge=2.0)
    x0, x1 = -6.0, 6.0
    zf, zb = 5.0, -5.0
    y_slab, y_roof = 4.2, 7.6
    # ground storey: piers + lintel in the front plane, recessed sectional doors
    for (xa, xb) in ((-6.0, -5.6), (-0.4, 0.4), (5.6, 6.0)):
        k.box("concrete", ((xa + xb) / 2, y_slab / 2, zf - 0.3), (xb - xa, y_slab, 0.6), faces="XxZ")
    k.box("concrete", (0, 3.9, zf - 0.3), (12.0, 0.6, 0.6), faces="Zy")
    for (xa, xb) in ((-5.6, -0.4), (0.4, 5.6)):
        zd = zf - 0.5
        nsl = 9
        for s in range(nsl):   # door slats (slight relief so the AO reads as a sectional door)
            ya, yb = 3.6 * s / nsl, 3.6 * (s + 1) / nsl
            k.quad("garage_door", (xa, ya + 0.015, zd), (xb, ya + 0.015, zd), (xb, yb - 0.015, zd), (xa, yb - 0.015, zd), out=Z,
                   uv=[(0, 1 - s / nsl), (1, 1 - s / nsl), (1, 1 - (s + 1) / nsl), (0, 1 - (s + 1) / nsl)], max_edge=2.7)
        k.quad("steel_dark", (xa, 0, zd - 0.02), (xb, 0, zd - 0.02), (xb, 3.6, zd - 0.02), (xa, 3.6, zd - 0.02), out=Z)
        # bay number plate
        k.box("panel_dark", ((xa + xb) / 2, 3.9, zf + 0.02), (1.1, 0.42, 0.04), faces="XxYyZ")
    # side + back walls (panel cladding), full height
    for xe, o in ((x0, -X), (x1, X)):
        k.quad("panel", (xe, 0, zb), (xe, y_roof, zb), (xe, y_roof, zf - 0.6), (xe, 0, zf - 0.6), out=o)
    k.quad("panel", (x0, 0, zb), (x1, 0, zb), (x1, y_roof, zb), (x0, y_roof, zb), out=-Z)
    k.box("glass", (0, 5.9, zb - 0.02), (10.4, 1.3, 0.04), faces="z")
    k.box("panel_dark", (0, 5.9, zb - 0.01), (10.7, 1.6, 0.02), faces="z")
    # first-floor slab with balcony / canopy over the pit lane
    k.box("concrete", (0, y_slab + 0.15, (zb + zf + 1.4) / 2), (12.0, 0.3, zf + 1.4 - zb), faces="YyZ")
    k.box("panel_dark", (0, y_slab + 0.15, zf + 1.42), (12.0, 0.34, 0.04), faces="ZYy")
    # glazed hospitality level
    zg = zf - 0.8
    yg0, yg1 = y_slab + 0.3, y_roof
    k.quad("glass", (x0, yg0, zg), (x1, yg0, zg), (x1, yg1, zg), (x0, yg1, zg), out=Z)
    for i in range(9):
        x = x0 + 12.0 * i / 8
        k.box("steel_dark", (x, (yg0 + yg1) / 2, zg + 0.04), (0.07, yg1 - yg0, 0.1), faces="XxZ")
    k.box("steel_dark", (0, yg0 + 1.05, zg + 0.04), (12.0, 0.06, 0.09), faces="YyZ")
    for xe, o in ((x0, -X), (x1, X)):   # upper side walls in front of the glazing line
        k.quad("panel", (xe, yg0, zg), (xe, yg1, zg), (xe, yg1, zf - 0.6), (xe, yg0, zf - 0.6), out=o)
    # balcony balustrade (glass + rail)
    zbal = zf + 1.3
    k.quad("glass", (x0, yg0, zbal), (x1, yg0, zbal), (x1, yg0 + 1.0, zbal), (x0, yg0 + 1.0, zbal), out=Z)
    k.beam("steel", (x0, yg0 + 1.04, zbal), (x1, yg0 + 1.04, zbal), 0.07, 0.05)
    for i in range(7):
        x = x0 + 12.0 * i / 6
        k.beam("steel", (x, yg0, zbal), (x, yg0 + 1.04, zbal), 0.05)
    # roof slab with overhang, terrace rail, plant
    k.box("concrete", (0, y_roof + 0.17, (zb - 0.3 + zf + 0.9) / 2), (12.0, 0.34, zf + 0.9 - (zb - 0.3)), faces="YyZz")
    k.box("panel_dark", (0, y_roof + 0.17, zf + 0.92), (12.0, 0.38, 0.04), faces="Z")
    rail(k, "steel", (x0, y_roof + 0.34, zf + 0.75), (x1, y_roof + 0.34, zf + 0.75), 1.05)
    rail(k, "steel", (x0, y_roof + 0.34, zb - 0.15), (x1, y_roof + 0.34, zb - 0.15), 1.05)
    k.box("panel", (-3.2, y_roof + 0.34 + 0.55, -2.6), (2.2, 1.1, 1.5), faces="XxYZz")
    k.box("steel_dark", (-3.2, y_roof + 0.34 + 1.13, -2.6), (2.3, 0.06, 1.6), faces="XxYZz")
    k.box("panel", (2.6, y_roof + 0.34 + 0.4, -3.0), (1.4, 0.8, 1.0), faces="XxYZz")
    return k


# ============================================================================ start / finish gantry

def lattice_tower(k, cx, height, half=0.5, post=0.16, web=0.08, levels=6):
    for sx in (-1, 1):
        for sz in (-1, 1):
            k.beam("steel", (cx + sx * half, 0.3, sz * half), (cx + sx * half, height, sz * half), post)
    for lv in range(levels + 1):
        y = 0.3 + (height - 0.3) * lv / levels
        for a, b in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            p0 = v3(cx + a[0] * half, y, a[1] * half)
            p1 = v3(cx + b[0] * half, y, b[1] * half)
            k.beam("steel", p0, p1, web)
            if lv < levels:
                y2 = 0.3 + (height - 0.3) * (lv + 1) / levels
                q = v3(p1[0], y2, p1[2]) if lv % 2 == 0 else v3(p0[0], y2, p0[2])
                k.beam("steel", p0 if lv % 2 == 0 else p1, q, web * 0.8)
    k.box("concrete", (cx, 0.15, 0), (1.7, 0.3, 1.7), faces="XxYZz")


def gantry():
    k = Kit(max_edge=1.8)
    tx = 9.6                      # tower centres; inner faces at +-9.02 -> clear span 18.04 m
    hb, ht = 6.5, 7.7             # beam bottom / top chord heights (clearance 6.43 m under the chords)
    lattice_tower(k, -tx, 8.5, half=0.5)
    lattice_tower(k, tx, 8.5, half=0.5)
    zc = 0.5
    panels = 16
    xa, xb = -tx, tx
    for z in (-zc, zc):
        for y in (hb, ht):
            k.beam("steel", (xa, y, z), (xb, y, z), 0.14)
        for i in range(panels + 1):
            x = xa + (xb - xa) * i / panels
            k.beam("steel", (x, hb, z), (x, ht, z), 0.08)
            if i < panels:
                x2 = xa + (xb - xa) * (i + 1) / panels
                k.beam("steel", (x, hb if i % 2 == 0 else ht, z), (x2, ht if i % 2 == 0 else hb, z), 0.07)
    for i in range(panels + 1):
        x = xa + (xb - xa) * i / panels
        for y in (hb, ht):
            k.beam("steel", (x, y, -zc), (x, y, zc), 0.07)
        if i < panels:
            x2 = xa + (xb - xa) * (i + 1) / panels
            k.beam("steel", (x, ht, -zc if i % 2 == 0 else zc), (x2, ht, zc if i % 2 == 0 else -zc), 0.06)
    # sign board above the beam, readable from both sides
    sx, sy0, sy1, sz = 7.0, 7.98, 9.58, 0.16
    k.quad("ad", (-sx, sy0, sz), (sx, sy0, sz), (sx, sy1, sz), (-sx, sy1, sz), uv=[(0, 1), (1, 1), (1, 0), (0, 0)], out=Z)
    k.quad("ad", (sx, sy0, -sz), (-sx, sy0, -sz), (-sx, sy1, -sz), (sx, sy1, -sz), uv=[(0, 1), (1, 1), (1, 0), (0, 0)], out=-Z)
    fr = 0.1
    k.box("steel_dark", (0, sy1 + fr / 2, 0), (2 * sx + 2 * fr, fr, 2 * sz + 0.06), faces="XxYyZz")
    k.box("steel_dark", (0, sy0 - fr / 2, 0), (2 * sx + 2 * fr, fr, 2 * sz + 0.06), faces="XxYyZz")
    for s in (-1, 1):
        k.box("steel_dark", (s * (sx + fr / 2), (sy0 + sy1) / 2, 0), (fr, sy1 - sy0, 2 * sz + 0.06), faces="XxZz")
    for x in (-6.0, -2.0, 2.0, 6.0):
        k.beam("steel", (x, ht, 0), (x, sy0 - fr, 0), 0.12)
    # start light board on the approach (+Z) face: 5 round lamps with hoods
    k.box("steel_dark", (0, 7.1, zc + 0.16), (4.7, 1.05, 0.14), faces="XxYyZz")
    for i in range(5):
        x = -1.72 + i * 0.86
        k.cyl("steel_dark", (x, 7.1, zc + 0.23), (x, 7.1, zc + 0.42), 0.36, 0.36, sides=14, smooth=True)
        k.cyl(f"startlight_{i}", (x, 7.1, zc + 0.24), (x, 7.1, zc + 0.27), 0.31, 0.31, sides=14, cap_b=f"startlight_{i}")
    return k


# ============================================================================ smaller props

def billboard():
    k = Kit(max_edge=2.0)
    w, hgt, y0, zp = 8.0, 2.5, 1.2, 0.0
    k.quad("ad", (-w / 2, y0, zp), (w / 2, y0, zp), (w / 2, y0 + hgt, zp), (-w / 2, y0 + hgt, zp),
           uv=[(0, 1), (1, 1), (1, 0), (0, 0)], out=Z)
    k.quad("steel", (-w / 2, y0, zp - 0.05), (w / 2, y0, zp - 0.05), (w / 2, y0 + hgt, zp - 0.05), (-w / 2, y0 + hgt, zp - 0.05), out=-Z)
    fr = 0.08
    k.box("steel_dark", (0, y0 + hgt + fr / 2, zp - 0.025), (w + 2 * fr, fr, 0.11), faces="XxYyZz")
    k.box("steel_dark", (0, y0 - fr / 2, zp - 0.025), (w + 2 * fr, fr, 0.11), faces="XxYyZz")
    for s in (-1, 1):
        k.box("steel_dark", (s * (w / 2 + fr / 2), y0 + hgt / 2, zp - 0.025), (fr, hgt, 0.11), faces="XxZz")
    for x in (-3.2, 0.0, 3.2):
        k.beam("steel", (x, 0, zp - 0.12), (x, y0 + hgt, zp - 0.12), 0.12)
        k.beam("steel", (x, 0.02, zp - 1.7), (x, y0 + hgt * 0.72, zp - 0.16), 0.09)
        k.beam("steel", (x, 0.05, zp - 1.7), (x, 0.05, zp - 0.12), 0.09)
        k.box("concrete", (x, 0.04, zp - 0.9), (0.5, 0.08, 2.0), faces="XxYZz")
    for y in (y0 + 0.35, y0 + hgt - 0.35):
        k.beam("steel", (-w / 2 + 0.1, y, zp - 0.12), (w / 2 - 0.1, y, zp - 0.12), 0.07)
    return k


def light_tower():
    k = Kit(max_edge=2.5)
    k.box("concrete", (0, 0.3, 0), (1.5, 0.6, 1.5), faces="XxYZz")
    k.cyl("steel", (0, 0.6, 0), (0, 21.2, 0), 0.34, 0.15, sides=10, segs=7)
    k.cyl("steel", (0, 0.6, 0), (0, 0.72, 0), 0.5, 0.5, sides=10, cap_b="steel")
    # service platform under the head
    k.box("steel_dark", (0, 19.3, 0), (1.5, 0.06, 1.5), faces="XxYyZz")
    for a, b in (((-0.75, -0.75), (0.75, -0.75)), ((0.75, -0.75), (0.75, 0.75)), ((0.75, 0.75), (-0.75, 0.75)), ((-0.75, 0.75), (-0.75, -0.75))):
        rail(k, "steel_dark", (a[0], 19.33, a[1]), (b[0], 19.33, b[1]), 1.0, posts_every=1.5, bars=(1.0, 0.5), post=0.035, bar=0.03)
    # lamp head: 4 x 3 floodlights on a frame tilted toward +Z / down
    R = rot_x(24.0)
    c = v3(0, 21.5, 0.3)

    def w(p):
        return c + R @ v3(p)

    for y in (-0.95, 0.0, 0.95):
        k.beam("steel_dark", w((-1.75, y, 0)), w((1.75, y, 0)), 0.1)
    for x in (-1.75, 1.75, 0.0):
        k.beam("steel_dark", w((x, -0.95, 0)), w((x, 0.95, 0)), 0.1)
    k.beam("steel", (0, 20.6, 0), w((0, -0.5, -0.05)), 0.14)
    for ix in range(4):
        for iy in range(3):
            p = ((ix - 1.5) * 0.86, (iy - 1) * 0.66, 0.2)
            k.box("steel_dark", w(p), (0.7, 0.54, 0.3), R=R, mats={"Z": "lamp"}, faces="XxYyZz")
    return k


def marshal_post():
    k = Kit(max_edge=1.5)
    pw, pd, ph = 2.8, 2.2, 0.9
    k.box("steel", (0, ph - 0.05, 0), (pw, 0.1, pd), faces="XxYyZz")
    for sx in (-1, 1):
        for sz in (-1, 1):
            k.beam("steel", (sx * (pw / 2 - 0.1), 0, sz * (pd / 2 - 0.1)), (sx * (pw / 2 - 0.1), ph - 0.1, sz * (pd / 2 - 0.1)), 0.09)
        k.beam("steel", (sx * (pw / 2 - 0.1), 0.08, -(pd / 2 - 0.1)), (sx * (pw / 2 - 0.1), ph - 0.14, pd / 2 - 0.1), 0.05)
    for i in range(4):   # steps at the back
        k.box("steel", (-0.75, 0.2 + i * 0.2, -pd / 2 - 0.9 + i * 0.26), (0.8, 0.04, 0.26), faces="XxYyZz")
    k.beam("steel", (-1.15, 0.05, -pd / 2 - 1.0), (-1.15, ph - 0.05, -pd / 2 - 0.05), 0.05)
    k.beam("steel", (-0.35, 0.05, -pd / 2 - 1.0), (-0.35, ph - 0.05, -pd / 2 - 0.05), 0.05)
    # cabin: three walls + parapet + roof, open window band toward the track
    cw, cd, ch, t = 2.2, 1.6, 2.15, 0.06
    y0 = ph
    k.box("panel", (0, y0 + ch / 2, -cd / 2), (cw, ch, t), faces="XxZzY")
    k.box("panel", (cw / 2, y0 + ch / 2, 0), (t, ch, cd), faces="XxZzY")
    k.box("panel", (-cw / 2, y0 + 0.5, 0), (t, 1.0, cd), faces="XxZzY")          # door-side wall is half height
    k.box("panel", (0, y0 + 0.5, cd / 2), (cw, 1.0, t), faces="XxZzY")           # front parapet
    k.box("marshal_orange", (0, y0 + 0.86, cd / 2 + 0.035), (cw + 0.02, 0.2, 0.012), faces="XxZYy")
    k.box("marshal_orange", (cw / 2 + 0.035, y0 + 0.86, 0), (0.012, 0.2, cd + 0.02), faces="XYyZz")
    for sx in (-1, 1):
        k.beam("panel", (sx * (cw / 2 - 0.03), y0 + 1.0, cd / 2), (sx * (cw / 2 - 0.03), y0 + ch, cd / 2), 0.07)
    k.beam("panel", (-cw / 2, y0 + 1.0, -cd / 2 + 0.03), (-cw / 2, y0 + ch, -cd / 2 + 0.03), 0.07)
    k.box("panel_dark", (0, y0 + ch + 0.05, 0.12), (cw + 0.5, 0.1, cd + 0.7), faces="XxYyZz")
    rail(k, "steel", (-pw / 2 + 0.05, y0, pd / 2 - 0.05), (pw / 2 - 0.05, y0, pd / 2 - 0.05), 1.0, posts_every=1.4)
    return k


def tire_stack():
    k = Kit(max_edge=9.0)
    # closed cross-section (radius, y): inner wall, top sidewall + shoulder, tread, bottom sidewall
    prof = [(0.2, 0.012), (0.2, 0.205), (0.3, 0.217), (0.335, 0.17), (0.335, 0.045)]
    rng = np.random.default_rng(4)
    for i in range(4):
        mats = ["tire"] * 5
        if i in (1, 2):
            mats[3] = "paint_band"
        k.lathe("tire", prof, sides=12, center=(rng.uniform(-0.012, 0.012), i * 0.217, rng.uniform(-0.012, 0.012)), closed=True, mats=mats)
    return k


def cone():
    k = Kit(max_edge=9.0)
    k.box("cone_orange", (0, 0.015, 0), (0.30, 0.03, 0.30), faces="XxYZz")
    k.cyl("cone_orange", (0, 0.03, 0), (0, 0.21, 0), 0.115, 0.082, sides=10)
    k.cyl("cone_white", (0, 0.21, 0), (0, 0.33, 0), 0.082, 0.06, sides=10)
    k.cyl("cone_orange", (0, 0.33, 0), (0, 0.50, 0), 0.06, 0.028, sides=10, cap_b="cone_orange")
    return k


def camera_tower():
    k = Kit(max_edge=2.2)
    s, lv, H = 1.0, [0.12, 2.0, 4.0, 6.0], 6.0
    corners = [(-s, -s), (s, -s), (s, s), (-s, s)]
    for (x, z) in corners:
        k.beam("steel", (x, 0.0, z), (x, H + 1.1, z), 0.06)
        k.box("steel_dark", (x, 0.02, z), (0.22, 0.04, 0.22), faces="XxYZz")
    for li, y in enumerate(lv):
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            k.beam("steel", (a[0], y, a[1]), (b[0], y, b[1]), 0.05)
            if li < len(lv) - 1:
                y2 = lv[li + 1]
                if (li + i) % 2 == 0:
                    k.beam("steel", (a[0], y, a[1]), (b[0], y2, b[1]), 0.04)
                else:
                    k.beam("steel", (b[0], y, b[1]), (a[0], y2, a[1]), 0.04)
    k.box("wood", (0, H + 0.03, 0), (2 * s + 0.2, 0.06, 2 * s + 0.2), faces="XxYyZz")
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        for y in (H + 0.6, H + 1.08):
            k.beam("steel", (a[0], y, a[1]), (b[0], y, b[1]), 0.045)
        k.beam("wood", (a[0], H + 0.14, a[1]), (b[0], H + 0.14, b[1]), 0.025, 0.16)
    # ladder on the back face
    for x in (-0.25, 0.25):
        k.beam("steel", (x, 0.0, -s - 0.08), (x, H + 1.0, -s - 0.08), 0.04)
    for i in range(18):
        y = 0.35 + i * 0.36
        k.beam("steel", (-0.25, y, -s - 0.08), (0.25, y, -s - 0.08), 0.028)
    # broadcast camera on a tripod, pointing +Z
    head = v3(0.15, H + 1.5, 0.35)
    for a in (0.0, 2.1, 4.2):
        k.beam("steel_dark", (head[0] + 0.45 * math.cos(a), H + 0.06, head[2] - 0.1 + 0.45 * math.sin(a)), head - Y * 0.12, 0.035)
    k.box("steel_dark", head + v3(0, 0.06, 0.05), (0.26, 0.24, 0.62), faces="XxYyZz")
    k.cyl("steel_dark", head + v3(0, 0.06, 0.36), head + v3(0, 0.06, 0.62), 0.09, 0.11, sides=10, cap_b="glass")
    k.box("steel_dark", head + v3(0, 0.21, -0.2), (0.1, 0.07, 0.3), faces="XxYyZz")
    return k


def flag_pole():
    k = Kit(max_edge=3.0)
    k.cyl("concrete", (0, 0, 0), (0, 0.1, 0), 0.22, 0.2, sides=10, cap_b="concrete")
    k.cyl("steel", (0, 0.1, 0), (0, 7.0, 0), 0.048, 0.03, sides=8, segs=4, cap_b="steel")
    k.lathe("steel", [(0.0, 7.0), (0.06, 7.03), (0.075, 7.08), (0.06, 7.13), (0.0, 7.16)], sides=8)
    k.set_node("flag")
    nx, ny = 12, 8
    xa, xb, yt, yb = 0.05, 1.85, 6.85, 5.65
    for i in range(nx):
        for j in range(ny):
            u0, u1, v0, v1 = i / nx, (i + 1) / nx, j / ny, (j + 1) / ny

            def P(u, v):
                return (xa + (xb - xa) * u, yt + (yb - yt) * v, 0.0)
            k.quad("flag", P(u0, v1), P(u1, v1), P(u1, v0), P(u0, v0), uv=[(u0, v1), (u1, v1), (u1, v0), (u0, v0)], out=Z, max_edge=1e9)
    return k


PROPS = {
    "grandstand": (grandstand, dict(rays=64, max_dist=5.0)),
    "pit_building": (pit_building, dict(rays=64, max_dist=4.5)),
    "gantry": (gantry, dict(rays=48, max_dist=3.0)),
    "billboard": (billboard, dict(rays=64, max_dist=3.0)),
    "light_tower": (light_tower, dict(rays=48, max_dist=3.0)),
    "marshal_post": (marshal_post, dict(rays=64, max_dist=2.5)),
    "tire_stack": (tire_stack, dict(rays=64, max_dist=0.8)),
    "cone": (cone, dict(rays=64, max_dist=0.5)),
    "camera_tower": (camera_tower, dict(rays=48, max_dist=2.5)),
    "flag_pole": (flag_pole, dict(rays=32, max_dist=1.5, ao_nodes={"main"})),
}


def build(pid):
    fn, ao = PROPS[pid]
    k = fn()
    parts = k.finalize(**ao)
    nodes = []
    for node, plist in parts.items():
        if plist:
            nodes.append({"name": pid if node == "main" else node, "parts": plist})
    size = write_glb(os.path.join(OUT, pid + ".glb"), nodes, MATS)
    allp = np.concatenate([p.positions for n in nodes for p in n["parts"]])
    tris = sum(p.triangles for n in nodes for p in n["parts"])
    mats = sorted({p.material for n in nodes for p in n["parts"]})
    print(f"PROP {pid}: tris={tris} verts={len(allp)} bounds={allp.min(0).round(2)}..{allp.max(0).round(2)} {size // 1024} KB mats={mats}")


if __name__ == "__main__":
    ids = argv[0].split(",") if argv else list(PROPS.keys())
    for pid in ids:
        build(pid)
    print("PROPS DONE")
