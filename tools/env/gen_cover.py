"""Ground cover -> GLB: grass_tuft (crossed blade cards on grass_atlas) and bush_a (leaf cards on foliage_atlas).
COLOR_0: R = wind weight, G = phase, B = AO (same convention as the trees)."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foliage_src as fs  # noqa: E402
from glb import write_glb, material_def  # noqa: E402
from treelib import UP, Acc, norm, smoothstep, make_card, fib_dirs, random_rotation  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))


def grass_tuft():
    rng = np.random.default_rng(5)
    acc = Acc()
    names = ["grass_0", "grass_1", "grass_2", "grass_3"]
    W, Hh = 0.8, 0.5
    for k in range(3):
        a = k * np.pi / 3 + rng.uniform(-0.12, 0.12)
        s = np.array([np.cos(a), 0.0, np.sin(a)])           # across the card
        out = np.array([-np.sin(a), 0.0, np.cos(a)])        # card front
        lean = np.radians(rng.uniform(6, 14)) * (1 if k % 2 == 0 else -1)
        d = norm(UP * np.cos(lean) + out * np.sin(lean))    # blades grow upward, leaning a little
        u0, v0, u1, v1 = fs.grass_tile_uv(names[k])
        # u across the tile, v from bottom (v1) to top (v0): pass a rect whose "length" axis is vertical
        off = out * rng.uniform(-0.05, 0.05)
        pos, _uv, t, edge, nl, tris = make_card(off, d, s, Hh, [(W / 2, 0.0, 0.0), (-W / 2, 0.0, 1.0)], (0, 0, 1, 1), 2,
                                                bend=np.radians(rng.uniform(4, 12)))
        col_frac = np.array([0.0, 1.0] * 3)                 # column index per vertex (2 columns x 3 stations)
        uv = np.stack([u0 + (u1 - u0) * col_frac, v1 + (v0 - v1) * t], 1)
        n = norm(UP[None, :] * 0.9 + norm(pos - np.array([0, -0.4, 0])) * 0.35)
        col = np.stack([t ** 1.3, np.full(len(pos), 0.2 + 0.3 * k), 0.5 + 0.5 * smoothstep(0.0, 0.7, t)], 1)
        acc.add(pos, n, uv, col, tris)
    mats = {"grass": material_def("grass", roughness=0.85, alpha_mode="MASK", alpha_cutoff=0.5, double_sided=True)}
    part = acc.part("grass")
    size = write_glb(os.path.join(OUT, "grass_tuft.glb"), [{"name": "grass_tuft", "parts": [part]}], mats)
    print("grass_tuft tris", part.triangles, "bounds", part.positions.min(0).round(2), part.positions.max(0).round(2), size, "B")


def bush():
    rng = np.random.default_rng(9)
    acc = Acc()
    c = np.array([0.0, 0.5, 0.0])
    rr = np.array([0.82, 0.62, 0.82])
    tiles = ["leaf_b1", "leaf_b2", "leaf_a1", "leaf_b1", "leaf_b2"]
    dirs = fib_dirs(150, rng) @ random_rotation(rng).T
    dirs = dirs[dirs[:, 1] > -0.3]
    for i, d in enumerate(dirs):
        p = c + d * rr * rng.uniform(0.55, 0.98)
        p[1] = max(p[1], 0.12)
        nc = norm(d + 0.5 * rng.normal(size=3))
        dd = np.array([d[0], 0.0, d[2]]) * 0.7 + rng.normal(size=3) * 0.6 - UP * 0.25
        dd = dd - nc * np.dot(dd, nc)
        dd = norm(dd if np.linalg.norm(dd) > 1e-3 else np.cross(nc, UP))
        s = np.cross(dd, nc)
        L = rng.uniform(0.5, 0.78)
        card = make_card(p - dd * L * 0.5, dd, s, L, [(L / 2, 0.0, 0.0), (-L / 2, 0.0, 1.0)], fs.tile_uv(tiles[i % len(tiles)]), 1,
                         bend=np.radians(rng.uniform(5, 25)))
        pos, uv, t, edge, nl, tris = card
        n = norm(0.75 * norm((pos - np.array([0.0, 0.15, 0.0])) / rr) + 0.35 * UP[None, :] + 0.1 * nl)
        rho = np.linalg.norm((pos - c) / rr, axis=1)
        hgt = np.clip(pos[:, 1] / 1.15, 0, 1)
        ao = (0.35 + 0.65 * smoothstep(0.35, 0.95, rho)) * (0.6 + 0.4 * smoothstep(0.0, 0.6, hgt))
        wind = np.clip(0.15 + 0.6 * hgt + 0.25 * t, 0, 1)
        acc.add(pos, n, uv, np.stack([wind, np.full(len(pos), rng.uniform()), np.clip(ao, 0, 1)], 1), tris)
    mats = {"leaves": material_def("leaves", roughness=0.78, alpha_mode="MASK", alpha_cutoff=0.5, double_sided=True)}
    part = acc.part("leaves")
    size = write_glb(os.path.join(OUT, "bush_a.glb"), [{"name": "bush_a", "parts": [part]}], mats)
    print("bush_a tris", part.triangles, "bounds", part.positions.min(0).round(2), part.positions.max(0).round(2), size, "B")


if __name__ == "__main__":
    grass_tuft()
    bush()
