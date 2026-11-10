"""Compose per-tile pass renders into the shipped atlases (system python3 + numpy + PIL).

  python3 compose_atlas.py foliage|grass

foliage -> public/assets/env/foliage_atlas.webp (RGBA, 2048^2), foliage_atlas_n.webp (RGB normals, 1024^2)
grass   -> public/assets/env/grass_atlas.webp (RGBA)
Colour is baked albedo * soft AO * depth cue; RGB is dilated under transparent texels (mip-safe);
alpha is gained so thin needles survive alpha-test / minification.
"""
import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import foliage_src as fs  # noqa: E402

SRC = os.path.join(HERE, "_work", "atlas")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
PREV = os.path.join(HERE, "_work", "preview")


def s2l(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def l2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def shift(a, dy, dx):
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    ys = slice(max(dy, 0), h + min(dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def dilate_fill(rgb, valid, iters=48, fallback=None):
    rgb = rgb.copy()
    valid = valid.copy()
    orig = valid.copy()
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for _ in range(iters):
        if valid.all():
            break
        acc = np.zeros_like(rgb)
        cnt = np.zeros(valid.shape, dtype=np.float64)
        masked = rgb * valid[..., None]
        for dy, dx in nb:
            acc += shift(masked, dy, dx)
            cnt += shift(valid.astype(np.float64), dy, dx)
        fill = (~valid) & (cnt > 0)
        rgb[fill] = acc[fill] / cnt[fill][:, None]
        valid |= fill
    if not valid.all():
        fb = rgb[orig].mean(axis=0) if fallback is None else fallback
        rgb[~valid] = fb
    return rgb


def load(name, p):
    return np.asarray(Image.open(os.path.join(SRC, f"{name}_{p}.png")).convert("RGBA"), dtype=np.float64) / 255.0


def compose(table, size, ao_mix=0.5, h_mix=0.22, gain=1.9, a_off=0.04):
    W, H = size
    atlas = np.zeros((H, W, 4), dtype=np.float64)
    natlas = np.zeros((H, W, 3), dtype=np.float64)
    natlas[..., 2] = 1.0
    stats = {}
    for name, (_b, _rect, (px, py, w, h)) in table.items():
        alb = load(name, "albedo")
        ao = load(name, "ao")[..., 0]
        hh = load(name, "height")[..., 0]
        nrm = load(name, "normal")[..., :3] * 2.0 - 1.0
        if alb.shape[0] != h or alb.shape[1] != w:
            raise SystemExit(f"{name}: render {alb.shape[1]}x{alb.shape[0]} != tile {w}x{h}")
        a = alb[..., 3]
        lin = s2l(alb[..., :3])
        shade = ((1 - ao_mix) + ao_mix * ao) * ((1 - h_mix) + h_mix * hh)
        lin = lin * shade[..., None]
        valid = a > 0.08
        a2 = np.clip((a - a_off) * gain, 0, 1)
        rgb = dilate_fill(l2s(lin), valid)
        # normals: face the camera, soften, dilate
        nrm[..., 2] = np.abs(nrm[..., 2])
        nrm[..., :2] *= 0.85
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        nrm = dilate_fill(nrm, valid, iters=12, fallback=np.array([0.0, 0.0, 1.0]))
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        atlas[py:py + h, px:px + w, :3] = rgb
        atlas[py:py + h, px:px + w, 3] = a2
        natlas[py:py + h, px:px + w] = nrm
        stats[name] = (float((a2 > 0.5).mean()), [float(x) for x in (rgb[valid].mean(axis=0) * 255)])
    # texels outside every tile: neutral foliage colour so mips never pull in black
    return atlas, natlas, stats


def save_rgba(arr, path, quality=86):
    im = Image.fromarray(np.clip(np.round(arr * 255), 0, 255).astype(np.uint8), "RGBA")
    im.save(path, "WEBP", quality=quality, method=6, exact=True)
    return im


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "foliage"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PREV, exist_ok=True)
    if which == "foliage":
        atlas, natlas, stats = compose(fs.tiles(), fs.ATLAS_SIZE)
        im = save_rgba(atlas, os.path.join(OUT, "foliage_atlas.webp"), quality=84)
        n8 = Image.fromarray(np.clip(np.round((natlas * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8), "RGB")
        n8 = n8.resize((1024, 1024), Image.LANCZOS)
        n8.save(os.path.join(OUT, "foliage_atlas_n.webp"), "WEBP", quality=82, method=6)
        n8.save(os.path.join(PREV, "foliage_atlas_n.png"))
        prev_name = "foliage_atlas_preview.png"
    else:
        atlas, natlas, stats = compose(fs.grass_tiles(), fs.GRASS_ATLAS_SIZE, ao_mix=0.35, h_mix=0.0, gain=1.7)
        im = save_rgba(atlas, os.path.join(OUT, "grass_atlas.webp"), quality=86)
        prev_name = "grass_atlas_preview.png"
    bg = Image.new("RGBA", im.size, (150, 170, 200, 255))
    bg.alpha_composite(im)
    bg.convert("RGB").save(os.path.join(PREV, prev_name))
    # also the cutout look (alpha test 0.5) on a dark background
    a = np.asarray(im)[..., 3] > 127
    cut = np.asarray(im)[..., :3] * a[..., None] + np.array([30, 34, 44], dtype=np.uint8) * (~a[..., None])
    Image.fromarray(cut.astype(np.uint8), "RGB").save(os.path.join(PREV, prev_name.replace("_preview", "_cutout")))
    for k, (cov, mean) in stats.items():
        print(f"{k:14s} coverage={cov:.3f} mean_rgb=({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})")
    for f in sorted(os.listdir(OUT)):
        if f.startswith(which):
            print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")


if __name__ == "__main__":
    main()
