"""Pack per-view impostor passes into <id>_imp.webp (RGBA colour) and <id>_imp_n.webp (view-space normals).

  python3 compose_impostors.py pine_a pine_b ...

Colour = albedo * softened ambient visibility (neutral lighting, no sun direction baked in).
Normals: RGB = n*0.5+0.5 in the view frame of that azimuth (x right, y up, z toward the camera).
View k is seen from azimuth phi = k*45deg around +Y, camera direction (sin phi, 0, cos phi) from the trunk.
"""
import json
import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compose_atlas import dilate_fill, s2l, l2s  # noqa: E402

IMP = os.path.join(HERE, "_work", "imp")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
PREV = os.path.join(HERE, "_work", "preview")


def load(path):
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0


def compose(tid):
    info = json.load(open(os.path.join(IMP, f"{tid}_info.json")))
    views, vw, vh = info["views"], info["viewWidth"], info["viewHeight"]
    col = np.zeros((vh, vw * views, 4))
    nrm = np.zeros((vh, vw * views, 3))
    for k in range(views):
        amb = load(os.path.join(IMP, f"{tid}_c{k}.png"))
        alb = load(os.path.join(IMP, f"{tid}_a{k}.png"))
        nw = load(os.path.join(IMP, f"{tid}_n{k}.png"))
        a = alb[..., 3]
        valid = a > 0.35
        alb_l = s2l(alb[..., :3])
        amb_l = s2l(amb[..., :3])
        vis = amb_l.sum(axis=2) / np.maximum(alb_l.sum(axis=2), 1e-4)
        shade = 0.45 + 0.55 * np.clip(vis * 1.8, 0.0, 1.0)
        rgb = l2s(alb_l * shade[..., None])
        rgb = dilate_fill(rgb, valid, iters=10)
        alpha = np.clip((a - 0.1) * 1.6, 0, 1)
        # world (Blender axes) -> glTF -> view space of this azimuth
        nb = nw[..., :3] * 2.0 - 1.0
        ng = np.stack([nb[..., 0], nb[..., 2], -nb[..., 1]], -1)
        phi = k * 2.0 * np.pi / views
        right = np.array([np.cos(phi), 0.0, -np.sin(phi)])
        toward = np.array([np.sin(phi), 0.0, np.cos(phi)])
        nv = np.stack([ng @ right, ng[..., 1], ng @ toward], -1)
        nv /= np.maximum(np.linalg.norm(nv, axis=-1, keepdims=True), 1e-6)
        nv = dilate_fill(nv, valid, iters=6, fallback=np.array([0.0, 0.0, 1.0]))
        nv /= np.maximum(np.linalg.norm(nv, axis=-1, keepdims=True), 1e-6)
        col[:, k * vw:(k + 1) * vw, :3] = rgb
        col[:, k * vw:(k + 1) * vw, 3] = alpha
        nrm[:, k * vw:(k + 1) * vw] = nv
    im = Image.fromarray(np.clip(np.round(col * 255), 0, 255).astype(np.uint8), "RGBA")
    im.save(os.path.join(OUT, f"{tid}_imp.webp"), "WEBP", quality=86, method=6, exact=True)
    n8 = Image.fromarray(np.clip(np.round((nrm * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8), "RGB")
    n8.save(os.path.join(OUT, f"{tid}_imp_n.webp"), "WEBP", quality=84, method=6)
    bg = Image.new("RGBA", im.size, (150, 170, 200, 255))
    bg.alpha_composite(im)
    sheet = Image.new("RGB", (im.size[0], im.size[1] * 2))
    sheet.paste(bg.convert("RGB"), (0, 0))
    sheet.paste(n8, (0, im.size[1]))
    sheet.save(os.path.join(PREV, f"{tid}_imp_preview.png"))
    info["atlasWidth"], info["atlasHeight"] = im.size
    json.dump(info, open(os.path.join(IMP, f"{tid}_info.json"), "w"), indent=1)
    print(tid, im.size, os.path.getsize(os.path.join(OUT, f"{tid}_imp.webp")) // 1024, "KB +",
          os.path.getsize(os.path.join(OUT, f"{tid}_imp_n.webp")) // 1024, "KB", info)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        compose(t)
