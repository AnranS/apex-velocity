"""Convert baked rock maps (PNG) to shipped WebP + a preview sheet."""
import os
import numpy as np
from PIL import Image


def neutralise(im, sat=0.42, tint=(1.0, 0.975, 0.93), gain=1.04):
    """Desert-orange scans -> weathered warm-grey granite: pull saturation down, keep luminance detail."""
    a = np.asarray(im, dtype=np.float64) / 255.0
    lum = (a * np.array([0.3, 0.59, 0.11])).sum(axis=2, keepdims=True)
    a = (lum + (a - lum) * sat) * np.array(tint) * gain
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8), "RGB")

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_work", "rocks")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
for rid in ("rock_a", "rock_b", "rock_c"):
    for suffix, q in (("d", 82), ("n", 84)):
        src = os.path.join(WORK, f"{rid}_{suffix}.png")
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("RGB")
        if suffix == "d":
            im = neutralise(im)
        dst = os.path.join(OUT, f"{rid}_{suffix}.webp")
        im.save(dst, "WEBP", quality=q, method=6)
        print(os.path.basename(dst), os.path.getsize(dst) // 1024, "KB")
sheet = Image.new("RGB", (800 * 3, 600))
for i, rid in enumerate(("rock_a", "rock_b", "rock_c")):
    p = os.path.join(HERE, "_work", "preview", f"{rid}_preview.png")
    if os.path.exists(p):
        sheet.paste(Image.open(p).convert("RGB"), (800 * i, 0))
sheet.resize((1500, 375), Image.LANCZOS).save(os.path.join(HERE, "_work", "preview", "rocks_sheet.png"))
