"""Convert downloaded CC0 bark maps into compact 512px WebP (run with python3 -I)."""
import os
import site
import sys

# -I drops the user site dir; add back only that one trusted location (Pillow lives there).
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def birchify(img):
    """Turn a brown bark photo into pale birch/aspen bark: desaturate + lift, add dark wrap-safe lenticels."""
    a = np.asarray(img, dtype=np.float64) / 255.0
    lum = a.mean(axis=2, keepdims=True)
    a = (lum * 0.7 + a * 0.3)
    a = np.clip((a - a.mean()) * 1.5 + 0.66, 0, 1) * np.array([1.0, 0.985, 0.95])
    h, w = a.shape[:2]
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w))
    for _ in range(46):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rx, ry = rng.uniform(14, 70), rng.uniform(2.0, 5.5)
        dx = np.minimum(np.abs(xx - cx), w - np.abs(xx - cx)) / rx
        dy = np.minimum(np.abs(yy - cy), h - np.abs(yy - cy)) / ry
        mask = np.maximum(mask, np.clip(1.0 - (dx * dx + dy * dy), 0, 1) ** 0.6 * rng.uniform(0.5, 1.0))
    for _ in range(5):  # a few larger dark scars
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rx, ry = rng.uniform(30, 60), rng.uniform(14, 30)
        dx = np.minimum(np.abs(xx - cx), w - np.abs(xx - cx)) / rx
        dy = np.minimum(np.abs(yy - cy), h - np.abs(yy - cy)) / ry
        mask = np.maximum(mask, np.clip(1.0 - (dx * dx + dy * dy), 0, 1) ** 1.5 * 0.9)
    a = a * (1.0 - 0.78 * mask[..., None])
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8), "RGB")

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "_work", "dl")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
PREV = os.path.join(HERE, "_work", "preview")
JOBS = [("pine_bark", "bark_pine"), ("jolcham_oak_bark_01", "bark_oak"), ("bark_platanus", "bark_birch")]
sheet = Image.new("RGB", (512 * 3, 1024))
for i, (src, dst) in enumerate(JOBS):
    d = Image.open(os.path.join(DL, src, f"{src}_diff_1k.jpg")).convert("RGB").resize((512, 512), Image.LANCZOS)
    n = Image.open(os.path.join(DL, src, f"{src}_nor_gl_1k.jpg")).convert("RGB").resize((512, 512), Image.LANCZOS)
    if dst == "bark_birch":
        d = birchify(d)
    d.save(os.path.join(OUT, dst + ".webp"), "WEBP", quality=82, method=6)
    n.save(os.path.join(OUT, dst + "_n.webp"), "WEBP", quality=80, method=6)
    sheet.paste(d, (512 * i, 0))
    sheet.paste(n, (512 * i, 512))
    px = d.resize((1, 1), Image.BOX).getpixel((0, 0))
    print(dst, "mean sRGB", px, os.path.getsize(os.path.join(OUT, dst + ".webp")) // 1024, "KB +",
          os.path.getsize(os.path.join(OUT, dst + "_n.webp")) // 1024, "KB")
sheet.resize((1152, 768), Image.LANCZOS).save(os.path.join(PREV, "bark_sheet.png"))
