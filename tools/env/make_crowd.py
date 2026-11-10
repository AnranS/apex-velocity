"""Procedural crowd strip for the grandstand `crowd` cards: 16 seated spectators, RGBA, 1024x128.
Also writes preview-only test textures (ad panel, flag) into _work/tmp."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
TMP = os.path.join(HERE, "_work", "tmp")
SS = 4
W, H, N = 1024, 128, 16
rng = np.random.default_rng(21)
im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
shirts = [(225, 225, 220), (190, 30, 28), (28, 60, 140), (235, 190, 40), (30, 30, 34), (40, 110, 60), (120, 124, 130),
          (230, 110, 20), (250, 250, 250), (90, 30, 110), (60, 140, 190), (200, 60, 90)]
skins = [(232, 190, 160), (214, 170, 135), (190, 140, 105), (150, 105, 75), (110, 75, 55), (240, 205, 180)]
hairs = [(30, 24, 20), (70, 48, 30), (120, 90, 50), (200, 180, 130), (20, 20, 22), (150, 150, 150)]
cw = W * SS / N
for i in range(N):
    if rng.uniform() < 0.14:
        continue
    cx = (i + 0.5) * cw + rng.uniform(-3, 3) * SS
    top = (22 + rng.uniform(-9, 9)) * SS
    shirt = shirts[int(rng.integers(0, len(shirts)))]
    skin = skins[int(rng.integers(0, len(skins)))]
    hair = hairs[int(rng.integers(0, len(hairs)))]
    hw = rng.uniform(19, 23) * SS
    sh_y = top + 30 * SS
    dark = tuple(int(c * 0.72) for c in shirt)
    # arms / sides, torso, shoulders
    d.rounded_rectangle([cx - hw - 3 * SS, sh_y + 4 * SS, cx + hw + 3 * SS, H * SS + 20 * SS], radius=9 * SS, fill=dark + (255,))
    d.rounded_rectangle([cx - hw + 3 * SS, sh_y, cx + hw - 3 * SS, H * SS + 20 * SS], radius=10 * SS, fill=shirt + (255,))
    d.ellipse([cx - hw, sh_y - 5 * SS, cx + hw, sh_y + 16 * SS], fill=shirt + (255,))
    # neck + head + hair / cap
    d.rectangle([cx - 4 * SS, top + 20 * SS, cx + 4 * SS, sh_y + 2 * SS], fill=tuple(int(c * 0.85) for c in skin) + (255,))
    d.ellipse([cx - 10.5 * SS, top, cx + 10.5 * SS, top + 26 * SS], fill=skin + (255,))
    if rng.uniform() < 0.35:
        cap = shirts[int(rng.integers(0, len(shirts)))]
        d.pieslice([cx - 11.5 * SS, top - 2 * SS, cx + 11.5 * SS, top + 22 * SS], 180, 360, fill=cap + (255,))
        d.rectangle([cx - 12 * SS, top + 9 * SS, cx + 12 * SS, top + 11.5 * SS], fill=tuple(int(c * 0.8) for c in cap) + (255,))
    else:
        d.pieslice([cx - 11 * SS, top - 1.5 * SS, cx + 11 * SS, top + 20 * SS], 180, 360, fill=hair + (255,))
    if rng.uniform() < 0.3:   # sunglasses
        d.rectangle([cx - 8 * SS, top + 11 * SS, cx + 8 * SS, top + 14.5 * SS], fill=(20, 20, 24, 255))
im = im.resize((W, H), Image.LANCZOS)
a = np.asarray(im).astype(np.float64) / 255.0
# soft top-down shading + colour bleed under transparent texels
shade = np.linspace(1.05, 0.7, H)[:, None, None]
rgb = np.clip(a[..., :3] * shade, 0, 1)
alpha = a[..., 3]
valid = alpha > 0.5
mean = rgb[valid].mean(axis=0)
rgb[~valid] = mean
out = np.concatenate([rgb, alpha[..., None]], 2)
Image.fromarray((out * 255).astype(np.uint8), "RGBA").save(os.path.join(OUT, "crowd_atlas.webp"), "WEBP", quality=88, method=6, exact=True)
print("crowd_atlas.webp", os.path.getsize(os.path.join(OUT, "crowd_atlas.webp")) // 1024, "KB")

# ---- preview-only textures
os.makedirs(TMP, exist_ok=True)
ad = Image.new("RGB", (1024, 256), (16, 20, 34))
dd = ImageDraw.Draw(ad)
dd.rectangle([0, 0, 1024, 18], fill=(230, 40, 30))
dd.rectangle([0, 238, 1024, 256], fill=(230, 40, 30))
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 120)
except Exception:
    font = ImageFont.load_default()
dd.text((512, 128), "TEST SPONSOR", fill=(245, 245, 245), font=font, anchor="mm")
ad.save(os.path.join(TMP, "test_ad.png"))
fl = Image.new("RGB", (384, 256), (255, 255, 255))
df = ImageDraw.Draw(fl)
for i in range(6):
    for j in range(4):
        if (i + j) % 2 == 0:
            df.rectangle([i * 64, j * 64, i * 64 + 64, j * 64 + 64], fill=(12, 12, 12))
fl.save(os.path.join(TMP, "test_flag.png"))
bg = Image.new("RGBA", (W, H), (60, 20, 20, 255))
bg.alpha_composite(Image.open(os.path.join(OUT, "crowd_atlas.webp")).convert("RGBA"))
bg.convert("RGB").save(os.path.join(HERE, "_work", "preview", "crowd_preview.png"))
