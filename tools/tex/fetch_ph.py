#!/usr/bin/env python3
"""Fetch CC0 skies / PBR textures from Poly Haven and bake them into web-ready assets.

    python3 -I tools/tex/fetch_ph.py [sky|tex|all]

Downloads go to tools/tex/_work/dl/<asset>/ (one directory per asset, treated as data only).
Outputs: public/assets/sky/<preset>.jpg + sky.json, public/assets/tex/<name>_d.jpg + <name>_nr.jpg + tex.json
"""
import json
import os
import subprocess
import sys
import urllib.request

import site  # noqa: E402

# isolated mode (-I) drops the user site directory: add the site-packages dirs back explicitly
for _p in site.getsitepackages() + [site.getusersitepackages()]:
    if _p not in sys.path:
        sys.path.append(_p)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DL = os.path.join(ROOT, "tools", "tex", "_work", "dl")
SKY_OUT = os.path.join(ROOT, "public", "assets", "sky")
TEX_OUT = os.path.join(ROOT, "public", "assets", "tex")
UA = {"User-Agent": "asset-fetch/1.0"}

SKIES = {
    # preset: (polyhaven id, source resolution)
    "sunset": ("kloppenheim_06_puresky", "4k"),
    "day": ("kloofendal_48d_partly_cloudy_puresky", "4k"),
    "storm": ("overcast_soil_puresky", "4k"),
    "night": ("qwantani_moonrise_puresky", "4k"),
}
TEXTURES = {
    # name: (polyhaven id, output size, write the normal+roughness map?)
    "asphalt": ("asphalt_track", 1024, True),
    "grass": ("leafy_grass", 1024, True),
    "rock": ("rock_face_03", 1024, True),
    "gravel": ("gravel_floor_02", 1024, True),
    "dirt": ("forest_ground_04", 512, False),
    "sand": ("coast_sand_01", 512, False),
    "concrete": ("concrete_floor_02", 512, False),
}


def http_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    tmp = path + ".part"
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)
    print("  downloaded", os.path.basename(path), os.path.getsize(path) // 1024, "KB")


def read_hdr(path):
    """Decode a Radiance .hdr to float32 HxWx3 via ffmpeg."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    w, h = [int(v) for v in probe.stdout.strip().split(",")[:2]]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"],
        capture_output=True, check=True,
    ).stdout
    a = np.frombuffer(raw, dtype="<f4").reshape(3, h, w)
    return np.stack([a[2], a[0], a[1]], axis=-1).astype(np.float32)  # planes are G,B,R


def lum(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def process_sky(preset, asset, res):
    print("sky", preset, asset)
    d = os.path.join(DL, asset)
    files = http_json(f"https://api.polyhaven.com/files/{asset}")
    url = files["hdri"][res]["hdr"]["url"]
    src = os.path.join(d, os.path.basename(url))
    download(url, src)
    img = read_hdr(src)
    H, W, _ = img.shape
    rows = (np.arange(H) + 0.5) / H
    elev = (0.5 - rows) * np.pi  # +pi/2 at the top row
    d_omega = (2 * np.pi / W) * (np.pi / H) * np.cos(elev)  # per pixel in a row
    L = lum(img)

    # --- locate the sun/moon: brightest spot of a lightly blurred luminance (upper hemisphere)
    k = 8
    Ls = L[: H // k * k, : W // k * k].reshape(H // k, k, W // k, k).mean(axis=(1, 3))
    Ls[H // k // 2 :, :] = 0
    iy, ix = np.unravel_index(np.argmax(Ls), Ls.shape)
    cy, cx = (iy + 0.5) * k, (ix + 0.5) * k
    # refine: luminance-weighted centroid in a window
    r = 48
    y0, y1 = int(max(0, cy - r)), int(min(H, cy + r))
    xs = (np.arange(int(cx - r), int(cx + r)) % W)
    win = L[y0:y1][:, xs]
    wy, wx = np.mgrid[y0:y1, int(cx - r) : int(cx + r)]
    wgt = np.maximum(win - np.percentile(win, 90), 0) ** 2
    if wgt.sum() > 0:
        cy = float((wy * wgt).sum() / wgt.sum()) + 0.5
        cx = float((wx * wgt).sum() / wgt.sum()) + 0.5
    u = (cx % W) / W
    v = 1 - cy / H
    sun_el = (v - 0.5) * np.pi
    sun_az = (u - 0.5) * 2 * np.pi
    sun_dir = [float(np.cos(sun_el) * np.cos(sun_az)), float(np.sin(sun_el)), float(np.cos(sun_el) * np.sin(sun_az))]

    # --- sky statistics (upper hemisphere)
    up = img[: H // 2]
    Lup = L[: H // 2]
    w_up = np.repeat(d_omega[: H // 2, None], W, axis=1)
    med = float(np.median(Lup))
    p997 = float(np.percentile(Lup, 99.7))
    # angular distance of every pixel to the sun
    cols = (np.arange(W) + 0.5) / W
    az = (cols - 0.5) * 2 * np.pi
    dirx = np.cos(elev)[:, None] * np.cos(az)[None, :]
    diry = np.repeat(np.sin(elev)[:, None], W, axis=1)
    dirz = np.cos(elev)[:, None] * np.sin(az)[None, :]
    cosang = dirx * sun_dir[0] + diry * sun_dir[1] + dirz * sun_dir[2]
    sun_mask = cosang > np.cos(np.radians(2.5))
    clamp_level = max(p997 * 1.6, med * 6)
    excess = np.maximum(img - clamp_level, 0) * sun_mask[..., None]
    d_om_full = np.repeat(d_omega[:, None], W, axis=1)
    sun_rgb = (excess * d_om_full[..., None]).sum(axis=(0, 1))  # irradiance removed from the disc
    sun_E = float(lum(sun_rgb[None, None, :])[0, 0])
    sky_E = float((np.minimum(Lup, clamp_level) * w_up * np.sin(np.clip(elev[: H // 2, None], 0, None))).sum())
    sun_col = sun_rgb / max(sun_rgb.max(), 1e-9)

    def band(e0, e1, mask=None):
        sel = (elev >= np.radians(e0)) & (elev < np.radians(e1))
        px = np.minimum(img[sel], clamp_level)
        ww = np.repeat(d_omega[sel, None], W, axis=1)
        if mask is not None:
            ww = ww * mask[sel]
        return [float(x) for x in (px * ww[..., None]).sum(axis=(0, 1)) / max(ww.sum(), 1e-9)]

    near = (cosang > np.cos(np.radians(50))).astype(np.float32)
    far = (cosang < np.cos(np.radians(110))).astype(np.float32)
    stats = {
        "zenith": band(55, 90),
        "horizon": band(1, 9),
        "horizonSun": band(1, 9, near),
        "horizonAway": band(1, 9, far),
        "mid": band(15, 45),
    }

    # --- encode the background: clamp, gamma 2.2, 8-bit JPG
    M = float(np.clip(clamp_level, 0.05, 48.0))
    enc = np.clip(img / M, 0, 1) ** (1 / 2.2)
    out = Image.fromarray((enc * 255 + 0.5).astype(np.uint8), "RGB")
    if W > 4096:
        out = out.resize((4096, 2048), Image.LANCZOS)
    os.makedirs(SKY_OUT, exist_ok=True)
    dst = os.path.join(SKY_OUT, f"{preset}.jpg")
    out.save(dst, quality=88, subsampling=1, optimize=True)
    meta = {
        "file": f"{preset}.jpg",
        "source": {"name": asset, "url": f"https://polyhaven.com/a/{asset}", "license": "CC0"},
        "range": M,
        "sunDir": sun_dir,
        "sunElevationDeg": float(np.degrees(sun_el)),
        "sunAzimuthDeg": float(np.degrees(sun_az)),
        "sunColor": [float(x) for x in sun_col],
        "sunIrradiance": sun_E,
        "skyIrradiance": sky_E,
        "medianLuminance": med,
        **stats,
    }
    print(f"  {W}x{H} median {med:.3f} p99.7 {p997:.2f} range {M:.2f} sunEl {np.degrees(sun_el):.1f} sunE {sun_E:.2f} skyE {sky_E:.2f} -> {os.path.getsize(dst)//1024} KB")
    return meta


def process_tex(name, asset, size, want_nr=True):
    print("tex", name, asset)
    d = os.path.join(DL, asset)
    files = http_json(f"https://api.polyhaven.com/files/{asset}")

    def get(keys):
        for k in keys:
            if k in files and "1k" in files[k]:
                fmt = files[k]["1k"].get("jpg") or files[k]["1k"].get("png")
                if fmt:
                    p = os.path.join(d, os.path.basename(fmt["url"]))
                    download(fmt["url"], p)
                    return Image.open(p)
        return None

    diff = get(["Diffuse", "diffuse", "diff", "col", "Color"])
    nor = get(["nor_gl"])
    rough = get(["Rough", "rough"])
    arm = None if rough is not None else get(["arm"])
    if diff is None or nor is None:
        raise RuntimeError(f"missing maps for {asset}: {list(files.keys())}")
    diff = diff.convert("RGB").resize((size, size), Image.LANCZOS)
    nor = nor.convert("RGB").resize((size, size), Image.LANCZOS)
    if rough is not None:
        r = np.asarray(rough.convert("L").resize((size, size), Image.LANCZOS))
    elif arm is not None:
        r = np.asarray(arm.convert("RGB").resize((size, size), Image.LANCZOS))[..., 1]
    else:
        r = np.full((size, size), 200, np.uint8)
    n = np.asarray(nor).copy()
    n[..., 2] = r  # pack roughness into B (normal Z is reconstructed in the shader)
    os.makedirs(TEX_OUT, exist_ok=True)
    pd = os.path.join(TEX_OUT, f"{name}_d.jpg")
    pn = os.path.join(TEX_OUT, f"{name}_nr.jpg")
    diff.save(pd, quality=82, subsampling=1, optimize=True)
    if want_nr:
        Image.fromarray(n, "RGB").save(pn, quality=80, subsampling=0, optimize=True)
    elif os.path.exists(pn):
        os.remove(pn)
    a = np.asarray(diff).astype(np.float32) / 255.0
    avg = [float(x) for x in (a ** 2.2).mean(axis=(0, 1))]
    print(f"  -> {os.path.getsize(pd)//1024} KB + {os.path.getsize(pn)//1024 if want_nr else 0} KB, avg linear {['%.3f' % c for c in avg]}, rough mean {r.mean()/255:.2f}")
    return {
        "albedo": f"{name}_d.jpg", "normalRough": f"{name}_nr.jpg" if want_nr else None, "avgLinear": avg, "roughMean": float(r.mean() / 255),
        "source": {"name": asset, "url": f"https://polyhaven.com/a/{asset}", "license": "CC0"},
    }


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("sky", "all"):
        meta = {p: process_sky(p, a, r) for p, (a, r) in SKIES.items()}
        with open(os.path.join(SKY_OUT, "sky.json"), "w") as f:
            json.dump(meta, f, indent=1)
    if what in ("tex", "all"):
        meta = {n: process_tex(n, a, s, nr) for n, (a, s, nr) in TEXTURES.items()}
        with open(os.path.join(TEX_OUT, "tex.json"), "w") as f:
            json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
