#!/usr/bin/env python3
"""Plot the track dump (tools/sim/dump_track.ts) to a PNG: python3 tools/sim/plot_track.py in.json out.png"""
import json
import sys

from PIL import Image, ImageDraw

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/apex_track.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "/tmp/apex_track.png"
d = json.load(open(src))
n = d["n"]
W, H = 1500, 1000
xs, zs = d["cx"], d["cz"]
pad = 90
minx, maxx, minz, maxz = min(xs), max(xs), min(zs), max(zs)
sc = min((W - 2 * pad) / (maxx - minx), (H - 2 * pad) / (maxz - minz))
ox = (W - (maxx - minx) * sc) / 2 - minx * sc
oz = (H - (maxz - minz) * sc) / 2 - minz * sc


def P(i, off):
    return (xs[i] + d["nx"][i] * off) * sc + ox, (zs[i] + d["nz"][i] * off) * sc + oz


img = Image.new("RGB", (W, H), (24, 44, 30))
dr = ImageDraw.Draw(img)
# corridor (grass / gravel), road, kerbs
for i in range(n):
    j = (i + 1) % n
    dr.polygon([P(i, d["wallL"][i]), P(j, d["wallL"][j]), P(j, -d["wallR"][j]), P(i, -d["wallR"][i])], fill=(52, 92, 48))
    if d["gravelL"][i] > 0:
        dr.polygon([P(i, d["gravelL"][i]), P(j, d["gravelL"][j] or d["gravelL"][i]), P(j, d["wallL"][j]), P(i, d["wallL"][i])], fill=(170, 150, 110))
    if d["gravelR"][i] > 0:
        dr.polygon([P(i, -d["gravelR"][i]), P(j, -(d["gravelR"][j] or d["gravelR"][i])), P(j, -d["wallR"][j]), P(i, -d["wallR"][i])], fill=(170, 150, 110))
for i in range(n):
    j = (i + 1) % n
    dr.polygon([P(i, d["hw"][i]), P(j, d["hw"][j]), P(j, -d["hw"][j]), P(i, -d["hw"][i])], fill=(58, 58, 62))
    if d["kerbL"][i] > 0.5:
        dr.line([P(i, d["hw"][i] + 0.7), P(j, d["hw"][j] + 0.7)], fill=(220, 40, 40) if (i // 2) % 2 else (240, 240, 240), width=3)
    if d["kerbR"][i] > 0.5:
        dr.line([P(i, -d["hw"][i] - 0.7), P(j, -d["hw"][j] - 0.7)], fill=(220, 40, 40) if (i // 2) % 2 else (240, 240, 240), width=3)
# walls
for i in range(n):
    j = (i + 1) % n
    dr.line([P(i, d["wallL"][i]), P(j, d["wallL"][j])], fill=(200, 200, 210), width=1)
    dr.line([P(i, -d["wallR"][i]), P(j, -d["wallR"][j])], fill=(200, 200, 210), width=1)
# racing line coloured by reference speed
vmax = max(d["vref"])
vmin = min(d["vref"])
for i in range(n):
    j = (i + 1) % n
    t = (d["vref"][i] - vmin) / (vmax - vmin + 1e-6)
    col = (int(255 * (1 - t)), int(80 + 175 * t), 60)
    dr.line([P(i, d["line"][i]), P(j, d["line"][j])], fill=col, width=2)
# start line, direction arrow, segment tags, elevation
dr.line([P(0, d["hw"][0] + 4), P(0, -d["hw"][0] - 4)], fill=(255, 255, 255), width=3)
a = P(0, 0)
b = P(20, 0)
dr.line([a, b], fill=(255, 255, 0), width=3)
for k, s in enumerate(d["segS"]):
    tag = d["segTag"][k]
    if not tag:
        continue
    i = int(s / d["ds"]) % n
    x, y = P(i, d["wallL"][i] + 14)
    dr.text((x, y), f"{tag} y={d['cy'][i]:.0f}", fill=(255, 255, 160))
for i in range(0, n, 200):
    x, y = P(i, -d["wallR"][i] - 16)
    dr.text((x, y), f"{int(i * d['ds'])}m", fill=(160, 200, 255))
dr.text((10, 10), f"N up (z south). len={n * d['ds']:.0f} m  v {vmin * 3.6:.0f}-{vmax * 3.6:.0f} km/h", fill=(255, 255, 255))
img.save(dst)
print("wrote", dst)
