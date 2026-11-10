"""Tile preview renders into one contact sheet (system python + PIL).
usage: python3 sheet.py out.png cols img1 img2 ... [--scale 0.75]"""
import sys
from PIL import Image

args = sys.argv[1:]
scale = 1.0
if '--scale' in args:
    i = args.index('--scale')
    scale = float(args[i + 1])
    del args[i:i + 2]
out, cols, files = args[0], int(args[1]), args[2:]
ims = [Image.open(f).convert('RGB') for f in files]
if scale != 1.0:
    ims = [im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS) for im in ims]
w, h = ims[0].size
rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (w * cols, h * rows), (20, 20, 22))
for k, im in enumerate(ims):
    sheet.paste(im, ((k % cols) * w, (k // cols) * h))
sheet.save(out)
print('sheet', out, sheet.size)
