"""Contact sheets of preview renders: python3 mk_sheet.py <tree_id> [lod] [tag]"""
import sys
from PIL import Image
tid = sys.argv[1]; lod = sys.argv[2] if len(sys.argv) > 2 else "0"; tag = sys.argv[3] if len(sys.argv) > 3 else ""
P = "_work/preview/"
a = Image.open(f"{P}{tid}_lod{lod}_full{tag}.png").convert("RGB")
b = Image.open(f"{P}{tid}_lod{lod}_close{tag}.png").convert("RGB")
f = Image.open(f"{P}{tid}_lod{lod}_far{tag}.png").convert("RGB"); w, h = f.size
f = f.crop((w // 2 - 110, h // 2 - 105, w // 2 + 110, h // 2 + 60)).resize((640, 480), Image.LANCZOS)
sheet = Image.new("RGB", (720 + 640, 960), (0, 0, 0))
sheet.paste(a, (0, 0)); sheet.paste(b.resize((640, 480), Image.LANCZOS), (720, 0)); sheet.paste(f, (720, 480))
sheet.save(f"{P}{tid}_lod{lod}_sheet{tag}.png")
print("sheet", f"{P}{tid}_lod{lod}_sheet{tag}.png")
