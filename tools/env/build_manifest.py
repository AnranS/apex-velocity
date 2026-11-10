"""Validate every shipped GLB / texture and write public/assets/env/manifest.json.

  python3 build_manifest.py        (exit code 1 if a check fails)
"""
import datetime
import json
import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glb import read_glb  # noqa: E402

ENV = os.path.abspath(os.path.join(HERE, "..", "..", "public", "assets", "env"))
IMP = os.path.join(HERE, "_work", "imp")

PH = "https://polyhaven.com/a/"
CC0 = "CC0-1.0"
TEX_SOURCES = {
    "bark_pine.webp": {"name": "Pine Bark (Poly Haven)", "url": PH + "pine_bark", "license": CC0},
    "bark_pine_n.webp": {"name": "Pine Bark (Poly Haven)", "url": PH + "pine_bark", "license": CC0},
    "bark_oak.webp": {"name": "Jolcham Oak Bark 01 (Poly Haven)", "url": PH + "jolcham_oak_bark_01", "license": CC0},
    "bark_oak_n.webp": {"name": "Jolcham Oak Bark 01 (Poly Haven)", "url": PH + "jolcham_oak_bark_01", "license": CC0},
    "bark_birch.webp": {"name": "Bark Platanus (Poly Haven), recoloured", "url": PH + "bark_platanus", "license": CC0},
    "bark_birch_n.webp": {"name": "Bark Platanus (Poly Haven)", "url": PH + "bark_platanus", "license": CC0},
}
LEAVES = {"albedo": "foliage_atlas.webp", "normal": "foliage_atlas_n.webp"}


def tree(tid, bark, lod0_max=7000, lod1_max=1500):
    return {"id": tid, "kind": "tree", "files": [f"{tid}_lod0.glb", f"{tid}_lod1.glb"], "budgets": [lod0_max, lod1_max],
            "impostor": True, "materials": ["bark", "leaves"], "vcolor": True,
            "textures": {"leaves": LEAVES, "bark": {"albedo": f"{bark}.webp", "normal": f"{bark}_n.webp"}},
            "source": "procedural"}


def prop(pid, mats, budget=None, textures=None, nodes=None):
    return {"id": pid, "kind": "prop", "files": [f"{pid}.glb"], "budgets": [budget], "impostor": False, "materials": mats,
            "vcolor": True, "textures": textures or {}, "source": "procedural", "nodes": nodes}


def rock(rid, name, slug):
    return {"id": rid, "kind": "rock", "files": [f"{rid}.glb"], "budgets": [1200], "impostor": False, "materials": ["rock"],
            "vcolor": False, "textures": {"rock": {"albedo": f"{rid}_d.webp", "normal": f"{rid}_n.webp"}},
            "source": {"name": name + " (Poly Haven photoscan, decimated + rebaked, albedo neutralised)", "url": PH + slug, "license": CC0}}


REGISTRY = [
    tree("pine_a", "bark_pine"),
    tree("pine_b", "bark_pine"),
    tree("broadleaf_a", "bark_oak"),
    tree("broadleaf_b", "bark_birch"),
    {"id": "grass_tuft", "kind": "cover", "files": ["grass_tuft.glb"], "budgets": [40], "impostor": False, "materials": ["grass"],
     "vcolor": True, "textures": {"grass": {"albedo": "grass_atlas.webp"}}, "source": "procedural"},
    {"id": "bush_a", "kind": "cover", "files": ["bush_a.glb"], "budgets": [600], "impostor": False, "materials": ["leaves"],
     "vcolor": True, "textures": {"leaves": LEAVES}, "source": "procedural"},
    rock("rock_a", "Boulder 01", "boulder_01"),
    rock("rock_b", "Rock 07", "rock_07"),
    rock("rock_c", "Namaqualand Boulder 04", "namaqualand_boulder_04"),
    prop("grandstand", ["concrete", "crowd", "panel", "panel_dark", "seats", "steel", "steel_dark"],
         textures={"crowd": {"albedo": "crowd_atlas.webp"}}),
    prop("pit_building", ["concrete", "garage_door", "glass", "panel", "panel_dark", "steel", "steel_dark"]),
    prop("gantry", ["ad", "concrete", "startlight_0", "startlight_1", "startlight_2", "startlight_3", "startlight_4", "steel",
                    "steel_dark"]),
    prop("billboard", ["ad", "concrete", "steel", "steel_dark"]),
    prop("light_tower", ["concrete", "lamp", "steel", "steel_dark"]),
    prop("marshal_post", ["marshal_orange", "panel", "panel_dark", "steel"]),
    prop("tire_stack", ["paint_band", "tire"], budget=500),
    prop("cone", ["cone_orange", "cone_white"], budget=150),
    prop("camera_tower", ["glass", "steel", "steel_dark", "wood"]),
    prop("flag_pole", ["concrete", "flag", "steel"], nodes=["flag_pole", "flag"]),
]

CONVENTIONS = {
    "axes": "glTF: +Y up, metres. Origin = ground centre (trunk base for trees). 'Front' of props = +Z "
            "(grandstand seats, pit garages, gantry start lights and billboard face +Z).",
    "uv": "glTF UVs (v = 0 at the image top): load the shared textures with flipY = false. GLBs carry no images; "
          "assign textures from this manifest per material name. baseColorFactor is white on textured materials.",
    "normalMaps": "OpenGL convention (+Y / green up). Rocks ship a TANGENT attribute (MikkTSpace, matches the bake). "
                  "Foliage/bark have no tangents: with derivative tangents and flipY = false use normalScale.y = -1 "
                  "(what GLTFLoader does).",
    "treeVertexColor": "COLOR_0 (linear, 8-bit): R = wind sway weight (0 at the trunk base -> 1 at branch / leaf tips), "
                       "G = per-branch phase 0..1 (0 on the trunk), B = ambient occlusion (darker deep inside the crown). "
                       "grass_tuft / bush_a use the same channels. Do NOT multiply the albedo by COLOR_0.rgb.",
    "foliageNormals": "Foliage normals are 'puffed' (outward from the crown, same on both faces). Render the cards "
                      "double-sided WITHOUT flipping the normal on back faces; alpha-test 0.5 (alpha is pre-gained) or "
                      "alpha-to-coverage.",
    "propVertexColor": "COLOR_0 = baked ambient occlusion as grayscale (R = G = B), includes ground occlusion. "
                       "Multiply ambient / diffuse by it.",
    "impostor": "<id>_imp.webp: `views` azimuth views side by side in one row (view k occupies u in [k/views, (k+1)/views]), "
                "orthographic, colour = albedo with soft ambient shading, RGB dilated under alpha. View k is seen from "
                "azimuth k*45 deg around +Y measured from +Z toward +X, i.e. camera direction from the trunk = "
                "(sin a, 0, cos a); pick k = round(atan2(dx, dz) / 45deg) mod 8 in the tree's local frame. Quad size = "
                "worldWidth x worldHeight, pivot bottom-centre on the tree origin, image right = camera right. "
                "width/height = whole atlas in pixels, viewWidth/viewHeight = one view. <id>_imp_n.webp holds "
                "view-space normals (x right, y up, z toward the camera, encoded n*0.5+0.5; z may be < 0 at the rim).",
    "crowd": "grandstand `crowd` cards: u repeats every 8 m (16 seats per texture repeat, random start per row), v 0 top "
             "-> 1 bottom; crowd_atlas.webp is RGBA (alpha cutout), tint/offset freely.",
    "startLights": "gantry materials startlight_0..4 (left to right seen from +Z): dark red lenses, drive emissive.",
    "flag": "flag_pole.glb has two nodes: `flag_pole` and `flag` (12x8 grid, UV 0..1 with u = 0 at the pole) for animation.",
}


def analyse(path):
    js, acc = read_glb(path)
    tris = 0
    mn, mx = np.full(3, 1e9), np.full(3, -1e9)
    mats, has_col, has_tan, has_uv = set(), True, True, True
    verts = 0
    for mesh in js["meshes"]:
        for prim in mesh["primitives"]:
            a = prim["attributes"]
            pos = acc(a["POSITION"])
            verts += len(pos)
            mn, mx = np.minimum(mn, pos.min(0)), np.maximum(mx, pos.max(0))
            tris += js["accessors"][prim["indices"]]["count"] // 3
            mats.add(js["materials"][prim["material"]]["name"])
            has_col &= "COLOR_0" in a
            has_tan &= "TANGENT" in a
            has_uv &= "TEXCOORD_0" in a
            nrm = acc(a["NORMAL"])
            ln = np.linalg.norm(nrm, axis=1)
            if np.abs(ln - 1).max() > 0.02:
                raise SystemExit(f"{path}: non-unit normals")
            if "COLOR_0" in a:
                c = acc(a["COLOR_0"])
                if not np.isfinite(c).all():
                    raise SystemExit(f"{path}: bad colours")
    images = len(js.get("images", []))
    return {"triangles": int(tris), "vertices": int(verts), "min": mn, "max": mx, "materials": sorted(mats), "color": has_col,
            "tangent": has_tan, "uv": has_uv, "nodes": [n["name"] for n in js["nodes"] if "mesh" in n], "images": images,
            "bytes": os.path.getsize(path)}


def main():
    errors, assets, tex_used = [], [], {}

    def need(cond, msg):
        if not cond:
            errors.append(msg)

    for reg in REGISTRY:
        lods, missing = [], False
        infos = []
        for f in reg["files"]:
            p = os.path.join(ENV, f)
            if not os.path.exists(p):
                missing = True
                break
            infos.append(analyse(p))
        if missing:
            print("skip (not built yet):", reg["id"])
            continue
        mn = np.min([i["min"] for i in infos], axis=0)
        mx = np.max([i["max"] for i in infos], axis=0)
        for f, info, budget in zip(reg["files"], infos, reg["budgets"]):
            lods.append({"file": f, "triangles": info["triangles"], "vertices": info["vertices"], "bytes": info["bytes"]})
            if budget:
                need(info["triangles"] <= budget, f"{f}: {info['triangles']} tris > budget {budget}")
            need(info["materials"] == sorted(reg["materials"]) or set(info["materials"]) <= set(reg["materials"]),
                 f"{f}: materials {info['materials']} != {reg['materials']}")
            need(info["images"] == 0, f"{f}: embeds images")
            need(info["uv"], f"{f}: missing TEXCOORD_0")
            if reg["vcolor"]:
                need(info["color"], f"{f}: missing COLOR_0")
            if reg["kind"] == "rock":
                need(info["tangent"], f"{f}: missing TANGENT")
            need(info["min"][1] > -0.2, f"{f}: geometry {info['min'][1]:.2f} m below the origin plane")
            if reg.get("nodes"):
                need(info["nodes"] == reg["nodes"], f"{f}: nodes {info['nodes']} != {reg['nodes']}")
        entry = {
            "id": reg["id"], "kind": reg["kind"], "lods": lods, "impostor": None,
            "size": [round(float(v), 3) for v in (mx - mn)],
            "bounds": {"min": [round(float(v), 3) for v in mn], "max": [round(float(v), 3) for v in mx]},
            "materials": sorted(set(m for i in infos for m in i["materials"])),
            "textures": reg["textures"], "source": reg["source"],
        }
        if len(infos[0]["nodes"]) > 1:
            entry["nodes"] = infos[0]["nodes"]
        if reg["impostor"]:
            ip = os.path.join(IMP, f"{reg['id']}_info.json")
            f_c, f_n = f"{reg['id']}_imp.webp", f"{reg['id']}_imp_n.webp"
            if os.path.exists(ip) and os.path.exists(os.path.join(ENV, f_c)):
                ii = json.load(open(ip))
                w, h = Image.open(os.path.join(ENV, f_c)).size
                need(w == ii["viewWidth"] * ii["views"] and h == ii["viewHeight"], f"{f_c}: atlas size mismatch")
                entry["impostor"] = {"file": f_c, "normal": f_n if os.path.exists(os.path.join(ENV, f_n)) else None,
                                     "views": ii["views"], "width": w, "height": h, "viewWidth": ii["viewWidth"],
                                     "viewHeight": ii["viewHeight"], "worldWidth": ii["worldWidth"],
                                     "worldHeight": ii["worldHeight"], "layout": "row", "pivot": "bottom-center",
                                     "azimuthStepDeg": 360 / ii["views"]}
                tex_used[f_c] = None
                if entry["impostor"]["normal"]:
                    tex_used[f_n] = None
            else:
                errors.append(f"{reg['id']}: impostor missing")
        tsrc = []
        for mat, td in reg["textures"].items():
            for role, f in td.items():
                need(os.path.exists(os.path.join(ENV, f)), f"{reg['id']}: texture {f} missing")
                tex_used[f] = None
                if f in TEX_SOURCES and TEX_SOURCES[f] not in tsrc:
                    tsrc.append(TEX_SOURCES[f])
        if tsrc:
            entry["textureSources"] = tsrc
        assets.append(entry)

    textures = {}
    for f in sorted(tex_used):
        p = os.path.join(ENV, f)
        if os.path.exists(p):
            im = Image.open(p)
            textures[f] = {"width": im.size[0], "height": im.size[1], "channels": len(im.getbands()), "bytes": os.path.getsize(p)}
    shipped = sorted(f for f in os.listdir(ENV) if f != "manifest.json" and not f.startswith("."))
    referenced = set(textures) | {l["file"] for a in assets for l in a["lods"]}
    for f in shipped:
        need(f in referenced, f"unreferenced file in pack: {f}")
    total = sum(os.path.getsize(os.path.join(ENV, f)) for f in shipped)
    manifest = {
        "version": 1,
        "generated": datetime.date.today().isoformat(),
        "generator": "tools/env/build_manifest.py",
        "conventions": CONVENTIONS,
        "assets": assets,
        "textures": textures,
        "credits": [{"what": "bark textures + rock scans", "source": "Poly Haven (polyhaven.com)", "license": CC0}],
        "totalBytes": total,
    }
    with open(os.path.join(ENV, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{'asset':14s} {'kind':6s} {'tris':>18s} {'size (x,y,z) m':>22s} {'KB':>6s}")
    for a in assets:
        tr = " / ".join(str(l["triangles"]) for l in a["lods"])
        kb = sum(l["bytes"] for l in a["lods"]) // 1024
        print(f"{a['id']:14s} {a['kind']:6s} {tr:>18s} {str(a['size']):>22s} {kb:>6d}")
    print("textures:", ", ".join(f"{k} {v['bytes'] // 1024}KB" for k, v in textures.items()))
    print(f"TOTAL pack: {total / 1e6:.2f} MB in {len(shipped)} files (+ manifest)")
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("VALIDATION OK")


if __name__ == "__main__":
    main()
