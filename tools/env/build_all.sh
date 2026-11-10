#!/bin/bash
# Rebuild the whole environment asset pack (public/assets/env) from scratch.
# Needs: Blender 5.2 (headless), python3 with numpy + Pillow, network only for the CC0 downloads (first run).
# Scratch files, downloads and previews live in tools/env/_work (git-ignored).
set -euo pipefail
cd "$(dirname "$0")"
BL="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
bl() { "$BL" -b --factory-startup -P "$@" 2>&1 | grep -E "TILE|DONE|PROP|PACK|rock_|Traceback|rror" || true; }

# 0. CC0 sources (Poly Haven): bark photos + boulder scans, each in its own directory under _work/dl
python3 fetch_cc0.py 1k pine_bark:Diffuse,nor_gl jolcham_oak_bark_01:Diffuse,nor_gl bark_platanus:Diffuse,nor_gl
python3 fetch_cc0.py 2k model boulder_01 rock_07 namaqualand_boulder_04
python3 -I make_bark.py                                  # downloaded images are read in isolated mode

# 1. foliage / grass atlases: high-detail clusters -> per-tile passes -> composed atlases
bl render_foliage_tiles.py -- --set foliage
python3 compose_atlas.py foliage
bl render_foliage_tiles.py -- --set grass
python3 compose_atlas.py grass

# 2. trees (LOD0 + LOD1), ground cover
python3 gen_trees.py
python3 gen_cover.py

# 3. impostors (8 azimuth views: colour + view-space normals)
bl render_trees.py -- --mode impostor --ids pine_a,pine_b,broadleaf_a,broadleaf_b --samples 64
python3 compose_impostors.py pine_a pine_b broadleaf_a broadleaf_b

# 4. rocks: decimate + rebake the scans
bl gen_rocks.py
python3 finish_rocks.py

# 5. race facility props (AO baked to COLOR_0) + crowd strip
python3 make_crowd.py
bl gen_props.py

# 6. validate everything, write manifest.json, parse with three.js
python3 build_manifest.py
node check_three.mjs | tail -1

# optional previews: bl render_trees.py -- --mode single|stand ; bl render_pack.py ; bl render_scene.py -- --cam start
