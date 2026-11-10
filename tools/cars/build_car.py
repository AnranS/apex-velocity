"""Build a car and export the game assets.

  blender -b -P tools/cars/build_car.py -- --car aurion            # LOD0 + LOD1 GLB + JSON into public/assets/cars
  blender -b -P tools/cars/build_car.py -- --car aurion --draft    # quick coarse build into tools/cars/_work (shape iteration)
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import bpy
import bl, carlib, designs

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'public', 'assets', 'cars')
WORK = os.path.join(HERE, '_work')
LOD1_BUDGET = 17900


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--car', required=True)
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--lod', default='0,1')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    car = designs.CARS[args.car]()
    if getattr(car, 'paint_color', None):
        m = list(bl.MATERIALS['paint'])
        m[0] = tuple(car.paint_color)
        bl.MATERIALS['paint'] = tuple(m)
    out_dir = args.out or (WORK if args.draft else OUT)
    os.makedirs(out_dir, exist_ok=True)
    tris = {}
    lods = [0] if args.draft else [int(v) for v in args.lod.split(',')]
    body0 = None
    skin0 = None
    if not args.draft and 0 not in lods:
        bl.reset_scene()
        skin0 = carlib.build_body(car, 0, False)          # reference for material transfer
    for lod in lods:
        bl.reset_scene()
        objs, stats, body, skin = carlib.assemble(car, lod=lod, draft=args.draft, ref=skin0)
        if lod > 0:
            # coarser base until the whole LOD fits its triangle budget
            for target in (4700, 3900, 3200, 2600, 2100):
                if stats['total'] <= LOD1_BUDGET:
                    break
                car.skin_tris_lod1 = target
                bl.reset_scene()
                objs, stats, body, skin = carlib.assemble(car, lod=lod, draft=args.draft, ref=skin0)
        if lod == 0:
            body0 = body
            skin0 = skin
        name = f'{car.id}_draft.glb' if args.draft else (f'{car.id}.glb' if lod == 0 else f'{car.id}_lod{lod}.glb')
        path = os.path.join(out_dir, name)
        bl.export_glb(path, objs)
        tris[f'lod{lod}'] = int(stats['total'])
        print(f'[car] exported {path} {os.path.getsize(path) / 1e6:.2f} MB tris={stats["total"]}', flush=True)
    meta_path = os.path.join(out_dir, f'{car.id}.json')
    if not args.draft and body0 is not None and len(lods) > 1:
        meta = carlib.metadata(car, body0, tris, skin0)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=1)
        print('[car] wrote metadata', flush=True)
    elif not args.draft and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        meta['triangles'].update(tris)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=1)
        print('[car] updated metadata triangle counts', flush=True)


main()
