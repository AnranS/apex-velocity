// Parse every shipped GLB with three's GLTFLoader (the real consumer) and report what a scene would get.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const env = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../public/assets/env');
const manifest = JSON.parse(fs.readFileSync(path.join(env, 'manifest.json'), 'utf8'));
const loader = new GLTFLoader();
let bad = 0;
for (const asset of manifest.assets) {
  for (const lod of asset.lods) {
    const buf = fs.readFileSync(path.join(env, lod.file));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const gltf = await new Promise((res, rej) => loader.parse(ab, '', res, rej));
    let tris = 0; const mats = new Set(); const attrs = new Set(); const names = [];
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      names.push(o.name);
      const g = o.geometry;
      tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
      for (const k of Object.keys(g.attributes)) attrs.add(k);
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) mats.add(m.name);
    });
    const ok = tris === lod.triangles && [...mats].sort().join() === [...asset.materials].sort().filter((m) => mats.has(m)).join();
    if (!ok) bad++;
    console.log(`${ok ? 'ok ' : 'BAD'} ${lod.file.padEnd(22)} tris=${String(tris).padStart(5)} attrs=${[...attrs].sort().join(',')} mats=${[...mats].sort().join(',')}`);
  }
}
console.log(bad ? `${bad} problem(s)` : 'three.js GLTFLoader: all GLBs parsed, triangle counts match the manifest');
process.exit(bad ? 1 : 0);
