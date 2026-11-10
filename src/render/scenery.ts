// Trackside scenery from the environment asset pack: instanced vegetation with three LOD levels
// (mesh -> simplified mesh -> 8-view billboard impostor), rocks, and race facilities placed from track zones.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { fbm2, mod, Rng, smoothstep } from '../sim/math';
import type { Track, ZoneDef } from '../sim/track';
import type { Engine } from './engine';
import { inject, loadTextureAsync, patchMaterial, ShaderHook, U } from './shared';
import type { World } from './world';

const ENV = () => `${import.meta.env.BASE_URL}assets/env/`;

interface ManifestAsset {
  id: string;
  kind: 'tree' | 'rock' | 'cover' | 'prop';
  lods: { file: string; triangles: number }[];
  impostor: { file: string; normal?: string; views: number; worldWidth: number; worldHeight: number } | null;
  size: number[];
  materials: string[];
  textures: Record<string, { albedo?: string; normal?: string }>;
}
interface Manifest {
  assets: ManifestAsset[];
}

const gltf = new GLTFLoader();

interface Part {
  geometry: THREE.BufferGeometry;
  material: string;
}

async function loadParts(file: string): Promise<Part[]> {
  const g = await gltf.loadAsync(ENV() + file);
  const parts: Part[] = [];
  g.scene.updateMatrixWorld(true);
  g.scene.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    const geo = m.geometry.clone();
    geo.applyMatrix4(m.matrixWorld);
    parts.push({ geometry: geo, material: (m.material as THREE.Material).name });
  });
  return parts;
}

// ------------------------------------------------------------------------------ shader hooks

/** Wind sway from the tree vertex-colour convention (aTree: R sway weight, G phase, B occlusion). */
const windHook: ShaderHook = (shader) => {
  shader.vertexShader = inject(shader.vertexShader, 'common', 'attribute vec4 aTree;\nuniform float uTime;\nuniform vec4 uWind;\nvarying float vTreeAO;');
  shader.vertexShader = inject(
    shader.vertexShader,
    'begin_vertex',
    /* glsl */ `
  vTreeAO = aTree.b;
  #ifdef USE_INSTANCING
    vec2 wOrg = vec2( instanceMatrix[3][0], instanceMatrix[3][2] );
  #else
    vec2 wOrg = vec2( modelMatrix[3][0], modelMatrix[3][2] );
  #endif
  float wPh = aTree.g * 6.2831 + dot( wOrg, vec2( 0.131, 0.173 ) );
  float wGust = 0.6 + 0.4 * sin( uTime * 0.31 + wOrg.x * 0.011 + wOrg.y * 0.013 );
  float wAmt = aTree.r * uWind.z * wGust;
  transformed.xz += uWind.xy * ( sin( uTime * 1.25 + wPh ) * 0.11 + sin( uTime * 3.1 + wPh * 1.7 ) * 0.035 ) * wAmt;
  transformed.y += sin( uTime * 2.2 + wPh * 1.3 ) * 0.03 * wAmt;
  // whole-tree lean grows with height
  transformed.xz += uWind.xy * sin( uTime * 0.7 + wOrg.x * 0.05 ) * 0.0018 * position.y * position.y * uWind.z;
`,
  );
  shader.fragmentShader = inject(shader.fragmentShader, 'common', 'varying float vTreeAO;');
};

/** Foliage shading: soft occlusion from the crown interior plus back-lit translucency. */
const leafHook: ShaderHook = (shader) => {
  shader.fragmentShader = inject(shader.fragmentShader, 'common', 'uniform vec3 uSunColor;\nuniform vec3 uAmbSky;\nuniform vec3 uAmbGround;');
  // puffed crown normals are shared by both faces of a card: undo three's back-face flip
  shader.fragmentShader = inject(shader.fragmentShader, 'normal_fragment_begin', '#ifdef DOUBLE_SIDED\n  normal *= faceDirection;\n#endif');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'aomap_fragment',
    /* glsl */ `
  float tAO = mix( 0.35, 1.0, vTreeAO );
  reflectedLight.directDiffuse *= mix( 0.6, 1.0, vTreeAO );
  {
    float upN = dot( normal, normalize( ( viewMatrix * vec4( 0.0, 1.0, 0.0, 0.0 ) ).xyz ) );
    reflectedLight.indirectDiffuse += diffuseColor.rgb * mix( uAmbGround, uAmbSky, 0.5 + 0.5 * upN ) * tAO;
  }
  {
    vec3 sunV = normalize( ( viewMatrix * vec4( uSunDir, 0.0 ) ).xyz );
    float back = pow( max( dot( normalize( vViewPosition ), - sunV ), 0.0 ), 2.5 );
    reflectedLight.directDiffuse += diffuseColor.rgb * uSunColor * back * 0.11 * vTreeAO;
  }`,
  );
};

/** Hemisphere ambient for Lambert materials (they ignore scene.environment). */
const ambientHook: ShaderHook = (shader) => {
  shader.fragmentShader = inject(shader.fragmentShader, 'common', 'uniform vec3 uAmbSky;\nuniform vec3 uAmbGround;');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'aomap_fragment',
    /* glsl */ `
  {
    float upN = dot( normal, normalize( ( viewMatrix * vec4( 0.0, 1.0, 0.0, 0.0 ) ).xyz ) );
    reflectedLight.indirectDiffuse += diffuseColor.rgb * mix( uAmbGround, uAmbSky, 0.5 + 0.5 * upN );
  }`,
  );
};

const barkHook: ShaderHook = (shader) => {
  shader.fragmentShader = inject(shader.fragmentShader, 'aomap_fragment', 'reflectedLight.indirectDiffuse *= mix( 0.45, 1.0, vTreeAO );');
};

/** Camera-facing 8-view impostor card (instances carry translation + uniform scale only). */
const impostorHook = (views: number): ShaderHook => (shader) => {
  shader.vertexShader = inject(shader.vertexShader, 'common', 'attribute float aYaw;\nattribute float aShow;\nvarying vec2 vImpUv;');
  shader.vertexShader = inject(
    shader.vertexShader,
    'beginnormal_vertex',
    /* glsl */ `
  vec3 iOrg = vec3( instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2] );
  vec2 toCam = normalize( cameraPosition.xz - iOrg.xz );
  objectNormal = normalize( vec3( toCam.x, 0.35, toCam.y ) );
`,
  );
  shader.vertexShader = inject(
    shader.vertexShader,
    'begin_vertex',
    /* glsl */ `
  vec3 iRight = vec3( toCam.y, 0.0, - toCam.x );
  transformed = ( iRight * position.x + vec3( 0.0, position.y, 0.0 ) ) * aShow;
  float iAng = atan( toCam.x, toCam.y ) - aYaw;
  float iView = mod( floor( iAng / ${(Math.PI * 2 / views).toFixed(6)} + 0.5 ), ${views.toFixed(1)} );
  vImpUv = vec2( ( uv.x + iView ) / ${views.toFixed(1)}, uv.y );
`,
  );
  shader.fragmentShader = inject(shader.fragmentShader, 'common', 'varying vec2 vImpUv;');
  shader.fragmentShader = shader.fragmentShader.replace(/vMapUv/g, 'vImpUv').replace(/vNormalMapUv/g, 'vImpUv');
};

// ------------------------------------------------------------------------------ helpers

interface Inst {
  x: number;
  y: number;
  z: number;
  yaw: number;
  s: number;
}

function instanced(geo: THREE.BufferGeometry, mat: THREE.Material, list: Inst[], tilt = false): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(geo, mat, list.length);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const p = new THREE.Vector3();
  const sc = new THREE.Vector3();
  for (let i = 0; i < list.length; i++) {
    const it = list[i];
    e.set(tilt ? Math.sin(it.yaw * 7.1) * 0.12 : 0, it.yaw, tilt ? Math.cos(it.yaw * 5.3) * 0.12 : 0);
    q.setFromEuler(e);
    p.set(it.x, it.y, it.z);
    sc.setScalar(it.s);
    m.compose(p, q, sc);
    mesh.setMatrixAt(i, m);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.computeBoundingSphere();
  mesh.matrixAutoUpdate = false;
  return mesh;
}

function makeAdTexture(lines: string[], bg: string, fg: string, accent: string, w = 1024, h = 320): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d')!;
  g.fillStyle = bg;
  g.fillRect(0, 0, w, h);
  // slanted accent blocks
  g.fillStyle = accent;
  g.beginPath();
  g.moveTo(0, 0);
  g.lineTo(w * 0.16, 0);
  g.lineTo(w * 0.09, h);
  g.lineTo(0, h);
  g.fill();
  g.beginPath();
  g.moveTo(w * 0.19, 0);
  g.lineTo(w * 0.215, 0);
  g.lineTo(w * 0.145, h);
  g.lineTo(w * 0.12, h);
  g.fill();
  g.fillStyle = fg;
  g.textBaseline = 'middle';
  g.textAlign = 'center';
  const big = lines[0];
  let size = h * 0.5;
  g.font = `italic 900 ${size}px "Avenir Next Condensed","Helvetica Neue","Arial Black","PingFang SC",sans-serif`;
  while (g.measureText(big).width > w * 0.7 && size > 20) {
    size -= 6;
    g.font = `italic 900 ${size}px "Avenir Next Condensed","Helvetica Neue","Arial Black","PingFang SC",sans-serif`;
  }
  g.fillText(big, w * 0.59, lines[1] ? h * 0.42 : h * 0.52);
  if (lines[1]) {
    g.font = `600 ${h * 0.14}px "Avenir Next","Helvetica Neue","PingFang SC",sans-serif`;
    g.fillStyle = accent;
    g.fillText(lines[1], w * 0.59, h * 0.8);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  t.flipY = false;
  return t;
}

const ADS: [string[], string, string, string][] = [
  [['VOLTARA', 'ELECTRIC PERFORMANCE'], '#0b1220', '#f4f6fa', '#19c2ff'],
  [['KINETIQ', 'RACING LUBRICANTS'], '#f3f1ea', '#15171a', '#e3172a'],
  [['NORDWIND', 'TIMING · TELEMETRY'], '#101314', '#ffd21f', '#ffffff'],
  [['HELIOX 98', 'HIGH OCTANE'], '#c2141c', '#ffffff', '#ffd21f'],
  [['ZENITH', 'TYRE TECHNOLOGY'], '#f6f6f6', '#0a2b6b', '#f08a00'],
  [['PULSAR', 'BRAKING SYSTEMS'], '#141414', '#ff5a1f', '#f2f2f2'],
  [['赤霞湾', 'CRIMSON BAY CIRCUIT'], '#7a0f18', '#fff4e6', '#ffb347'],
  [['AEROLITH', 'CARBON COMPOSITES'], '#0f2a22', '#d8ffe9', '#39e08f'],
];

// ------------------------------------------------------------------------------ scenery

interface LodCell {
  cx: number;
  cz: number;
  lod0: THREE.Mesh[];
  /** simplified meshes; also used as shadow-only proxies while lod0 is shown */
  lod1: { mesh: THREE.Mesh; mat: THREE.Material; shadowMat: THREE.Material }[];
  /** ranges inside the global per-species impostor meshes */
  imp: { attr: THREE.InstancedBufferAttribute; start: number; count: number }[];
  level: number;
  shadow: boolean;
}

const CELL = 112;

export class Scenery {
  readonly engine: Engine;
  readonly world: World;
  readonly track: Track;
  readonly group = new THREE.Group();
  private cells: LodCell[] = [];
  private mats: THREE.Material[] = [];
  private startLights: THREE.MeshStandardMaterial[] = [];
  private lampMat: THREE.MeshStandardMaterial | null = null;
  private timer = 0;
  private floods: THREE.SpotLight[] = [];
  treeCount = 0;

  constructor(engine: Engine, world: World) {
    this.engine = engine;
    this.world = world;
    this.track = world.track;
    this.group.name = 'scenery';
    engine.scene.add(this.group);
  }

  async build(): Promise<void> {
    let manifest: Manifest;
    try {
      manifest = (await (await fetch(`${ENV()}manifest.json`)).json()) as Manifest;
    } catch {
      console.warn('[scenery] environment pack missing');
      return;
    }
    const byId = new Map(manifest.assets.map((a) => [a.id, a]));
    await Promise.all([this.buildTrees(byId), this.buildProps(byId), this.buildRocksAndCover(byId)]);
    this.updateLod(this.engine.camera.position, true);
  }

  private keep<T extends THREE.Material>(m: T): T {
    this.mats.push(m);
    return m;
  }

  // ---------------------------------------------------------------- vegetation
  private inZone(s: number, d: number, kinds: ZoneDef['kind'][], margin: number): boolean {
    const tr = this.track;
    for (const z of tr.def.zones) {
      if (!kinds.includes(z.kind)) continue;
      if (z.side === 'L' && d < 0) continue;
      if (z.side === 'R' && d > 0) continue;
      let a = s;
      if (a > tr.length * 0.5 && z.s0 < 0) a -= tr.length;
      if (a >= z.s0 - margin && a <= z.s1 + margin) return true;
    }
    return false;
  }

  private async buildTrees(byId: Map<string, ManifestAsset>): Promise<void> {
    const q = this.engine.q;
    const tr = this.track;
    const world = this.world;
    const seed = tr.def.seed;
    const rng = new Rng(seed * 7 + 3);
    const species = ['pine_a', 'pine_b', 'broadleaf_a', 'broadleaf_b'].filter((id) => byId.has(id));
    if (species.length === 0) return;

    // ---- placement
    const cellMap = new Map<string, Map<string, Inst[]>>();
    const step = 10.6 / Math.sqrt(q.treeDensity);
    const x0 = tr.minX - 250;
    const x1 = tr.maxX + 250;
    const z0 = tr.minZ - 250;
    const z1 = tr.maxZ + 150;
    for (let gz = z0; gz < z1; gz += step) {
      for (let gx = x0; gx < x1; gx += step) {
        const x = gx + rng.range(-0.45, 0.45) * step;
        const z = gz + rng.range(-0.45, 0.45) * step;
        const r1 = rng.next();
        const r2 = rng.next();
        const r3 = rng.next();
        const r4 = rng.next();
        const cd = world.corridorDist(x, z);
        if (cd < 5.5) continue;
        const h = world.height(x, z);
        if (h < 2.6) continue;
        // keep the shore open: the coast straight looks out over the beach to the sea
        if (world.inland(x, z) < 64) continue;
        if (world.slope(x, z) > 0.55) continue;
        const forest = fbm2(x / 170, z / 170, 3, seed + 21) * 0.5 + 0.5;
        let prob = smoothstep(0.36, 0.6, forest);
        // a ragged tree line follows the circuit, thinning with distance
        prob = Math.max(prob, 0.55 * smoothstep(60, 9, cd) * smoothstep(0.25, 0.5, fbm2(x / 60, z / 60, 2, seed + 5) * 0.5 + 0.5));
        prob *= 1 - 0.72 * smoothstep(140, 320, cd);
        if (r1 > prob) continue;
        if (cd < 70) {
          tr.locateGlobal(x, z);
          const s = (tr.oi + tr.ot) * tr.ds;
          if (this.inZone(s, tr.od, ['grandstand', 'pit'], 18)) continue;
        }
        const kx = Math.floor(x / CELL);
        const kz = Math.floor(z / CELL);
        const key = `${kx}_${kz}`;
        // each cell is a stand of two species picked from its altitude + a low-frequency mix noise
        const ccx = (kx + 0.5) * CELL;
        const ccz = (kz + 0.5) * CELL;
        const hill = smoothstep(11, 38, world.height(ccx, ccz)) + fbm2(ccx / 260, ccz / 260, 2, seed + 40) * 0.9;
        const pair = hill > 0.62 ? ['pine_a', 'pine_b'] : hill > 0.3 ? ['pine_a', 'broadleaf_b'] : hill > 0.05 ? ['broadleaf_a', 'pine_b'] : ['broadleaf_a', 'broadleaf_b'];
        const want = pair[r2 < 0.62 ? 0 : 1];
        const id = species.includes(want) ? want : species[Math.floor(r3 * species.length) % species.length];
        let cell = cellMap.get(key);
        if (!cell) cellMap.set(key, (cell = new Map()));
        let list = cell.get(id);
        if (!list) cell.set(id, (list = []));
        list.push({ x, y: h - 0.15, z, yaw: r4 * Math.PI * 2, s: 0.78 + rng.next() * 0.5 });
        this.treeCount++;
      }
    }

    // ---- assets + materials
    const aTex = async (file: string | undefined, srgb: boolean) => (file ? loadTextureAsync(ENV() + file, { srgb, flipY: false, aniso: q.aniso }) : null);
    const msaa = q.msaa > 0;
    const leafMats = new Map<string, THREE.MeshLambertMaterial>();
    const depthMats = new Map<string, THREE.MeshDepthMaterial>();
    const shadowOnly = new Map<string, THREE.Material>();
    const shadowOnlyBark = this.keep(new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false }));
    const barkMats = new Map<string, THREE.MeshStandardMaterial>();
    const data = new Map<string, { lod0: Part[]; lod1: Part[]; a: ManifestAsset; impMat: THREE.MeshLambertMaterial | null }>();
    await Promise.all(
      species.map(async (id) => {
        const a = byId.get(id)!;
        const useLod0 = q.farTrees >= 100;
        const [lod0, lod1] = await Promise.all([useLod0 ? loadParts(a.lods[0].file) : Promise.resolve([] as Part[]), loadParts(a.lods[1].file)]);
        for (const part of [...lod0, ...lod1]) {
          const col = part.geometry.getAttribute('color');
          if (col) {
            part.geometry.setAttribute('aTree', col);
            part.geometry.deleteAttribute('color');
          }
        }
        const lt = a.textures.leaves;
        const bt = a.textures.bark;
        const lk = lt?.albedo ?? 'none';
        if (!leafMats.has(lk)) {
          const map = await aTex(lt?.albedo, true);
          const m = this.keep(
            patchMaterial(
              // Lambert + hemisphere ambient: foliage is the biggest source of overdraw, keep its shader cheap
              new THREE.MeshLambertMaterial({ map, alphaTest: 0.42, alphaToCoverage: msaa, side: THREE.DoubleSide, color: 0xc4d2b4 }),
              [windHook, leafHook],
              'leaf',
            ),
          );
          leafMats.set(lk, m);
          depthMats.set(lk, new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map, alphaTest: 0.42 }));
          shadowOnly.set(lk, this.keep(new THREE.MeshBasicMaterial({ map, alphaTest: 0.42, side: THREE.DoubleSide, colorWrite: false, depthWrite: false })));
        }
        const bk = bt?.albedo ?? 'none';
        if (!barkMats.has(bk)) {
          const [map, nmap] = await Promise.all([aTex(bt?.albedo, true), aTex(bt?.normal, false)]);
          const m = this.keep(
            patchMaterial(new THREE.MeshStandardMaterial({ map, normalMap: nmap, normalScale: new THREE.Vector2(1, -1), roughness: 0.92, metalness: 0 }), [windHook, barkHook], 'bark'),
          );
          barkMats.set(bk, m);
        }
        let impMat: THREE.MeshLambertMaterial | null = null;
        if (a.impostor) {
          const [map, nmap] = await Promise.all([aTex(a.impostor.file, true), aTex(a.impostor.normal, false)]);
          impMat = this.keep(
            patchMaterial(
              new THREE.MeshLambertMaterial({ map, normalMap: nmap, normalScale: new THREE.Vector2(1, -1), alphaTest: 0.5, alphaToCoverage: msaa, color: 0xb9c4ae }),
              [impostorHook(a.impostor.views), ambientHook],
              `imp${a.impostor.views}`,
            ),
          );
        }
        data.set(id, { lod0, lod1, a, impMat });
      }),
    );

    // ---- build LOD cells
    const impLists = new Map<string, Inst[]>();
    const impCells: { id: string; cell: LodCell; slot: number }[] = [];
    for (const [key, perSpecies] of cellMap) {
      const [kx, kz] = key.split('_').map(Number);
      const cell: LodCell = { cx: (kx + 0.5) * CELL, cz: (kz + 0.5) * CELL, lod0: [], lod1: [], imp: [], level: -1, shadow: false };
      for (const [id, list] of perSpecies) {
        const d = data.get(id)!;
        const lt = d.a.textures.leaves?.albedo ?? 'none';
        const bt = d.a.textures.bark?.albedo ?? 'none';
        for (const part of d.lod0) {
          const leaf = part.material === 'leaves';
          const mesh = instanced(part.geometry, leaf ? leafMats.get(lt)! : barkMats.get(bt)!, list);
          mesh.castShadow = false;
          mesh.receiveShadow = !leaf;
          mesh.visible = false;
          this.group.add(mesh);
          cell.lod0.push(mesh);
        }
        for (const part of d.lod1) {
          const leaf = part.material === 'leaves';
          const mat = leaf ? leafMats.get(lt)! : barkMats.get(bt)!;
          const mesh = instanced(part.geometry, mat, list);
          mesh.castShadow = false;
          mesh.receiveShadow = !leaf;
          if (leaf) mesh.customDepthMaterial = depthMats.get(lt)!;
          mesh.visible = false;
          this.group.add(mesh);
          cell.lod1.push({ mesh, mat, shadowMat: leaf ? shadowOnly.get(lt)! : shadowOnlyBark });
        }
        if (d.impMat && d.a.impostor) {
          let acc = impLists.get(id);
          if (!acc) impLists.set(id, (acc = []));
          cell.imp.push({ attr: null as unknown as THREE.InstancedBufferAttribute, start: acc.length, count: list.length });
          impCells.push({ id, cell, slot: cell.imp.length - 1 });
          for (const it of list) acc.push(it);
        }
      }
      this.cells.push(cell);
    }
    // hidden one-instance dummies so the engine prewarm compiles the shadow-proxy programs up front
    const seenProxy = new Set<THREE.Material>();
    for (const c of this.cells) {
      for (const o of c.lod1) {
        if (seenProxy.has(o.shadowMat)) continue;
        seenProxy.add(o.shadowMat);
        const dummy = instanced(o.mesh.geometry, o.shadowMat, [{ x: 0, y: -900, z: 0, yaw: 0, s: 0.001 }]);
        dummy.castShadow = true;
        dummy.customDepthMaterial = o.mesh.customDepthMaterial;
        dummy.visible = false;
        dummy.frustumCulled = false;
        this.group.add(dummy);
      }
    }

    // one impostor draw call per species; cells toggle their instance range through the aShow attribute
    for (const [id, list] of impLists) {
      const d = data.get(id)!;
      const imp = d.a.impostor!;
      const geo = new THREE.PlaneGeometry(imp.worldWidth, imp.worldHeight);
      geo.translate(0, imp.worldHeight / 2, 0);
      // GLB-style UVs (v = 0 at the image top) to match flipY = false textures
      const uv = geo.getAttribute('uv') as THREE.BufferAttribute;
      for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
      const yaw = new Float32Array(list.length);
      for (let i = 0; i < list.length; i++) yaw[i] = list[i].yaw;
      geo.setAttribute('aYaw', new THREE.InstancedBufferAttribute(yaw, 1));
      const show = new THREE.InstancedBufferAttribute(new Float32Array(list.length).fill(1), 1);
      show.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute('aShow', show);
      const mesh = instanced(geo, d.impMat!, list.map((it) => ({ ...it, yaw: 0 })));
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.frustumCulled = false;
      this.group.add(mesh);
      for (const ic of impCells) if (ic.id === id) ic.cell.imp[ic.slot].attr = show;
    }
  }

  private updateLod(cam: THREE.Vector3, force = false): void {
    const q = this.engine.q;
    const near = q.farTrees * 0.3;
    const mid = Math.max(q.farTrees, 130);
    const shadowR = q.shadows ? q.shadowFar * 0.8 : -1;
    for (const c of this.cells) {
      const d = Math.max(0, Math.hypot(c.cx - cam.x, c.cz - cam.z) - CELL * 0.6); // distance to the cell (approx)
      let level = d < near && c.lod0.length ? 0 : d < mid ? 1 : 2;
      // hysteresis
      if (!force && c.level >= 0 && level !== c.level) {
        const edge = level > c.level ? (c.level === 0 ? near : mid) + 10 : (level === 0 ? near : mid) - 10;
        if ((level > c.level && d < edge) || (level < c.level && d > edge)) level = c.level;
      }
      const shadow = level < 2 && d < shadowR;
      if (level === c.level && shadow === c.shadow) continue;
      c.level = level;
      c.shadow = shadow;
      for (const o of c.lod0) o.visible = level === 0;
      for (const o of c.lod1) {
        // the simplified mesh doubles as the shadow caster while the detailed mesh is on screen
        o.mesh.visible = level === 1 || (level === 0 && shadow);
        o.mesh.material = level === 0 ? o.shadowMat : o.mat;
        o.mesh.castShadow = shadow;
      }
      for (const o of c.imp) {
        const v = level === 2 ? 1 : 0;
        const arr = o.attr.array as Float32Array;
        if (arr[o.start] !== v) {
          arr.fill(v, o.start, o.start + o.count);
          o.attr.needsUpdate = true;
        }
      }
    }
  }

  // ---------------------------------------------------------------- rocks, bushes, grass
  private async buildRocksAndCover(byId: Map<string, ManifestAsset>): Promise<void> {
    const q = this.engine.q;
    const tr = this.track;
    const world = this.world;
    const rng = new Rng(tr.def.seed * 13 + 1);
    const tex = (file: string | undefined, srgb: boolean) => (file ? loadTextureAsync(ENV() + file, { srgb, flipY: false, aniso: q.aniso }) : Promise.resolve(null));

    // rocks: shoreline boulders and outcrops on steep ground near the track
    const rockIds = ['rock_a', 'rock_b', 'rock_c'].filter((id) => byId.has(id));
    const rockLists: Inst[][] = rockIds.map(() => []);
    if (rockIds.length) {
      for (let k = 0; k < 2600 && rockLists.reduce((a, l) => a + l.length, 0) < 260; k++) {
        const x = rng.range(tr.minX - 200, tr.maxX + 200);
        const z = rng.range(tr.minZ - 200, tr.maxZ + 130);
        const cd = world.corridorDist(x, z);
        if (cd < 5 || cd > 190) continue;
        const h = world.height(x, z);
        const sl = world.slope(x, z);
        const shore = h > -0.4 && h < 2.6;
        if (!shore && sl < 0.3) continue;
        if (h < -0.4) continue;
        const big = rng.next();
        rockLists[Math.floor(rng.next() * rockIds.length) % rockIds.length].push({
          x, y: h - 0.25, z, yaw: rng.range(0, 6.28), s: shore ? 0.9 + big * big * 3.2 : 1.2 + big * 3.4,
        });
      }
      await Promise.all(
        rockIds.map(async (id, i) => {
          if (!rockLists[i].length) return;
          const a = byId.get(id)!;
          const [parts, map, nmap] = await Promise.all([loadParts(a.lods[0].file), tex(a.textures.rock?.albedo, true), tex(a.textures.rock?.normal, false)]);
          const mat = this.keep(patchMaterial(new THREE.MeshStandardMaterial({ map, normalMap: nmap, roughness: 0.9, metalness: 0 })));
          for (const p of parts) {
            p.geometry.deleteAttribute('color');
            const mesh = instanced(p.geometry, mat, rockLists[i], true);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            this.group.add(mesh);
          }
        }),
      );
    }

    // bushes hugging the outside of the barriers
    const bush = byId.get('bush_a');
    if (bush) {
      const list: Inst[] = [];
      const o = { x: 0, y: 0, z: 0, psi: 0 };
      for (let s = 0; s < tr.length; s += 5.5) {
        for (const side of [1, -1]) {
          if (rng.next() > 0.42 * q.treeDensity) continue;
          const i = tr.indexAt(s);
          const pad = side > 0 ? world.padL[i] : world.padR[i];
          if (pad > 1) continue;
          const d = side * ((side > 0 ? tr.wallL[i] : tr.wallR[i]) + 3.4 + rng.next() * 5);
          tr.posAt(s + rng.range(-2, 2), d, o);
          const h = world.height(o.x, o.z);
          if (h < 2.2 || (world.inland(o.x, o.z) < 50 && rng.next() < 0.7)) continue;
          list.push({ x: o.x, y: h - 0.1, z: o.z, yaw: rng.range(0, 6.28), s: 0.8 + rng.next() * 0.9 });
        }
      }
      if (list.length) {
        const [parts, map] = await Promise.all([loadParts(bush.lods[0].file), tex(bush.textures.leaves?.albedo, true)]);
        const mat = this.keep(
          patchMaterial(
            new THREE.MeshLambertMaterial({ map, alphaTest: 0.42, alphaToCoverage: q.msaa > 0, side: THREE.DoubleSide, color: 0xb8c8a8 }),
            [windHook, leafHook],
            'leaf',
          ),
        );
        for (const p of parts) {
          const col = p.geometry.getAttribute('color');
          if (col) {
            p.geometry.setAttribute('aTree', col);
            p.geometry.deleteAttribute('color');
          }
          const mesh = instanced(p.geometry, mat, list);
          mesh.castShadow = false;
          mesh.receiveShadow = true;
          this.group.add(mesh);
        }
      }
    }

    // grass tufts on the verges (cheap cards, high tiers only)
    const tuft = byId.get('grass_tuft');
    if (tuft && q.grass) {
      const list: Inst[] = [];
      const o = { x: 0, y: 0, z: 0, psi: 0 };
      const stepS = 1.9 / q.treeDensity;
      for (let s = 0; s < tr.length; s += stepS) {
        const i = tr.indexAt(s);
        for (const side of [1, -1]) {
          const wall = side > 0 ? tr.wallL[i] : tr.wallR[i];
          const grav = side > 0 ? tr.gravelL[i] : tr.gravelR[i];
          const kerb = side > 0 ? tr.kerbL[i] : tr.kerbR[i];
          // the verge inside the barriers is mown; rough grass grows right behind them
          void grav;
          void kerb;
          const pad = side > 0 ? world.padL[i] : world.padR[i];
          if (pad > 1) continue;
          for (let k = 0; k < 2; k++) {
            const d = side * (wall + 0.9 + rng.next() * 4.5);
            tr.posAt(s + rng.range(-0.5, 0.5), d, o);
            const h = world.height(o.x, o.z);
            if (h < 2) continue;
            list.push({ x: o.x, y: h - 0.04, z: o.z, yaw: rng.range(0, 6.28), s: 0.8 + rng.next() * 1.0 });
          }
        }
      }
      const [parts, map] = await Promise.all([loadParts(tuft.lods[0].file), tex(tuft.textures.grass?.albedo, true)]);
      const mat = this.keep(
        patchMaterial(
          new THREE.MeshLambertMaterial({ map, alphaTest: 0.35, alphaToCoverage: q.msaa > 0, side: THREE.DoubleSide, color: 0x9fb37a }),
          [windHook, leafHook],
          'leaf',
        ),
      );
      // split along the lap so distant tufts are culled
      const per = 1400;
      for (const p of parts) {
        const col = p.geometry.getAttribute('color');
        if (col) {
          p.geometry.setAttribute('aTree', col);
          p.geometry.deleteAttribute('color');
        }
        for (let a = 0; a < list.length; a += per) {
          const mesh = instanced(p.geometry, mat, list.slice(a, a + per));
          mesh.castShadow = false;
          mesh.receiveShadow = true;
          this.group.add(mesh);
        }
      }
    }
  }

  // ---------------------------------------------------------------- facilities
  private propMaterial(name: string, extra: { ad?: THREE.Texture; crowd?: THREE.Texture | null; seat?: number }): THREE.Material {
    const std = (p: THREE.MeshStandardMaterialParameters) => this.keep(patchMaterial(new THREE.MeshStandardMaterial({ vertexColors: true, ...p })));
    switch (name) {
      case 'concrete':
        return std({ color: 0xa7a8a6, roughness: 0.9 });
      case 'steel':
        return std({ color: 0x9aa0a8, roughness: 0.45, metalness: 0.7 });
      case 'steel_dark':
        return std({ color: 0x26292e, roughness: 0.5, metalness: 0.6 });
      case 'panel':
        return std({ color: 0xdfe2e6, roughness: 0.55, metalness: 0.1 });
      case 'panel_dark':
        return std({ color: 0x1c2026, roughness: 0.5, metalness: 0.2 });
      case 'seats':
        return std({ color: extra.seat ?? 0x1c4c9a, roughness: 0.6 });
      case 'crowd':
        return std({ map: extra.crowd ?? null, alphaTest: extra.crowd ? 0.4 : 0, side: THREE.DoubleSide, roughness: 0.85, color: 0xffffff, vertexColors: false });
      case 'glass':
        return std({ color: 0x0c1a24, roughness: 0.05, metalness: 0.2, envMapIntensity: 1.4, vertexColors: false });
      case 'garage_door':
        return std({ color: 0x3b4047, roughness: 0.5, metalness: 0.4 });
      case 'ad':
        return std({ map: extra.ad ?? null, roughness: 0.5, vertexColors: false, emissive: 0xffffff, emissiveMap: extra.ad ?? null, emissiveIntensity: 0.06 + U.uNight.value * 0.5 });
      case 'lamp': {
        if (!this.lampMat) this.lampMat = std({ color: 0xf5f3ea, emissive: 0xfff2d6, emissiveIntensity: 0, roughness: 0.3, vertexColors: false }) as THREE.MeshStandardMaterial;
        return this.lampMat;
      }
      case 'marshal_orange':
        return std({ color: 0xe86a12, roughness: 0.6 });
      case 'tire':
        return std({ color: 0x0c0c0d, roughness: 0.85 });
      case 'paint_band':
        return std({ color: 0xe9e9e6, roughness: 0.6 });
      case 'cone_orange':
        return std({ color: 0xf06018, roughness: 0.6 });
      case 'cone_white':
        return std({ color: 0xf2f2f2, roughness: 0.6 });
      case 'wood':
        return std({ color: 0x6b5236, roughness: 0.85 });
      case 'flag':
        return std({ color: 0xf2f2f2, roughness: 0.7, side: THREE.DoubleSide, vertexColors: false });
      default:
        if (name.startsWith('startlight_')) {
          const m = std({ color: 0x1a0606, emissive: 0xff1408, emissiveIntensity: 0, roughness: 0.3, vertexColors: false }) as THREE.MeshStandardMaterial;
          this.startLights[Number(name.slice(-1))] = m;
          return m;
        }
        return std({ color: 0x777777, roughness: 0.7 });
    }
  }

  private batches = new Map<string, { mat: THREE.Material; geos: THREE.BufferGeometry[]; shadow: boolean }>();

  /** Queue one prop for batching: `front` (+Z of the asset) points along `yaw`. */
  private placeProp(parts: Part[], mats: Map<string, THREE.Material>, x: number, y: number, z: number, yaw: number, shadow = true): void {
    const m = new THREE.Matrix4().makeRotationY(yaw).setPosition(x, y, z);
    const region = `${Math.floor(x / 260)}_${Math.floor(z / 260)}`;
    for (const p of parts) {
      const mat = mats.get(p.material)!;
      const key = `${mat.uuid}|${region}|${shadow ? 1 : 0}`;
      let b = this.batches.get(key);
      if (!b) this.batches.set(key, (b = { mat, geos: [], shadow }));
      // normalise attributes so different assets can share a batch
      const src = p.geometry;
      const g = new THREE.BufferGeometry();
      const n = src.getAttribute('position').count;
      g.setAttribute('position', src.getAttribute('position').clone());
      g.setAttribute('normal', src.getAttribute('normal').clone());
      const uv = src.getAttribute('uv');
      g.setAttribute('uv', uv ? uv.clone() : new THREE.BufferAttribute(new Float32Array(n * 2), 2));
      const col = src.getAttribute('color');
      const cArr = new Float32Array(n * 3).fill(1);
      if (col) for (let i = 0; i < n; i++) cArr.set([col.getX(i), col.getY(i), col.getZ(i)], i * 3);
      g.setAttribute('color', new THREE.BufferAttribute(cArr, 3));
      g.setIndex(src.getIndex() ? src.getIndex()!.clone() : null);
      g.applyMatrix4(m);
      b.geos.push(g);
    }
  }

  private flushBatches(): void {
    for (const b of this.batches.values()) {
      const merged = b.geos.length === 1 ? b.geos[0] : mergeGeometries(b.geos, false);
      if (!merged) continue;
      for (const g of b.geos) if (g !== merged) g.dispose();
      merged.computeBoundingSphere();
      const mesh = new THREE.Mesh(merged, b.mat);
      mesh.castShadow = b.shadow;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      this.group.add(mesh);
    }
    this.batches.clear();
  }

  private async buildProps(byId: Map<string, ManifestAsset>): Promise<void> {
    const tr = this.track;
    const rng = new Rng(tr.def.seed * 3 + 11);
    const q = this.engine.q;
    const o = { x: 0, y: 0, z: 0, psi: 0 };
    const load = async (id: string) => (byId.has(id) ? loadParts(byId.get(id)!.lods[0].file) : null);
    const [grand, pit, gantry, board, tower, marshal, tires, camTower] = await Promise.all(
      ['grandstand', 'pit_building', 'gantry', 'billboard', 'light_tower', 'marshal_post', 'tire_stack', 'camera_tower'].map(load),
    );
    const crowdFile = byId.get('grandstand')?.textures.crowd?.albedo;
    const crowd = crowdFile ? await loadTextureAsync(ENV() + crowdFile, { srgb: true, flipY: false, aniso: q.aniso }) : null;
    const ads = ADS.map(([lines, bg, fg, ac]) => makeAdTexture(lines, bg, fg, ac));
    // shared materials: one instance per (material name, variant) so props batch together
    const shared = new Map<string, THREE.Material>();
    const matsFor = (parts: Part[], extra: { ad?: THREE.Texture; crowd?: THREE.Texture | null; seat?: number }) => {
      const m = new Map<string, THREE.Material>();
      for (const p of parts) {
        if (m.has(p.material)) continue;
        const variant = p.material === 'ad' ? `ad:${extra.ad?.uuid ?? ''}` : p.material === 'seats' ? `seats:${extra.seat ?? 0}` : p.material;
        let mat = shared.get(variant);
        if (!mat) shared.set(variant, (mat = this.propMaterial(p.material, extra)));
        m.set(p.material, mat);
      }
      return m;
    };
    /** yaw so that the asset's +Z faces the direction (dx, dz) */
    const yawTo = (dx: number, dz: number) => Math.atan2(dx, dz);

    for (const z of tr.def.zones) {
      const sides = z.side === 'both' ? [1, -1] : [z.side === 'L' ? 1 : -1];
      for (const side of sides) {
        if (z.kind === 'grandstand' && grand) {
          const seatCols = [0x1c4c9a, 0xb8232b, 0x1c4c9a, 0xe0a800];
          const mats = seatCols.map((c) => matsFor(grand, { crowd, seat: c }));
          let k = 0;
          for (let s = z.s0 + 12; s <= z.s1 - 12; s += 24.6) {
            const i = tr.indexAt(s);
            const wall = side > 0 ? tr.wallL[i] : tr.wallR[i];
            const d = side * (wall + (z.offset ?? 9) + 6.5);
            tr.posAt(s, d, o);
            this.placeProp(grand, mats[k++ % mats.length], o.x, o.y + 0.02, o.z, yawTo(-side * tr.nx[i], -side * tr.nz[i]));
          }
        } else if (z.kind === 'pit' && pit) {
          const mats = matsFor(pit, {});
          for (let s = z.s0 + 6; s <= z.s1 - 6; s += 12.1) {
            const i = tr.indexAt(s);
            const wall = side > 0 ? tr.wallL[i] : tr.wallR[i];
            const d = side * (wall + (z.offset ?? 7) + 6);
            tr.posAt(s, d, o);
            this.placeProp(pit, mats, o.x, o.y + 0.02, o.z, yawTo(-side * tr.nx[i], -side * tr.nz[i]));
          }
        } else if (z.kind === 'lights' && tower) {
          const mats = matsFor(tower, {});
          for (let s = z.s0; s <= z.s1; s += 58) {
            const i = tr.indexAt(s);
            const wall = side > 0 ? tr.wallL[i] : tr.wallR[i];
            const pad = side > 0 ? this.world.padL[i] : this.world.padR[i];
            const d = side * (wall + (pad > 1 ? 2.2 : z.offset ?? 5));
            tr.posAt(s, d, o);
            this.placeProp(tower, mats, o.x, o.y, o.z, yawTo(-side * tr.nx[i], -side * tr.nz[i]), false);
          }
        } else if (z.kind === 'billboards' && board) {
          for (let s = z.s0; s <= z.s1; s += 34) {
            const i = tr.indexAt(s);
            const wall = side > 0 ? tr.wallL[i] : tr.wallR[i];
            tr.posAt(s, side * (wall + (z.offset ?? 3) + 1.2), o);
            const mats = matsFor(board, { ad: ads[Math.floor(rng.next() * ads.length) % ads.length] });
            // angle the board slightly toward oncoming cars
            this.placeProp(board, mats, o.x, this.world.height(o.x, o.z) - 0.1, o.z, yawTo(-side * tr.nx[i] - tr.tx[i] * 0.5, -side * tr.nz[i] - tr.tz[i] * 0.5));
          }
        }
      }
    }

    // floodlighting for the pit straight: a few wide spots high above the track
    const nFlood = q.name === 'low' ? 0 : q.name === 'medium' ? 2 : 4;
    for (let k = 0; k < nFlood; k++) {
      const s = -190 + (k + 0.5) * (460 / Math.max(1, nFlood));
      tr.posAt(s, 0, o);
      const l = new THREE.SpotLight(0xfff1dc, 0, 150, 1.05, 0.6, 1.6);
      l.position.set(o.x, o.y + 30, o.z);
      l.target.position.set(o.x, o.y, o.z);
      l.castShadow = false;
      l.visible = false;
      this.group.add(l, l.target);
      l.target.updateMatrixWorld();
      this.floods.push(l);
    }

    // start/finish gantry just past the line, lights facing the grid
    if (gantry) {
      const mats = matsFor(gantry, { ad: makeAdTexture(['APEX VELOCITY', '赤霞湾国际赛道 · CRIMSON BAY CIRCUIT'], '#0a0c10', '#f5f7fa', '#ff3b2f', 2048, 320) });
      tr.posAt(10, 0, o);
      this.placeProp(gantry, mats, o.x, o.y, o.z, yawTo(-tr.tx[0], -tr.tz[0]));
    }

    // corner furniture: marshal posts, billboards and camera towers on the outside of each corner
    for (let k = 0; k < tr.segTag.length; k++) {
      const tag = tr.segTag[k];
      if (!tag || !/^T\d+$/.test(tag)) continue;
      const sMid = (tr.segS[k] + (tr.segS[k + 1] ?? tr.length)) * 0.5;
      const i = tr.indexAt(sMid);
      const outer = tr.kappa[i] > 0 ? -1 : 1; // outside of the turn
      const wall = outer > 0 ? tr.wallL[i] : tr.wallR[i];
      if (marshal) {
        tr.posAt(tr.segS[k] - 8, outer * (wall + 3.2), o);
        const j = tr.indexAt(tr.segS[k] - 8);
        this.placeProp(marshal, matsFor(marshal, {}), o.x, this.world.height(o.x, o.z) - 0.05, o.z, yawTo(-outer * tr.nx[j], -outer * tr.nz[j]));
      }
      if (board && rng.next() < 0.85) {
        for (let b = -1; b <= 1; b++) {
          const sb = sMid + b * 11;
          const jb = tr.indexAt(sb);
          const wb = outer > 0 ? tr.wallL[jb] : tr.wallR[jb];
          tr.posAt(sb, outer * (wb + 4.6), o);
          const mats = matsFor(board, { ad: ads[Math.floor(rng.next() * ads.length) % ads.length] });
          this.placeProp(board, mats, o.x, this.world.height(o.x, o.z) - 0.1, o.z, yawTo(-outer * tr.nx[jb], -outer * tr.nz[jb]));
        }
      }
      if (camTower && rng.next() < 0.6) {
        const sc = (tr.segS[k + 1] ?? tr.length) + 14;
        const jc = tr.indexAt(sc);
        const wc = outer > 0 ? tr.wallL[jc] : tr.wallR[jc];
        tr.posAt(sc, outer * (wc + 5.5), o);
        this.placeProp(camTower, matsFor(camTower, {}), o.x, this.world.height(o.x, o.z) - 0.05, o.z, rng.range(0, 6.28));
      }
    }

    this.flushBatches();

    // tyre walls in front of the barriers at gravel run-offs
    if (tires) {
      const list: Inst[] = [];
      for (let s = 0; s < tr.length && list.length < 520; s += 0.78) {
        const i = tr.indexAt(s);
        for (const side of [1, -1]) {
          const g = side > 0 ? tr.gravelL[i] : tr.gravelR[i];
          if (g <= 0) continue;
          const wall = side > 0 ? tr.wallL[i] : tr.wallR[i];
          tr.posAt(s, side * (wall - 0.42), o);
          list.push({ x: o.x, y: o.y, z: o.z, yaw: rng.range(0, 6.28), s: 1.25 });
        }
      }
      if (list.length) {
        const mats = matsFor(tires, {});
        for (const p of tires) {
          const mesh = instanced(p.geometry, mats.get(p.material)!, list);
          mesh.castShadow = false;
          mesh.receiveShadow = true;
          this.group.add(mesh);
        }
      }
    }
  }

  /** Start lights: `lit` = number of red lamps on (0..5). */
  setStartLights(lit: number, green = false): void {
    for (let i = 0; i < this.startLights.length; i++) {
      const m = this.startLights[i];
      if (!m) continue;
      m.emissive.setHex(green ? 0x19ff3c : 0xff1408);
      m.emissiveIntensity = green || i < lit ? 9 : 0;
    }
  }

  /** Recompute LOD levels immediately for a camera position (used by the load-time warm-up). */
  forceLod(cam: THREE.Vector3): void {
    this.updateLod(cam, true);
  }

  update(dt: number, cam: THREE.Vector3): void {
    this.timer -= dt;
    if (this.timer <= 0) {
      this.timer = 0.2;
      this.updateLod(cam);
      const night = U.uNight.value;
      if (this.lampMat) this.lampMat.emissiveIntensity = night > 0.2 ? 14 * night : 0;
      for (const l of this.floods) {
        l.visible = night > 0.4;
        l.intensity = 2600 * Math.min(1, Math.max(0, night - 0.4) * 2.2);
      }
    }
  }

  dispose(): void {
    this.engine.scene.remove(this.group);
    this.group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
    });
    for (const m of this.mats) m.dispose();
  }
}

void mod;
