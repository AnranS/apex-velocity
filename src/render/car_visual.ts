// Car visuals: loads the GLB model (or builds a placeholder), swaps in game materials by name,
// and animates body pose, wheels, steering, suspension and lamps from the simulation state.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Car } from '../sim/car';
import type { CarModelMeta, CarSpec } from '../sim/car_specs';
import { inject, patchMaterial, ShaderHook, U } from './shared';

const loader = new GLTFLoader();
const gltfCache = new Map<string, Promise<THREE.Group | null>>();
const metaCache = new Map<string, Promise<CarModelMeta | null>>();

export function loadCarMeta(id: string): Promise<CarModelMeta | null> {
  let p = metaCache.get(id);
  if (!p) {
    p = fetch(`${import.meta.env.BASE_URL}assets/cars/${id}.json`)
      .then((r) => (r.ok ? (r.json() as Promise<CarModelMeta>) : null))
      .catch(() => null);
    metaCache.set(id, p);
  }
  return p;
}

function loadCarModel(id: string, lod: number): Promise<THREE.Group | null> {
  const key = `${id}:${lod}`;
  let p = gltfCache.get(key);
  if (!p) {
    const file = `${id}${lod > 0 ? '_lod1' : ''}.glb`;
    const tryLoad = (name: string) =>
      new Promise<THREE.Group | null>((resolve) => {
        loader.load(`${import.meta.env.BASE_URL}assets/cars/${name}`, (g) => resolve(g.scene), undefined, () => resolve(null));
      });
    // a missing LOD falls back to the full model; a missing model falls back to the placeholder
    p = tryLoad(file).then((m) => m ?? (lod > 0 ? tryLoad(`${id}.glb`) : null));
    gltfCache.set(key, p);
  }
  return p;
}

/** Vertex colour = baked ambient occlusion (applied to indirect light, not to albedo). */
const aoHook: ShaderHook = (shader) => {
  shader.fragmentShader = inject(shader.fragmentShader, 'color_fragment', '', 'replace');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'aomap_fragment',
    /* glsl */ `
  #ifdef USE_COLOR
    float apxAO = clamp( vColor.r, 0.0, 1.0 );
    reflectedLight.indirectDiffuse *= apxAO;
    reflectedLight.indirectSpecular *= mix( 1.0, apxAO, 0.8 );
    reflectedLight.directDiffuse *= mix( 1.0, apxAO, 0.3 );
    #if defined( USE_CLEARCOAT )
      clearcoatSpecularIndirect *= mix( 1.0, apxAO, 0.8 );
    #endif
  #endif`,
  );
};

/** Metallic flake sparkle + subtle pearl shift for the paint base layer; rain beads when wet. */
const paintHook: ShaderHook = (shader) => {
  shader.vertexShader = inject(shader.vertexShader, 'common', 'varying vec3 vObjPos;');
  shader.vertexShader = inject(shader.vertexShader, 'begin_vertex', 'vObjPos = position;');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'common',
    /* glsl */ `
varying vec3 vObjPos;
float apxHash3( vec3 p ) { p = fract( p * 0.3183099 + 0.1 ); p *= 17.0; return fract( p.x * p.y * p.z * ( p.x + p.y + p.z ) ); }
`,
  );
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'normal_fragment_maps',
    /* glsl */ `
  {
    vec3 cell = floor( vObjPos * 900.0 );
    vec3 fl = vec3( apxHash3( cell ), apxHash3( cell + 17.0 ), apxHash3( cell + 41.0 ) ) - 0.5;
    float fd = 1.0 - smoothstep( 4.0, 22.0, length( vViewPosition ) );
    normal = normalize( normal + fl * 0.11 * fd );
  }`,
  );
};

const aoSoftHook: ShaderHook = (shader) => {
  shader.fragmentShader = inject(shader.fragmentShader, 'color_fragment', '', 'replace');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'aomap_fragment',
    /* glsl */ `
  #ifdef USE_COLOR
    float apxAO = mix( 1.0, clamp( vColor.r, 0.0, 1.0 ), 0.45 );
    reflectedLight.indirectDiffuse *= apxAO;
    reflectedLight.indirectSpecular *= apxAO;
  #endif`,
  );
};

let poolTex: THREE.CanvasTexture | null = null;
/** Soft elongated light pool used as a cheap headlight beam on the road. */
function headPoolTexture(): THREE.CanvasTexture {
  if (poolTex) return poolTex;
  const c = document.createElement('canvas');
  c.width = 128;
  c.height = 256;
  const g = c.getContext('2d')!;
  const img = g.createImageData(128, 256);
  for (let y = 0; y < 256; y++) {
    for (let x = 0; x < 128; x++) {
      const u = (x - 64) / 64;
      const v = 1 - y / 255; // 0 = far end, 1 = at the car (canvas top row maps to the edge nearest the car)
      const spread = 0.35 + (1 - v) * 0.65;
      const lat = Math.exp(-((u / spread) * (u / spread)) * 2.6);
      const lon = Math.pow(v, 0.7) * Math.min(1, (1 - v) * 9) * Math.min(1, v * 3 + 0.05);
      const a = lat * lon;
      const i = (y * 128 + x) * 4;
      img.data[i] = 255 * a;
      img.data[i + 1] = 255 * a;
      img.data[i + 2] = 255 * a;
      img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  poolTex = new THREE.CanvasTexture(c);
  poolTex.colorSpace = THREE.SRGBColorSpace;
  return poolTex;
}

export interface PaintOptions {
  color: THREE.ColorRepresentation;
  accent?: THREE.ColorRepresentation;
  caliper?: THREE.ColorRepresentation;
}

interface WheelRig {
  pivot: THREE.Object3D;
  spin: THREE.Object3D;
  baseY: number;
}

export class CarVisual {
  readonly root = new THREE.Group();
  readonly spec: CarSpec;
  private wheels: WheelRig[] = [];
  private paintMat: THREE.MeshPhysicalMaterial;
  private brakeMat: THREE.MeshStandardMaterial;
  private tailMat: THREE.MeshStandardMaterial;
  private headMat: THREE.MeshStandardMaterial;
  private mats: THREE.Material[] = [];
  private q = new THREE.Quaternion();
  private qa = new THREE.Quaternion();
  private qb = new THREE.Quaternion();
  private poolMat: THREE.MeshBasicMaterial | null = null;
  private bodyNode: THREE.Object3D | null = null;
  private lean = 0;
  private dive = 0;
  /** the hero car gets a real spot light, so its fake beam is toned down */
  hero = false;
  ready = false;
  /** points from the model metadata (model space) */
  points: Record<string, number[]> = {};

  constructor(spec: CarSpec, paint: PaintOptions) {
    this.spec = spec;
    this.root.name = `car_${spec.id}`;
    this.paintMat = patchMaterial(
      new THREE.MeshPhysicalMaterial({
        color: paint.color,
        metalness: 0.62,
        roughness: 0.44,
        clearcoat: 1,
        clearcoatRoughness: 0.035,
        envMapIntensity: 1.5,
        vertexColors: true,
      }),
      [aoHook, paintHook],
      'paint',
    );
    this.brakeMat = patchMaterial(new THREE.MeshStandardMaterial({ color: 0x2a0303, emissive: 0xff1a0a, emissiveIntensity: 0, roughness: 0.3 }));
    this.tailMat = patchMaterial(new THREE.MeshStandardMaterial({ color: 0x300404, emissive: 0xff1608, emissiveIntensity: 0.9, roughness: 0.3 }));
    this.headMat = patchMaterial(new THREE.MeshStandardMaterial({ color: 0xdfe8f0, emissive: 0xdfeaff, emissiveIntensity: 1.2, roughness: 0.2 }));
    this.mats.push(this.paintMat, this.brakeMat, this.tailMat, this.headMat);
  }

  setPaint(color: THREE.ColorRepresentation): void {
    this.paintMat.color.set(color);
  }

  private material(name: string, paint: PaintOptions): THREE.Material {
    const std = (p: THREE.MeshStandardMaterialParameters, ao = true) => {
      const m = patchMaterial(new THREE.MeshStandardMaterial({ ...p, vertexColors: ao }), ao ? [aoHook] : [], ao ? 'ao' : '');
      this.mats.push(m);
      return m;
    };
    switch (name) {
      case 'paint':
        return this.paintMat;
      case 'paint_accent': {
        const m = patchMaterial(
          new THREE.MeshPhysicalMaterial({ color: paint.accent ?? 0x101012, metalness: 0.5, roughness: 0.4, clearcoat: 1, clearcoatRoughness: 0.05, vertexColors: true }),
          [aoHook],
          'ao',
        );
        this.mats.push(m);
        return m;
      }
      case 'glass': {
        const m = patchMaterial(
          new THREE.MeshPhysicalMaterial({ color: 0x06090c, metalness: 0, roughness: 0.02, transparent: true, opacity: 0.8, envMapIntensity: 1.5, clearcoat: 1, clearcoatRoughness: 0 }),
        );
        this.mats.push(m);
        return m;
      }
      case 'trim_black':
        return std({ color: 0x0b0b0c, roughness: 0.62, metalness: 0 });
      case 'carbon': {
        const m = patchMaterial(
          new THREE.MeshPhysicalMaterial({ color: 0x0c0d0f, metalness: 0.35, roughness: 0.42, clearcoat: 0.8, clearcoatRoughness: 0.12, vertexColors: true }),
          [aoHook],
          'ao',
        );
        this.mats.push(m);
        return m;
      }
      case 'metal':
        return std({ color: 0xc9ccd2, roughness: 0.16, metalness: 1 });
      case 'light_head':
        return this.headMat;
      case 'light_tail':
        return this.tailMat;
      case 'light_brake':
        return this.brakeMat;
      case 'interior':
        return std({ color: 0x0a0a0b, roughness: 0.8, metalness: 0 });
      case 'underbody':
        return std({ color: 0x040404, roughness: 0.9, metalness: 0 }, false);
      case 'tire':
        return std({ color: 0x0b0b0c, roughness: 0.84, metalness: 0 });
      case 'rim': {
        // rims sit deep in the arches: soften the baked occlusion so they keep their metallic sparkle
        const m = patchMaterial(new THREE.MeshStandardMaterial({ color: 0xc2c6cc, roughness: 0.3, metalness: 0.88, envMapIntensity: 1.5, vertexColors: true }), [aoSoftHook], 'aosoft');
        this.mats.push(m);
        return m;
      }
      case 'brake_disc':
        return std({ color: 0x8a8e95, roughness: 0.45, metalness: 0.7 });
      case 'caliper':
        return std({ color: paint.caliper ?? 0xd41414, roughness: 0.4, metalness: 0.2 });
      default:
        return std({ color: 0x555555, roughness: 0.6, metalness: 0 });
    }
  }

  /** Build the visual from the GLB if present, otherwise a placeholder. `lod` 0 = hero, 1 = traffic. */
  async build(paint: PaintOptions, lod: number): Promise<void> {
    this.hero = lod === 0;
    const model = await loadCarModel(this.spec.id, lod);
    const meta = await loadCarMeta(this.spec.id);
    if (meta?.points) this.points = meta.points;
    if (model) this.buildFromModel(model.clone(true), paint);
    else this.buildPlaceholder(paint);
    this.root.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        m.castShadow = true;
        m.receiveShadow = true;
      }
    });
    // headlight pool on the road (additive, fades in with darkness)
    const sp = this.spec;
    this.poolMat = new THREE.MeshBasicMaterial({
      map: headPoolTexture(), color: 0xcfe0ff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, fog: false,
      polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -6,
    });
    const pool = new THREE.Mesh(new THREE.PlaneGeometry(7.5, 17), this.poolMat);
    pool.rotation.x = -Math.PI / 2;
    pool.position.set(0, 0.04, sp.bodyFrontZ + 8.2);
    pool.renderOrder = 3;
    pool.castShadow = false;
    pool.receiveShadow = false;
    this.root.add(pool);
    this.mats.push(this.poolMat);
    this.ready = true;
  }

  private buildFromModel(model: THREE.Object3D, paint: PaintOptions): void {
    const cache = new Map<string, THREE.Material>();
    model.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      const swap = (src: THREE.Material) => {
        const name = src.name;
        let dst = cache.get(name);
        if (!dst) {
          dst = this.material(name, paint);
          cache.set(name, dst);
        }
        return dst;
      };
      m.material = Array.isArray(m.material) ? m.material.map(swap) : swap(m.material);
      const mat = m.material as THREE.Material | THREE.Material[];
      const hasGlass = Array.isArray(mat) ? mat.some((x) => x.transparent) : mat.transparent;
      if (hasGlass) m.renderOrder = 5;
    });
    const sp = this.spec;
    const names = ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr'];
    for (let i = 0; i < 4; i++) {
      const node = model.getObjectByName(names[i]);
      if (!node) continue;
      const spin = new THREE.Group();
      spin.name = `${names[i]}_spin`;
      const meshNode = node.getObjectByName(`${names[i]}_mesh`);
      if (meshNode) {
        node.add(spin);
        spin.add(meshNode);
      }
      this.wheels[i] = { pivot: node, spin, baseY: node.position.y };
    }
    void sp;
    this.bodyNode = model.getObjectByName('body') ?? null;
    this.root.add(model);
  }

  private buildPlaceholder(paint: PaintOptions): void {
    const sp = this.spec;
    const len = sp.bodyFrontZ - sp.bodyRearZ;
    const w = sp.bodyHalfWidth * 2;
    const zc = (sp.bodyFrontZ + sp.bodyRearZ) / 2;
    const body = new THREE.Mesh(new THREE.BoxGeometry(w, 0.42, len, 1, 1, 1), this.paintMat);
    body.position.set(0, sp.rideHeight + 0.3, zc);
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(w * 0.74, 0.4, len * 0.42), this.material('glass', paint));
    cabin.position.set(0, sp.rideHeight + 0.7, zc - 0.25);
    const tail = new THREE.Mesh(new THREE.BoxGeometry(w * 0.86, 0.07, 0.05), this.tailMat);
    tail.position.set(0, sp.rideHeight + 0.42, sp.bodyRearZ - 0.02);
    const brake = new THREE.Mesh(new THREE.BoxGeometry(w * 0.5, 0.05, 0.05), this.brakeMat);
    brake.position.set(0, sp.rideHeight + 0.52, sp.bodyRearZ - 0.02);
    const head = new THREE.Mesh(new THREE.BoxGeometry(w * 0.8, 0.06, 0.05), this.headMat);
    head.position.set(0, sp.rideHeight + 0.36, sp.bodyFrontZ + 0.02);
    const wing = new THREE.Mesh(new THREE.BoxGeometry(w * 0.9, 0.04, 0.32), this.material('carbon', paint));
    wing.position.set(0, sp.rideHeight + 0.84, sp.bodyRearZ + 0.25);
    this.root.add(body, cabin, tail, brake, head, wing);
    const tireMat = this.material('tire', paint);
    const rimMat = this.material('rim', paint);
    for (let i = 0; i < 4; i++) {
      const front = i < 2;
      const left = i % 2 === 0;
      const r = front ? sp.wheelRadiusF : sp.wheelRadiusR;
      const pivot = new THREE.Group();
      pivot.position.set((left ? 1 : -1) * (front ? sp.trackFront : sp.trackRear) * 0.5, r, front ? sp.axleFrontZ : sp.axleRearZ);
      const spin = new THREE.Group();
      const tire = new THREE.Mesh(new THREE.CylinderGeometry(r, r, front ? 0.25 : 0.31, 28), tireMat);
      tire.rotation.z = Math.PI / 2;
      const rim = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.62, r * 0.62, (front ? 0.25 : 0.31) + 0.01, 10), rimMat);
      rim.rotation.z = Math.PI / 2;
      spin.add(tire, rim);
      pivot.add(spin);
      this.root.add(pivot);
      this.wheels[i] = { pivot, spin, baseY: r };
    }
  }

  /** Update pose from the simulation, interpolating between the last two physics states. */
  update(car: Car, alpha: number): void {
    const a = alpha;
    const px = car.ppx + (car.px - car.ppx) * a;
    const py = car.ppy + (car.py - car.ppy) * a;
    const pz = car.ppz + (car.pz - car.ppz) * a;
    this.qa.set(car.pqx, car.pqy, car.pqz, car.pqw);
    this.qb.set(car.qx, car.qy, car.qz, car.qw);
    this.q.copy(this.qa).slerp(this.qb, a);
    this.root.quaternion.copy(this.q);
    // model origin = centre of mass - R * (0, comY, comZ)
    const o = _v.set(0, -car.comY, -car.comZ).applyQuaternion(this.q);
    this.root.position.set(px + o.x, py + o.y, pz + o.z);
    for (let i = 0; i < 4; i++) {
      const rig = this.wheels[i];
      if (!rig) continue;
      const w = car.wheels[i];
      const len = w.prevLen + (w.len - w.prevLen) * a;
      rig.pivot.position.y = w.radius + w.bump - len;
      rig.pivot.rotation.y = w.prevSteer + (w.steer - w.prevSteer) * a;
      rig.spin.rotation.x = w.prevAngle + (w.angle - w.prevAngle) * a;
    }
    // exaggerated body roll / dive on top of the simulated suspension (visual only)
    if (this.bodyNode) {
      const roll = Math.max(-0.07, Math.min(0.07, car.latG * 0.02 + (car.drifting ? car.driftDir * 0.012 : 0)));
      const pitch = Math.max(-0.035, Math.min(0.05, -car.longG * 0.016));
      this.lean += (roll - this.lean) * 0.18;
      this.dive += (pitch - this.dive) * 0.18;
      this.bodyNode.rotation.set(this.dive, 0, this.lean);
    }
    const night = U.uNight.value;
    this.brakeMat.emissiveIntensity = car.brk > 0.05 ? 7.5 : 0.0;
    this.tailMat.emissiveIntensity = 0.5 + night * 2.2 + (car.brk > 0.05 ? 1.5 : 0);
    this.headMat.emissiveIntensity = 0.8 + night * 6;
    if (this.poolMat) this.poolMat.opacity = Math.max(0, night - 0.3) * (this.hero ? 0.16 : 0.42);
  }

  dispose(): void {
    this.root.removeFromParent();
    for (const m of this.mats) m.dispose();
  }
}

const _v = new THREE.Vector3();
