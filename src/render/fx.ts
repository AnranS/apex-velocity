// Visual effects: tyre smoke / dust / spray, skid marks, sparks, exhaust flames and rain.
// Particles are stateless on the GPU: each instance stores its birth state and the vertex shader
// integrates motion, so the CPU only writes a particle once when it is spawned.
import * as THREE from 'three';
import type { Car } from '../sim/car';
import type { RaceEvent, RaceSim } from '../sim/race';
import { SURF_GRASS, SURF_GRAVEL } from '../sim/track';
import type { CarVisual } from './car_visual';
import type { Engine } from './engine';
import { U, makeNoiseTexture, patchMaterial } from './shared';

// ------------------------------------------------------------------------------ particles

const PART_VERT = /* glsl */ `
attribute vec3 aPos;
attribute vec3 aVel;
attribute vec4 aLife;   // birth time, life, size0, size1
attribute vec4 aColor;
attribute vec2 aMisc;   // gravity, drag
uniform float uTime;
uniform vec4 uFogParams;
uniform vec3 uFogColA;
uniform vec3 uFogColB;
uniform vec3 uSunDir;
varying vec2 vUv;
varying vec4 vColor;
varying float vAge;
varying float vSeed;
varying vec4 vFog;
void main() {
  float age = uTime - aLife.x;
  float t = age / max( aLife.y, 1e-3 );
  vAge = t;
  vSeed = fract( aLife.x * 13.37 + aPos.x * 0.71 );
  vUv = uv;
  vColor = aColor;
  float k = aMisc.y;
  // closed-form motion with linear drag
  float e = k > 1e-3 ? ( 1.0 - exp( - k * age ) ) / k : age;
  vec3 p = aPos + aVel * e + vec3( 0.0, - aMisc.x, 0.0 ) * 0.5 * age * age;
  float size = mix( aLife.z, aLife.w, sqrt( clamp( t, 0.0, 1.0 ) ) );
  if ( t < 0.0 || t > 1.0 ) size = 0.0;
  vec4 mv = viewMatrix * vec4( p, 1.0 );
  // shrink and fade sprites right in front of the lens (keeps overdraw bounded, avoids smoke in the face)
  float nearF = smoothstep( 1.0, 7.5, - mv.z );
  size *= mix( 0.4, 1.0, nearF );
  vColor.a *= mix( 0.3, 1.0, nearF );
  // never let one sprite cover a large part of the screen
  size = min( size, - mv.z * 0.2 );
  float rot = vSeed * 6.2831 + age * ( vSeed - 0.5 ) * 1.5;
  vec2 c = position.xy * size;
  mv.xy += vec2( c.x * cos( rot ) - c.y * sin( rot ), c.x * sin( rot ) + c.y * cos( rot ) );
  gl_Position = projectionMatrix * mv;
  // fog (same model as the scene)
  vec3 fv = p - cameraPosition;
  float fd = length( fv );
  float hf = uFogParams.y;
  float tt = hf * fv.y;
  float integ = abs( tt ) > 1e-3 ? ( 1.0 - exp( - tt ) ) / tt : 1.0;
  float od = uFogParams.x * exp( - hf * max( cameraPosition.y - uFogParams.z, -40.0 ) ) * integ * fd;
  vFog = vec4( mix( uFogColA, uFogColB, pow( max( dot( fv / max( fd, 1e-4 ), uSunDir ), 0.0 ), 5.0 ) ), 1.0 - exp( - od ) );
}`;

const PART_FRAG_SMOKE = /* glsl */ `
uniform sampler2D uNoise;
uniform vec3 uLight;
varying vec2 vUv;
varying vec4 vColor;
varying float vAge;
varying float vSeed;
varying vec4 vFog;
void main() {
  vec2 q = vUv - 0.5;
  float r = length( q ) * 2.0;
  float n = texture2D( uNoise, vUv * 0.6 + vSeed ).r;
  float shape = smoothstep( 1.0, 0.2, r + ( n - 0.5 ) * 0.75 );
  float fade = smoothstep( 0.0, 0.08, vAge ) * ( 1.0 - smoothstep( 0.35, 1.0, vAge ) );
  float a = shape * fade * vColor.a;
  if ( a < 0.004 ) discard;
  vec3 c = vColor.rgb * uLight * ( 0.75 + 0.5 * n );
  c = mix( c, vFog.rgb, vFog.a );
  gl_FragColor = vec4( c, a );
}`;

const PART_FRAG_GLOW = /* glsl */ `
varying vec2 vUv;
varying vec4 vColor;
varying float vAge;
varying vec4 vFog;
void main() {
  vec2 q = vUv - 0.5;
  float r = length( q ) * 2.0;
  float core = smoothstep( 1.0, 0.0, r );
  float a = core * core * ( 1.0 - smoothstep( 0.5, 1.0, vAge ) ) * vColor.a;
  gl_FragColor = vec4( vColor.rgb * a * ( 1.0 - vFog.a ), 1.0 );
}`;

class ParticlePool {
  readonly mesh: THREE.Mesh;
  private readonly n: number;
  private cursor = 0;
  private pos: Float32Array;
  private vel: Float32Array;
  private life: Float32Array;
  private col: Float32Array;
  private misc: Float32Array;
  private attrs: THREE.InstancedBufferAttribute[];
  private dirty = false;

  constructor(n: number, additive: boolean, noise: THREE.Texture) {
    this.n = n;
    const g = new THREE.InstancedBufferGeometry();
    const quad = new THREE.PlaneGeometry(1, 1);
    g.index = quad.index;
    g.setAttribute('position', quad.getAttribute('position'));
    g.setAttribute('uv', quad.getAttribute('uv'));
    this.pos = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.life = new Float32Array(n * 4).fill(-1000);
    this.col = new Float32Array(n * 4);
    this.misc = new Float32Array(n * 2);
    const mk = (arr: Float32Array, size: number) => {
      const a = new THREE.InstancedBufferAttribute(arr, size);
      a.setUsage(THREE.DynamicDrawUsage);
      return a;
    };
    this.attrs = [mk(this.pos, 3), mk(this.vel, 3), mk(this.life, 4), mk(this.col, 4), mk(this.misc, 2)];
    g.setAttribute('aPos', this.attrs[0]);
    g.setAttribute('aVel', this.attrs[1]);
    g.setAttribute('aLife', this.attrs[2]);
    g.setAttribute('aColor', this.attrs[3]);
    g.setAttribute('aMisc', this.attrs[4]);
    g.instanceCount = n;
    const mat = new THREE.ShaderMaterial({
      vertexShader: PART_VERT,
      fragmentShader: additive ? PART_FRAG_GLOW : PART_FRAG_SMOKE,
      uniforms: {
        uTime: U.uTime,
        uFogParams: U.uFogParams,
        uFogColA: U.uFogColA,
        uFogColB: U.uFogColB,
        uSunDir: U.uSunDir,
        uNoise: { value: noise },
        uLight: { value: new THREE.Color(1, 1, 1) },
      },
      transparent: true,
      depthWrite: false,
      blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
      fog: false,
    });
    this.mesh = new THREE.Mesh(g, mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = additive ? 21 : 20;
  }

  get light(): THREE.Color {
    return (this.mesh.material as THREE.ShaderMaterial).uniforms.uLight.value as THREE.Color;
  }

  spawn(
    x: number, y: number, z: number, vx: number, vy: number, vz: number,
    life: number, size0: number, size1: number, r: number, g: number, b: number, a: number, gravity: number, drag: number,
  ): void {
    const i = this.cursor;
    this.cursor = (i + 1) % this.n;
    this.pos.set([x, y, z], i * 3);
    this.vel.set([vx, vy, vz], i * 3);
    this.life.set([U.uTime.value, life, size0, size1], i * 4);
    this.col.set([r, g, b, a], i * 4);
    this.misc.set([gravity, drag], i * 2);
    this.dirty = true;
  }

  flush(): void {
    if (!this.dirty) return;
    for (const a of this.attrs) a.needsUpdate = true;
    this.dirty = false;
  }

  clear(): void {
    this.life.fill(-1000);
    this.dirty = true;
    this.flush();
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

// ------------------------------------------------------------------------------ skid marks

class SkidMarks {
  readonly mesh: THREE.Mesh;
  private readonly max: number;
  private cursor = 0;
  private pos: Float32Array;
  private col: Float32Array;
  private posAttr: THREE.BufferAttribute;
  private colAttr: THREE.BufferAttribute;
  private dirty = false;

  constructor(max: number) {
    this.max = max;
    this.pos = new Float32Array(max * 4 * 3);
    this.col = new Float32Array(max * 4 * 4);
    const idx = new Uint32Array(max * 6);
    for (let i = 0; i < max; i++) idx.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 2, i * 4 + 1, i * 4 + 3], i * 6);
    const g = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.pos, 3);
    this.colAttr = new THREE.BufferAttribute(this.col, 4);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.colAttr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', this.posAttr);
    g.setAttribute('color', this.colAttr);
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    const m = patchMaterial(
      new THREE.MeshBasicMaterial({ color: 0x050505, vertexColors: true, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 }),
    );
    this.mesh = new THREE.Mesh(g, m);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
  }

  /** Add one quad between the previous edge (a, b) and the new edge (c, d). */
  add(ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number, dx: number, dy: number, dz: number, a0: number, a1: number): void {
    const i = this.cursor;
    this.cursor = (i + 1) % this.max;
    this.pos.set([ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz], i * 12);
    this.col.set([1, 1, 1, a0, 1, 1, 1, a0, 1, 1, 1, a1, 1, 1, 1, a1], i * 16);
    this.dirty = true;
  }

  flush(): void {
    if (!this.dirty) return;
    this.posAttr.needsUpdate = true;
    this.colAttr.needsUpdate = true;
    this.dirty = false;
  }

  clear(): void {
    this.pos.fill(0);
    this.col.fill(0);
    this.cursor = 0;
    this.dirty = true;
    this.flush();
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

// ------------------------------------------------------------------------------ rain

const RAIN_VERT = /* glsl */ `
attribute vec3 aOffset;
uniform float uTime;
uniform vec3 uCam;
uniform vec3 uVel;
varying float vAlpha;
void main() {
  const vec3 box = vec3( 30.0, 20.0, 30.0 );
  vec3 fall = vec3( 0.0, - 26.0, 0.0 ) + uVel;
  vec3 p = aOffset + fall * uTime * ( 0.85 + fract( aOffset.x * 7.3 ) * 0.3 );
  // wrap the drop into a box centred on the camera
  p = mod( p - uCam + box * 0.5, box ) - box * 0.5 + uCam;
  vec3 dir = normalize( fall );
  float len = 0.5 + fract( aOffset.z * 3.1 ) * 0.5;
  vec3 wp = p + dir * position.y * len;
  vec4 mv = viewMatrix * vec4( wp, 1.0 );
  // widen the streak in view space
  vec3 vd = normalize( ( viewMatrix * vec4( dir, 0.0 ) ).xyz );
  mv.xy += normalize( vec2( - vd.y, vd.x ) + 1e-5 ) * position.x * 0.026;
  gl_Position = projectionMatrix * mv;
  vAlpha = smoothstep( 15.0, 3.0, length( p - uCam ) ) * ( 0.5 + 0.5 * fract( aOffset.y * 5.7 ) );
}`;
const RAIN_FRAG = /* glsl */ `
uniform float uAmount;
varying float vAlpha;
void main() { gl_FragColor = vec4( vec3( 0.75, 0.8, 0.88 ) * vAlpha * uAmount * 1.5, 1.0 ); }`;

class Rain {
  readonly mesh: THREE.Mesh;
  constructor(count: number) {
    const g = new THREE.InstancedBufferGeometry();
    const quad = new THREE.PlaneGeometry(1, 1);
    g.index = quad.index;
    g.setAttribute('position', quad.getAttribute('position'));
    const off = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) off.set([Math.random() * 30, Math.random() * 20, Math.random() * 30], i * 3);
    g.setAttribute('aOffset', new THREE.InstancedBufferAttribute(off, 3));
    g.instanceCount = count;
    const m = new THREE.ShaderMaterial({
      vertexShader: RAIN_VERT,
      fragmentShader: RAIN_FRAG,
      uniforms: { uTime: U.uTime, uCam: { value: new THREE.Vector3() }, uVel: { value: new THREE.Vector3() }, uAmount: U.uRain },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      fog: false,
    });
    this.mesh = new THREE.Mesh(g, m);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 25;
  }
  update(cam: THREE.Vector3, vx: number, vz: number): void {
    const u = (this.mesh.material as THREE.ShaderMaterial).uniforms;
    (u.uCam.value as THREE.Vector3).copy(cam);
    // streaks lean against the direction of travel
    (u.uVel.value as THREE.Vector3).set(-vx * 0.45 + 3, 0, -vz * 0.45 + 1.5);
    this.mesh.visible = U.uRain.value > 0.01;
  }
  dispose(): void {
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

// ------------------------------------------------------------------------------ manager

interface WheelTrail {
  on: boolean;
  lx: number; ly: number; lz: number; // last left edge
  rx: number; ry: number; rz: number; // last right edge
  px: number; pz: number;
  alpha: number;
}

const _v = new THREE.Vector3();
const _q = new THREE.Quaternion();

export class Fx {
  readonly engine: Engine;
  readonly group = new THREE.Group();
  private smoke: ParticlePool;
  private glow: ParticlePool;
  private skids: SkidMarks;
  private rain: Rain | null = null;
  private trails: WheelTrail[][] = [];
  private acc: number[] = [];
  private flame: number[] = [];
  private scale: number;
  private budget = 0;
  private order: number[] = [];

  constructor(engine: Engine) {
    this.engine = engine;
    this.scale = engine.q.particles;
    const noise = makeNoiseTexture(128, 77);
    this.smoke = new ParticlePool(Math.round(2200 * this.scale), false, noise);
    this.glow = new ParticlePool(Math.round(900 * this.scale), true, noise);
    this.skids = new SkidMarks(Math.round(3500 * this.scale));
    this.group.add(this.skids.mesh, this.smoke.mesh, this.glow.mesh);
    this.group.name = 'fx';
    engine.scene.add(this.group);
  }

  /** Remove every skid mark and live particle (new race / back to the menu). */
  clear(): void {
    this.skids.clear();
    this.smoke.clear();
    this.glow.clear();
  }

  reset(cars: number): void {
    this.clear();
    this.trails = [];
    this.acc = [];
    this.flame = [];
    for (let i = 0; i < cars; i++) {
      this.trails.push(Array.from({ length: 4 }, () => ({ on: false, lx: 0, ly: 0, lz: 0, rx: 0, ry: 0, rz: 0, px: 0, pz: 0, alpha: 0 })));
      this.acc.push(0);
      this.flame.push(0);
    }
  }

  /** React to race events (impacts -> sparks, backfire/nitro -> flames). */
  onEvent(e: RaceEvent, race: RaceSim, visuals: CarVisual[]): void {
    if (e.type === 'impact') {
      const n = Math.min(26, Math.round((e.a / 900) * this.scale));
      const car = race.cars[e.car];
      for (let k = 0; k < n; k++) {
        const s = 2.5 + Math.random() * 6;
        this.glow.spawn(
          e.x, e.y + 0.05, e.z,
          car.vx * 0.55 + (Math.random() - 0.5) * s * 2, 1 + Math.random() * s * 0.7, car.vz * 0.55 + (Math.random() - 0.5) * s * 2,
          0.25 + Math.random() * 0.4, 0.05, 0.02, 3.2, 1.6, 0.5, 2.2, 9.8, 0.6,
        );
      }
    } else if (e.type === 'backfire' || e.type === 'mini_boost' || e.type === 'nitro_start') {
      this.flame[e.car] = Math.max(this.flame[e.car] ?? 0, e.type === 'backfire' ? 0.12 : 0.3);
      void visuals;
    }
  }

  update(dt: number, race: RaceSim, visuals: CarVisual[], cam: THREE.Vector3, sunColor: THREE.Color, ambient: number): void {
    const cars = race.cars;
    if (this.trails.length !== cars.length) this.reset(cars.length);
    // smoke is lit by the sun colour blended with ambient so it sits in the scene
    this.smoke.light.setRGB(0.5 * ambient + sunColor.r * 0.05, 0.52 * ambient + sunColor.g * 0.05, 0.56 * ambient + sunColor.b * 0.05);
    const wet = U.uWet.value;
    // smoke budget per second, shared by all cars (the player's car is served first)
    this.budget = Math.min(this.budget + dt * 90 * this.scale, 5);
    const order = this.order;
    order.length = 0;
    order.push(race.playerIdx);
    for (let i = 0; i < cars.length; i++) if (i !== race.playerIdx) order.push(i);
    for (let oi = 0; oi < order.length; oi++) {
      const ci = order[oi];
      const car = cars[ci];
      const vis = visuals[ci];
      const dx = vis.root.position.x - cam.x;
      const dz = vis.root.position.z - cam.z;
      const dist2 = dx * dx + dz * dz;
      const near = dist2 < 150 * 150;
      const isPlayer = ci === race.playerIdx;
      // other cars emit less the further away they are
      const share = isPlayer ? 1 : dist2 < 30 * 30 ? 0.3 : dist2 < 70 * 70 ? 0.16 : 0.06;
      const speed = car.speed;
      this.acc[ci] += dt;
      const emit = this.acc[ci] >= 1 / 60;
      if (emit) this.acc[ci] = 0;
      for (let wi = 0; wi < 4; wi++) {
        const w = car.wheels[wi];
        const tr = this.trails[ci][wi];
        const skid = w.contact ? w.skid : 0;
        const offroad = w.contact && (w.surface === SURF_GRASS || w.surface === SURF_GRAVEL);
        // ---- skid marks (tarmac only)
        const want = skid > 0.22 && !offroad && near;
        if (want) {
          // tyre edges: contact point +/- half width along the body's left axis
          const hw = (w.front ? 0.125 : 0.155) * 1.0;
          const ex = car.m00 * hw;
          const ey = car.m10 * hw;
          const ez = car.m20 * hw;
          const ax = w.cpx + ex + w.nx * 0.02;
          const ay = w.cpy + ey + w.ny * 0.02;
          const az = w.cpz + ez + w.nz * 0.02;
          const bx = w.cpx - ex + w.nx * 0.02;
          const by = w.cpy - ey + w.ny * 0.02;
          const bz = w.cpz - ez + w.nz * 0.02;
          const a = Math.min(0.75, (skid - 0.15) * 0.9);
          if (!tr.on) {
            tr.on = true;
            tr.alpha = 0;
            tr.lx = ax; tr.ly = ay; tr.lz = az;
            tr.rx = bx; tr.ry = by; tr.rz = bz;
            tr.px = w.cpx; tr.pz = w.cpz;
          } else {
            const d2 = (w.cpx - tr.px) * (w.cpx - tr.px) + (w.cpz - tr.pz) * (w.cpz - tr.pz);
            if (d2 > 0.3 * 0.3) {
              if (d2 < 9) this.skids.add(tr.lx, tr.ly, tr.lz, tr.rx, tr.ry, tr.rz, ax, ay, az, bx, by, bz, tr.alpha, a);
              tr.alpha = a;
              tr.lx = ax; tr.ly = ay; tr.lz = az;
              tr.rx = bx; tr.ry = by; tr.rz = bz;
              tr.px = w.cpx; tr.pz = w.cpz;
            }
          }
        } else tr.on = false;
        if (!emit || !near || !w.contact) continue;
        // ---- smoke / dust / spray
        if (this.budget < 1) continue;
        if (skid > 0.3 && !offroad && Math.random() < skid * 0.55 * this.scale * share) {
          this.budget -= 1;
          const s = 0.3 + Math.random() * 0.2;
          this.smoke.spawn(
            w.cpx, w.cpy + 0.12, w.cpz,
            car.vx * 0.22 + (Math.random() - 0.5) * 1.6, 0.7 + Math.random() * 1.1, car.vz * 0.22 + (Math.random() - 0.5) * 1.6,
            0.7 + Math.random() * 0.6, s, s * 3.8, 0.86, 0.86, 0.88, 0.26 + skid * 0.26, -0.6, 1.4,
          );
        } else if (offroad && speed > 6 && Math.random() < Math.min(1, speed / 30) * 0.22 * this.scale * share) {
          this.budget -= 1;
          const gravel = w.surface === SURF_GRAVEL;
          this.smoke.spawn(
            w.cpx, w.cpy + 0.1, w.cpz,
            car.vx * 0.3 + (Math.random() - 0.5) * 2, 0.6 + Math.random() * 1.4, car.vz * 0.3 + (Math.random() - 0.5) * 2,
            0.5 + Math.random() * 0.4, 0.35, 1.5 + Math.random() * 0.6, gravel ? 0.78 : 0.5, gravel ? 0.7 : 0.42, gravel ? 0.58 : 0.3, gravel ? 0.4 : 0.34, -0.3, 1.6,
          );
        } else if (wet > 0.3 && speed > 12 && wi >= 2 && Math.random() < Math.min(1, speed / 45) * 0.5 * this.scale * share) {
          this.budget -= 1;
          this.smoke.spawn(
            w.cpx, w.cpy + 0.15, w.cpz,
            car.vx * 0.5 + (Math.random() - 0.5) * 1.5, 1.2 + Math.random() * 1.6, car.vz * 0.5 + (Math.random() - 0.5) * 1.5,
            0.5 + Math.random() * 0.4, 0.5, 2.6, 0.9, 0.93, 0.97, 0.17 * wet, 1.5, 1.2,
          );
        }
      }
      // ---- exhaust flames (nitro: blue jet, backfire/mini boost: orange pop)
      if (this.flame[ci] > 0) this.flame[ci] -= dt;
      const nitro = car.nitroOn;
      if ((nitro || this.flame[ci] > 0) && near && emit) {
        _q.copy(vis.root.quaternion);
        const pts = [vis.points.exhaustL, vis.points.exhaustR];
        for (const p of pts) {
          const lp = p ?? [0, 0.45, car.spec.bodyRearZ];
          _v.set(lp[0], lp[1], lp[2] - 0.05).applyQuaternion(_q).add(vis.root.position);
          const bx = -car.m02;
          const by = -car.m12;
          const bz = -car.m22;
          for (let k = 0; k < 2; k++) {
            const sp = 6 + Math.random() * 9;
            if (nitro)
              this.glow.spawn(_v.x, _v.y, _v.z, car.vx + bx * sp, car.vy + by * sp + 0.2, car.vz + bz * sp, 0.06 + Math.random() * 0.06, 0.13, 0.3, 0.22, 0.6, 2.2, 0.8, 0, 3);
            else
              this.glow.spawn(_v.x, _v.y, _v.z, car.vx + bx * sp, car.vy + by * sp + 0.3, car.vz + bz * sp, 0.06 + Math.random() * 0.07, 0.12, 0.32, 2.2, 0.8, 0.18, 0.9, 0, 3);
          }
        }
      }
      // ---- wall scrape sparks
      if (car.scrape > 0.25 && near && emit && Math.random() < car.scrape) {
        const side = car.d > 0 ? 1 : -1;
        const hw = car.spec.bodyHalfWidth;
        const z = (Math.random() - 0.5) * (car.spec.bodyFrontZ - car.spec.bodyRearZ) * 0.8;
        const x = vis.root.position.x + car.m00 * hw * side + car.m02 * z;
        const y = vis.root.position.y + 0.35;
        const zz = vis.root.position.z + car.m20 * hw * side + car.m22 * z;
        for (let k = 0; k < 3; k++)
          this.glow.spawn(x, y, zz, car.vx * 0.7 + (Math.random() - 0.5) * 4, 0.5 + Math.random() * 3, car.vz * 0.7 + (Math.random() - 0.5) * 4, 0.2 + Math.random() * 0.35, 0.045, 0.015, 3.2, 1.7, 0.5, 2, 9.8, 0.8);
      }
    }
    // ---- rain
    if (U.uRain.value > 0.01) {
      if (!this.rain) {
        this.rain = new Rain(Math.round(7000 * this.scale));
        this.group.add(this.rain.mesh);
      }
      const p = race.player;
      this.rain.update(cam, p.vx, p.vz);
    } else if (this.rain) this.rain.mesh.visible = false;
    this.smoke.flush();
    this.glow.flush();
    this.skids.flush();
  }

  dispose(): void {
    this.smoke.dispose();
    this.glow.dispose();
    this.skids.dispose();
    this.rain?.dispose();
    this.engine.scene.remove(this.group);
  }
}

export type { Car };
