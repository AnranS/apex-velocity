// Shared render state: global uniforms, the atmospheric fog shader chunks that replace three's
// built-in fog, and the material patcher that wires every material into fog / cascaded shadows /
// feature hooks (wetness, wind, ...).
import * as THREE from 'three';
import type { CSM } from 'three/examples/jsm/csm/CSM.js';

/** Uniform objects shared (by reference) between all patched materials. */
export const U = {
  uTime: { value: 0 },
  uWet: { value: 0 }, // 0..1 surface wetness
  uRain: { value: 0 }, // 0..1 rain intensity
  uSunDir: { value: new THREE.Vector3(0.3, 0.5, 0.4).normalize() }, // direction TO the sun (world)
  uSunColor: { value: new THREE.Color(1, 1, 1) },
  uFogColA: { value: new THREE.Color(0.55, 0.65, 0.78) }, // away from the sun
  uFogColB: { value: new THREE.Color(1.0, 0.82, 0.62) }, // toward the sun
  uFogParams: { value: new THREE.Vector4(0.00045, 0.018, 0, 1) }, // density, height falloff, base height, max opacity
  uWind: { value: new THREE.Vector4(0.8, 0.6, 0.6, 0) }, // dir.x, dir.z, strength, gust phase
  uNight: { value: 0 }, // 0 day .. 1 night (drives emissive windows/lamps)
  // hemisphere ambient (radiance) for cheap Lambert foliage, which does not sample the environment map
  uAmbSky: { value: new THREE.Color(0.5, 0.6, 0.8) },
  uAmbGround: { value: new THREE.Color(0.1, 0.1, 0.08) },
};

export type ShaderHook = (shader: THREE.WebGLProgramParametersWithUniforms) => void;

let chunksInstalled = false;

/** Replace three's fog chunks with height fog + sun in-scattering. Call once before any material compiles. */
export function installShaderChunks(): void {
  if (chunksInstalled) return;
  chunksInstalled = true;
  const C = THREE.ShaderChunk as Record<string, string>;
  C.fog_pars_vertex = /* glsl */ `
#ifdef USE_FOG
  varying vec3 vFogWorldPos;
#endif`;
  C.fog_vertex = /* glsl */ `
#ifdef USE_FOG
  vFogWorldPos = cameraPosition + mvPosition.xyz * mat3( viewMatrix );
#endif`;
  C.fog_pars_fragment = /* glsl */ `
#ifdef USE_FOG
  uniform vec3 fogColor;
  varying vec3 vFogWorldPos;
  #ifdef FOG_EXP2
    uniform float fogDensity;
  #else
    uniform float fogNear;
    uniform float fogFar;
  #endif
  uniform vec3 uFogColA;
  uniform vec3 uFogColB;
  uniform vec3 uSunDir;
  uniform vec4 uFogParams;
  vec4 apexFog( vec3 worldPos ) {
    vec3 fv = worldPos - cameraPosition;
    float fd = length( fv );
    vec3 fdir = fv / max( fd, 1e-4 );
    float hf = uFogParams.y;
    float camH = max( cameraPosition.y - uFogParams.z, -40.0 );
    float t = hf * fv.y;
    float integ = abs( t ) > 1e-3 ? ( 1.0 - exp( - t ) ) / t : 1.0;
    float od = uFogParams.x * exp( - hf * camH ) * integ * fd;
    float amount = ( 1.0 - exp( - od ) ) * uFogParams.w;
    float sunAmt = pow( max( dot( fdir, uSunDir ), 0.0 ), 5.0 );
    return vec4( mix( uFogColA, uFogColB, sunAmt ), amount );
  }
#endif`;
  C.fog_fragment = /* glsl */ `
#ifdef USE_FOG
  vec4 apexF = apexFog( vFogWorldPos );
  gl_FragColor.rgb = mix( gl_FragColor.rgb, apexF.rgb, apexF.a );
#endif`;
}

let csmRef: CSM | null = null;
const patched = new Set<THREE.Material>();

export function setCSM(csm: CSM | null): void {
  csmRef = csm;
}

function isLit(m: THREE.Material): boolean {
  const a = m as unknown as Record<string, boolean>;
  return !!(a.isMeshStandardMaterial || a.isMeshPhysicalMaterial || a.isMeshLambertMaterial || a.isMeshPhongMaterial || a.isMeshToonMaterial);
}

/**
 * Wire a material into the global systems. `hooks` may rewrite the shader source; `key` must uniquely
 * identify the set of hooks (it becomes part of the program cache key).
 */
export function patchMaterial<T extends THREE.Material>(mat: T, hooks: ShaderHook[] = [], key = ''): T {
  if (mat.userData.apexPatched) return mat;
  mat.userData.apexPatched = true;
  let csmHook: ((shader: THREE.WebGLProgramParametersWithUniforms, renderer: THREE.WebGLRenderer) => void) | null = null;
  if (csmRef && isLit(mat)) {
    csmRef.setupMaterial(mat);
    csmHook = mat.onBeforeCompile;
  }
  mat.onBeforeCompile = (shader, renderer) => {
    if (csmHook) csmHook.call(mat, shader, renderer);
    Object.assign(shader.uniforms, U);
    for (const h of hooks) h(shader);
  };
  const k = `apex|${key}|${csmHook ? 'csm' : ''}`;
  mat.customProgramCacheKey = () => k;
  patched.add(mat);
  return mat;
}

/** Patch every material under `root` that has not been patched yet. */
export function patchTree(root: THREE.Object3D): void {
  root.traverse((o) => {
    const m = (o as THREE.Mesh).material;
    if (!m) return;
    if (Array.isArray(m)) m.forEach((x) => patchMaterial(x));
    else patchMaterial(m);
  });
}

export function forgetMaterials(): void {
  patched.clear();
}

/** Insert `code` right after `#include <chunk>` (or replace the include when `replace` is true). */
export function inject(src: string, chunk: string, code: string, mode: 'after' | 'before' | 'replace' = 'after'): string {
  const token = `#include <${chunk}>`;
  if (!src.includes(token)) {
    console.warn(`[shader] chunk <${chunk}> not found`);
    return src;
  }
  if (mode === 'replace') return src.replace(token, code);
  return src.replace(token, mode === 'after' ? `${token}\n${code}` : `${code}\n${token}`);
}

// ----------------------------------------------------------------------------- small texture helpers

const texLoader = new THREE.TextureLoader();
/** every texture loaded through the helpers below, so the engine can upload them ahead of first use */
export const loadedTextures = new Set<THREE.Texture>();

export function loadTexture(url: string, opts: { srgb?: boolean; repeat?: boolean; aniso?: number; flipY?: boolean } = {}): THREE.Texture {
  const t = texLoader.load(url);
  loadedTextures.add(t);
  t.colorSpace = opts.srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  if (opts.repeat !== false) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = opts.aniso ?? 8;
  if (opts.flipY !== undefined) t.flipY = opts.flipY;
  return t;
}

export function loadTextureAsync(url: string, opts: { srgb?: boolean; repeat?: boolean; aniso?: number; flipY?: boolean } = {}): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    texLoader.load(
      url,
      (t) => {
        t.colorSpace = opts.srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
        if (opts.repeat !== false) t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.anisotropy = opts.aniso ?? 8;
        if (opts.flipY !== undefined) t.flipY = opts.flipY;
        t.needsUpdate = true;
        loadedTextures.add(t);
        resolve(t);
      },
      undefined,
      (e) => reject(e),
    );
  });
}

/** Tileable value-noise texture (RGBA: four independent octave mixes), generated once on the CPU. */
export function makeNoiseTexture(size = 256, seed = 1): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  const rand = (x: number, y: number, s: number) => {
    let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(s, 362437);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  };
  const noise = (x: number, y: number, period: number, s: number) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const fx = x - xi;
    const fy = y - yi;
    const ux = fx * fx * (3 - 2 * fx);
    const uy = fy * fy * (3 - 2 * fy);
    const p = (v: number) => ((v % period) + period) % period;
    const a = rand(p(xi), p(yi), s);
    const b = rand(p(xi + 1), p(yi), s);
    const c = rand(p(xi), p(yi + 1), s);
    const d = rand(p(xi + 1), p(yi + 1), s);
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
  };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      for (let ch = 0; ch < 4; ch++) {
        let amp = 0.5;
        let f = 4 * (ch + 1);
        let v = 0;
        let norm = 0;
        for (let o = 0; o < 5; o++) {
          v += noise((x / size) * f, (y / size) * f, f, seed + ch * 13 + o) * amp;
          norm += amp;
          amp *= 0.5;
          f *= 2;
        }
        data[i + ch] = Math.max(0, Math.min(255, Math.round((v / norm) * 255)));
      }
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}
