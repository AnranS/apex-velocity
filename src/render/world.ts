// Track world: road, kerbs, verges, gravel traps, barriers, terrain and sea, all generated from the
// simulation Track so that what you see is exactly what the physics drives on.
import * as THREE from 'three';
import { clamp, fbm2, lerp, mod, ridged2, smoothstep } from '../sim/math';
import { Track } from '../sim/track';
import type { Engine } from './engine';
import { inject, loadTexture, makeNoiseTexture, patchMaterial, ShaderHook } from './shared';

const TEX = () => `${import.meta.env.BASE_URL}assets/tex/`;
const CHUNK = 64; // track samples per mesh chunk

export interface TerrainField {
  /** natural + corridor-blended ground height at world (x, z) */
  height(x: number, z: number): number;
  /** distance (m) from the track corridor edge, 0 inside */
  corridorDist(x: number, z: number): number;
  slope(x: number, z: number): number;
}

// --------------------------------------------------------------------------- shader hooks

/** Declares the shared noise sampler + helpers used by the ground shaders. */
function groundCommon(noise: THREE.Texture): ShaderHook {
  return (shader) => {
    shader.uniforms.uNoise = { value: noise };
    shader.fragmentShader = inject(
      shader.fragmentShader,
      'common',
      /* glsl */ `
uniform sampler2D uNoise;
uniform float uWet;
uniform float uRain;
uniform float uTime;
vec3 apxNormalRG( vec4 nr ) { vec2 n = nr.xy * 2.0 - 1.0; return vec3( n, sqrt( max( 1.0 - dot( n, n ), 0.0 ) ) ); }
`,
    );
  };
}

const roadHook: ShaderHook = (shader) => {
  shader.vertexShader = inject(shader.vertexShader, 'common', 'attribute vec2 aInfo;\nvarying vec2 vInfo;');
  shader.vertexShader = inject(shader.vertexShader, 'begin_vertex', 'vInfo = aInfo;');
  shader.fragmentShader = inject(shader.fragmentShader, 'common', 'varying vec2 vInfo;');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'map_fragment',
    /* glsl */ `
  vec2 sd = vMapUv;                       // (lateral d, distance s) in metres
  vec2 uvA = sd / 3.4;
  vec2 uvB = sd / 21.0 + 0.37;
  vec4 texA = texture2D( map, uvA );
  vec4 texB = texture2D( map, uvB );
  vec4 nrA = texture2D( normalMap, uvA );
  vec4 nrB = texture2D( normalMap, uvB );
  vec4 nz1 = texture2D( uNoise, sd / 130.0 );
  vec4 nz2 = texture2D( uNoise, sd / vec2( 3.5, 46.0 ) );
  // the source asphalt is very dark; normalise it and rebuild a believable sun-bleached tarmac grey
  vec3 detail = texA.rgb / 0.0115;
  float macro = texB.g / 0.0105;
  vec3 alb = vec3( 0.082, 0.083, 0.088 ) * mix( vec3( 1.0 ), detail, 0.75 ) * mix( 1.0, macro, 0.4 ) * ( 0.78 + 0.44 * nz1.r );
  // rubbered-in racing line
  float lineDist = abs( sd.x - vInfo.x );
  float rubber = smoothstep( 2.9, 0.5, lineDist ) * ( 0.5 + 0.5 * nz2.g );
  alb *= 1.0 - rubber * 0.4;
  // painted edge lines
  float edge = vInfo.y - abs( sd.x );
  float paint = smoothstep( 0.2, 0.24, edge ) * ( 1.0 - smoothstep( 0.4, 0.44, edge ) ) * ( 0.7 + 0.3 * texA.g );
  alb = mix( alb, vec3( 0.78 ), paint );
  // wetness: darker, glossier, with standing water in the low spots
  float puddle = smoothstep( 0.52, 0.7, nz1.b * 0.7 + nz2.r * 0.3 );
  float wetMask = uWet * ( 0.82 + 0.18 * puddle );
  alb *= mix( 1.0, 0.45, uWet );
  diffuseColor.rgb *= alb;
  float apxRough = mix( 0.62, 0.98, nrA.b ) - rubber * 0.14 - paint * 0.2;
  apxRough = mix( apxRough, 0.045 + 0.13 * ( 1.0 - puddle ), wetMask );
`,
    'replace',
  );
  shader.fragmentShader = inject(shader.fragmentShader, 'roughnessmap_fragment', 'roughnessFactor = clamp( apxRough, 0.04, 1.0 );');
  // a film of water is a stronger mirror than dry tarmac: lift the specular reflectance where it is wet
  shader.fragmentShader = inject(shader.fragmentShader, 'lights_physical_fragment', 'material.specularColor = mix( material.specularColor, vec3( 0.13 ), wetMask );');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'normal_fragment_maps',
    /* glsl */ `
  vec3 mapN = apxNormalRG( nrA ) + vec3( apxNormalRG( nrB ).xy * 0.6, 0.0 );
  mapN.xy *= normalScale * ( 1.0 - wetMask * 0.88 );
  // rain ripples on standing water
  vec2 rp = texture2D( uNoise, sd * 0.9 + vec2( 0.0, uTime * 0.9 ) ).rg + texture2D( uNoise, sd * 1.7 - vec2( uTime * 0.7, 0.0 ) ).ba - 1.0;
  mapN.xy += rp * 0.22 * uRain * wetMask;
  normal = normalize( tbn * normalize( mapN ) );
`,
    'replace',
  );
};

/** Grass verge blending into gravel traps (aInfo.x = gravel weight). */
const vergeHook = (gravelD: THREE.Texture, gravelNR: THREE.Texture): ShaderHook => (shader) => {
  shader.uniforms.uGravelD = { value: gravelD };
  shader.uniforms.uGravelNR = { value: gravelNR };
  shader.vertexShader = inject(shader.vertexShader, 'common', 'attribute vec2 aInfo;\nvarying vec2 vInfo;');
  shader.vertexShader = inject(shader.vertexShader, 'begin_vertex', 'vInfo = aInfo;');
  shader.fragmentShader = inject(shader.fragmentShader, 'common', 'varying vec2 vInfo;\nuniform sampler2D uGravelD;\nuniform sampler2D uGravelNR;');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'map_fragment',
    /* glsl */ `
  vec2 wuv = vMapUv;                       // world xz in metres
  vec4 nz1 = texture2D( uNoise, wuv / 90.0 );
  vec4 nz2 = texture2D( uNoise, wuv / 9.0 );
  vec3 g1 = texture2D( map, wuv / 2.6 ).rgb;
  vec3 g2 = texture2D( map, wuv / 13.0 + 0.21 ).rgb;
  vec3 grass = g1 * ( 0.55 + 0.9 * g2.g ) * vec3( 0.62, 0.92, 0.5 ) * ( 0.75 + 0.5 * nz1.g );
  vec4 gnr = texture2D( normalMap, wuv / 2.6 );
  vec3 grav = texture2D( uGravelD, wuv / 2.2 ).rgb * ( 0.8 + 0.3 * nz1.r ) * vec3( 0.9, 0.86, 0.78 );
  vec4 vnr = texture2D( uGravelNR, wuv / 2.2 );
  float gw = smoothstep( 0.42, 0.58, vInfo.x + ( nz2.r - 0.5 ) * 0.3 );
  vec3 alb = mix( grass, grav, gw );
  vec4 nrA = mix( gnr, vnr, gw );
  float pv = smoothstep( 0.35, 0.65, vInfo.y );
  alb = mix( alb, vec3( 0.105, 0.105, 0.11 ) * ( 0.8 + 0.4 * nz1.r ) * ( 0.85 + 0.3 * nz2.g ), pv );
  nrA = mix( nrA, vec4( 0.5, 0.5, 0.7, 1.0 ), pv );
  alb *= mix( 1.0, 0.68, uWet );
  diffuseColor.rgb *= alb;
  float apxRough = mix( mix( 0.78, 1.0, nrA.b ), 0.5, uWet * 0.5 );
`,
    'replace',
  );
  shader.fragmentShader = inject(shader.fragmentShader, 'roughnessmap_fragment', 'roughnessFactor = clamp( apxRough, 0.04, 1.0 );');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'normal_fragment_maps',
    'vec3 mapN = apxNormalRG( nrA );\n  mapN.xy *= normalScale;\n  normal = normalize( tbn * mapN );',
    'replace',
  );
};

/** Terrain: grass / rock / dirt / sand chosen from slope, height and noise in world space. */
const terrainHook = (rockD: THREE.Texture, rockNR: THREE.Texture, sandD: THREE.Texture, dirtD: THREE.Texture): ShaderHook => (shader) => {
  shader.uniforms.uRockD = { value: rockD };
  shader.uniforms.uRockNR = { value: rockNR };
  shader.uniforms.uSandD = { value: sandD };
  shader.uniforms.uDirtD = { value: dirtD };
  shader.vertexShader = inject(shader.vertexShader, 'common', 'varying vec3 vWPos;\nvarying vec3 vWNrm;');
  shader.vertexShader = inject(shader.vertexShader, 'begin_vertex', 'vWPos = ( modelMatrix * vec4( position, 1.0 ) ).xyz;\nvWNrm = normal;');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'common',
    'varying vec3 vWPos;\nvarying vec3 vWNrm;\nuniform sampler2D uRockD;\nuniform sampler2D uRockNR;\nuniform sampler2D uSandD;\nuniform sampler2D uDirtD;',
  );
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'map_fragment',
    /* glsl */ `
  vec2 wuv = vWPos.xz;
  float camDist = length( vWPos - cameraPosition );
  float farMix = smoothstep( 60.0, 420.0, camDist );
  vec4 nz1 = texture2D( uNoise, wuv / 220.0 );
  vec4 nz2 = texture2D( uNoise, wuv / 31.0 );
  vec3 wn = normalize( vWNrm );
  float slope = 1.0 - wn.y;
  // layers (near tiling fades into a larger, calmer tiling in the distance)
  vec3 gN = texture2D( map, wuv / 2.6 ).rgb;
  vec3 gF = texture2D( map, wuv / 23.0 + 0.4 ).rgb;
  vec3 grass = mix( gN * ( 0.55 + 0.9 * gF.g ), gF * 1.05, farMix ) * vec3( 0.6, 0.9, 0.5 ) * ( 0.7 + 0.6 * nz1.g );
  vec3 dirt = mix( texture2D( uDirtD, wuv / 3.1 ).rgb, texture2D( uDirtD, wuv / 27.0 ).rgb, farMix ) * 1.25;
  vec3 rN = texture2D( uRockD, wuv / 6.0 ).rgb;
  vec3 rF = texture2D( uRockD, wuv / 47.0 + 0.3 ).rgb;
  // steep faces: project along the dominant horizontal axis to avoid stretching
  vec3 rS = mix( texture2D( uRockD, vWPos.xy / 7.0 ).rgb, texture2D( uRockD, vWPos.zy / 7.0 ).rgb, step( abs( wn.z ), abs( wn.x ) ) );
  vec3 rock = mix( mix( rN, rS, smoothstep( 0.35, 0.6, slope ) ), rF, farMix * 0.8 ) * vec3( 1.0, 0.94, 0.86 ) * ( 0.8 + 0.45 * nz2.b );
  vec3 sand = texture2D( uSandD, wuv / 4.0 ).rgb * vec3( 1.25, 1.15, 0.98 );
  float wRock = smoothstep( 0.16, 0.34, slope + ( nz2.r - 0.5 ) * 0.14 ) + smoothstep( 120.0, 230.0, vWPos.y + nz1.r * 60.0 ) * 0.7;
  wRock = clamp( wRock, 0.0, 1.0 );
  float wDirt = smoothstep( 0.56, 0.72, nz1.b * 0.65 + nz2.g * 0.35 ) * ( 1.0 - wRock ) * 0.8;
  float wSand = ( 1.0 - smoothstep( 1.2, 3.6, vWPos.y + ( nz2.r - 0.5 ) * 1.6 ) ) * ( 1.0 - wRock * 0.6 );
  vec3 alb = mix( grass, dirt, wDirt );
  alb = mix( alb, rock, wRock );
  alb = mix( alb, sand, wSand );
  alb *= mix( 1.0, 0.72, uWet );
  // darker wet sand band right at the waterline
  alb *= mix( 0.55, 1.0, smoothstep( -0.2, 0.9, vWPos.y ) );
  diffuseColor.rgb *= alb;
  vec4 nrA = mix( texture2D( normalMap, wuv / 2.6 ), texture2D( uRockNR, wuv / 6.0 ), wRock );
  float apxRough = mix( 0.82, 1.0, nrA.b );
`,
    'replace',
  );
  shader.fragmentShader = inject(shader.fragmentShader, 'roughnessmap_fragment', 'roughnessFactor = clamp( apxRough, 0.04, 1.0 );');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'normal_fragment_maps',
    'vec3 mapN = apxNormalRG( nrA );\n  mapN.xy *= normalScale * ( 1.0 - farMix * 0.7 );\n  normal = normalize( tbn * mapN );',
    'replace',
  );
};

const kerbHook: ShaderHook = (shader) => {
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'map_fragment',
    /* glsl */ `
  vec2 sd = vMapUv;
  vec3 kt = texture2D( map, sd / 2.0 ).rgb;
  float stripe = step( 0.5, fract( sd.y / 5.0 ) );
  vec3 kc = mix( vec3( 0.86, 0.86, 0.84 ), vec3( 0.72, 0.045, 0.04 ), stripe );
  kc *= 0.55 + 0.9 * kt.g;
  kc *= mix( 1.0, 0.6, uWet );
  diffuseColor.rgb *= kc;
`,
    'replace',
  );
  shader.fragmentShader = inject(shader.fragmentShader, 'roughnessmap_fragment', 'roughnessFactor = mix( 0.62, 0.2, uWet );');
};

/** Concrete barrier: soft staining from noise plus a painted band along the top edge (uv = distance, height). */
const wallHook: ShaderHook = (shader) => {
  shader.vertexShader = inject(shader.vertexShader, 'common', 'varying vec2 vWallUv;');
  shader.vertexShader = inject(shader.vertexShader, 'begin_vertex', 'vWallUv = uv;');
  shader.fragmentShader = inject(shader.fragmentShader, 'common', 'varying vec2 vWallUv;');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'color_fragment',
    /* glsl */ `
  {
    vec4 wn = texture2D( uNoise, vWallUv * vec2( 0.045, 0.6 ) );
    vec4 wn2 = texture2D( uNoise, vWallUv * vec2( 0.7, 2.3 ) );
    vec3 wc = diffuseColor.rgb * ( 0.78 + 0.34 * wn.r ) * ( 0.92 + 0.16 * wn2.g );
    // streaks running down from the top
    wc *= 1.0 - 0.22 * smoothstep( 0.55, 0.8, wn2.b ) * smoothstep( 1.05, 0.2, vWallUv.y );
    // red / white warning band on the upper part, in blocks of 2 m
    float band = smoothstep( 0.72, 0.76, vWallUv.y ) * ( 1.0 - smoothstep( 1.0, 1.04, vWallUv.y ) );
    vec3 bc = mix( vec3( 0.8, 0.8, 0.78 ), vec3( 0.62, 0.05, 0.045 ), step( 0.5, fract( vWallUv.x / 4.0 ) ) );
    wc = mix( wc, bc * ( 0.8 + 0.3 * wn2.r ), band * 0.92 );
    diffuseColor.rgb = wc * mix( 1.0, 0.72, uWet );
  }`,
  );
};

const waterHook: ShaderHook = (shader) => {
  shader.vertexShader = inject(shader.vertexShader, 'common', 'varying vec3 vWPos;');
  shader.vertexShader = inject(shader.vertexShader, 'begin_vertex', 'vWPos = ( modelMatrix * vec4( position, 1.0 ) ).xyz;');
  shader.fragmentShader = inject(shader.fragmentShader, 'common', 'varying vec3 vWPos;');
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'map_fragment',
    /* glsl */ `
  vec2 wuv = vWPos.xz;
  float wd = length( vWPos - cameraPosition );
  vec2 w1 = texture2D( uNoise, wuv / 38.0 + vec2( uTime * 0.012, uTime * 0.008 ) ).rg - 0.5;
  vec2 w2 = texture2D( uNoise, wuv / 9.0 - vec2( uTime * 0.03, uTime * 0.021 ) ).ba - 0.5;
  vec2 w3 = texture2D( uNoise, wuv / 150.0 + vec2( -uTime * 0.004, uTime * 0.006 ) ).gb - 0.5;
  vec2 wave = w1 * 0.5 + w2 * 0.22 * ( 1.0 - smoothstep( 80.0, 500.0, wd ) ) + w3 * 0.7;
  diffuseColor.rgb *= mix( vec3( 0.012, 0.07, 0.085 ), vec3( 0.02, 0.11, 0.12 ), w3.x + 0.5 );
`,
    'replace',
  );
  shader.fragmentShader = inject(
    shader.fragmentShader,
    'normal_fragment_maps',
    'normal = normalize( normal + ( viewMatrix * vec4( wave.x, 0.0, wave.y, 0.0 ) ).xyz * 0.42 );',
  );
};

// --------------------------------------------------------------------------- geometry helpers

interface StripOpts {
  cols: number;
  /** lateral offset (m, + left) of column c at sample row i */
  offset: (i: number, c: number) => number;
  /** extra height added to the track ground */
  lift?: (i: number, c: number) => number;
  /** per-vertex aInfo attribute */
  info?: (i: number, c: number, d: number) => [number, number];
  /** uv mode: 'sd' = (d, s) metres, 'xz' = world metres, 'sh' = (s, height) */
  uv: 'sd' | 'xz' | 'sh';
  flip?: boolean;
  /** use a fixed up normal blend instead of the physics normal */
  flatNormal?: boolean;
}

function buildStrip(track: Track, i0: number, i1: number, o: StripOpts): THREE.BufferGeometry {
  const rows = i1 - i0 + 1;
  const cols = o.cols;
  const pos = new Float32Array(rows * cols * 3);
  const nrm = new Float32Array(rows * cols * 3);
  const uv = new Float32Array(rows * cols * 2);
  const info = o.info ? new Float32Array(rows * cols * 2) : null;
  const n = track.n;
  for (let r = 0; r < rows; r++) {
    const iRaw = i0 + r;
    const i = mod(iRaw, n);
    for (let c = 0; c < cols; c++) {
      const d = o.offset(i, c);
      track.groundAt(i, 0, d);
      const k = r * cols + c;
      const x = track.cx[i] + track.nx[i] * d;
      const z = track.cz[i] + track.nz[i] * d;
      const y = track.oy + (o.lift ? o.lift(i, c) : 0);
      pos[k * 3] = x;
      pos[k * 3 + 1] = y;
      pos[k * 3 + 2] = z;
      nrm[k * 3] = track.onx;
      nrm[k * 3 + 1] = track.ony;
      nrm[k * 3 + 2] = track.onz;
      if (o.uv === 'sd') {
        uv[k * 2] = d;
        uv[k * 2 + 1] = iRaw * track.ds;
      } else if (o.uv === 'xz') {
        uv[k * 2] = x;
        uv[k * 2 + 1] = z;
      } else {
        uv[k * 2] = iRaw * track.ds;
        uv[k * 2 + 1] = o.lift ? o.lift(i, c) : 0;
      }
      if (info && o.info) {
        const v = o.info(i, c, d);
        info[k * 2] = v[0];
        info[k * 2 + 1] = v[1];
      }
    }
  }
  const idx: number[] = [];
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c;
      const b = a + 1;
      const cc = a + cols;
      const dd = cc + 1;
      // columns run right -> left when offsets increase; winding chosen so faces point up
      if (o.flip) idx.push(a, b, cc, b, dd, cc);
      else idx.push(a, cc, b, b, cc, dd);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  if (info) g.setAttribute('aInfo', new THREE.BufferAttribute(info, 2));
  g.setIndex(idx);
  if (o.flatNormal) g.computeVertexNormals();
  g.computeBoundingSphere();
  g.computeBoundingBox();
  return g;
}

// --------------------------------------------------------------------------- world

export class World implements TerrainField {
  readonly engine: Engine;
  readonly track: Track;
  readonly group = new THREE.Group();
  readonly noise: THREE.DataTexture;
  // terrain field (inner grid)
  private gx0 = 0;
  private gz0 = 0;
  private gcs = 4;
  private gnx = 0;
  private gnz = 0;
  private gh!: Float32Array;
  private gdist!: Float32Array;
  readonly seaLevel = 0;
  /** extra flat ground (m) beyond the wall for paddock / grandstand areas, per track sample */
  readonly padL: Float32Array;
  readonly padR: Float32Array;

  constructor(engine: Engine, track: Track) {
    this.engine = engine;
    this.track = track;
    this.group.name = 'world';
    this.noise = makeNoiseTexture(256, track.def.seed);
    this.padL = new Float32Array(track.n);
    this.padR = new Float32Array(track.n);
    for (const z of track.def.zones) {
      if (z.kind !== 'grandstand' && z.kind !== 'pit') continue;
      const pad = (z.offset ?? 8) + (z.kind === 'grandstand' ? 17 : 16);
      const i0 = Math.floor(z.s0 / track.ds);
      const i1 = Math.ceil(z.s1 / track.ds);
      for (let i = i0 - 6; i <= i1 + 6; i++) {
        const k = mod(i, track.n);
        const f = Math.min(1, Math.min(i - (i0 - 6), i1 + 6 - i) / 6);
        if (z.side !== 'R') this.padL[k] = Math.max(this.padL[k], pad * f);
        if (z.side !== 'L') this.padR[k] = Math.max(this.padR[k], pad * f);
      }
    }
    engine.scene.add(this.group);
  }

  build(): void {
    const tex = (name: string, srgb: boolean) => loadTexture(`${TEX()}${name}.jpg`, { srgb, aniso: this.engine.q.aniso });
    const asphaltD = tex('asphalt_d', true);
    const asphaltNR = tex('asphalt_nr', false);
    const grassD = tex('grass_d', true);
    const grassNR = tex('grass_nr', false);
    const gravelD = tex('gravel_d', true);
    const gravelNR = tex('gravel_nr', false);
    const rockD = tex('rock_d', true);
    const rockNR = tex('rock_nr', false);
    const sandD = tex('sand_d', true);
    const dirtD = tex('dirt_d', true);
    const concD = tex('concrete_d', true);
    const common = groundCommon(this.noise);

    const roadMat = patchMaterial(
      new THREE.MeshStandardMaterial({ map: asphaltD, normalMap: asphaltNR, roughness: 1, metalness: 0, normalScale: new THREE.Vector2(0.9, 0.9) }),
      [common, roadHook],
      'road',
    );
    const vergeMat = patchMaterial(
      new THREE.MeshStandardMaterial({ map: grassD, normalMap: grassNR, roughness: 1, metalness: 0 }),
      [common, vergeHook(gravelD, gravelNR)],
      'verge',
    );
    const kerbMat = patchMaterial(
      new THREE.MeshStandardMaterial({ map: concD, roughness: 0.6, metalness: 0, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }),
      [common, kerbHook],
      'kerb',
    );
    const wallMat = patchMaterial(new THREE.MeshStandardMaterial({ roughness: 0.86, metalness: 0, color: 0xb4b5b3 }), [common, wallHook], 'wall');

    const terrainMat = patchMaterial(
      new THREE.MeshStandardMaterial({ map: grassD, normalMap: grassNR, roughness: 1, metalness: 0 }),
      [common, terrainHook(rockD, rockNR, sandD, dirtD)],
      'terrain',
    );

    this.buildCorridor(roadMat, vergeMat, kerbMat, wallMat);
    this.buildTerrain(terrainMat);
    this.buildSea(common);
  }

  // ---------------------------------------------------------------- corridor
  private buildCorridor(roadMat: THREE.Material, vergeMat: THREE.Material, kerbMat: THREE.Material, wallMat: THREE.Material): void {
    const tr = this.track;
    const n = tr.n;
    const add = (g: THREE.BufferGeometry, m: THREE.Material, cast = false, receive = true, name = '') => {
      const mesh = new THREE.Mesh(g, m);
      mesh.castShadow = cast;
      mesh.receiveShadow = receive;
      mesh.name = name;
      mesh.matrixAutoUpdate = false;
      this.group.add(mesh);
      return mesh;
    };
    for (let i0 = 0; i0 < n; i0 += CHUNK) {
      const i1 = Math.min(i0 + CHUNK, n);
      // road: right edge -> left edge
      const RC = 7;
      add(
        buildStrip(tr, i0, i1, {
          cols: RC,
          offset: (i, c) => {
            const hw = tr.hw[i];
            const f = [-1, -0.94, -0.5, 0, 0.5, 0.94, 1][c];
            return hw * f;
          },
          info: (i) => [tr.line[i], tr.hw[i]],
          uv: 'sd',
        }),
        roadMat,
        false,
        true,
        'road',
      );
      // verges: road edge -> wall -> hidden skirt, both sides
      for (const side of [1, -1]) {
        const VC = 9;
        add(
          buildStrip(tr, i0, i1, {
            cols: VC,
            offset: (i, c) => {
              const hw = tr.hw[i];
              const wall = side > 0 ? tr.wallL[i] : tr.wallR[i];
              const span = wall - hw;
              const pad = side > 0 ? this.padL[i] : this.padR[i];
              const o = [0, 0.3, 1.1, 1.36, Math.min(2.6, span * 0.5), span * 0.72, span, span + 2.8 + pad, span + 7.5 + pad][c];
              return side * (hw + o);
            },
            lift: (_i, c) => (c === 8 ? -4.2 : c === 7 ? -0.05 : 0),
            info: (i, c, d) => {
              const g = side > 0 ? tr.gravelL[i] : tr.gravelR[i];
              const wall = side > 0 ? tr.wallL[i] : tr.wallR[i];
              const pad = side > 0 ? this.padL[i] : this.padR[i];
              const ad = Math.abs(d);
              const w = g > 0 && ad >= g - 0.2 && ad <= wall - 0.3 && c < 7 ? 1 : 0;
              // paved paddock/grandstand apron beyond the wall
              const paved = pad > 1 && c >= 6 ? 1 : 0;
              return [w, paved];
            },
            uv: 'xz',
            flip: side < 0,
          }),
          vergeMat,
          false,
          true,
          'verge',
        );
        // barrier
        const prof: [number, number][] = [
          [0, 0],
          [0.07, 0.32],
          [0.17, 1.05],
          [0.45, 1.05],
          [0.62, 0],
        ];
        const wg = buildStrip(tr, i0, i1, {
          cols: prof.length,
          offset: (i, c) => side * ((side > 0 ? tr.wallL[i] : tr.wallR[i]) + prof[c][0]),
          lift: (_i, c) => prof[c][1],
          uv: 'sh',
          flip: side < 0,
          flatNormal: true,
        });
        add(wg, wallMat, true, true, 'wall');
      }
    }
    // kerbs: contiguous runs where the kerb exists
    for (const side of [1, -1]) {
      const arr = side > 0 ? tr.kerbL : tr.kerbR;
      let start = -1;
      for (let i = 0; i <= n; i++) {
        const on = i < n && arr[i] > 0.5;
        if (on && start < 0) start = i;
        if ((!on || i - start >= CHUNK) && start >= 0) {
          const end = on ? i : i - 1;
          if (end > start) {
            add(
              buildStrip(tr, start, end, {
                cols: 4,
                offset: (ii, c) => side * (tr.hw[ii] + [0, 0.3, 1.1, 1.35][c]),
                uv: 'sd',
                flip: side < 0,
              }),
              kerbMat,
              false,
              true,
              'kerb',
            );
          }
          start = on ? i : -1;
        }
      }
    }
    this.buildMarkings();
  }

  /** Start/finish chequer band and grid slot marks as thin decals over the road. */
  private buildMarkings(): void {
    const tr = this.track;
    const pos: number[] = [];
    const col: number[] = [];
    const idx: number[] = [];
    const o = { x: 0, y: 0, z: 0, psi: 0 };
    const quad = (s0: number, s1: number, d0: number, d1: number, c: number) => {
      const base = pos.length / 3;
      const pts: [number, number][] = [
        [s0, d0],
        [s0, d1],
        [s1, d1],
        [s1, d0],
      ];
      for (const [s, d] of pts) {
        tr.posAt(s, d, o);
        pos.push(o.x, o.y + 0.012, o.z);
        col.push(c, c, c);
      }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    };
    const hw = tr.hw[0];
    const sq = 0.7;
    const colsN = Math.floor((hw * 2) / sq);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < colsN; c++) {
        const d0 = -hw + c * sq;
        quad(-1.05 + r * sq, -1.05 + (r + 1) * sq, d0, d0 + sq, (r + c) % 2 === 0 ? 0.92 : 0.03);
      }
    }
    for (let k = 0; k < 12; k++) {
      const s = tr.length - 9 - k * 8.5 + 2.45;
      const d = (k % 2 === 0 ? 1 : -1) * 3.3;
      quad(s, s + 0.14, d - 1.25, d + 1.25, 0.9);
      quad(s - 1.4, s, d - 1.25, d - 1.11, 0.9);
      quad(s - 1.4, s, d + 1.11, d + 1.25, 0.9);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    const m = patchMaterial(
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 }),
    );
    const mesh = new THREE.Mesh(g, m);
    mesh.receiveShadow = true;
    mesh.name = 'markings';
    this.group.add(mesh);
  }

  // ---------------------------------------------------------------- terrain
  /** Natural landscape before the track is cut into it. */
  private natural(x: number, z: number): number {
    const tr = this.track;
    const seed = tr.def.seed;
    const inland = this.inland(x, z); // metres inland of the waterline
    const rolling = 10 + 9 * fbm2(x / 240, z / 240, 4, seed) + 2.2 * fbm2(x / 37, z / 37, 3, seed + 1);
    const ex = Math.max(0, Math.abs(x) - tr.maxX);
    const ez = Math.max(0, tr.minZ - z);
    const out = Math.hypot(ex, ez);
    const far = smoothstep(40, 1100, out);
    const ridge = ridged2(x / 780 + 3.1, z / 780 + 1.7, 5, seed + 9);
    const mountains = far * (40 + 430 * ridge * (0.35 + 0.65 * far)) + far * 60 * fbm2(x / 150, z / 150, 3, seed + 4);
    let h = rolling + mountains;
    // coast: beach rising out of the sea
    const c = smoothstep(-6, 120, inland);
    const beach = inland > 0 ? Math.min(inland * 0.05, 3.2) : Math.max(inland * 0.09, -30);
    h = lerp(beach, h, c * c);
    return h;
  }

  /** Distance inland of the waterline (negative = out at sea). The coast straight runs right along it. */
  inland(x: number, z: number): number {
    const tr = this.track;
    const seed = tr.def.seed;
    const shore = tr.maxZ + 36 + 15 * fbm2(x / 380, 7.3, 3, seed + 3) + 7 * fbm2(x / 90, 2.1, 2, seed + 5);
    return shore - z;
  }

  private buildTerrain(mat: THREE.Material): void {
    const tr = this.track;
    const cs = this.gcs;
    const margin = 300;
    this.gx0 = Math.floor((tr.minX - margin) / cs) * cs;
    this.gz0 = Math.floor((tr.minZ - margin) / cs) * cs;
    const gnx = Math.ceil((tr.maxX + margin - this.gx0) / cs) + 1;
    const gnz = Math.ceil((tr.maxZ + margin - this.gz0) / cs) + 1;
    this.gnx = gnx;
    this.gnz = gnz;
    const N = gnx * gnz;
    const dist = new Float32Array(N).fill(1e9);
    const hc = new Float32Array(N); // corridor height carried from the nearest seed
    const sx = new Float32Array(N); // nearest seed position (for exact distances)
    const sz = new Float32Array(N);
    const padF = new Float32Array(N); // flat apron width carried from the nearest seed

    // seeds: rasterise the corridor (road + verges out to the wall + 2.5 m)
    const n = tr.n;
    for (let i = 0; i < n; i++) {
      const dl = tr.wallL[i] + 2.6;
      const dr = tr.wallR[i] + 2.6;
      for (let d = -dr; d <= dl + 0.01; d += 1.9) {
        const dd = Math.min(d, dl);
        tr.groundAt(i, 0, dd);
        const x = tr.cx[i] + tr.nx[i] * dd;
        const z = tr.cz[i] + tr.nz[i] * dd;
        const gx = Math.round((x - this.gx0) / cs);
        const gz = Math.round((z - this.gz0) / cs);
        for (let oz = -1; oz <= 1; oz++) {
          for (let ox = -1; ox <= 1; ox++) {
            const ix = gx + ox;
            const iz = gz + oz;
            if (ix < 0 || iz < 0 || ix >= gnx || iz >= gnz) continue;
            const k = iz * gnx + ix;
            const px = this.gx0 + ix * cs;
            const pz = this.gz0 + iz * cs;
            const q = Math.hypot(px - x, pz - z);
            const inside = Math.abs(dd) < (dd > 0 ? dl : dr) - 1.5 ? 0 : q;
            const cur = dist[k];
            if (inside < cur || (inside === cur && track_lower(tr.oy, hc[k]))) {
              dist[k] = inside;
              hc[k] = tr.oy;
              sx[k] = x;
              sz[k] = z;
              padF[k] = dd > 0 ? this.padL[i] : this.padR[i];
            }
          }
        }
      }
    }
    function track_lower(a: number, b: number): boolean {
      return a < b;
    }
    // propagate nearest-seed (two chamfer sweeps, twice)
    const relax = (k: number, kn: number, px: number, pz: number) => {
      if (dist[kn] >= 1e8) return;
      const d = dist[kn] === 0 ? Math.hypot(px - sx[kn], pz - sz[kn]) : Math.hypot(px - sx[kn], pz - sz[kn]);
      if (d < dist[k]) {
        dist[k] = d;
        hc[k] = hc[kn];
        sx[k] = sx[kn];
        sz[k] = sz[kn];
        padF[k] = padF[kn];
      }
    };
    for (let pass = 0; pass < 2; pass++) {
      for (let iz = 0; iz < gnz; iz++) {
        for (let ix = 0; ix < gnx; ix++) {
          const k = iz * gnx + ix;
          if (dist[k] === 0) continue;
          const px = this.gx0 + ix * cs;
          const pz = this.gz0 + iz * cs;
          if (ix > 0) relax(k, k - 1, px, pz);
          if (iz > 0) {
            relax(k, k - gnx, px, pz);
            if (ix > 0) relax(k, k - gnx - 1, px, pz);
            if (ix < gnx - 1) relax(k, k - gnx + 1, px, pz);
          }
        }
      }
      for (let iz = gnz - 1; iz >= 0; iz--) {
        for (let ix = gnx - 1; ix >= 0; ix--) {
          const k = iz * gnx + ix;
          if (dist[k] === 0) continue;
          const px = this.gx0 + ix * cs;
          const pz = this.gz0 + iz * cs;
          if (ix < gnx - 1) relax(k, k + 1, px, pz);
          if (iz < gnz - 1) {
            relax(k, k + gnx, px, pz);
            if (ix < gnx - 1) relax(k, k + gnx + 1, px, pz);
            if (ix > 0) relax(k, k + gnx - 1, px, pz);
          }
        }
      }
    }
    // final heights
    const h = new Float32Array(N);
    for (let iz = 0; iz < gnz; iz++) {
      for (let ix = 0; ix < gnx; ix++) {
        const k = iz * gnx + ix;
        const px = this.gx0 + ix * cs;
        const pz = this.gz0 + iz * cs;
        const nat = this.natural(px, pz);
        const d = Math.max(0, dist[k] - padF[k]);
        if (d <= 0) h[k] = hc[k] - 1.6 - Math.min(dist[k], 20) * 0.025;
        else {
          // follow the corridor closely near the track, release into the landscape with distance
          const w = smoothstep(0, 70, d) * (0.3 + 0.7 * smoothstep(40, 260, d));
          const edge = hc[k] - 0.45 - 1.2 * (1 - smoothstep(0, 5, d));
          h[k] = lerp(edge, nat, w);
        }
      }
    }
    this.gh = h;
    this.gdist = dist;

    // chunked mesh
    const TC = 48;
    for (let cz = 0; cz < gnz - 1; cz += TC) {
      for (let cx = 0; cx < gnx - 1; cx += TC) {
        const w = Math.min(TC, gnx - 1 - cx) + 1;
        const d = Math.min(TC, gnz - 1 - cz) + 1;
        const pos = new Float32Array(w * d * 3);
        const nrm = new Float32Array(w * d * 3);
        const uv = new Float32Array(w * d * 2);
        for (let j = 0; j < d; j++) {
          for (let i = 0; i < w; i++) {
            const ix = cx + i;
            const iz = cz + j;
            const k = j * w + i;
            const px = this.gx0 + ix * cs;
            const pz = this.gz0 + iz * cs;
            pos[k * 3] = px;
            pos[k * 3 + 1] = h[iz * gnx + ix];
            pos[k * 3 + 2] = pz;
            const hl = h[iz * gnx + Math.max(ix - 1, 0)];
            const hr = h[iz * gnx + Math.min(ix + 1, gnx - 1)];
            const hd = h[Math.max(iz - 1, 0) * gnx + ix];
            const hu = h[Math.min(iz + 1, gnz - 1) * gnx + ix];
            let nx = hl - hr;
            let nz = hd - hu;
            const il = 1 / Math.hypot(nx, 2 * cs, nz);
            nrm[k * 3] = nx * il;
            nrm[k * 3 + 1] = 2 * cs * il;
            nrm[k * 3 + 2] = nz * il;
            uv[k * 2] = px;
            uv[k * 2 + 1] = pz;
          }
        }
        const idx: number[] = [];
        for (let j = 0; j < d - 1; j++) {
          for (let i = 0; i < w - 1; i++) {
            const a = j * w + i;
            idx.push(a, a + w, a + 1, a + 1, a + w, a + w + 1);
          }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
        g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
        g.setIndex(idx);
        g.computeBoundingSphere();
        const mesh = new THREE.Mesh(g, mat);
        mesh.receiveShadow = true;
        mesh.castShadow = false;
        mesh.name = 'terrain';
        mesh.matrixAutoUpdate = false;
        this.group.add(mesh);
      }
    }

    // far terrain ring (low resolution), meeting the inner grid at its border
    const fcs = 40;
    const R = 4200;
    const fn = Math.floor((2 * R) / fcs) + 1;
    const fpos = new Float32Array(fn * fn * 3);
    const fuv = new Float32Array(fn * fn * 2);
    const x1 = this.gx0 + (gnx - 1) * cs;
    const z1 = this.gz0 + (gnz - 1) * cs;
    for (let j = 0; j < fn; j++) {
      for (let i = 0; i < fn; i++) {
        const px = -R + i * fcs;
        const pz = -R + j * fcs;
        const k = j * fn + i;
        let y = this.natural(px, pz);
        // sink the part hidden under the inner grid
        const insideX = Math.min(px - this.gx0, x1 - px);
        const insideZ = Math.min(pz - this.gz0, z1 - pz);
        const ins = Math.min(insideX, insideZ);
        if (ins > 0) y -= Math.min(ins * 0.25, 30);
        fpos[k * 3] = px;
        fpos[k * 3 + 1] = y;
        fpos[k * 3 + 2] = pz;
        fuv[k * 2] = px;
        fuv[k * 2 + 1] = pz;
      }
    }
    const fidx: number[] = [];
    for (let j = 0; j < fn - 1; j++) {
      for (let i = 0; i < fn - 1; i++) {
        const a = j * fn + i;
        fidx.push(a, a + fn, a + 1, a + 1, a + fn, a + fn + 1);
      }
    }
    const fg = new THREE.BufferGeometry();
    fg.setAttribute('position', new THREE.BufferAttribute(fpos, 3));
    fg.setAttribute('uv', new THREE.BufferAttribute(fuv, 2));
    fg.setIndex(fidx);
    fg.computeVertexNormals();
    const far = new THREE.Mesh(fg, mat);
    far.name = 'terrain_far';
    far.receiveShadow = false;
    far.matrixAutoUpdate = false;
    this.group.add(far);
  }

  private buildSea(common: ShaderHook): void {
    const g = new THREE.PlaneGeometry(16000, 16000, 1, 1);
    g.rotateX(-Math.PI / 2);
    const m = patchMaterial(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.09, metalness: 0.0, envMapIntensity: 1.25 }), [common, waterHook], 'water');
    const sea = new THREE.Mesh(g, m);
    sea.position.y = this.seaLevel;
    sea.name = 'sea';
    sea.receiveShadow = false;
    this.group.add(sea);
  }

  // ---------------------------------------------------------------- field queries
  private sampleGrid(arr: Float32Array, x: number, z: number): number {
    const fx = clamp((x - this.gx0) / this.gcs, 0, this.gnx - 1.001);
    const fz = clamp((z - this.gz0) / this.gcs, 0, this.gnz - 1.001);
    const ix = Math.floor(fx);
    const iz = Math.floor(fz);
    const tx = fx - ix;
    const tz = fz - iz;
    const k = iz * this.gnx + ix;
    const a = arr[k];
    const b = arr[k + 1];
    const c = arr[k + this.gnx];
    const d = arr[k + this.gnx + 1];
    return a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz;
  }
  private inGrid(x: number, z: number): boolean {
    return x >= this.gx0 && z >= this.gz0 && x <= this.gx0 + (this.gnx - 1) * this.gcs && z <= this.gz0 + (this.gnz - 1) * this.gcs;
  }
  height(x: number, z: number): number {
    return this.inGrid(x, z) ? this.sampleGrid(this.gh, x, z) : this.natural(x, z);
  }
  corridorDist(x: number, z: number): number {
    return this.inGrid(x, z) ? this.sampleGrid(this.gdist, x, z) : 400;
  }
  slope(x: number, z: number): number {
    const e = 3;
    const dx = this.height(x + e, z) - this.height(x - e, z);
    const dz = this.height(x, z + e) - this.height(x, z - e);
    return Math.hypot(dx, dz) / (2 * e);
  }

  dispose(): void {
    this.engine.scene.remove(this.group);
    this.group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
    });
  }
}
