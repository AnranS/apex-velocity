// Rendering engine: WebGL2 renderer, cascaded sun shadows, HDR post-processing chain
// (AO -> per-pixel motion blur -> bloom -> tone mapping -> grade) and quality tiers.
import * as THREE from 'three';
import { CSM } from 'three/examples/jsm/csm/CSM.js';
import {
  BlendFunction,
  BloomEffect,
  ChromaticAberrationEffect,
  DepthOfFieldEffect,
  Effect,
  EffectAttribute,
  EffectComposer,
  EffectPass,
  RenderPass,
  SMAAEffect,
  SMAAPreset,
  ToneMappingEffect,
  ToneMappingMode,
} from 'postprocessing';
import { N8AOPostPass } from 'n8ao';
import { U, installShaderChunks, loadedTextures, setCSM } from './shared';

export type QualityName = 'low' | 'medium' | 'high' | 'ultra';

export interface Quality {
  name: QualityName;
  label: string;
  dprCap: number;
  maxPixels: number; // cap on rendered pixels (dynamic resolution target)
  msaa: number;
  smaa: boolean;
  shadows: boolean;
  cascades: number;
  shadowSize: number;
  shadowFar: number;
  ao: boolean;
  aoHalf: boolean;
  motionBlur: boolean;
  bloom: boolean;
  chroma: boolean;
  dynamicEnv: boolean;
  envSize: number;
  treeDensity: number;
  grass: boolean;
  farTrees: number; // metres of full-detail trees
  aniso: number;
  particles: number; // particle budget scale
}

export const QUALITY: Record<QualityName, Quality> = {
  low: {
    name: 'low', label: '流畅', dprCap: 1, maxPixels: 1280 * 720, msaa: 0, smaa: false, shadows: true, cascades: 1, shadowSize: 1024, shadowFar: 90,
    ao: false, aoHalf: true, motionBlur: false, bloom: true, chroma: false, dynamicEnv: false, envSize: 128, treeDensity: 0.4, grass: false, farTrees: 60, aniso: 2, particles: 0.5,
  },
  medium: {
    name: 'medium', label: '均衡', dprCap: 1.25, maxPixels: 1920 * 1080, msaa: 0, smaa: true, shadows: true, cascades: 2, shadowSize: 2048, shadowFar: 160,
    ao: false, aoHalf: true, motionBlur: true, bloom: true, chroma: true, dynamicEnv: false, envSize: 256, treeDensity: 0.7, grass: true, farTrees: 110, aniso: 4, particles: 0.8,
  },
  high: {
    name: 'high', label: '高画质', dprCap: 1.5, maxPixels: 2560 * 1440, msaa: 4, smaa: false, shadows: true, cascades: 3, shadowSize: 2048, shadowFar: 260,
    ao: true, aoHalf: true, motionBlur: true, bloom: true, chroma: true, dynamicEnv: true, envSize: 256, treeDensity: 1.0, grass: true, farTrees: 160, aniso: 8, particles: 1.0,
  },
  ultra: {
    name: 'ultra', label: '极致', dprCap: 2, maxPixels: 3840 * 2160, msaa: 4, smaa: false, shadows: true, cascades: 4, shadowSize: 4096, shadowFar: 340,
    ao: true, aoHalf: false, motionBlur: true, bloom: true, chroma: true, dynamicEnv: true, envSize: 512, treeDensity: 1.25, grass: true, farTrees: 220, aniso: 16, particles: 1.3,
  },
};

// ------------------------------------------------------------------------------------ effects

const MOTION_FRAG = /* glsl */ `
uniform mat4 uProjInv;
uniform mat4 uCamWorld;
uniform mat4 uPrevViewProj;
uniform mat4 uProtectInv;
uniform vec3 uProtectHalf;
uniform vec4 uSpheres[8];
uniform int uSphereCount;
uniform float uStrength;

vec3 mbWorld( vec2 uv, float d ) {
  vec4 vp = uProjInv * vec4( uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
  return ( uCamWorld * vec4( vp.xyz / vp.w, 1.0 ) ).xyz;
}
float mbProtected( vec3 wp ) {
  vec3 q = abs( ( uProtectInv * vec4( wp, 1.0 ) ).xyz ) - uProtectHalf;
  float inside = step( max( q.x, max( q.y, q.z ) ), 0.0 );
  for ( int i = 0; i < 8; i ++ ) {
    if ( i >= uSphereCount ) break;
    vec3 dv = wp - uSpheres[ i ].xyz;
    // spheres sit 0.6 m above the road: leave the tarmac under the car unprotected
    inside = max( inside, step( dot( dv, dv ), uSpheres[ i ].w * uSpheres[ i ].w ) * step( -0.47, dv.y ) );
  }
  return inside;
}
void mainImage( const in vec4 inputColor, const in vec2 uv, const in float depth, out vec4 outputColor ) {
  vec3 wp = mbWorld( uv, depth );
  vec4 prev = uPrevViewProj * vec4( wp, 1.0 );
  vec2 vel = prev.w > 0.0 ? ( uv - ( prev.xy / prev.w * 0.5 + 0.5 ) ) * uStrength : vec2( 0.0 );
  float len = length( vel );
  if ( len < 0.0006 ) { outputColor = inputColor; return; }
  // cars travel with the camera: pixels on them are left sharp (tested once per pixel)
  if ( mbProtected( wp ) > 0.5 ) { outputColor = inputColor; return; }
  if ( len > 0.045 ) vel *= 0.045 / len;
  // view distance of this pixel; taps that are clearly nearer (a car in front of the road) are rejected
  float zc = cameraNear / max( 1.0 - depth, 1e-6 );
  vec3 acc = inputColor.rgb;
  float wsum = 1.0;
  for ( int i = 0; i < MB_SAMPLES; i ++ ) {
    float t = ( float( i ) + 0.5 ) / float( MB_SAMPLES ) - 0.5;
    vec2 suv = clamp( uv + vel * t, vec2( 0.001 ), vec2( 0.999 ) );
    float zs = cameraNear / max( 1.0 - texture2D( depthBuffer, suv ).r, 1e-6 );
    float w = step( zc * 0.9 - 0.4, zs );
    acc += texture2D( inputBuffer, suv ).rgb * w;
    wsum += w;
  }
  outputColor = vec4( acc / wsum, inputColor.a );
}`;

export class MotionBlurEffect extends Effect {
  private prevViewProj = new THREE.Matrix4();
  private tmp = new THREE.Matrix4();
  private first = true;
  /** blur amount: fraction of the per-frame screen motion at 60 fps */
  shutter = 0.5;
  /** 0..1 fade so slow driving stays crisp; set from the player's speed */
  speedScale = 1;
  constructor(samples = 8) {
    super('MotionBlurEffect', MOTION_FRAG, {
      attributes: EffectAttribute.CONVOLUTION | EffectAttribute.DEPTH,
      defines: new Map([['MB_SAMPLES', String(samples)]]),
      uniforms: new Map<string, THREE.Uniform>([
        ['uProjInv', new THREE.Uniform(new THREE.Matrix4())],
        ['uCamWorld', new THREE.Uniform(new THREE.Matrix4())],
        ['uPrevViewProj', new THREE.Uniform(new THREE.Matrix4())],
        ['uProtectInv', new THREE.Uniform(new THREE.Matrix4().makeScale(1e-6, 1e-6, 1e-6))],
        ['uProtectHalf', new THREE.Uniform(new THREE.Vector3(1, 0.7, 2.4))],
        ['uSpheres', new THREE.Uniform(Array.from({ length: 8 }, () => new THREE.Vector4(0, -1e6, 0, 0)))],
        ['uSphereCount', new THREE.Uniform(0)],
        ['uStrength', new THREE.Uniform(0)],
      ]),
    });
  }
  /** Call once per frame after the camera has been updated and before rendering. */
  updateCamera(camera: THREE.PerspectiveCamera, dt: number): void {
    const u = this.uniforms;
    (u.get('uProjInv')!.value as THREE.Matrix4).copy(camera.projectionMatrixInverse);
    (u.get('uCamWorld')!.value as THREE.Matrix4).copy(camera.matrixWorld);
    this.tmp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    if (this.first) {
      this.prevViewProj.copy(this.tmp);
      this.first = false;
    }
    (u.get('uPrevViewProj')!.value as THREE.Matrix4).copy(this.prevViewProj);
    this.prevViewProj.copy(this.tmp);
    u.get('uStrength')!.value = this.shutter * this.speedScale * Math.min(2, 1 / 60 / Math.max(dt, 1 / 240));
  }
  /** After a camera cut, skip one frame of blur. */
  reset(): void {
    this.first = true;
  }
  setProtectBox(worldToLocal: THREE.Matrix4, half: THREE.Vector3): void {
    (this.uniforms.get('uProtectInv')!.value as THREE.Matrix4).copy(worldToLocal);
    (this.uniforms.get('uProtectHalf')!.value as THREE.Vector3).copy(half);
  }
  setSpheres(list: { x: number; y: number; z: number; r: number }[]): void {
    const arr = this.uniforms.get('uSpheres')!.value as THREE.Vector4[];
    const n = Math.min(list.length, 8);
    for (let i = 0; i < n; i++) arr[i].set(list[i].x, list[i].y, list[i].z, list[i].r);
    this.uniforms.get('uSphereCount')!.value = n;
  }
}

const GRADE_FRAG = /* glsl */ `
uniform vec3 uShadowTint;
uniform vec3 uHighTint;
uniform float uContrast;
uniform float uSaturation;
uniform float uVignette;
uniform float uGrain;
uniform float uFade;
uniform float uSpeedFx;
float gHash( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.x + p3.y ) * p3.z );
}
void mainImage( const in vec4 inputColor, const in vec2 uv, out vec4 outputColor ) {
  vec3 c = inputColor.rgb;
  float l = dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
  // split toning
  c *= mix( uShadowTint, uHighTint, smoothstep( 0.0, 0.7, l ) );
  // saturation + filmic contrast around mid grey
  c = mix( vec3( l ), c, uSaturation );
  c = ( c - 0.18 ) * uContrast + 0.18;
  c = max( c, 0.0 );
  // vignette (stronger at speed)
  vec2 q = uv - 0.5;
  float v = dot( q, q );
  c *= 1.0 - smoothstep( 0.12, 0.62, v ) * ( uVignette + uSpeedFx * 0.22 );
  // grain
  float n = gHash( uv * resolution + fract( time ) * 311.7 ) - 0.5;
  c += n * uGrain * ( 1.0 - l * 0.6 );
  c *= 1.0 - uFade;
  outputColor = vec4( c, inputColor.a );
}`;

export class GradeEffect extends Effect {
  constructor() {
    super('GradeEffect', GRADE_FRAG, {
      blendFunction: BlendFunction.NORMAL,
      uniforms: new Map<string, THREE.Uniform>([
        ['uShadowTint', new THREE.Uniform(new THREE.Vector3(1, 1, 1))],
        ['uHighTint', new THREE.Uniform(new THREE.Vector3(1, 1, 1))],
        ['uContrast', new THREE.Uniform(1.06)],
        ['uSaturation', new THREE.Uniform(1.08)],
        ['uVignette', new THREE.Uniform(0.32)],
        ['uGrain', new THREE.Uniform(0.011)],
        ['uFade', new THREE.Uniform(0)],
        ['uSpeedFx', new THREE.Uniform(0)],
      ]),
    });
  }
  set(name: 'uContrast' | 'uSaturation' | 'uVignette' | 'uGrain' | 'uFade' | 'uSpeedFx', v: number): void {
    this.uniforms.get(name)!.value = v;
  }
  setTints(shadow: THREE.Vector3Like, high: THREE.Vector3Like): void {
    (this.uniforms.get('uShadowTint')!.value as THREE.Vector3).copy(shadow as THREE.Vector3);
    (this.uniforms.get('uHighTint')!.value as THREE.Vector3).copy(high as THREE.Vector3);
  }
}

// ------------------------------------------------------------------------------------ engine

export class Engine {
  readonly canvas: HTMLCanvasElement;
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly composer: EffectComposer;
  readonly q: Quality;
  csm: CSM | null = null;
  sun: THREE.DirectionalLight | null = null;
  /** player headlights: always in the scene (intensity 0 by day) so toggling never recompiles shaders */
  readonly headlight: THREE.SpotLight;
  /** showroom key light (menu only) */
  readonly keyLight: THREE.SpotLight;
  readonly sunDir = new THREE.Vector3(0.4, 0.5, 0.3).normalize();

  readonly renderPass: RenderPass;
  ao: N8AOPostPass | null = null;
  motion: MotionBlurEffect | null = null;
  bloom: BloomEffect | null = null;
  chroma: ChromaticAberrationEffect | null = null;
  tone: ToneMappingEffect;
  grade: GradeEffect;
  dof: DepthOfFieldEffect | null = null;
  private dofPass: EffectPass | null = null;

  width = 1;
  height = 1;
  dpr = 1;
  renderScale = 1;
  /** GPU string reported by the driver (for diagnostics) */
  gpu = '';

  constructor(canvas: HTMLCanvasElement, quality: QualityName) {
    this.canvas = canvas;
    this.q = QUALITY[quality];
    installShaderChunks();

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', stencil: false, depth: true, alpha: false });
    this.renderer = renderer;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.shadowMap.enabled = this.q.shadows;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.info.autoReset = false;
    renderer.setClearColor(0x05070b, 1);
    try {
      const gl = renderer.getContext();
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      this.gpu = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : 'unknown';
    } catch {
      this.gpu = 'unknown';
    }

    this.camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.25, 7000);
    this.camera.position.set(0, 3, -8);
    this.scene.add(this.camera);
    // FogExp2 only switches USE_FOG on; the actual fog model lives in shared.ts
    this.scene.fog = new THREE.FogExp2(0x8899aa, 0.0001);

    this.setupSun();
    this.headlight = new THREE.SpotLight(0xdfeaff, 0, 120, 0.52, 0.7, 1.7);
    this.headlight.castShadow = false;
    this.headlight.visible = false;
    this.scene.add(this.headlight, this.headlight.target);
    this.keyLight = new THREE.SpotLight(0xfff1e2, 0, 40, 0.62, 0.9, 2);
    this.keyLight.castShadow = false;
    this.keyLight.visible = false;
    this.scene.add(this.keyLight, this.keyLight.target);

    const composer = new EffectComposer(renderer, { frameBufferType: THREE.HalfFloatType, multisampling: this.q.msaa, depthBuffer: true, stencilBuffer: false });
    this.composer = composer;
    this.renderPass = new RenderPass(this.scene, this.camera);
    composer.addPass(this.renderPass);

    if (this.q.ao) {
      const ao = new N8AOPostPass(this.scene, this.camera, 1, 1);
      ao.configuration.aoRadius = 1.6;
      ao.configuration.distanceFalloff = 1.2;
      ao.configuration.intensity = 2.4;
      ao.configuration.aoSamples = 12;
      ao.configuration.denoiseSamples = 6;
      ao.configuration.denoiseRadius = 10;
      ao.configuration.halfRes = this.q.aoHalf;
      ao.configuration.depthAwareUpsampling = true;
      ao.configuration.gammaCorrection = false;
      ao.configuration.color = new THREE.Color(0.02, 0.025, 0.04);
      this.ao = ao;
      composer.addPass(ao);
    }
    if (this.q.motionBlur) {
      this.motion = new MotionBlurEffect(this.q.name === 'medium' ? 5 : this.q.name === 'ultra' ? 9 : 7);
      composer.addPass(new EffectPass(this.camera, this.motion));
    }

    const effects: Effect[] = [];
    if (this.q.bloom) {
      this.bloom = new BloomEffect({
        blendFunction: BlendFunction.ADD,
        mipmapBlur: true,
        luminanceThreshold: 1.0,
        luminanceSmoothing: 0.35,
        intensity: 0.55,
        radius: 0.72,
        levels: this.q.name === 'low' ? 5 : 7,
      });
      effects.push(this.bloom);
    }
    if (this.q.chroma) {
      this.chroma = new ChromaticAberrationEffect({ offset: new THREE.Vector2(0.0006, 0.0006), radialModulation: true, modulationOffset: 0.35 });
    }
    this.tone = new ToneMappingEffect({ mode: ToneMappingMode.AGX });
    this.grade = new GradeEffect();
    effects.push(this.tone, this.grade);
    if (this.q.smaa) effects.push(new SMAAEffect({ preset: SMAAPreset.MEDIUM }));
    composer.addPass(new EffectPass(this.camera, ...effects));
    if (this.chroma) composer.addPass(new EffectPass(this.camera, this.chroma));

    this.resize();
  }

  private setupSun(): void {
    const q = this.q;
    if (q.shadows && q.cascades >= 2) {
      const splits = q.cascades === 2 ? [0.16, 1] : q.cascades === 3 ? [0.065, 0.27, 1] : [0.045, 0.15, 0.42, 1];
      const csm = new CSM({
        camera: this.camera,
        parent: this.scene,
        cascades: q.cascades,
        maxFar: q.shadowFar,
        mode: 'custom',
        customSplitsCallback: (_n: number, _near: number, _far: number, target: number[]) => {
          for (const s of splits) target.push(s);
        },
        shadowMapSize: q.shadowSize,
        lightDirection: this.sunDir.clone().negate(),
        lightIntensity: 3,
        lightNear: 1,
        lightFar: 1200,
        lightMargin: 320,
        shadowBias: -0.00012,
      });
      csm.fade = true;
      for (let i = 0; i < csm.lights.length; i++) {
        const sh = csm.lights[i].shadow;
        sh.normalBias = 0.02 + 0.07 * i * i;
        sh.bias = -0.00008 - 0.00012 * i;
        sh.radius = 1.6;
      }
      this.csm = csm;
      setCSM(csm);
    } else {
      const sun = new THREE.DirectionalLight(0xffffff, 3);
      sun.castShadow = q.shadows;
      if (q.shadows) {
        sun.shadow.mapSize.set(q.shadowSize, q.shadowSize);
        const r = q.shadowFar * 0.5;
        sun.shadow.camera.left = -r;
        sun.shadow.camera.right = r;
        sun.shadow.camera.top = r;
        sun.shadow.camera.bottom = -r;
        sun.shadow.camera.near = 1;
        sun.shadow.camera.far = 900;
        sun.shadow.bias = -0.0003;
        sun.shadow.normalBias = 0.06;
        sun.shadow.radius = 1.5;
      }
      this.sun = sun;
      this.scene.add(sun, sun.target);
      setCSM(null);
    }
  }

  /** Direction TO the sun (world space), colour and intensity of the key light. */
  setSun(dirToSun: THREE.Vector3, color: THREE.Color, intensity: number): void {
    this.sunDir.copy(dirToSun).normalize();
    U.uSunDir.value.copy(this.sunDir);
    U.uSunColor.value.copy(color).multiplyScalar(intensity);
    if (this.csm) {
      this.csm.lightDirection.copy(this.sunDir).negate();
      for (const l of this.csm.lights) {
        l.color.copy(color);
        l.intensity = intensity;
      }
    } else if (this.sun) {
      this.sun.color.copy(color);
      this.sun.intensity = intensity;
    }
  }

  setExposure(e: number): void {
    this.renderer.toneMappingExposure = e;
  }

  /** Depth of field for showroom / photo shots. Pass null to disable. */
  setDof(focusDistance: number | null, focalRange = 3, bokeh = 3): void {
    if (focusDistance === null) {
      if (this.dofPass) this.dofPass.enabled = false;
      return;
    }
    if (!this.dof) {
      this.dof = new DepthOfFieldEffect(this.camera, { worldFocusDistance: focusDistance, worldFocusRange: focalRange, bokehScale: bokeh, resolutionScale: 0.5 });
      this.dofPass = new EffectPass(this.camera, this.dof);
      // insert before the final (tone mapping) pass
      const idx = this.composer.passes.length - (this.chroma ? 2 : 1);
      this.composer.addPass(this.dofPass, idx);
    }
    this.dofPass!.enabled = true;
    this.dof.cocMaterial.worldFocusDistance = focusDistance;
    this.dof.cocMaterial.worldFocusRange = focalRange;
    this.dof.bokehScale = bokeh;
  }

  resize(): void {
    const w = Math.max(1, this.canvas.clientWidth || window.innerWidth);
    const h = Math.max(1, this.canvas.clientHeight || window.innerHeight);
    let dpr = Math.min(window.devicePixelRatio || 1, this.q.dprCap);
    const px = w * h * dpr * dpr;
    if (px > this.q.maxPixels) dpr *= Math.sqrt(this.q.maxPixels / px);
    dpr *= this.renderScale;
    this.width = w;
    this.height = h;
    this.dpr = dpr;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.csm) this.csm.updateFrustums();
  }

  /**
   * Compile every material program and upload every texture now, so nothing hitches the first time
   * an object comes into view mid-race.
   */
  async prewarm(): Promise<void> {
    const r = this.renderer;
    for (const t of loadedTextures) {
      const img = t.image as { width?: number } | undefined;
      if (img && img.width) r.initTexture(t);
    }
    this.scene.traverse((o) => {
      const m = (o as THREE.Mesh).material;
      const each = (x: THREE.Material) => {
        const sm = x as THREE.MeshStandardMaterial;
        for (const t of [sm.map, sm.normalMap, sm.emissiveMap, sm.roughnessMap, sm.alphaMap]) {
          const img = t?.image as { width?: number } | undefined;
          if (t && img && img.width) r.initTexture(t);
        }
      };
      if (Array.isArray(m)) m.forEach(each);
      else if (m) each(m);
    });
    // compile with everything visible so LOD levels that start hidden are included
    const hidden: THREE.Object3D[] = [];
    this.scene.traverse((o) => {
      if (!o.visible && !(o as THREE.Light).isLight) {
        hidden.push(o);
        o.visible = true;
      }
    });
    try {
      await r.compileAsync(this.scene, this.camera);
    } catch (e) {
      console.warn('[engine] prewarm failed', e);
    }
    for (const o of hidden) o.visible = false;
  }

  /** Call when the camera FOV/near/far changed. */
  cameraChanged(): void {
    this.camera.updateProjectionMatrix();
    if (this.csm) this.csm.updateFrustums();
  }

  render(dt: number): void {
    U.uTime.value += dt;
    const cam = this.camera;
    cam.updateMatrixWorld();
    if (this.csm) this.csm.update();
    else if (this.sun) {
      // keep the single shadow frustum centred ahead of the camera, snapped to texels
      const fwd = new THREE.Vector3();
      cam.getWorldDirection(fwd);
      const c = cam.position.clone().addScaledVector(fwd, this.q.shadowFar * 0.35);
      this.sun.target.position.copy(c);
      this.sun.position.copy(c).addScaledVector(this.sunDir, 400);
    }
    if (this.motion) this.motion.updateCamera(cam, dt);
    this.renderer.info.reset();
    this.composer.render(dt);
  }

  dispose(): void {
    this.composer.dispose();
    this.renderer.dispose();
  }
}
