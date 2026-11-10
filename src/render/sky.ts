// Sky dome (HDRI encoded as gamma/range JPG), image-based lighting, sun/fog setup per weather preset.
import * as THREE from 'three';
import type { Engine } from './engine';
import { U, loadTextureAsync } from './shared';

export type WeatherId = 'sunset' | 'day' | 'storm' | 'night';

export interface Weather {
  id: WeatherId;
  name: string;
  nameCn: string;
  desc: string;
  /** world azimuth of the key light, degrees: direction to sun = (sin az, ., cos az) */
  sunAzimuth: number;
  sunElevation: number;
  sunColor: [number, number, number];
  sunIntensity: number;
  skyIntensity: number;
  envIntensity: number;
  exposure: number;
  disc: number;
  fogDensity: number;
  fogFalloff: number;
  fogColA: [number, number, number];
  fogColB: [number, number, number];
  wet: number;
  rain: number;
  night: number;
  shadowTint: [number, number, number];
  highTint: [number, number, number];
  contrast: number;
  saturation: number;
  bloom: number;
  grip: number; // physics wetness 0..1
}

export const WEATHER: Record<WeatherId, Weather> = {
  sunset: {
    id: 'sunset', name: 'GOLDEN HOUR', nameCn: '黄昏', desc: '落日余晖 · 干地',
    sunAzimuth: -66, sunElevation: 13, sunColor: [1.0, 0.56, 0.27], sunIntensity: 6.6,
    skyIntensity: 1.0, envIntensity: 0.86, exposure: 0.7, disc: 0,
    fogDensity: 0.00105, fogFalloff: 0.016, fogColA: [0.36, 0.44, 0.6], fogColB: [1.25, 0.72, 0.4],
    wet: 0, rain: 0, night: 0.25,
    shadowTint: [0.92, 0.98, 1.1], highTint: [1.09, 1.0, 0.9], contrast: 1.09, saturation: 1.16, bloom: 0.62, grip: 0,
  },
  day: {
    id: 'day', name: 'CLEAR NOON', nameCn: '晴日', desc: '碧空流云 · 干地',
    sunAzimuth: -50, sunElevation: 46, sunColor: [1.0, 0.96, 0.9], sunIntensity: 5.4,
    skyIntensity: 1.0, envIntensity: 1.0, exposure: 0.6, disc: 1,
    fogDensity: 0.00042, fogFalloff: 0.014, fogColA: [0.5, 0.62, 0.8], fogColB: [0.86, 0.86, 0.84],
    wet: 0, rain: 0, night: 0,
    shadowTint: [0.97, 1.0, 1.05], highTint: [1.03, 1.0, 0.97], contrast: 1.06, saturation: 1.1, bloom: 0.45, grip: 0,
  },
  storm: {
    id: 'storm', name: 'STORM', nameCn: '暴雨', desc: '阴云骤雨 · 湿地',
    sunAzimuth: -100, sunElevation: 38, sunColor: [0.82, 0.86, 0.92], sunIntensity: 0.9,
    skyIntensity: 0.72, envIntensity: 0.8, exposure: 0.62, disc: 0,
    fogDensity: 0.0026, fogFalloff: 0.008, fogColA: [0.5, 0.53, 0.56], fogColB: [0.62, 0.63, 0.63],
    wet: 1, rain: 1, night: 0.45,
    shadowTint: [0.93, 0.99, 1.06], highTint: [0.98, 1.0, 1.02], contrast: 1.1, saturation: 0.86, bloom: 0.7, grip: 1,
  },
  night: {
    id: 'night', name: 'MOONLIGHT', nameCn: '月夜', desc: '月色泛光 · 干地',
    sunAzimuth: -70, sunElevation: 27, sunColor: [0.6, 0.73, 1.0], sunIntensity: 1.15,
    skyIntensity: 0.22, envIntensity: 0.5, exposure: 1.45, disc: 0,
    fogDensity: 0.0011, fogFalloff: 0.02, fogColA: [0.012, 0.02, 0.04], fogColB: [0.05, 0.07, 0.11],
    wet: 0, rain: 0, night: 1,
    shadowTint: [0.9, 0.98, 1.12], highTint: [1.04, 1.0, 0.96], contrast: 1.1, saturation: 1.05, bloom: 0.9, grip: 0,
  },
};
export const WEATHER_IDS: WeatherId[] = ['sunset', 'day', 'storm', 'night'];

interface SkyMeta {
  file: string;
  range: number;
  sunDir: number[];
  sunAzimuthDeg: number;
  sunElevationDeg: number;
  mid: number[];
  horizon: number[];
}

const SKY_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = position;
  vec4 p = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  gl_Position = p.xyww;
}`;
const SKY_FRAG = /* glsl */ `
uniform sampler2D uMap;
uniform float uRange;
uniform float uIntensity;
uniform float uYaw;
uniform vec3 uGround;
uniform vec3 uSunDir;
uniform vec3 uDiscColor;
uniform float uDisc;
varying vec3 vDir;
void main() {
  vec3 d = normalize( vDir );
  float cs = cos( uYaw ), sn = sin( uYaw );
  vec3 l = vec3( d.x * cs - d.z * sn, d.y, d.x * sn + d.z * cs );
  vec2 uv = vec2( atan( l.z, l.x ) * 0.15915494 + 0.5, asin( clamp( l.y, -1.0, 1.0 ) ) * 0.31830989 + 0.5 );
  vec3 c = pow( texture2D( uMap, uv ).rgb, vec3( 2.2 ) ) * uRange * uIntensity;
  float g = smoothstep( 0.0, -0.05, d.y );
  c = mix( c, uGround, g );
  float cosA = dot( d, uSunDir );
  float disc = smoothstep( 0.99988, 0.99996, cosA );
  float corona = pow( max( cosA, 0.0 ), 1400.0 ) * 0.9 + pow( max( cosA, 0.0 ), 120.0 ) * 0.05;
  c += uDiscColor * uDisc * ( disc * 60.0 + corona ) * ( 1.0 - g );
  gl_FragColor = vec4( c, 1.0 );
}`;

export class Sky {
  readonly engine: Engine;
  readonly dome: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  weather: Weather = WEATHER.sunset;
  private meta: Record<string, SkyMeta> | null = null;
  private envRT: THREE.WebGLRenderTarget | null = null;
  private textures = new Map<string, THREE.Texture>();
  readonly sunDir = new THREE.Vector3();

  constructor(engine: Engine) {
    this.engine = engine;
    this.material = new THREE.ShaderMaterial({
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      uniforms: {
        uMap: { value: null },
        uRange: { value: 8 },
        uIntensity: { value: 1 },
        uYaw: { value: 0 },
        uGround: { value: new THREE.Color(0.05, 0.055, 0.05) },
        uSunDir: { value: new THREE.Vector3(0, 1, 0) },
        uDiscColor: { value: new THREE.Color(1, 1, 1) },
        uDisc: { value: 0 },
      },
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: true,
      fog: false,
    });
    this.dome = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), this.material);
    this.dome.scale.setScalar(3000);
    this.dome.frustumCulled = false;
    this.dome.renderOrder = 900;
    this.dome.name = 'sky';
    engine.scene.add(this.dome);
  }

  async load(id: WeatherId): Promise<void> {
    const base = `${import.meta.env.BASE_URL}assets/sky/`;
    if (!this.meta) this.meta = (await (await fetch(`${base}sky.json`)).json()) as Record<string, SkyMeta>;
    const w = WEATHER[id];
    const m = this.meta[id];
    let tex = this.textures.get(id);
    if (!tex) {
      tex = await loadTextureAsync(`${base}${m.file}`, { srgb: false, repeat: false });
      // low tier (phones): halve the 4096x2048 sky to save ~24 MB of GPU memory
      const img = tex.image as HTMLImageElement;
      if (this.engine.q.name === 'low' && img && img.width > 2048) {
        const c = document.createElement('canvas');
        c.width = 2048;
        c.height = 1024;
        c.getContext('2d')!.drawImage(img, 0, 0, 2048, 1024);
        tex.dispose();
        tex = new THREE.CanvasTexture(c);
        tex.colorSpace = THREE.NoColorSpace;
      }
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.generateMipmaps = false;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.needsUpdate = true;
      this.textures.set(id, tex);
    }
    this.apply(w, m, tex);
  }

  private apply(w: Weather, m: SkyMeta, tex: THREE.Texture): void {
    this.weather = w;
    const eng = this.engine;
    const u = this.material.uniforms;
    // key light direction in world space
    const az = (w.sunAzimuth * Math.PI) / 180;
    const el = (w.sunElevation * Math.PI) / 180;
    this.sunDir.set(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
    // rotate the dome so the HDRI's own sun sits at the same azimuth
    const thetaWorld = Math.atan2(this.sunDir.z, this.sunDir.x);
    const thetaLocal = (m.sunAzimuthDeg * Math.PI) / 180;
    u.uMap.value = tex;
    u.uRange.value = m.range;
    u.uIntensity.value = w.skyIntensity;
    u.uYaw.value = thetaLocal - thetaWorld;
    const hel = (m.sunElevationDeg * Math.PI) / 180;
    (u.uSunDir.value as THREE.Vector3).set(Math.cos(thetaWorld) * Math.cos(hel), Math.sin(hel), Math.sin(thetaWorld) * Math.cos(hel));
    (u.uDiscColor.value as THREE.Color).setRGB(w.sunColor[0], w.sunColor[1], w.sunColor[2]);
    u.uDisc.value = w.disc;
    (u.uGround.value as THREE.Color).setRGB(m.mid[0] * 0.2 * w.skyIntensity, m.mid[1] * 0.21 * w.skyIntensity, m.mid[2] * 0.17 * w.skyIntensity);

    eng.setSun(this.sunDir, new THREE.Color(w.sunColor[0], w.sunColor[1], w.sunColor[2]), w.sunIntensity);
    eng.setExposure(w.exposure);
    U.uFogColA.value.setRGB(w.fogColA[0], w.fogColA[1], w.fogColA[2]);
    U.uFogColB.value.setRGB(w.fogColB[0], w.fogColB[1], w.fogColB[2]);
    U.uFogParams.value.set(w.fogDensity, w.fogFalloff, 0, 1);
    U.uWet.value = w.wet;
    U.uRain.value = w.rain;
    U.uNight.value = w.night;
    const amb = w.skyIntensity * w.envIntensity;
    U.uAmbSky.value.setRGB((m.mid[0] * 0.6 + m.horizon[0] * 0.4) * amb, (m.mid[1] * 0.6 + m.horizon[1] * 0.4) * amb, (m.mid[2] * 0.6 + m.horizon[2] * 0.4) * amb);
    U.uAmbGround.value.setRGB(m.mid[0] * 0.2 * amb, m.mid[1] * 0.21 * amb, m.mid[2] * 0.16 * amb);
    eng.grade.setTints(new THREE.Vector3(...w.shadowTint), new THREE.Vector3(...w.highTint));
    eng.grade.set('uContrast', w.contrast);
    eng.grade.set('uSaturation', w.saturation);
    if (eng.bloom) eng.bloom.intensity = w.bloom;

    // image-based lighting from the same dome
    const envScene = new THREE.Scene();
    const envDome = new THREE.Mesh(this.dome.geometry, this.material);
    envDome.scale.setScalar(50);
    envScene.add(envDome);
    const pmrem = new THREE.PMREMGenerator(eng.renderer);
    const rt = pmrem.fromScene(envScene, 0, 0.1, 200);
    pmrem.dispose();
    if (this.envRT) this.envRT.dispose();
    this.envRT = rt;
    eng.scene.environment = rt.texture;
    eng.scene.environmentIntensity = w.envIntensity;
  }

  update(camera: THREE.Camera): void {
    this.dome.position.copy(camera.position);
  }
}
