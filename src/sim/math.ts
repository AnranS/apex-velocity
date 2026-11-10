// Small allocation-free math helpers shared by the simulation (DOM-free).

export const TAU = Math.PI * 2;
export const DEG = Math.PI / 180;

export function clamp(x: number, a: number, b: number): number {
  return x < a ? a : x > b ? b : x;
}
export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
export function invLerp(a: number, b: number, x: number): number {
  return (x - a) / (b - a);
}
export function remap(x: number, a: number, b: number, c: number, d: number): number {
  return c + (d - c) * clamp01((x - a) / (b - a));
}
export function smoothstep(a: number, b: number, x: number): number {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
}
/** Frame-rate independent exponential approach. `rate` = 1/time-constant. */
export function damp(current: number, target: number, rate: number, dt: number): number {
  return target + (current - target) * Math.exp(-rate * dt);
}
export function moveToward(current: number, target: number, maxDelta: number): number {
  const d = target - current;
  if (Math.abs(d) <= maxDelta) return target;
  return current + Math.sign(d) * maxDelta;
}
export function wrapAngle(a: number): number {
  a = (a + Math.PI) % TAU;
  if (a < 0) a += TAU;
  return a - Math.PI;
}
export function mod(a: number, n: number): number {
  return ((a % n) + n) % n;
}
/** Signed shortest difference a-b on a ring of circumference `len`, in (-len/2, len/2]. */
export function ringDiff(a: number, b: number, len: number): number {
  let d = (a - b) % len;
  if (d > len * 0.5) d -= len;
  else if (d <= -len * 0.5) d += len;
  return d;
}

/** Deterministic PRNG (mulberry32). */
export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = seed >>> 0 || 1;
  }
  next(): number {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }
  int(a: number, b: number): number {
    return Math.floor(this.range(a, b + 1));
  }
  pick<T>(arr: readonly T[]): T {
    return arr[Math.floor(this.next() * arr.length)];
  }
  sign(): number {
    return this.next() < 0.5 ? -1 : 1;
  }
}

/** Cheap 2D value-noise based fBm, deterministic, used for terrain and scatter masks. */
function hash2(ix: number, iz: number, seed: number): number {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iz, 668265263) ^ Math.imul(seed, 2147483647);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
export function valueNoise2(x: number, z: number, seed = 0): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10);
  const a = hash2(ix, iz, seed);
  const b = hash2(ix + 1, iz, seed);
  const c = hash2(ix, iz + 1, seed);
  const d = hash2(ix + 1, iz + 1, seed);
  return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}
/** fBm in [-1, 1] (approximately). */
export function fbm2(x: number, z: number, octaves: number, seed = 0, lacunarity = 2.0, gain = 0.5): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += (valueNoise2(x * freq, z * freq, seed + o * 17) * 2 - 1) * amp;
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm;
}
/** Ridged fBm in [0, 1], good for mountain crests. */
export function ridged2(x: number, z: number, octaves: number, seed = 0): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  let prev = 1;
  for (let o = 0; o < octaves; o++) {
    let n = 1 - Math.abs(valueNoise2(x * freq, z * freq, seed + o * 31) * 2 - 1);
    n *= n;
    sum += n * amp * prev;
    prev = n;
    norm += amp;
    amp *= 0.5;
    freq *= 2.03;
  }
  return sum / norm;
}
