/**
 * Tiny offline synthesis toolkit (pure functions on Float32Array, no Web Audio) used to bake one-shot SFX,
 * drum hits, ambience loops and impulse responses at start-up. Deterministic (seeded RNG) and Node-testable.
 */

export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = seed | 0 || 0x1234567;
  }
  /** uniform 0..1 */
  next(): number {
    let s = this.s;
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    this.s = s;
    return (s >>> 0) / 4294967296;
  }
  /** uniform -1..1 */
  bi(): number {
    return this.next() * 2 - 1;
  }
  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }
}

export type Fn = (t: number) => number;
export type Wave = 'sine' | 'tri' | 'saw' | 'square';
export interface Stereo {
  L: Float32Array;
  R: Float32Array;
}

export const midiHz = (m: number): number => 440 * Math.pow(2, (m - 69) / 12);
export const TAU = Math.PI * 2;

export function buf(fs: number, sec: number): Float32Array {
  return new Float32Array(Math.max(1, Math.round(fs * sec)));
}

function polyblep(t: number, dt: number): number {
  if (t < dt) {
    t /= dt;
    return t + t - t * t - 1;
  }
  if (t > 1 - dt) {
    t = (t - 1) / dt;
    return t * t + t + t + 1;
  }
  return 0;
}

/** Add a band-limited oscillator. `freq`/`amp` may be constants or functions of the time since `start` (s). */
export function osc(out: Float32Array, fs: number, start: number, dur: number, wave: Wave, freq: number | Fn, amp: number | Fn, phase0 = 0, duty = 0.5): void {
  const i0 = Math.max(0, Math.round(start * fs));
  const i1 = Math.min(out.length, Math.round((start + dur) * fs));
  const fF = typeof freq === 'function' ? freq : null;
  const aF = typeof amp === 'function' ? amp : null;
  let f = fF ? 0 : (freq as number);
  let a = aF ? 0 : (amp as number);
  let ph = phase0 - Math.floor(phase0);
  for (let i = i0; i < i1; i++) {
    const t = (i - i0) / fs;
    if (fF) f = fF(t);
    if (aF) a = aF(t);
    const dt = f / fs;
    let y: number;
    if (wave === 'sine') y = Math.sin(TAU * ph);
    else if (wave === 'tri') y = 4 * Math.abs(ph - 0.5) - 1;
    else if (wave === 'saw') y = 2 * ph - 1 - polyblep(ph, dt);
    else {
      let p2 = ph + 1 - duty;
      p2 -= Math.floor(p2);
      y = (ph < duty ? 1 : -1) + polyblep(ph, dt) - polyblep(p2, dt);
    }
    out[i] += y * a;
    ph += dt;
    if (ph >= 1) ph -= 1;
  }
}

/** Add white noise with an amplitude envelope. */
export function noise(out: Float32Array, fs: number, start: number, dur: number, amp: number | Fn, rng: Rng): void {
  const i0 = Math.max(0, Math.round(start * fs));
  const i1 = Math.min(out.length, Math.round((start + dur) * fs));
  const aF = typeof amp === 'function' ? amp : null;
  let a = aF ? 0 : (amp as number);
  for (let i = i0; i < i1; i++) {
    if (aF) a = aF((i - i0) / fs);
    out[i] += rng.bi() * a;
  }
}

export type FiltType = 'lp' | 'hp' | 'bp';

function coefs(type: FiltType, f: number, q: number, fs: number, c: Float64Array): void {
  const fc = Math.min(fs * 0.45, Math.max(10, f));
  const w = (TAU * fc) / fs;
  const cs = Math.cos(w);
  const al = Math.sin(w) / (2 * q);
  const a0 = 1 + al;
  let b0: number, b1: number, b2: number;
  if (type === 'lp') {
    b1 = 1 - cs;
    b0 = b1 / 2;
    b2 = b0;
  } else if (type === 'hp') {
    b1 = -(1 + cs);
    b0 = (1 + cs) / 2;
    b2 = b0;
  } else {
    b0 = al;
    b1 = 0;
    b2 = -al;
  }
  c[0] = b0 / a0;
  c[1] = b1 / a0;
  c[2] = b2 / a0;
  c[3] = (-2 * cs) / a0;
  c[4] = (1 - al) / a0;
}

/**
 * In-place biquad. `f` may be a function of time (coefficients refreshed every 32 samples).
 * `circular` runs the filter once over the buffer to prime its state first, so a periodic input gives a seamless loop.
 */
export function filt(x: Float32Array, fs: number, type: FiltType, f: number | Fn, q: number, circular = false): Float32Array {
  const c = new Float64Array(5);
  const fF = typeof f === 'function' ? f : null;
  coefs(type, fF ? fF(0) : (f as number), q, fs, c);
  let z1 = 0;
  let z2 = 0;
  const n = x.length;
  if (circular) {
    for (let i = 0; i < n; i++) {
      if (fF && (i & 31) === 0) coefs(type, fF(i / fs), q, fs, c);
      const xi = x[i];
      const y = c[0] * xi + z1;
      z1 = c[1] * xi - c[3] * y + z2;
      z2 = c[2] * xi - c[4] * y;
    }
  }
  for (let i = 0; i < n; i++) {
    if (fF && (i & 31) === 0) coefs(type, fF(i / fs), q, fs, c);
    const xi = x[i];
    const y = c[0] * xi + z1;
    z1 = c[1] * xi - c[3] * y + z2;
    z2 = c[2] * xi - c[4] * y;
    x[i] = y;
  }
  return x;
}

/** Filtered copy. */
export function filtered(x: Float32Array, fs: number, type: FiltType, f: number | Fn, q: number, circular = false): Float32Array {
  return filt(x.slice(), fs, type, f, q, circular);
}

// ───────────── envelopes ─────────────
/** Linear attack to 1, then exponential decay with time constant `tau` (optionally hold before decaying). */
export function envAD(attack: number, tau: number, hold = 0): Fn {
  return (t) => (t < attack ? t / Math.max(1e-6, attack) : t < attack + hold ? 1 : Math.exp(-(t - attack - hold) / tau));
}
/** Attack / sustain / release (linear attack, cosine release) for a note of length `dur`. */
export function envASR(attack: number, dur: number, release: number): Fn {
  return (t) => {
    if (t < 0) return 0;
    if (t < attack) return t / Math.max(1e-6, attack);
    if (t < dur) return 1;
    const r = (t - dur) / Math.max(1e-6, release);
    return r >= 1 ? 0 : 0.5 + 0.5 * Math.cos(Math.PI * r);
  };
}
export const smooth = (a: number, b: number, x: number): number => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// ───────────── utilities ─────────────
export function mixInto(dst: Float32Array, src: Float32Array, offset = 0, gain = 1): void {
  const n = Math.min(src.length, dst.length - offset);
  for (let i = 0; i < n; i++) dst[offset + i] += src[i] * gain;
}
export function gain(x: Float32Array, g: number): Float32Array {
  for (let i = 0; i < x.length; i++) x[i] *= g;
  return x;
}
export function peakOf(...xs: Float32Array[]): number {
  let p = 0;
  for (const x of xs)
    for (let i = 0; i < x.length; i++) {
      const v = x[i] < 0 ? -x[i] : x[i];
      if (v > p) p = v;
    }
  return p;
}
export function normalize(s: Stereo, peak: number): Stereo {
  const p = peakOf(s.L, s.R);
  if (p > 1e-9) {
    gain(s.L, peak / p);
    gain(s.R, peak / p);
  }
  return s;
}
/** Short fades so a buffer always starts and ends at zero. */
export function fadeEdges(x: Float32Array, fs: number, inSec = 0.002, outSec = 0.01): Float32Array {
  const a = Math.min(x.length, Math.round(inSec * fs));
  const b = Math.min(x.length, Math.round(outSec * fs));
  for (let i = 0; i < a; i++) x[i] *= i / a;
  for (let i = 0; i < b; i++) x[x.length - 1 - i] *= i / b;
  return x;
}
export function softClip(x: Float32Array, drive: number): Float32Array {
  for (let i = 0; i < x.length; i++) {
    const v = x[i] * drive;
    x[i] = v / (1 + Math.abs(v));
  }
  return x;
}
/** Constant-power pan of a mono buffer into a stereo pair (pan -1..1). */
export function panInto(dst: Stereo, src: Float32Array, pan: number, offset = 0, g = 1): void {
  const th = ((Math.max(-1, Math.min(1, pan)) + 1) * Math.PI) / 4;
  mixInto(dst.L, src, offset, g * Math.cos(th) * Math.SQRT2);
  mixInto(dst.R, src, offset, g * Math.sin(th) * Math.SQRT2);
}
export function stereo(fs: number, sec: number): Stereo {
  return { L: buf(fs, sec), R: buf(fs, sec) };
}

/**
 * Small Freeverb-style reverb (6 damped combs + 3 all-passes per channel).
 * Returns dry*(1-mix) + wet*mix, extended by the tail.
 */
export function reverb(inp: Stereo, fs: number, rt60: number, mix: number, damp = 0.35, tail = rt60): Stereo {
  const n = inp.L.length + Math.round(tail * fs);
  const out: Stereo = { L: new Float32Array(n), R: new Float32Array(n) };
  const combT = [0.0253, 0.0269, 0.029, 0.0307, 0.0322, 0.0338];
  const apT = [0.0126, 0.01, 0.0077];
  for (let ch = 0; ch < 2; ch++) {
    const src = ch === 0 ? inp.L : inp.R;
    const dst = ch === 0 ? out.L : out.R;
    const spread = ch === 0 ? 0 : 0.00052;
    const wet = new Float32Array(n);
    for (const ct of combT) {
      const len = Math.max(8, Math.round((ct + spread) * fs));
      const fb = Math.pow(10, (-3 * len) / (fs * rt60));
      const line = new Float32Array(len);
      let idx = 0;
      let lp = 0;
      for (let i = 0; i < n; i++) {
        const x = i < src.length ? src[i] : 0;
        const y = line[idx];
        lp = y * (1 - damp) + lp * damp;
        line[idx] = x + lp * fb;
        if (++idx >= len) idx = 0;
        wet[i] += y;
      }
    }
    for (const at of apT) {
      const len = Math.max(4, Math.round((at + spread) * fs));
      const line = new Float32Array(len);
      let idx = 0;
      for (let i = 0; i < n; i++) {
        const x = wet[i];
        const d = line[idx];
        const y = -x + d;
        line[idx] = x + d * 0.5;
        if (++idx >= len) idx = 0;
        wet[i] = y;
      }
    }
    const wg = mix / combT.length;
    for (let i = 0; i < n; i++) dst[i] = (i < src.length ? src[i] * (1 - mix) : 0) + wet[i] * wg * 1.6;
  }
  return out;
}

/** Inharmonic bell/chime partials (struck-bar ratios) with exponential decays. */
export function bell(out: Float32Array, fs: number, start: number, f: number, amp: number, decay: number): void {
  const parts: [number, number, number][] = [
    [1, 1, 1],
    [2.0, 0.42, 0.7],
    [3.01, 0.28, 0.45],
    [4.17, 0.16, 0.3],
    [5.43, 0.1, 0.22],
  ];
  for (const [r, a, d] of parts) {
    if (f * r > fs * 0.45) continue;
    osc(out, fs, start, decay * d * 6, 'sine', f * r, envADScaled(0.002, decay * d, amp * a));
  }
}
function envADScaled(attack: number, tau: number, a: number): Fn {
  return (t) => a * (t < attack ? t / attack : Math.exp(-(t - attack) / tau));
}

/** Periodic noise: white noise buffer of exactly n samples (use with circular filters for seamless loops). */
export function whiteLoop(n: number, rng: Rng): Float32Array {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = rng.bi();
  return x;
}
