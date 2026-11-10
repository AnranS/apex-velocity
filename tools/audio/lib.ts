/**
 * Shared helpers for the audio verification tools (Node, run with `npx tsx`).
 * FFT / spectral metrics / WAV writer / offline driver for the car synth DSP.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { CarSynthDSP, type EngineCfg } from '../../src/audio/engine-dsp';
import { engineConfig, NOMINAL_RPM_MAX, type EngineType } from '../../src/audio/engine-configs';

export const WORK = new URL('./_work/', import.meta.url).pathname;

// ───────────────────────── basic metrics ─────────────────────────
export function rms(x: ArrayLike<number>, a = 0, b = x.length): number {
  let s = 0;
  for (let i = a; i < b; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, b - a));
}
export function peak(x: ArrayLike<number>, a = 0, b = x.length): number {
  let p = 0;
  for (let i = a; i < b; i++) {
    const v = Math.abs(x[i]);
    if (v > p) p = v;
  }
  return p;
}
export function dc(x: ArrayLike<number>, a = 0, b = x.length): number {
  let s = 0;
  for (let i = a; i < b; i++) s += x[i];
  return s / Math.max(1, b - a);
}
export function allFinite(x: ArrayLike<number>): boolean {
  for (let i = 0; i < x.length; i++) if (!Number.isFinite(x[i])) return false;
  return true;
}
export const dB = (v: number): number => 20 * Math.log10(Math.max(v, 1e-12));
export const fmt = (v: number, d = 2): string => (Number.isFinite(v) ? v.toFixed(d) : String(v));

/** Largest sample-to-sample jump relative to the signal RMS (click detector). */
export function maxStep(x: ArrayLike<number>, a = 1, b = x.length): number {
  let m = 0;
  for (let i = Math.max(1, a); i < b; i++) {
    const d = Math.abs(x[i] - x[i - 1]);
    if (d > m) m = d;
  }
  return m;
}

// ───────────────────────── FFT ─────────────────────────
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci;
        const xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

/** Hann-windowed magnitude spectrum (bins 0..N/2). A sine of amplitude A peaks at ≈ A. */
export function spectrum(x: ArrayLike<number>, start: number, N: number): Float64Array {
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
    re[i] = (x[start + i] ?? 0) * w;
  }
  fft(re, im);
  const out = new Float64Array(N / 2 + 1);
  for (let i = 0; i <= N / 2; i++) out[i] = (Math.hypot(re[i], im[i]) * 4) / N;
  return out;
}

/** Average power spectrum over several half-overlapping frames (Welch). Returns power per bin. */
export function welch(x: ArrayLike<number>, start: number, len: number, N: number): Float64Array {
  const out = new Float64Array(N / 2 + 1);
  let frames = 0;
  for (let s = start; s + N <= start + len; s += N / 2) {
    const m = spectrum(x, s, N);
    for (let i = 0; i < out.length; i++) out[i] += m[i] * m[i];
    frames++;
  }
  if (frames > 0) for (let i = 0; i < out.length; i++) out[i] /= frames;
  return out;
}

export function bandPower(pw: Float64Array, fs: number, f0: number, f1: number): number {
  const N = (pw.length - 1) * 2;
  const a = Math.max(1, Math.floor((f0 * N) / fs));
  const b = Math.min(pw.length - 1, Math.ceil((f1 * N) / fs));
  let s = 0;
  for (let i = a; i <= b; i++) s += pw[i];
  return s;
}

export function centroid(pw: Float64Array, fs: number, f0 = 20, f1 = 16000): number {
  const N = (pw.length - 1) * 2;
  const a = Math.max(1, Math.floor((f0 * N) / fs));
  const b = Math.min(pw.length - 1, Math.ceil((f1 * N) / fs));
  let s = 0;
  let w = 0;
  for (let i = a; i <= b; i++) {
    s += pw[i] * ((i * fs) / N);
    w += pw[i];
  }
  return w > 0 ? s / w : 0;
}

/**
 * Power captured by the harmonics k*f0 (k = 1..) up to fmax, each within ±(tol*f + 1.5 bins),
 * optionally only every `stride`-th harmonic starting at `first` (to separate engine orders).
 */
export function harmonicPower(pw: Float64Array, fs: number, f0: number, fmax: number, tol = 0.012, first = 1, stride = 1): number {
  const N = (pw.length - 1) * 2;
  const binHz = fs / N;
  let s = 0;
  for (let k = first; k * f0 <= fmax; k += stride) {
    const f = k * f0;
    const half = Math.max(1.5 * binHz, tol * f);
    const a = Math.max(1, Math.round((f - half) / binHz));
    const b = Math.min(pw.length - 1, Math.round((f + half) / binHz));
    for (let i = a; i <= b; i++) s += pw[i];
  }
  return s;
}

/** Frequency of the strongest bin in [f0,f1] with parabolic interpolation. */
export function peakFreq(pw: Float64Array, fs: number, f0: number, f1: number): { f: number; p: number } {
  const N = (pw.length - 1) * 2;
  const a = Math.max(1, Math.floor((f0 * N) / fs));
  const b = Math.min(pw.length - 2, Math.ceil((f1 * N) / fs));
  let bi = a;
  for (let i = a; i <= b; i++) if (pw[i] > pw[bi]) bi = i;
  const y0 = Math.log(pw[bi - 1] + 1e-30);
  const y1 = Math.log(pw[bi] + 1e-30);
  const y2 = Math.log(pw[bi + 1] + 1e-30);
  const d = (0.5 * (y0 - y2)) / (y0 - 2 * y1 + y2 || 1);
  return { f: ((bi + d) * fs) / N, p: pw[bi] };
}

/** Log-spectral distance (dB RMS) between two power spectra on third-octave-ish bands. */
export function bandProfile(pw: Float64Array, fs: number): number[] {
  const out: number[] = [];
  for (let f = 40; f < 12000; f *= Math.pow(2, 1 / 3)) out.push(10 * Math.log10(bandPower(pw, fs, f, f * Math.pow(2, 1 / 3)) + 1e-20));
  return out;
}
export function profileDistance(a: number[], b: number[]): number {
  // remove overall level, compare shapes
  const ma = a.reduce((s, v) => s + v, 0) / a.length;
  const mb = b.reduce((s, v) => s + v, 0) / b.length;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - ma - (b[i] - mb)) ** 2;
  return Math.sqrt(s / a.length);
}

// ───────────────────────── WAV ─────────────────────────
export function writeWav(path: string, channels: ArrayLike<number>[], fs: number): void {
  const n = channels[0].length;
  const ch = channels.length;
  const buf = Buffer.alloc(44 + n * ch * 2);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * ch * 2, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(ch, 22);
  buf.writeUInt32LE(fs, 24);
  buf.writeUInt32LE(fs * ch * 2, 28);
  buf.writeUInt16LE(ch * 2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(n * ch * 2, 40);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < ch; c++) {
      const v = Math.max(-1, Math.min(1, channels[c][i]));
      buf.writeInt16LE(Math.round(v * 32767), o);
      o += 2;
    }
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, buf);
}

// ───────────────────────── car synth offline driver ─────────────────────────
export interface DriveState {
  rpm: number;
  norm?: number;
  load: number;
  speed?: number;
  nitro?: number;
  slip?: number;
  surface?: number;
  scrape?: number;
  gain?: number;
  pan?: number;
  pitch?: number;
  air?: number;
  active?: number;
}
export interface RenderOpts {
  type: EngineType;
  player?: boolean;
  fs?: number;
  dur: number;
  /** state as a function of time */
  state: (t: number) => DriveState;
  /** state update rate (Hz); the game updates once per rendered frame */
  updateHz?: number;
  debug?: number;
  cfg?: EngineCfg;
  rpmMax?: number;
  events?: { t: number; code: number; val?: number }[];
}
export interface Rendered {
  fs: number;
  eL: Float32Array;
  eR: Float32Array;
  tL: Float32Array;
  tR: Float32Array;
}

export function packState(slot: number, s: DriveState, rpmMax: number): Float32Array {
  const d = new Float32Array(14);
  d[0] = slot;
  d[1] = s.active ?? 1;
  d[2] = s.rpm;
  d[3] = s.norm ?? s.rpm / rpmMax;
  d[4] = s.load;
  d[5] = s.speed ?? 0;
  d[6] = s.nitro ?? 0;
  d[7] = s.slip ?? 0;
  d[8] = s.surface ?? 0;
  d[9] = s.scrape ?? 0;
  d[10] = s.gain ?? 1;
  d[11] = s.pan ?? 0;
  d[12] = s.pitch ?? 1;
  d[13] = s.air ?? 1;
  return d;
}

/** Render one voice through the DSP exactly as the worklet would (128-sample quanta). */
export function renderVoice(o: RenderOpts): Rendered {
  const fs = o.fs ?? 48000;
  const n = Math.round(o.dur * fs);
  const dsp = new CarSynthDSP(fs, 4);
  const rpmMax = o.rpmMax ?? NOMINAL_RPM_MAX[o.type];
  dsp.configure(0, o.cfg ?? engineConfig(o.type), o.player ?? true, o.debug ?? 0);
  const out: Rendered = { fs, eL: new Float32Array(n), eR: new Float32Array(n), tL: new Float32Array(n), tR: new Float32Array(n) };
  const q0 = new Float32Array(128);
  const q1 = new Float32Array(128);
  const q2 = new Float32Array(128);
  const q3 = new Float32Array(128);
  const upd = Math.max(1, Math.round(fs / (o.updateHz ?? 60)));
  let nextUpd = 0;
  const evs = (o.events ?? []).slice().sort((a, b) => a.t - b.t);
  let ei = 0;
  for (let pos = 0; pos < n; pos += 128) {
    const t = pos / fs;
    if (pos >= nextUpd) {
      dsp.setStates(packState(0, o.state(t), rpmMax));
      nextUpd += upd;
    }
    while (ei < evs.length && evs[ei].t <= t) {
      dsp.event(0, evs[ei].code, evs[ei].val ?? 1);
      ei++;
    }
    const m = Math.min(128, n - pos);
    dsp.process(q0, q1, q2, q3, m);
    out.eL.set(q0.subarray(0, m), pos);
    out.eR.set(q1.subarray(0, m), pos);
    out.tL.set(q2.subarray(0, m), pos);
    out.tR.set(q3.subarray(0, m), pos);
  }
  return out;
}

// ───────────────────────── tiny test harness ─────────────────────────
let failed = 0;
let passed = 0;
export function check(name: string, ok: boolean, detail = ''): void {
  if (ok) passed++;
  else failed++;
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${name}${detail ? '  — ' + detail : ''}`);
}
export function summary(title: string): number {
  console.log(`\n${title}: ${passed} passed, ${failed} failed`);
  return failed;
}
