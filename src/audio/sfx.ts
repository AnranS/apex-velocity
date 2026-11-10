/**
 * One-shot sound effects and drum hits, synthesised offline into stereo Float32Arrays (no asset files).
 * Pure + deterministic → verified numerically in Node (tools/audio/test-sfx.ts).
 */
import { Rng, buf, osc, noise, filt, envAD, envASR, mixInto, normalize, fadeEdges, reverb, bell, stereo, panInto, midiHz, softClip, smooth, type Stereo, type Fn } from './synth';

export type SfxName =
  | 'countdown_beep'
  | 'countdown_go'
  | 'ui_move'
  | 'ui_select'
  | 'ui_back'
  | 'lap'
  | 'best_lap'
  | 'final_lap'
  | 'finish'
  | 'crash_light'
  | 'crash_heavy'
  | 'position_up'
  | 'position_down'
  | 'wrong_way'
  | 'cheer'
  | 'nitro_ready';

export const SFX_NAMES: SfxName[] = [
  'countdown_beep',
  'countdown_go',
  'ui_move',
  'ui_select',
  'ui_back',
  'lap',
  'best_lap',
  'final_lap',
  'finish',
  'crash_light',
  'crash_heavy',
  'position_up',
  'position_down',
  'wrong_way',
  'cheer',
  'nitro_ready',
];

/** Sounds with several random variants. */
export const SFX_VARIANTS: Partial<Record<SfxName, number>> = { crash_light: 3, crash_heavy: 3 };
/** Sounds that belong to the UI bus (not ducked / low-passed while paused). */
export const SFX_UI: Partial<Record<SfxName, boolean>> = { ui_move: true, ui_select: true, ui_back: true };

const dup = (m: Float32Array): Stereo => ({ L: m, R: m.slice() });
const scaled = (e: Fn, a: number): Fn => (t) => a * e(t);

function tone3(m: Float32Array, fs: number, t0: number, f: number, env: Fn, dur: number, a = 1): void {
  osc(m, fs, t0, dur, 'sine', f, scaled(env, a));
  osc(m, fs, t0, dur, 'sine', f * 2, scaled(env, a * 0.28));
  osc(m, fs, t0, dur, 'sine', f * 3, scaled(env, a * 0.1));
}

function beep(fs: number, f: number, hold: number, rel: number): Float32Array {
  const m = buf(fs, hold + rel + 0.03);
  const env: Fn = (t) => (t < 0.004 ? t / 0.004 : t < hold ? 1 : Math.exp(-(t - hold) / (rel * 0.28)));
  tone3(m, fs, 0, f, env, hold + rel);
  return fadeEdges(m, fs, 0.001, 0.02);
}

function brass(out: Float32Array, fs: number, t: number, dur: number, midi: number, amp: number): void {
  const f = midiHz(midi);
  const tmp = buf(fs, dur + 0.3);
  const env = envASR(0.028, dur, 0.2);
  osc(tmp, fs, 0, dur + 0.25, 'saw', f, scaled(env, 0.5));
  osc(tmp, fs, 0, dur + 0.25, 'saw', f * 1.006, scaled(env, 0.35), 0.31);
  osc(tmp, fs, 0, dur + 0.25, 'saw', f * 0.5, scaled(env, 0.18), 0.6);
  filt(tmp, fs, 'lp', (tt) => 600 + 2700 * Math.min(1, tt / 0.06) * (0.45 + 0.55 * Math.exp(-tt / 0.9)), 0.9);
  mixInto(out, tmp, Math.round(t * fs), amp);
}

function crashLight(fs: number, variant: number): Stereo {
  const rng = new Rng(0x51ed + variant * 7919);
  const m = buf(fs, 0.5);
  osc(m, fs, 0, 0.22, 'sine', (t) => 58 + 75 * Math.exp(-t / 0.02), scaled(envAD(0.001, 0.05), 0.5));
  const n1 = buf(fs, 0.2);
  noise(n1, fs, 0, 0.15, envAD(0.0005, 0.028), rng);
  filt(n1, fs, 'lp', 1700 * rng.range(0.8, 1.3), 0.8);
  mixInto(m, n1, 0, 1.5);
  const base = [540, 1270, 2130, 3320];
  for (let k = 0; k < 4; k++) {
    const f = base[k] * rng.range(0.85, 1.2);
    osc(m, fs, rng.range(0, 0.018), 0.35, 'sine', f, scaled(envAD(0.001, rng.range(0.02, 0.06)), 0.16 / (k + 1)));
  }
  const n2 = buf(fs, 0.2);
  noise(n2, fs, 0.004, 0.1, envAD(0.001, 0.022), rng);
  filt(n2, fs, 'bp', 2600 * rng.range(0.8, 1.2), 1.1);
  mixInto(m, n2, 0, 1.3);
  fadeEdges(m, fs, 0.0005, 0.03);
  return normalize(reverb(dup(m), fs, 0.35, 0.1, 0.5, 0.2), 0.6);
}

function crashHeavy(fs: number, variant: number): Stereo {
  const rng = new Rng(0xc4a5 + variant * 104729);
  const dur = 1.35;
  const out = stereo(fs, dur);
  const m = buf(fs, dur);
  // body impact
  osc(m, fs, 0, 0.7, 'sine', (t) => 36 + 44 * Math.exp(-t / 0.05), scaled(envAD(0.001, 0.16, 0.02), 0.5));
  osc(m, fs, 0, 0.3, 'sine', (t) => 95 + 60 * Math.exp(-t / 0.03), scaled(envAD(0.001, 0.07), 0.3));
  // crunch: noise gated by a crackly random envelope
  const cr = buf(fs, dur);
  let g = 1;
  let target = 1;
  const n = cr.length;
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    if ((i & 127) === 0) target = rng.next() < 0.45 ? rng.range(0.1, 0.5) : rng.range(0.6, 1.2);
    g += (target - g) * 0.02;
    cr[i] = rng.bi() * g * Math.exp(-t / 0.2) * (t < 0.002 ? t / 0.002 : 1);
  }
  filt(cr, fs, 'lp', (t) => 900 + 3200 * Math.exp(-t / 0.15), 0.8);
  mixInto(m, cr, 0, 2.2);
  // sheet-metal resonances
  for (let k = 0; k < 6; k++) {
    const f = rng.range(260, 3100);
    const wob = rng.range(18, 60);
    const ph = rng.range(0, 6.28);
    const e = envAD(0.002, rng.range(0.04, 0.13));
    osc(m, fs, rng.range(0, 0.04), 0.6, 'sine', (t) => f * (1 + 0.012 * Math.sin(wob * t + ph)), (t) => rng.range(0.03, 0.09) * e(t) * (0.55 + 0.45 * Math.sin(wob * 2.3 * t + ph)));
  }
  // mid-band plastic / body-panel crunch
  const pl = buf(fs, 0.5);
  noise(pl, fs, 0.004, 0.4, (t) => Math.exp(-t / 0.09) * (0.6 + 0.4 * Math.sin(310 * t) * Math.sin(47 * t)), rng);
  filt(pl, fs, 'bp', (t) => 2400 - 2200 * t, 0.9);
  mixInto(m, pl, 0, 1.8);
  panInto(out, m, 0);
  // debris / glass
  const cnt = 14 + Math.floor(rng.next() * 8);
  for (let k = 0; k < cnt; k++) {
    const t0 = 0.06 + Math.pow(rng.next(), 1.6) * 0.95;
    const p = buf(fs, 0.09);
    const pf = rng.range(2800, 7200);
    const pe = scaled(envAD(0.0005, rng.range(0.005, 0.018)), 0.1 * (1 - t0 * 0.7));
    osc(p, fs, 0, 0.08, 'sine', pf, pe);
    osc(p, fs, 0, 0.08, 'sine', pf * rng.range(1.31, 1.62), scaled(pe, 0.6));
    panInto(out, p, rng.range(-0.8, 0.8), Math.round(t0 * fs));
  }
  fadeEdges(out.L, fs, 0.0005, 0.05);
  fadeEdges(out.R, fs, 0.0005, 0.05);
  softClip(out.L, 1.4);
  softClip(out.R, 1.4);
  return normalize(reverb(out, fs, 0.6, 0.16, 0.45, 0.4), 0.8);
}

function cheer(fs: number): Stereo {
  const rng = new Rng(0xbeef);
  const dur = 3.2;
  const out = stereo(fs, dur);
  const voices = 14;
  for (let v = 0; v < voices; v++) {
    const n1 = buf(fs, dur);
    const on = rng.range(0, 0.45);
    const len = rng.range(1.3, 2.5);
    const syl = rng.range(2.6, 5.5);
    const ph = rng.range(0, 6.28);
    const env: Fn = (t) => {
      const x = t - on;
      if (x < 0 || x > len) return 0;
      return smooth(0, 0.18, x) * (1 - smooth(len - 0.7, len, x)) * (0.62 + 0.38 * Math.sin(6.283 * syl * x + ph));
    };
    noise(n1, fs, 0, dur, env, rng);
    const f1 = rng.range(480, 1150);
    const a = filt(n1.slice(), fs, 'bp', (t) => f1 * (1 + 0.12 * Math.sin(3 * t + ph)), 2.6);
    const b = filt(n1, fs, 'bp', rng.range(2300, 3300), 3.2);
    const pan = rng.range(-0.9, 0.9);
    panInto(out, a, pan, 0, 1.0);
    panInto(out, b, pan, 0, 0.45);
  }
  // claps
  for (let k = 0; k < 110; k++) {
    const t0 = 0.15 + rng.next() * 2.5;
    const c = buf(fs, 0.03);
    noise(c, fs, 0, 0.02, envAD(0.0004, 0.005), rng);
    filt(c, fs, 'bp', rng.range(1500, 2700), 1.4);
    panInto(out, c, rng.range(-1, 1), Math.round(t0 * fs), 1.4 * smooth(0, 0.5, t0) * (1 - smooth(1.6, 2.9, t0)));
  }
  // whistles
  for (let k = 0; k < 3; k++) {
    const t0 = rng.range(0.3, 1.6);
    const d = rng.range(0.28, 0.5);
    const f0 = rng.range(1900, 2600);
    const f1 = f0 * rng.range(1.15, 1.4);
    const w = buf(fs, d + 0.02);
    osc(w, fs, 0, d, 'sine', (t) => f0 + (f1 - f0) * (t / d) + 25 * Math.sin(40 * t), (t) => 0.22 * smooth(0, 0.04, t) * (1 - smooth(d - 0.08, d, t)));
    panInto(out, w, rng.range(-0.7, 0.7), Math.round(t0 * fs));
  }
  const n = out.L.length;
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const e = smooth(0, 0.3, t) * (1 - smooth(1.5, dur - 0.02, t));
    out.L[i] *= e;
    out.R[i] *= e;
  }
  return normalize(reverb(out, fs, 0.9, 0.2, 0.4, 0.5), 0.6);
}

/** Render a sound effect. Peak level is baked in (see the `normalize` targets). */
export function renderSfx(name: SfxName, fs: number, variant = 0): Stereo {
  switch (name) {
    case 'countdown_beep':
      return normalize(reverb(dup(beep(fs, 660, 0.14, 0.14)), fs, 0.4, 0.1, 0.4, 0.3), 0.5);
    case 'countdown_go': {
      const m = buf(fs, 1.0);
      const env: Fn = (t) => (t < 0.004 ? t / 0.004 : t < 0.5 ? 1 : Math.exp(-(t - 0.5) / 0.11));
      tone3(m, fs, 0, 1320, env, 0.95);
      osc(m, fs, 0, 0.95, 'sine', 1980, scaled(env, 0.22));
      fadeEdges(m, fs, 0.001, 0.03);
      return normalize(reverb(dup(m), fs, 0.6, 0.16, 0.4, 0.45), 0.6);
    }
    case 'ui_move': {
      const rng = new Rng(11);
      const m = buf(fs, 0.08);
      osc(m, fs, 0, 0.07, 'sine', (t) => 1550 - 5000 * t, envAD(0.0015, 0.016));
      noise(m, fs, 0, 0.005, 0.25, rng);
      filt(m, fs, 'hp', 500, 0.7);
      fadeEdges(m, fs, 0.0005, 0.01);
      return normalize(dup(m), 0.25);
    }
    case 'ui_select': {
      const m = buf(fs, 0.3);
      osc(m, fs, 0, 0.1, 'tri', 880, scaled(envAD(0.002, 0.03), 0.8));
      osc(m, fs, 0.055, 0.22, 'tri', 1320, scaled(envAD(0.002, 0.06), 0.8));
      osc(m, fs, 0.055, 0.22, 'sine', 2640, scaled(envAD(0.002, 0.04), 0.25));
      fadeEdges(m, fs, 0.0005, 0.02);
      return normalize(reverb(dup(m), fs, 0.35, 0.12, 0.4, 0.2), 0.35);
    }
    case 'ui_back': {
      const m = buf(fs, 0.28);
      osc(m, fs, 0, 0.1, 'tri', 660, scaled(envAD(0.002, 0.03), 0.8));
      osc(m, fs, 0.06, 0.2, 'tri', 440, scaled(envAD(0.002, 0.055), 0.8));
      fadeEdges(m, fs, 0.0005, 0.02);
      return normalize(reverb(dup(m), fs, 0.3, 0.1, 0.4, 0.18), 0.3);
    }
    case 'lap': {
      const m = buf(fs, 0.9);
      bell(m, fs, 0, 1318.5, 0.8, 0.11);
      bell(m, fs, 0.13, 1975.5, 0.7, 0.17);
      fadeEdges(m, fs, 0.0005, 0.05);
      return normalize(reverb(dup(m), fs, 0.8, 0.22, 0.35, 0.5), 0.45);
    }
    case 'best_lap': {
      const m = buf(fs, 1.3);
      const notes = [1046.5, 1318.5, 1568, 2093];
      notes.forEach((f, i) => bell(m, fs, i * 0.085, f, 0.7, i === 3 ? 0.3 : 0.1));
      osc(m, fs, 0.26, 0.9, 'sine', 4186, (t) => 0.06 * Math.exp(-t / 0.3) * (0.6 + 0.4 * Math.sin(50 * t)));
      fadeEdges(m, fs, 0.0005, 0.06);
      return normalize(reverb(dup(m), fs, 1.0, 0.26, 0.3, 0.6), 0.5);
    }
    case 'final_lap': {
      const m = buf(fs, 1.0);
      for (let k = 0; k < 4; k++) osc(m, fs, k * 0.12, 0.13, 'square', k % 2 ? 1318.5 : 880, scaled(envASR(0.004, 0.09, 0.025), 0.5));
      osc(m, fs, 0.48, 0.45, 'square', 1318.5, scaled(envAD(0.004, 0.12, 0.12), 0.5));
      filt(m, fs, 'lp', 3200, 0.7);
      fadeEdges(m, fs, 0.0005, 0.04);
      return normalize(reverb(dup(m), fs, 0.5, 0.14, 0.4, 0.35), 0.45);
    }
    case 'finish': {
      const rng = new Rng(77);
      const m = buf(fs, 3.0);
      // ta-ta-ta TAAA — ta TAAAA (C major)
      brass(m, fs, 0.0, 0.11, 67, 0.8);
      brass(m, fs, 0.16, 0.11, 67, 0.8);
      brass(m, fs, 0.32, 0.11, 67, 0.8);
      for (const k of [60, 64, 67, 72]) brass(m, fs, 0.48, 0.5, k, k === 72 ? 0.9 : 0.5);
      brass(m, fs, 1.05, 0.17, 76, 0.85);
      for (const k of [48, 64, 67, 72, 79]) brass(m, fs, 1.27, 1.15, k, k === 79 ? 1.0 : 0.5);
      for (const t0 of [0.48, 1.27]) {
        osc(m, fs, t0, 0.8, 'sine', (t) => 82 + 22 * Math.exp(-t / 0.03), scaled(envAD(0.002, 0.2), 0.9));
        const th = buf(fs, 0.2);
        noise(th, fs, 0, 0.1, envAD(0.001, 0.02), rng);
        filt(th, fs, 'lp', 900, 0.7);
        mixInto(m, th, Math.round(t0 * fs), 0.5);
      }
      const cy = buf(fs, 1.7);
      noise(cy, fs, 0, 1.6, scaled(envAD(0.004, 0.45), 0.35), rng);
      filt(cy, fs, 'hp', 4500, 0.7);
      mixInto(m, cy, Math.round(1.27 * fs), 1);
      fadeEdges(m, fs, 0.001, 0.1);
      return normalize(reverb(dup(m), fs, 1.4, 0.24, 0.35, 0.9), 0.7);
    }
    case 'crash_light':
      return crashLight(fs, variant);
    case 'crash_heavy':
      return crashHeavy(fs, variant);
    case 'position_up': {
      const m = buf(fs, 0.24);
      osc(m, fs, 0, 0.08, 'sine', 1174.7, scaled(envAD(0.002, 0.022), 0.8));
      osc(m, fs, 0.06, 0.16, 'sine', 1568, scaled(envAD(0.002, 0.045), 0.8));
      osc(m, fs, 0.06, 0.16, 'sine', 3136, scaled(envAD(0.002, 0.03), 0.2));
      fadeEdges(m, fs, 0.0005, 0.02);
      return normalize(reverb(dup(m), fs, 0.3, 0.1, 0.4, 0.15), 0.35);
    }
    case 'position_down': {
      const m = buf(fs, 0.26);
      osc(m, fs, 0, 0.09, 'tri', 784, scaled(envAD(0.002, 0.025), 0.8));
      osc(m, fs, 0.065, 0.18, 'tri', 587.3, scaled(envAD(0.002, 0.05), 0.8));
      filt(m, fs, 'lp', 2200, 0.7);
      fadeEdges(m, fs, 0.0005, 0.02);
      return normalize(reverb(dup(m), fs, 0.3, 0.1, 0.4, 0.15), 0.3);
    }
    case 'wrong_way': {
      const m = buf(fs, 1.25);
      for (let k = 0; k < 8; k++) {
        osc(m, fs, k * 0.15, 0.15, 'square', k % 2 ? 392 : 523.3, scaled(envASR(0.006, 0.125, 0.02), 0.5), 0, 0.4);
      }
      filt(m, fs, 'lp', 2400, 0.8);
      fadeEdges(m, fs, 0.001, 0.03);
      return normalize(reverb(dup(m), fs, 0.3, 0.08, 0.4, 0.2), 0.3);
    }
    case 'cheer':
      return cheer(fs);
    case 'nitro_ready': {
      const rng = new Rng(5);
      const m = buf(fs, 0.6);
      bell(m, fs, 0.03, 1760, 0.7, 0.09);
      osc(m, fs, 0.03, 0.4, 'sine', 2637, scaled(envAD(0.002, 0.07), 0.3));
      const w = buf(fs, 0.26);
      noise(w, fs, 0, 0.24, (t) => 0.5 * smooth(0, 0.1, t) * (1 - smooth(0.14, 0.24, t)), rng);
      filt(w, fs, 'bp', (t) => 700 + 9000 * t, 2.5);
      mixInto(m, w, 0, 0.6);
      fadeEdges(m, fs, 0.001, 0.04);
      return normalize(reverb(dup(m), fs, 0.5, 0.16, 0.4, 0.3), 0.4);
    }
  }
}

// ───────────────────────── drum kit (for the music sequencer) ─────────────────────────
export type DrumName = 'kick' | 'snare' | 'clap' | 'hat' | 'hatOpen' | 'tom' | 'crash' | 'boom' | 'rim';
export const DRUM_NAMES: DrumName[] = ['kick', 'snare', 'clap', 'hat', 'hatOpen', 'tom', 'crash', 'boom', 'rim'];

/** peak level of each drum hit (relative balance of the kit) */
const DRUM_PEAK: Record<DrumName, number> = { kick: 0.95, snare: 0.8, clap: 0.7, hat: 0.5, hatOpen: 0.5, tom: 0.85, crash: 0.6, boom: 0.9, rim: 0.6 };

export function renderDrum(name: DrumName, fs: number): Float32Array {
  const x = drumRaw(name, fs);
  let p = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i] < 0 ? -x[i] : x[i];
    if (v > p) p = v;
  }
  const g = p > 1e-9 ? DRUM_PEAK[name] / p : 1;
  for (let i = 0; i < x.length; i++) x[i] *= g;
  return x;
}

function drumRaw(name: DrumName, fs: number): Float32Array {
  const rng = new Rng(0xd00d + name.length * 131 + name.charCodeAt(0));
  switch (name) {
    case 'kick': {
      const m = buf(fs, 0.42);
      osc(m, fs, 0, 0.4, 'sine', (t) => 52 + 150 * Math.exp(-t / 0.034), envAD(0.0008, 0.11, 0.03));
      const c = buf(fs, 0.01);
      noise(c, fs, 0, 0.004, 0.35, rng);
      filt(c, fs, 'hp', 1800, 0.7);
      mixInto(m, c);
      softClip(m, 1.6);
      return fadeEdges(m, fs, 0.0003, 0.03);
    }
    case 'snare': {
      const m = buf(fs, 0.32);
      const nz = buf(fs, 0.32);
      noise(nz, fs, 0, 0.3, envAD(0.0008, 0.065), rng);
      filt(nz, fs, 'bp', 1900, 0.6);
      filt(nz, fs, 'hp', 700, 0.7);
      mixInto(m, nz, 0, 1.6);
      osc(m, fs, 0, 0.15, 'tri', (t) => 165 + 30 * Math.exp(-t / 0.02), scaled(envAD(0.0008, 0.04), 0.7));
      return fadeEdges(softClip(m, 1.3), fs, 0.0003, 0.03);
    }
    case 'clap': {
      const m = buf(fs, 0.32);
      for (const t0 of [0, 0.011, 0.023]) noise(m, fs, t0, 0.02, envAD(0.0005, 0.005), rng);
      noise(m, fs, 0.03, 0.26, scaled(envAD(0.001, 0.075), 0.7), rng);
      filt(m, fs, 'bp', 1300, 1.0);
      filt(m, fs, 'hp', 650, 0.7);
      return fadeEdges(softClip(m, 2.2), fs, 0.0003, 0.03);
    }
    case 'hat': {
      const m = buf(fs, 0.08);
      noise(m, fs, 0, 0.07, envAD(0.0004, 0.013), rng);
      filt(m, fs, 'hp', 7500, 0.9);
      return fadeEdges(m, fs, 0.0002, 0.01);
    }
    case 'hatOpen': {
      const m = buf(fs, 0.38);
      noise(m, fs, 0, 0.36, envAD(0.0006, 0.1), rng);
      filt(m, fs, 'hp', 6500, 0.9);
      return fadeEdges(m, fs, 0.0002, 0.04);
    }
    case 'tom': {
      const m = buf(fs, 0.3);
      osc(m, fs, 0, 0.28, 'sine', (t) => 92 + 60 * Math.exp(-t / 0.04), envAD(0.001, 0.1));
      return fadeEdges(m, fs, 0.0003, 0.03);
    }
    case 'crash': {
      const m = buf(fs, 1.9);
      noise(m, fs, 0, 1.85, envAD(0.002, 0.5), rng);
      filt(m, fs, 'hp', 3600, 0.7);
      const b = buf(fs, 1.9);
      noise(b, fs, 0, 1.85, envAD(0.002, 0.3), rng);
      filt(b, fs, 'bp', 6200, 1.5);
      mixInto(m, b, 0, 0.8);
      return fadeEdges(m, fs, 0.0005, 0.2);
    }
    case 'boom': {
      const m = buf(fs, 2.6);
      osc(m, fs, 0, 2.5, 'sine', (t) => 34 + 22 * Math.exp(-t / 0.12), envAD(0.004, 0.75, 0.05));
      const th = buf(fs, 0.6);
      noise(th, fs, 0, 0.5, envAD(0.002, 0.09), rng);
      filt(th, fs, 'lp', 260, 0.7);
      mixInto(m, th, 0, 1.2);
      return fadeEdges(softClip(m, 1.3), fs, 0.001, 0.3);
    }
    case 'rim': {
      const m = buf(fs, 0.07);
      osc(m, fs, 0, 0.06, 'sine', 1720, scaled(envAD(0.0004, 0.008), 0.8));
      osc(m, fs, 0, 0.06, 'tri', 820, scaled(envAD(0.0004, 0.012), 0.5));
      return fadeEdges(m, fs, 0.0002, 0.01);
    }
  }
}
