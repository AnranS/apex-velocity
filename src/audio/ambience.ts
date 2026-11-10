/**
 * Seamless ambience loops (wind / rain / crowd) and reverb impulse responses, synthesised offline.
 * Loops are built from periodic noise + "circular" filtering, so they wrap without a click.
 */
import { Rng, whiteLoop, filt, filtered, mixInto, panInto, normalize, noise, buf, smooth, type Stereo } from './synth';

/** Wind bed: slow gusting low-passed noise. Suggested buffer rate: 16 kHz. */
export function renderWindLoop(fs: number, sec = 6): Stereo {
  const n = Math.round(fs * sec);
  const out: Stereo = { L: new Float32Array(n), R: new Float32Array(n) };
  for (let ch = 0; ch < 2; ch++) {
    const rng = new Rng(0x77aa + ch * 9973);
    const dst = ch === 0 ? out.L : out.R;
    const p1 = rng.range(0, 6.28);
    const p2 = rng.range(0, 6.28);
    const p3 = rng.range(0, 6.28);
    const gust = (i: number): number => {
      const x = (i / n) * Math.PI * 2;
      return 0.62 + 0.22 * Math.sin(x + p1) + 0.12 * Math.sin(3 * x + p2) + 0.06 * Math.sin(7 * x + p3);
    };
    const a = whiteLoop(n, rng);
    for (let i = 0; i < n; i++) a[i] *= gust(i);
    filt(a, fs, 'lp', 520, 0.7, true);
    mixInto(dst, a, 0, 1.0);
    const b = whiteLoop(n, rng);
    for (let i = 0; i < n; i++) {
      const g = gust((i + n / 3) % n);
      b[i] *= g * g;
    }
    filt(b, fs, 'bp', 1350, 1.4, true);
    mixInto(dst, b, 0, 0.3);
    const c = whiteLoop(n, rng);
    filt(c, fs, 'lp', 110, 0.7, true);
    mixInto(dst, c, 0, 1.2);
  }
  return normalize(out, 0.5);
}

/** Rain: hiss + dense droplets + low patter. Suggested buffer rate: context rate. */
export function renderRainLoop(fs: number, sec = 5): Stereo {
  const n = Math.round(fs * sec);
  const out: Stereo = { L: new Float32Array(n), R: new Float32Array(n) };
  for (let ch = 0; ch < 2; ch++) {
    const rng = new Rng(0x4a1 + ch * 7477);
    const dst = ch === 0 ? out.L : out.R;
    const hiss = whiteLoop(n, rng);
    mixInto(dst, filtered(hiss, fs, 'hp', 2600, 0.7, true), 0, 0.22);
    mixInto(dst, filt(hiss, fs, 'bp', 6500, 0.8, true), 0, 0.3);
    // droplets
    const drops = new Float32Array(n);
    const cnt = Math.round(380 * sec);
    for (let k = 0; k < cnt; k++) drops[Math.floor(rng.next() * n)] += rng.range(0.3, 1) * (rng.next() < 0.5 ? -1 : 1);
    mixInto(dst, filt(drops, fs, 'bp', 3600, 2.4, true), 0, 2.2);
    const pat = new Float32Array(n);
    const cnt2 = Math.round(70 * sec);
    for (let k = 0; k < cnt2; k++) pat[Math.floor(rng.next() * n)] += rng.range(0.4, 1);
    mixInto(dst, filt(pat, fs, 'bp', 480, 1.3, true), 0, 3.0);
  }
  return normalize(out, 0.5);
}

/** Crowd murmur: many syllabic formant-noise "voices". Suggested buffer rate: 16 kHz. */
export function renderCrowdLoop(fs: number, sec = 6): Stereo {
  const n = Math.round(fs * sec);
  const out: Stereo = { L: new Float32Array(n), R: new Float32Array(n) };
  const rng = new Rng(0xc0de);
  const voices = 14;
  for (let v = 0; v < voices; v++) {
    const k1 = 8 + Math.floor(rng.next() * 18); // syllables per loop
    const k2 = 1 + Math.floor(rng.next() * 3);
    const p1 = rng.range(0, 6.28);
    const p2 = rng.range(0, 6.28);
    const x = whiteLoop(n, rng);
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const s = Math.sin(k1 * a + p1);
      x[i] *= (s > 0 ? Math.pow(s, 1.5) : 0) * (0.55 + 0.45 * Math.sin(k2 * a + p2));
    }
    const lo = filtered(x, fs, 'bp', rng.range(330, 880), 3.0, true);
    const hi = filt(x, fs, 'bp', rng.range(1100, 2400), 4.0, true);
    const pan = rng.range(-1, 1);
    panInto(out, lo, pan, 0, 1);
    panInto(out, hi, pan, 0, 0.4);
  }
  // distant wash
  for (let ch = 0; ch < 2; ch++) {
    const w = whiteLoop(n, rng);
    filt(w, fs, 'bp', 700, 0.8, true);
    mixInto(ch === 0 ? out.L : out.R, w, 0, 0.5);
  }
  return normalize(out, 0.5);
}

/** Concrete tunnel / gallery: strong early slap + ~1.3 s dark tail. Must be rendered at the context rate. */
export function renderTunnelIR(fs: number): Stereo {
  const sec = 1.5;
  const out: Stereo = { L: buf(fs, sec), R: buf(fs, sec) };
  for (let ch = 0; ch < 2; ch++) {
    const rng = new Rng(0x7a11 + ch * 31337);
    const dst = ch === 0 ? out.L : out.R;
    const taps = [0.009, 0.019, 0.032, 0.047, 0.066, 0.089];
    taps.forEach((t, k) => {
      const i = Math.round((t + (ch ? 0.0017 : 0) + rng.range(0, 0.002)) * fs);
      dst[i] += (k % 2 ? -1 : 1) * 0.75 * Math.pow(0.8, k);
    });
    const tail = buf(fs, sec);
    noise(tail, fs, 0.012, sec - 0.02, (t) => smooth(0, 0.03, t) * Math.exp(-t / 0.27), rng);
    filt(tail, fs, 'lp', 3400, 0.7);
    filt(tail, fs, 'hp', 110, 0.7);
    mixInto(dst, tail, 0, 0.55);
    const m = dst.length;
    for (let i = 0; i < Math.round(0.05 * fs); i++) dst[m - 1 - i] *= i / (0.05 * fs);
  }
  return out;
}

/** Bright-ish hall/plate for the music bus. Must be rendered at the context rate. */
export function renderHallIR(fs: number): Stereo {
  const sec = 2.4;
  const out: Stereo = { L: buf(fs, sec), R: buf(fs, sec) };
  for (let ch = 0; ch < 2; ch++) {
    const rng = new Rng(0x4a11 + ch * 65537);
    const dst = ch === 0 ? out.L : out.R;
    noise(dst, fs, 0.012, sec - 0.02, (t) => smooth(0, 0.015, t) * Math.exp(-t / 0.42), rng);
    filt(dst, fs, 'hp', 220, 0.7);
    filt(dst, fs, 'lp', 6500, 0.7);
    const m = dst.length;
    for (let i = 0; i < Math.round(0.08 * fs); i++) dst[m - 1 - i] *= i / (0.08 * fs);
  }
  return out;
}
