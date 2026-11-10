/**
 * Node verification of the baked sounds: every one-shot SFX, drum hit, ambience loop and impulse response.
 *   npx tsx tools/audio/test-sfx.ts [wav]
 * Checks: finite samples, peak headroom, sane RMS, zero-valued edges (no clicks), expected spectral balance,
 * seamless loop boundaries, render cost. With `wav` every sound is also written to tools/audio/_work/.
 */
import { renderSfx, renderDrum, SFX_NAMES, SFX_VARIANTS, DRUM_NAMES, type SfxName } from '../../src/audio/sfx';
import { renderWindLoop, renderRainLoop, renderCrowdLoop, renderTunnelIR, renderHallIR } from '../../src/audio/ambience';
import { rms, peak, dB, fmt, allFinite, welch, bandPower, centroid, maxStep, check, summary, writeWav, WORK } from './lib';

const FS = 48000;
const wav = process.argv.includes('wav');

/** expected: [min duration s, max duration s, min centroid Hz, max centroid Hz] */
const EXPECT: Record<SfxName, [number, number, number, number]> = {
  countdown_beep: [0.25, 0.8, 500, 2200],
  countdown_go: [0.8, 1.6, 1000, 3200],
  ui_move: [0.04, 0.15, 900, 4000],
  ui_select: [0.2, 0.6, 800, 3200],
  ui_back: [0.2, 0.6, 400, 2200],
  lap: [0.6, 1.6, 1200, 5000],
  best_lap: [0.9, 2.2, 1200, 5500],
  final_lap: [0.7, 1.6, 900, 3500],
  finish: [2.2, 4.5, 400, 3500],
  crash_light: [0.3, 0.9, 350, 3500],
  crash_heavy: [1.0, 2.2, 300, 3000],
  position_up: [0.15, 0.5, 1000, 3500],
  position_down: [0.15, 0.5, 450, 1800],
  wrong_way: [1.0, 1.7, 400, 2200],
  cheer: [2.8, 4.2, 600, 3200],
  nitro_ready: [0.4, 1.1, 1400, 6000],
};

console.log('— one-shot SFX —');
let total = 0;
let totalSec = 0;
for (const name of SFX_NAMES) {
  const variants = SFX_VARIANTS[name] ?? 1;
  for (let k = 0; k < variants; k++) {
    const t0 = performance.now();
    const s = renderSfx(name, FS, k);
    const ms = performance.now() - t0;
    total += ms;
    const n = s.L.length;
    totalSec += n / FS;
    const pk = Math.max(peak(s.L), peak(s.R));
    const r = Math.sqrt((rms(s.L) ** 2 + rms(s.R) ** 2) / 2);
    const N = n >= 16384 ? 8192 : n >= 4096 ? 2048 : 1024;
    const pw = welch(s.L, 0, n, N);
    const cen = centroid(pw, FS, 30, 16000);
    const e = EXPECT[name];
    const edge = Math.max(Math.abs(s.L[0]), Math.abs(s.R[0]), Math.abs(s.L[n - 1]), Math.abs(s.R[n - 1]));
    const tag = variants > 1 ? `${name}#${k}` : name;
    check(
      tag,
      allFinite(s.L) && allFinite(s.R) && pk <= 0.95 && pk > 0.15 && dB(r) > -45 && edge < 0.01 && n / FS >= e[0] && n / FS <= e[1] && cen >= e[2] && cen <= e[3],
      `${fmt(n / FS, 2)}s peak ${fmt(dB(pk), 1)}dB rms ${fmt(dB(r), 1)}dB centroid ${fmt(cen, 0)}Hz edge ${edge.toExponential(1)} render ${fmt(ms, 0)}ms`,
    );
    if (wav) writeWav(`${WORK}sfx_${name}${variants > 1 ? '_' + k : ''}.wav`, [s.L, s.R], FS);
  }
}
// crash variants must actually differ
{
  const a = renderSfx('crash_heavy', FS, 0).L;
  const b = renderSfx('crash_heavy', FS, 1).L;
  let d = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) d += (a[i] - b[i]) ** 2;
  check('crash variants differ', Math.sqrt(d / a.length) > 0.01, `rms diff ${fmt(dB(Math.sqrt(d / a.length)), 1)}dB`);
  const low = welch(a, 0, a.length, 8192);
  check('crash_heavy has low-end weight but is not all sub-bass', bandPower(low, FS, 25, 160) / bandPower(low, FS, 25, 12000) > 0.15 && bandPower(low, FS, 25, 160) / bandPower(low, FS, 25, 12000) < 0.75, `sub-160Hz share ${fmt(bandPower(low, FS, 25, 160) / bandPower(low, FS, 25, 12000), 2)}`);
}
// deterministic
{
  const a = renderSfx('cheer', FS).L;
  const b = renderSfx('cheer', FS).L;
  let same = a.length === b.length;
  for (let i = 0; same && i < a.length; i++) if (a[i] !== b[i]) same = false;
  check('rendering is deterministic', same);
}
console.log(`  (all SFX: ${fmt(totalSec, 1)}s of audio rendered in ${fmt(total, 0)}ms)`);

console.log('— drum kit —');
let dms = 0;
for (const d of DRUM_NAMES) {
  const t0 = performance.now();
  const x = renderDrum(d, FS);
  dms += performance.now() - t0;
  const pk = peak(x);
  check(`drum ${d}`, allFinite(x) && pk <= 1.0 && pk > 0.2 && Math.abs(x[0]) < 0.01 && Math.abs(x[x.length - 1]) < 0.01, `${fmt(x.length / FS, 2)}s peak ${fmt(dB(pk), 1)}dB rms ${fmt(dB(rms(x)), 1)}dB`);
  if (wav) writeWav(`${WORK}drum_${d}.wav`, [x], FS);
}
console.log(`  (drum kit rendered in ${fmt(dms, 0)}ms)`);

console.log('— ambience loops (seamless) —');
const loops: [string, () => { L: Float32Array; R: Float32Array }, number][] = [
  ['wind', () => renderWindLoop(16000), 16000],
  ['rain', () => renderRainLoop(32000), 32000],
  ['crowd', () => renderCrowdLoop(16000), 16000],
];
for (const [name, fn, fs] of loops) {
  const t0 = performance.now();
  const s = fn();
  const ms = performance.now() - t0;
  const n = s.L.length;
  // boundary step vs. the largest step inside the loop
  const inner = maxStep(s.L, 1, n);
  const wrap = Math.abs(s.L[0] - s.L[n - 1]);
  // click test: high-band energy of a window straddling the loop point vs. the loop average
  const W = 2048;
  const strad = new Float32Array(W);
  for (let i = 0; i < W; i++) strad[i] = s.L[(n - W / 2 + i) % n];
  const pS = welch(strad, 0, W, 1024);
  const pA = welch(s.L, 0, n, 1024);
  const hiS = bandPower(pS, fs, fs * 0.3, fs * 0.48);
  const hiA = bandPower(pA, fs, fs * 0.3, fs * 0.48);
  const ratio = 10 * Math.log10((hiS + 1e-20) / (hiA + 1e-20));
  // stereo decorrelation
  let c = 0;
  for (let i = 0; i < n; i++) c += s.L[i] * s.R[i];
  const corr = c / (n * rms(s.L) * rms(s.R));
  check(
    `loop ${name}`,
    allFinite(s.L) && allFinite(s.R) && peak(s.L) <= 0.51 && wrap <= inner * 1.2 && ratio < 6 && Math.abs(corr) < 0.6,
    `${fmt(n / fs, 1)}s @${fs}Hz peak ${fmt(dB(peak(s.L)), 1)}dB rms ${fmt(dB(rms(s.L)), 1)}dB wrap-step ${fmt(wrap / inner, 2)}x max inner step, HF at seam ${fmt(ratio, 1)}dB vs avg, L/R corr ${fmt(corr, 2)}, render ${fmt(ms, 0)}ms`,
  );
  if (wav) {
    // write two repetitions so the seam can be auditioned
    const L2 = new Float32Array(n * 2);
    const R2 = new Float32Array(n * 2);
    L2.set(s.L);
    L2.set(s.L, n);
    R2.set(s.R);
    R2.set(s.R, n);
    writeWav(`${WORK}loop_${name}_x2.wav`, [L2, R2], fs);
  }
}

console.log('— impulse responses —');
for (const [name, fn] of [
  ['tunnel', renderTunnelIR],
  ['hall', renderHallIR],
] as const) {
  const t0 = performance.now();
  const s = fn(FS);
  const ms = performance.now() - t0;
  const n = s.L.length;
  const head = rms(s.L, 0, Math.round(n * 0.15));
  const tail = rms(s.L, Math.round(n * 0.85), n);
  // RT60 estimate from the energy decay curve (Schroeder integration)
  const edc = new Float64Array(n);
  let acc = 0;
  for (let i = n - 1; i >= 0; i--) {
    acc += s.L[i] * s.L[i];
    edc[i] = acc;
  }
  let i5 = 0;
  let i25 = 0;
  for (let i = 0; i < n; i++) {
    const d = 10 * Math.log10(edc[i] / edc[0] + 1e-30);
    if (!i5 && d <= -5) i5 = i;
    if (!i25 && d <= -25) {
      i25 = i;
      break;
    }
  }
  const rt60 = ((i25 - i5) / FS) * 3;
  check(`IR ${name}`, allFinite(s.L) && allFinite(s.R) && tail < head * 0.05 && Math.abs(s.L[n - 1]) < 1e-3 && rt60 > 0.4 && rt60 < 3.5, `${fmt(n / FS, 2)}s RT60≈${fmt(rt60, 2)}s head/tail ${fmt(dB(head / (tail + 1e-12)), 0)}dB render ${fmt(ms, 0)}ms`);
  if (wav) writeWav(`${WORK}ir_${name}.wav`, [s.L, s.R], FS);
}

process.exit(summary('SFX / ambience') ? 1 : 0);
