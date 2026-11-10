/**
 * Node verification of the car synth DSP (the exact code that runs inside the AudioWorklet).
 *   npx tsx tools/audio/test-dsp.ts
 * Renders steady states, rpm sweeps and scripted drives for every engine type and checks levels, harmonic
 * structure, fundamental tracking, load response, spatial parameters, events, robustness and CPU cost.
 */
import { CarSynthDSP } from '../../src/audio/engine-dsp';
import { engineConfig, ENGINE_VOICINGS, NOMINAL_RPM_MAX, type EngineType } from '../../src/audio/engine-configs';
import { renderVoice, packState, rms, peak, dc, dB, fmt, allFinite, welch, bandPower, centroid, peakFreq, bandProfile, profileDistance, spectrum, check, summary, writeWav, WORK } from './lib';
import { driveScenario } from './scenario';
import { carSynthClassSource } from '../../src/audio/index';

const TYPES: EngineType[] = ['v8', 'v12', 'i6t', 'v10'];
const FS = 48000;
const loud = (r: number, l: number): number => (0.3 + 0.7 * Math.pow(Math.min(1, r), 1.2)) * (0.5 + 0.5 * Math.pow(l, 0.75));

/** Line / inter-line power for the first K harmonics of f0 (+ the local noise floor under each line). */
function harmonicity(pw: Float64Array, fs: number, f0: number, K: number): { H: number; lines: number[]; floor: number[] } {
  const N = (pw.length - 1) * 2;
  const bin = fs / N;
  let P = 0;
  let B = 0;
  const lines: number[] = [0];
  const floor: number[] = [0];
  let prevGap = 0;
  for (let k = 1; k <= K; k++) {
    const f = k * f0;
    const half = Math.max(2 * bin, 0.006 * f);
    const a = Math.round((f - half) / bin);
    const b = Math.round((f + half) / bin);
    const c = Math.round((f + f0 - Math.max(2 * bin, 0.006 * (f + f0))) / bin);
    let p = 0;
    for (let i = a; i <= b; i++) p += pw[i] ?? 0;
    let g = 0;
    for (let i = b + 1; i < c; i++) g += pw[i] ?? 0;
    const gapPerBin = c - b - 1 > 0 ? g / (c - b - 1) : 0;
    B += g;
    lines.push(p);
    floor.push(((k === 1 ? gapPerBin : (prevGap + gapPerBin) / 2) * (b - a + 1)));
    prevGap = gapPerBin;
    P += p;
  }
  return { H: P / (P + B), lines, floor };
}

// ───────────────────────── A. steady states ─────────────────────────
console.log('— steady states: level calibration, harmonic structure, fundamental —');
const profiles: Record<string, number[]> = {};
const burble: Record<string, number> = {};
for (const type of TYPES) {
  const rm = NOMINAL_RPM_MAX[type];
  const cyl = ENGINE_VOICINGS[type].cyl;
  let worstLvl = 0;
  let minH = 1;
  let maxH = 0;
  let pk = 0;
  let fin = true;
  let dcMax = 0;
  let fundErr = 0;
  let fundRank = 0;
  const notes: string[] = [];
  for (const norm of [0.15, 0.35, 0.55, 0.75, 0.95]) {
    for (const load of [1, 0]) {
      const rpm = norm * rm;
      const r = renderVoice({ type, dur: 3.6, state: () => ({ rpm, norm, load }) });
      const a = Math.round(0.6 * FS);
      fin = fin && allFinite(r.eL) && allFinite(r.eR);
      pk = Math.max(pk, peak(r.eL, a), peak(r.eR, a));
      dcMax = Math.max(dcMax, Math.abs(dc(r.eL, a)));
      const lvl = Math.sqrt((rms(r.eL, a) ** 2 + rms(r.eR, a) ** 2) / 2);
      const err = dB(lvl / (0.2 * loud(norm, load)));
      if (Math.abs(err) > Math.abs(worstLvl)) worstLvl = err;
      if (load === 1 && norm >= 0.35) {
        const pw = welch(r.eL, a, r.eL.length - a, 65536);
        const fc = rpm / 120;
        const h = harmonicity(pw, FS, fc, 25);
        minH = Math.min(minH, h.H);
        maxH = Math.max(maxH, h.H);
        // the firing line (k = cyl) must be one of the strongest and sit at the firing frequency
        const sorted = h.lines.map((p, k) => [p, k]).sort((x, y) => y[0] - x[0]);
        const rank = sorted.findIndex((q) => q[1] === cyl) + 1;
        fundRank = Math.max(fundRank, rank);
        const pf = peakFreq(pw, FS, fc * cyl * 0.94, fc * cyl * 1.06);
        fundErr = Math.max(fundErr, Math.abs(pf.f / (fc * cyl) - 1));
        if (norm === 0.55) {
          profiles[type] = bandProfile(pw, FS);
          // sub-firing half-order lines within 20 dB of the firing line (the cross-plane "burble" signature)
          let cnt = 0;
          for (let k = 1; k < cyl; k++) if (h.lines[k] > h.lines[cyl] * 0.01 && h.lines[k] > h.floor[k] * 5) cnt++;
          burble[type] = cnt;
          notes.push(`@${Math.round(rpm)}rpm firing ${fmt(fc * cyl, 0)}Hz, top lines (engine order): ${sorted.slice(0, 5).map((q) => fmt(q[1] / 2, 1)).join(', ')}`);
        }
      }
    }
  }
  check(`${type}: finite, no clipping, no DC`, fin && pk < 0.95 && dcMax < 0.004, `peak ${fmt(dB(pk), 1)}dBFS, |dc| ${dcMax.toExponential(1)}`);
  check(`${type}: loudness curve hit on the whole rpm × load grid`, Math.abs(worstLvl) < 1.5, `worst deviation ${fmt(worstLvl, 2)}dB (idle/overrun ${fmt(dB(0.2 * loud(0.15, 0)), 1)} … redline/WOT ${fmt(dB(0.2 * loud(0.95, 1)), 1)} dBFS)`);
  check(`${type}: harmonic but not a pure tone (order lines over a noise floor)`, minH > 0.45 && maxH < 0.985, `line-power share ${fmt(minH, 2)}…${fmt(maxH, 2)} of the first 25 half-orders`);
  check(`${type}: fundamental sits on the firing frequency`, fundErr < 0.01 && fundRank <= 3, `max error ${fmt(fundErr * 100, 2)}%, firing line rank ≤ ${fundRank}; ${notes.join('; ')}`);
}
{
  let minD = 99;
  const ds: string[] = [];
  for (let i = 0; i < TYPES.length; i++)
    for (let j = i + 1; j < TYPES.length; j++) {
      const d = profileDistance(profiles[TYPES[i]], profiles[TYPES[j]]);
      minD = Math.min(minD, d);
      ds.push(`${TYPES[i]}/${TYPES[j]} ${fmt(d, 1)}`);
    }
  check('engine types have distinct spectra', minD > 2.5, `third-octave shape distance (dB rms): ${ds.join(', ')}`);
  check('cross-plane V8 burbles (≥2 strong sub-firing half-order lines), the others are smoother', burble.v8 >= 2 && burble.v8 > burble.v12 && burble.v8 > burble.i6t && burble.v8 > burble.v10, 'sub-firing lines within 20dB of the firing line: ' + TYPES.map((t) => `${t} ${burble[t]}`).join(', '));
}

// ───────────────────────── B. sweeps ─────────────────────────
console.log('— rpm sweeps: fundamental tracking, zipper noise —');
for (const type of TYPES) {
  const rm = NOMINAL_RPM_MAX[type];
  const cyl = ENGINE_VOICINGS[type].cyl;
  const dur = 8;
  const rpmAt = (t: number): number => rm * (0.15 + 0.83 * Math.min(1, t / (dur - 0.5)));
  const sweep = (hz: number) => renderVoice({ type, dur, updateHz: hz, state: (t) => ({ rpm: rpmAt(t), norm: rpmAt(t) / rm, load: 1 }) });
  const r = sweep(60);
  const N = 8192;
  const errs: number[] = [];
  let top = 0;
  let frames = 0;
  for (let s = Math.round(0.5 * FS); s + N < r.eL.length; s += 4096) {
    const m = spectrum(r.eL, s, N);
    const pw = new Float64Array(m.length);
    for (let i = 0; i < m.length; i++) pw[i] = m[i] * m[i];
    const tc = (s + N / 2) / FS - 0.03; // smoothing lag of the rpm input
    const fFire = (rpmAt(tc) / 120) * cyl;
    const pf = peakFreq(pw, FS, fFire * 0.93, fFire * 1.07);
    errs.push(Math.abs(pf.f / fFire - 1));
    const mx = peakFreq(pw, FS, 35, 5000);
    // strongest partial of the frame is the firing line or one of its first harmonics / the bank half-order
    const ratio = mx.f / fFire;
    if ([0.5, 1, 2, 3].some((q) => Math.abs(ratio / q - 1) < 0.04)) top++;
    frames++;
  }
  errs.sort((a, b) => a - b);
  const med = errs[errs.length >> 1];
  check(`${type}: spectral fundamental tracks firing frequency over the sweep`, med < 0.012 && errs[Math.floor(errs.length * 0.9)] < 0.03 && top / frames > 0.8, `median error ${fmt(med * 100, 2)}%, p90 ${fmt(errs[Math.floor(errs.length * 0.9)] * 100, 2)}%, strongest partial is a firing harmonic in ${fmt((top / frames) * 100, 0)}% of ${frames} frames`);
  if (type === 'v8' || type === 'v12') {
    const fine = sweep(375);
    const a = Math.round(0.5 * FS);
    const p60 = bandProfile(welch(r.eL, a, r.eL.length - a, 8192), FS);
    const p375 = bandProfile(welch(fine.eL, a, fine.eL.length - a, 8192), FS);
    const hf60 = bandPower(welch(r.eL, a, r.eL.length - a, 8192), FS, 9000, 20000);
    const hf375 = bandPower(welch(fine.eL, a, fine.eL.length - a, 8192), FS, 9000, 20000);
    check(`${type}: no zipper noise from 60 Hz state updates`, profileDistance(p60, p375) < 0.5 && Math.abs(10 * Math.log10(hf60 / hf375)) < 1, `spectrum vs. per-block updates differs ${fmt(profileDistance(p60, p375), 2)}dB, >9kHz ${fmt(10 * Math.log10(hf60 / hf375), 2)}dB`);
  }
  if (type === 'v10') writeWav(`${WORK}sweep_${type}.wav`, [r.eL, r.eR], FS);
}

// ───────────────────────── C. load response ─────────────────────────
console.log('— throttle on / off —');
for (const type of TYPES) {
  const rm = NOMINAL_RPM_MAX[type];
  const st = (load: number) => {
    const r = renderVoice({ type, dur: 2.6, state: () => ({ rpm: 0.6 * rm, norm: 0.6, load }) });
    const a = Math.round(0.6 * FS);
    const pw = welch(r.eL, a, r.eL.length - a, 8192);
    return { r: rms(r.eL, a), c: centroid(pw, FS, 30, 12000), hi: bandPower(pw, FS, 1500, 12000) / bandPower(pw, FS, 30, 12000) };
  };
  const on = st(1);
  const off = st(0);
  check(`${type}: on-throttle is louder and brighter than overrun`, dB(on.r / off.r) > 5 && on.c > off.c * 1.4, `level +${fmt(dB(on.r / off.r), 1)}dB, centroid ${fmt(off.c, 0)}→${fmt(on.c, 0)}Hz, >1.5kHz share ${fmt(off.hi * 100, 1)}→${fmt(on.hi * 100, 1)}%`);
}

// ───────────────────────── F. events / scripted drive ─────────────────────────
console.log('— scripted drives & events —');
for (const type of TYPES) {
  const sc = driveScenario(type);
  const r = renderVoice({ type, dur: sc.dur, state: sc.state, events: sc.events });
  const mixL = r.eL.map((v, i) => v + r.tL[i]);
  const mixR = r.eR.map((v, i) => v + r.tR[i]);
  writeWav(`${WORK}drive_${type}.wav`, [mixL, mixR], FS);
  check(`${type}: full drive (shifts, limiter, overrun, backfire, nitro) is finite and below full scale`, allFinite(mixL) && allFinite(mixR) && Math.max(peak(mixL), peak(mixR)) < 0.98, `peak ${fmt(dB(Math.max(peak(mixL), peak(mixR))), 1)}dBFS rms ${fmt(dB(rms(mixL)), 1)}dBFS`);
}
{
  // limiter: ignition-cut stutter → strong amplitude modulation at ~27 Hz
  const rm = NOMINAL_RPM_MAX.v10;
  const r = renderVoice({ type: 'v10', dur: 2.5, state: () => ({ rpm: rm, norm: 1.0, load: 1 }) });
  const hop = 240;
  const env: number[] = [];
  for (let s = Math.round(0.8 * FS); s + hop <= r.eL.length; s += hop) env.push(rms(r.eL, s, s + hop));
  const mean = env.reduce((a, b) => a + b, 0) / env.length;
  const depth = (Math.max(...env) - Math.min(...env)) / (2 * mean);
  // modulation frequency via zero crossings of the centred envelope
  let zc = 0;
  for (let i = 1; i < env.length; i++) if (env[i - 1] - mean < 0 !== env[i] - mean < 0) zc++;
  const fmod = zc / 2 / ((env.length * hop) / FS);
  check('rev limiter stutters at redline', depth > 0.3 && fmod > 15 && fmod < 45, `envelope depth ${fmt(depth * 100, 0)}%, ≈${fmt(fmod, 0)}Hz`);
  const quiet = renderVoice({ type: 'v10', dur: 2.5, state: () => ({ rpm: rm * 0.9, norm: 0.9, load: 1 }) });
  const env2: number[] = [];
  for (let s = Math.round(0.8 * FS); s + hop <= quiet.eL.length; s += hop) env2.push(rms(quiet.eL, s, s + hop));
  const mean2 = env2.reduce((a, b) => a + b, 0) / env2.length;
  check('…and not below it', (Math.max(...env2) - Math.min(...env2)) / (2 * mean2) < depth * 0.6, `depth at 90% rpm ${fmt(((Math.max(...env2) - Math.min(...env2)) / (2 * mean2)) * 100, 0)}%`);
}
{
  // shift_up: torque cut dip; backfire: burst over the overrun level; blow-off: HF burst on lift (turbo)
  const rm = NOMINAL_RPM_MAX.v8;
  const r = renderVoice({ type: 'v8', dur: 3, state: () => ({ rpm: 0.8 * rm, norm: 0.8, load: 1 }), events: [{ t: 1.5, code: 0 }] });
  const w = (a: number, b: number) => rms(r.eL, Math.round(a * FS), Math.round(b * FS));
  check('shift_up cuts torque briefly', dB(w(1.53, 1.59) / w(1.2, 1.45)) < -3 && dB(w(1.9, 2.3) / w(1.2, 1.45)) > -1, `level during cut ${fmt(dB(w(1.53, 1.59) / w(1.2, 1.45)), 1)}dB, recovered ${fmt(dB(w(1.9, 2.3) / w(1.2, 1.45)), 1)}dB`);
  const b = renderVoice({ type: 'v8', dur: 3, state: () => ({ rpm: 0.55 * rm, norm: 0.55, load: 0 }), events: [{ t: 1.5, code: 2, val: 1 }] });
  const wb = (a: number, c: number) => peak(b.eL, Math.round(a * FS), Math.round(c * FS));
  check('backfire pops over the overrun bed', dB(wb(1.5, 1.75) / wb(1.0, 1.45)) > 4, `peak +${fmt(dB(wb(1.5, 1.75) / wb(1.0, 1.45)), 1)}dB`);
  const d = renderVoice({ type: 'v8', dur: 3, state: () => ({ rpm: 0.55 * rm, norm: 0.55, load: 0 }), events: [{ t: 1.5, code: 1 }] });
  const wd = (a: number, c: number) => rms(d.eL, Math.round(a * FS), Math.round(c * FS));
  check('shift_down blips the throttle', dB(wd(1.52, 1.62) / wd(1.0, 1.45)) > 3, `blip +${fmt(dB(wd(1.52, 1.62) / wd(1.0, 1.45)), 1)}dB`);
  const rt = NOMINAL_RPM_MAX.i6t;
  const t = renderVoice({ type: 'i6t', dur: 4, state: (tt) => ({ rpm: 0.8 * rt, norm: 0.8, load: tt < 2.5 ? 1 : 0 }) });
  const hf = (a: number, c: number) => {
    const s = Math.round(a * FS);
    return bandPower(welch(t.eL, s, Math.round(c * FS) - s, 2048), FS, 1500, 9000);
  };
  check('turbo: whistle under boost, blow-off on lift', 10 * Math.log10(hf(2.55, 2.85) / hf(3.3, 3.9)) > 6, `HF burst after lift +${fmt(10 * Math.log10(hf(2.55, 2.85) / hf(3.3, 3.9)), 1)}dB over settled overrun`);
  const n = renderVoice({ type: 'v12', dur: 3, state: (tt) => ({ rpm: 5000, norm: 0.57, load: 0.3, nitro: tt > 1.5 ? 1 : 0 }) });
  const wn = (a: number, c: number) => rms(n.eL, Math.round(a * FS), Math.round(c * FS));
  check('nitro adds a whoosh layer', dB(wn(2, 2.9) / wn(0.6, 1.4)) > 1.5, `+${fmt(dB(wn(2, 2.9) / wn(0.6, 1.4)), 1)}dB`);
}

// ───────────────────────── G. cheap voices / spatial parameters ─────────────────────────
console.log('— non-player voices: level, pan, Doppler, air absorption —');
for (const type of TYPES) {
  const rm = NOMINAL_RPM_MAX[type];
  const cyl = ENGINE_VOICINGS[type].cyl;
  const a = Math.round(0.6 * FS);
  const base = { rpm: 0.7 * rm, norm: 0.7, load: 1 };
  const c = renderVoice({ type, player: false, dur: 2.6, state: () => ({ ...base, gain: 1, pan: 0 }) });
  const lvlErr = dB(rms(c.eL, a) / (0.2 * loud(0.7, 1) * Math.SQRT1_2));
  const right = renderVoice({ type, player: false, dur: 2.6, state: () => ({ ...base, gain: 1, pan: 1 }) });
  const dop = renderVoice({ type, player: false, dur: 2.6, state: () => ({ ...base, gain: 1, pan: 0, pitch: 1.2 }) });
  const fF = (0.7 * rm * cyl) / 120;
  const f1 = peakFreq(welch(c.eL, a, c.eL.length - a, 32768), FS, fF * 0.9, fF * 1.1).f;
  const f2 = peakFreq(welch(dop.eL, a, dop.eL.length - a, 32768), FS, fF * 1.1, fF * 1.3).f;
  const far = renderVoice({ type, player: false, dur: 2.6, state: () => ({ ...base, gain: 1, pan: 0, air: 0.1 }) });
  const hs = (x: Float32Array) => {
    const pw = welch(x, a, x.length - a, 4096);
    return bandPower(pw, FS, 2000, 12000) / bandPower(pw, FS, 30, 12000);
  };
  check(
    `${type} (cheap voice): calibrated level, equal-power pan, Doppler, distance low-pass`,
    Math.abs(lvlErr) < 1.5 && dB(rms(right.eL, a) / rms(right.eR, a)) < -40 && Math.abs(f2 / f1 - 1.2) < 0.012 && hs(far.eL) < hs(c.eL) * 0.35,
    `level ${fmt(lvlErr, 2)}dB vs target, hard-right L/R ${fmt(dB(rms(right.eL, a) / rms(right.eR, a)), 0)}dB, pitch×1.2 → fundamental ×${fmt(f2 / f1, 3)}, >2kHz share ${fmt(hs(c.eL) * 100, 1)}%→${fmt(hs(far.eL) * 100, 1)}% when far`,
  );
}

// ───────────────────────── H. tyre / surface layers ─────────────────────────
console.log('— tyre & surface layers —');
{
  const base = { rpm: 3000, norm: 0.4, load: 0.2 };
  const lvl = (extra: object, dur = 2.5) => {
    const r = renderVoice({ type: 'v8', dur, state: () => ({ ...base, ...extra }) });
    const a = Math.round(0.8 * FS);
    const pw = welch(r.tL, a, r.tL.length - a, 4096);
    return { r: Math.sqrt((rms(r.tL, a) ** 2 + rms(r.tR, a) ** 2) / 2), c: centroid(pw, FS, 30, 14000), pk: Math.max(peak(r.tL, a), peak(r.tR, a)), fin: allFinite(r.tL) && allFinite(r.tR) };
  };
  const still = lvl({ speed: 0 });
  const road = lvl({ speed: 70 });
  const sq = lvl({ speed: 30, slip: 1 });
  const sqHalf = lvl({ speed: 30, slip: 0.5 });
  const kerb = lvl({ speed: 40, surface: 1 });
  const grass = lvl({ speed: 40, surface: 2 });
  const gravel = lvl({ speed: 40, surface: 3 });
  const scrape = lvl({ speed: 40, scrape: 1 });
  const row = (n: string, x: { r: number; c: number; pk: number }) => `${n} rms ${fmt(dB(x.r), 1)} peak ${fmt(dB(x.pk), 1)}dB ${fmt(x.c, 0)}Hz`;
  check('standing still on asphalt: tyre bus silent', still.r < 1e-4, `${fmt(dB(still.r), 0)}dB`);
  check(
    'every layer audible, finite and within headroom',
    [road, sq, kerb, grass, gravel, scrape].every((x) => x.fin && x.pk < 0.9 && dB(x.r) > -40 && dB(x.r) < -12),
    [row('road+wind@70m/s', road), row('squeal', sq), row('kerb', kerb), row('grass', grass), row('gravel', gravel), row('scrape', scrape)].join(', '),
  );
  check('squeal grows with slip and is a mid-band screech', sq.r > sqHalf.r * 1.6 && sq.c > 700 && sq.c < 3200, `slip 0.5 → 1: ${fmt(dB(sqHalf.r), 1)} → ${fmt(dB(sq.r), 1)}dB, centroid ${fmt(sq.c, 0)}Hz`);
  check('surfaces sound different (kerb = low buzz, grass = dull, gravel = bright crackle, scrape = metallic)', kerb.c < grass.c * 1.2 && gravel.c > grass.c * 1.5 && scrape.c > 1200, `centroids kerb ${fmt(kerb.c, 0)} / grass ${fmt(grass.c, 0)} / gravel ${fmt(gravel.c, 0)} / scrape ${fmt(scrape.c, 0)} Hz`);
}

// ───────────────────────── I/J. robustness ─────────────────────────
console.log('— robustness —');
{
  // watchdog: updates stop → the voice fades out instead of droning forever
  const r = renderVoice({ type: 'v12', dur: 3, updateHz: 60, state: (t) => ({ rpm: 6000, norm: 0.7, load: 1, active: 1, gain: t < 1 ? 1 : NaN }) });
  // (NaN gain is sanitised to 0 → also covers bad input)
  check('NaN inputs are sanitised', allFinite(r.eL) && rms(r.eL, 2.5 * FS) < 1e-4, `tail ${fmt(dB(rms(r.eL, 2.5 * FS)), 0)}dB`);
  const dsp = new CarSynthDSP(FS, 2);
  dsp.configure(0, engineConfig('v8'), true, 0);
  const q = [0, 1, 2, 3].map(() => new Float32Array(128));
  dsp.setStates(packState(0, { rpm: 5000, norm: 0.7, load: 1 }, 7200));
  const out = new Float32Array(FS * 2);
  for (let p = 0; p < out.length; p += 128) {
    dsp.process(q[0], q[1], q[2], q[3], 128);
    out.set(q[0], p);
  }
  check('watchdog: voice fades out when the game stops sending updates', rms(out, 0.1 * FS, 0.4 * FS) > 0.05 && rms(out, 1.2 * FS, 2 * FS) < 1e-4, `first 0.4s ${fmt(dB(rms(out, 0.1 * FS, 0.4 * FS)), 1)}dB → after 1.2s ${fmt(dB(rms(out, 1.2 * FS, 2 * FS)), 0)}dB`);
  // inactive voices
  const off = renderVoice({ type: 'v8', dur: 2, state: () => ({ rpm: 5000, norm: 0.7, load: 1, active: 0, slip: 1, speed: 50 }) });
  check('setActive(false) → silent', rms(off.eL, 0.5 * FS) < 1e-5 && rms(off.tL, 0.5 * FS) < 1e-5);
  // extreme inputs
  const ex = renderVoice({ type: 'v10', dur: 2, state: (t) => ({ rpm: t < 1 ? 1e9 : -5, norm: t < 1 ? 50 : -3, load: 9, speed: 1e6, slip: 7, scrape: 5, nitro: 3, surface: 99, gain: 50, pan: -9, pitch: 100, air: -4 }) });
  check('absurd inputs stay finite and bounded', allFinite(ex.eL) && allFinite(ex.tL) && peak(ex.eL) < 4 && peak(ex.tL) < 4, `peaks ${fmt(peak(ex.eL), 2)} / ${fmt(peak(ex.tL), 2)} (gain clamps to 2)`);
  // sample-rate independence
  const a48 = renderVoice({ type: 'v8', dur: 2.5, fs: 48000, state: () => ({ rpm: 4500, norm: 0.625, load: 1 }) });
  const a44 = renderVoice({ type: 'v8', dur: 2.5, fs: 44100, state: () => ({ rpm: 4500, norm: 0.625, load: 1 }) });
  const f48 = peakFreq(welch(a48.eL, 24000, a48.eL.length - 24000, 32768), 48000, 280, 320).f;
  const f44 = peakFreq(welch(a44.eL, 22050, a44.eL.length - 22050, 32768), 44100, 280, 320).f;
  check('same pitch and level at 44.1 kHz and 48 kHz', Math.abs(dB(rms(a48.eL, 24000) / rms(a44.eL, 22050))) < 1 && Math.abs(f48 / f44 - 1) < 0.004, `level Δ ${fmt(dB(rms(a48.eL, 24000) / rms(a44.eL, 22050)), 2)}dB, fundamental ${fmt(f48, 1)} vs ${fmt(f44, 1)}Hz`);
}

// ───────────────────────── K. worklet source is self-contained ─────────────────────────
console.log('— worklet shipping —');
{
  const cs = carSynthClassSource();
  const src = cs ? cs.cls : '';
  let ok = false;
  let detail = '';
  try {
    if (!cs) throw new Error('class source unavailable');
    const Cls = new Function(`${cs.prelude}return (${cs.cls});`)() as typeof CarSynthDSP;
    const run = (C: typeof CarSynthDSP): Float32Array => {
      const d = new C(FS, 2);
      d.configure(0, engineConfig('v10'), true, 0);
      d.configure(1, engineConfig('i6t'), false, 0);
      const q = [0, 1, 2, 3].map(() => new Float32Array(128));
      const out = new Float32Array(FS);
      for (let p = 0; p < out.length; p += 128) {
        if (p % 768 === 0) {
          d.setStates(packState(0, { rpm: 3000 + p / 10, norm: 0.5, load: 1, slip: 0.6, speed: 40 }, 8800));
          d.setStates(packState(1, { rpm: 6000, norm: 0.8, load: 0.5, gain: 0.7, pan: 0.3, pitch: 1.1, air: 0.6, scrape: 0.5, speed: 30 }, 7600));
        }
        if (p === 9600) d.event(0, 2, 1);
        d.process(q[0], q[1], q[2], q[3], 128);
        for (let i = 0; i < 128; i++) out[p + i] = q[0][i] + q[1][i] + q[2][i] + q[3][i];
      }
      return out;
    };
    const a = run(CarSynthDSP);
    const b = run(Cls);
    let same = true;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) same = false;
    ok = same && rms(a) > 0.01;
    detail = `${src.length} chars${cs.prelude ? `, helper stubs added for this build: ${cs.prelude.trim().replace(/\n/g, ' ')}` : ''}; class re-created from text renders bit-identical output (rms ${fmt(dB(rms(a)), 1)}dB)`;
  } catch (e) {
    detail = String(e);
  }
  check('CarSynthDSP.toString() is a self-contained class', ok && /^\s*class\b/.test(src), detail);
}

// ───────────────────────── L. CPU ─────────────────────────
console.log('— DSP cost —');
{
  const dsp = new CarSynthDSP(FS, 12);
  const types: EngineType[] = ['v8', 'v12', 'i6t', 'v10', 'v8', 'v12', 'i6t', 'v10'];
  types.forEach((t, i) => dsp.configure(i, engineConfig(t), i === 0, 0));
  const q = [0, 1, 2, 3].map(() => new Float32Array(128));
  const st = (t: number): void => {
    types.forEach((ty, i) => {
      const rm = NOMINAL_RPM_MAX[ty];
      dsp.setStates(packState(i, { rpm: rm * (0.5 + 0.45 * Math.abs(Math.sin(t * 0.7 + i))), norm: 0.5 + 0.45 * Math.abs(Math.sin(t * 0.7 + i)), load: 1, speed: 60, slip: 0.7, scrape: i % 3 === 0 ? 0.6 : 0, surface: i % 4, nitro: i === 0 ? 1 : 0, gain: i === 0 ? 1 : 0.5, pan: Math.sin(i), pitch: 1 + 0.05 * Math.sin(t + i), air: 0.8 }, rm));
    });
  };
  const secs = 20;
  // warm-up for the JIT
  for (let p = 0; p < FS; p += 128) {
    if (p % 768 === 0) st(p / FS);
    dsp.process(q[0], q[1], q[2], q[3], 128);
  }
  const t0 = performance.now();
  let pk = 0;
  for (let p = 0; p < FS * secs; p += 128) {
    if (p % 768 === 0) st(p / FS);
    dsp.process(q[0], q[1], q[2], q[3], 128);
    pk = Math.max(pk, peak(q[0]));
  }
  const ms = (performance.now() - t0) / secs;
  check('8 cars (1 rich player voice + 7 cheap), everything active incl. squeal/scrape/surfaces/nitro', ms < 60, `${fmt(ms, 2)} ms of CPU per second of audio = ${fmt(ms / 10, 2)}% of one core (${fmt(1000 / ms, 0)}× real time), worst quantum budget use ≈ ${fmt((ms / 1000) * 100, 2)}%`);
  // all voices inaudible → near-zero cost
  types.forEach((ty, i) => dsp.setStates(packState(i, { rpm: 5000, norm: 0.6, load: 1, active: 0 }, 8000)));
  for (let p = 0; p < FS; p += 128) dsp.process(q[0], q[1], q[2], q[3], 128);
  const t1 = performance.now();
  for (let p = 0; p < FS * secs; p += 128) {
    if (p % 4800 === 0) types.forEach((ty, i) => dsp.setStates(packState(i, { rpm: 5000, norm: 0.6, load: 1, active: 0 }, 8000)));
    dsp.process(q[0], q[1], q[2], q[3], 128);
  }
  const msIdle = (performance.now() - t1) / secs;
  check('inactive voices are nearly free', msIdle < ms * 0.2, `${fmt(msIdle, 2)} ms per second`);
}

process.exit(summary('DSP') ? 1 : 0);
