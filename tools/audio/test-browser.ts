/**
 * Browser verification of the audio engine (headless Chromium via Playwright, audio muted).
 *   npx tsx tools/audio/test-browser.ts
 * Covers: the real AudioWorklet path (dev and minified production Vite bundles), the ScriptProcessor
 * fallback, pre-init safety, pause ducking, lifecycle, DSP cost in the browser, offline renders of every music cue,
 * a cue crossfade and the complete mix through compressor + limiter. WAVs land in tools/audio/_work/.
 */
import { rms, peak, dB, fmt, allFinite, welch, bandPower, maxStep, spectrum, check, summary, writeWav, WORK } from './lib';
import { ROOT, ENTRY, launch, openPage, decode, bundle } from './browser-lib';
import { MASTER_COMP_TRIM } from '../../src/audio/index';

const st = (L: Float32Array, R: Float32Array): string => `peak ${fmt(dB(Math.max(peak(L), peak(R))), 1)}dBFS rms ${fmt(dB(Math.sqrt((rms(L) ** 2 + rms(R) ** 2) / 2)), 1)}dBFS`;
const rms2 = (L: Float32Array, R: Float32Array): number => Math.sqrt((rms(L) ** 2 + rms(R) ** 2) / 2);

/** Beat tempo from the autocorrelation of the low-band onset envelope. */
function tempo(x: Float32Array, fs: number, lo: number, hi: number): number {
  const hop = 256;
  const k = 1 - Math.exp((-2 * Math.PI * 110) / fs);
  let lp = 0;
  const env: number[] = [];
  let acc = 0;
  for (let i = 0; i < x.length; i++) {
    lp += (x[i] - lp) * k;
    acc += lp * lp;
    if ((i + 1) % hop === 0) {
      env.push(Math.sqrt(acc / hop));
      acc = 0;
    }
  }
  const on = env.map((v, i) => Math.max(0, v - (env[i - 1] ?? v)));
  const rate = fs / hop;
  let best = 0;
  let bestBpm = 0;
  for (let bpm = lo; bpm <= hi; bpm += 0.25) {
    const lag = (60 / bpm) * rate;
    let s = 0;
    for (let m = 1; m <= 4; m++) {
      const l = lag * m;
      const l0 = Math.floor(l);
      const f = l - l0;
      for (let i = 0; i + l0 + 1 < on.length; i++) s += on[i] * (on[i + l0] * (1 - f) + on[i + l0 + 1] * f);
    }
    if (s > best) {
      best = s;
      bestBpm = bpm;
    }
  }
  return bestBpm;
}

/** Pitch-class energy (C=0 … B=11) of a segment, 60–1500 Hz. */
function chroma(x: Float32Array, fs: number, t0: number, t1: number): number[] {
  const N = 16384;
  const pc = new Array<number>(12).fill(0);
  for (let s = Math.round(t0 * fs); s + N <= Math.round(t1 * fs); s += N / 2) {
    const m = spectrum(x, s, N);
    for (let i = 1; i < m.length; i++) {
      const f = (i * fs) / N;
      if (f < 60 || f > 1500) continue;
      const midi = 69 + 12 * Math.log2(f / 440);
      const near = Math.round(midi);
      if (Math.abs(midi - near) > 0.3) continue;
      pc[((near % 12) + 12) % 12] += m[i] * m[i];
    }
  }
  return pc;
}
const NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
function topPcs(pc: number[], k: number): number[] {
  return pc
    .map((v, i) => [v, i])
    .sort((a, b) => b[0] - a[0])
    .slice(0, k)
    .map((q) => q[1]);
}

interface RT {
  info: { ready: boolean; state: string; sampleRate: number; synth: string; voices: number; sfxBuffers: number; music: string | null; initMs: number };
  errors: string[];
  fs: number;
  pcm: string;
  pausedPcm: string;
  soloEnginePcm: string;
  load: { avg: number; peak: number; underrun: number; samples: number } | null;
  workletSrcLen: number;
  frames: number;
}

async function main(): Promise<void> {
  const browser = await launch();
  console.log(`browser: ${browser.browserType().name()} ${browser.version()}`);
  const js = await bundle(false);
  const page = await openPage(browser, js);

  // ── compressor make-up (keeps the master chain at unity for small signals) ──
  const makeup = (await page.evaluate('AT.measureCompressorMakeup()')) as number;
  check('master compressor make-up compensated', Math.abs(dB(makeup * MASTER_COMP_TRIM)) < 1.0, `browser make-up ${fmt(dB(makeup), 2)}dB, trim ${fmt(dB(MASTER_COMP_TRIM), 2)}dB → net ${fmt(dB(makeup * MASTER_COMP_TRIM), 2)}dB`);

  // ── real-time: worklet path, 8 cars + music + ambience + sfx ──
  console.log('— real-time engine (AudioWorklet) —');
  const rt = (await page.evaluate('AT.testRealtime({ seconds: 4, cars: 8 })')) as RT;
  {
    const [L, R] = decode(rt.pcm);
    writeWav(`${WORK}rt_mix_worklet.wav`, [L, R], rt.fs);
    check('init → ready, AudioWorklet synth, context running', rt.info.ready && rt.info.synth === 'worklet' && rt.info.state === 'running', JSON.stringify(rt.info));
    check('no exceptions (pre-init calls, per-frame driving, lifecycle)', rt.errors.length === 0, rt.errors.join(' | ') || `${rt.frames} frames driven`);
    check('all 16 SFX baked', rt.info.sfxBuffers === 16, `${rt.info.sfxBuffers} buffers, init ${rt.info.initMs}ms`);
    check('mix is finite, audible and never clips', allFinite(L) && allFinite(R) && Math.max(peak(L), peak(R)) < 0.999 && dB(rms2(L, R)) > -30 && dB(rms2(L, R)) < -8, `${fmt(L.length / rt.fs, 1)}s captured, ${st(L, R)}`);
    const [pL, pR] = decode(rt.pausedPcm);
    check('setPaused ducks + low-passes the world', dB(rms2(pL, pR)) < dB(rms2(L, R)) - 5, `paused ${st(pL, pR)}`);
    const pwP = welch(pL, 0, pL.length, 4096);
    const pwM = welch(L, 0, L.length, 4096);
    const hfP = bandPower(pwP, rt.fs, 3000, 12000) / bandPower(pwP, rt.fs, 40, 12000);
    const hfM = bandPower(pwM, rt.fs, 3000, 12000) / bandPower(pwM, rt.fs, 40, 12000);
    check('paused mix is duller', hfP < hfM * 0.5, `>3kHz share ${fmt(hfM, 3)} → ${fmt(hfP, 3)}`);
    const [sL, sR] = decode(rt.soloEnginePcm);
    writeWav(`${WORK}rt_engines_worklet.wav`, [sL, sR], rt.fs);
    check('engines alone (worklet output) audible and clean', allFinite(sL) && dB(rms2(sL, sR)) > -32 && Math.max(peak(sL), peak(sR)) < 0.999, st(sL, sR));
    if (rt.load) check('audio thread load, 8 cars + music + ambience', rt.load.avg < 0.35 && rt.load.underrun < 0.02, `avg ${fmt(rt.load.avg * 100, 1)}% peak ${fmt(rt.load.peak * 100, 1)}% underruns ${fmt(rt.load.underrun * 100, 2)}% (${rt.load.samples} samples)`);
    else {
      const b = (await page.evaluate('AT.benchDsp()')) as { typical: number; worst: number };
      check('car synth DSP cost in this browser (8 cars)', b.worst < 80, `typical race ${fmt(b.typical, 1)} ms per second of audio (${fmt(b.typical / 10, 1)}% of the audio thread), worst case ${fmt(b.worst, 1)} ms (${fmt(b.worst / 10, 1)}%)`);
    }
  }

  // ── real-time: ScriptProcessor fallback ──
  console.log('— real-time engine (ScriptProcessor fallback) —');
  {
    const fb = (await page.evaluate('AT.testRealtime({ seconds: 2, cars: 8, noWorklet: true })')) as RT;
    const [L, R] = decode(fb.soloEnginePcm);
    writeWav(`${WORK}rt_engines_fallback.wav`, [L, R], fb.fs);
    check('fallback path selected and running', fb.info.ready && fb.info.synth === 'script' && fb.errors.length === 0, JSON.stringify(fb.info) + (fb.errors.length ? ' ' + fb.errors.join('|') : ''));
    check('fallback engines audible and clean', allFinite(L) && dB(rms2(L, R)) > -32 && Math.max(peak(L), peak(R)) < 0.999, st(L, R));
  }
  await page.close();

  // ── the worklet is shipped as Function.toString(): it must survive the production (minified) build ──
  console.log('— bundler robustness —');
  {
    const p2 = await openPage(browser, await bundle(true));
    const r = (await p2.evaluate('AT.testRealtime({ seconds: 1, cars: 4 })')) as RT;
    const [L, R] = decode(r.soloEnginePcm);
    check('Vite production build (minified): worklet synth works', r.info.synth === 'worklet' && r.errors.length === 0 && dB(rms2(L, R)) > -32, `${JSON.stringify(r.info.synth)} worklet source ${r.workletSrcLen} chars, engines ${st(L, R)}`);
    await p2.close();
  }

  // ── offline music renders ──
  console.log('— music (offline renders) —');
  const p4 = await openPage(browser, js);
  const music = async (expr: string): Promise<{ fs: number; L: Float32Array; R: Float32Array }> => {
    const r = (await p4.evaluate(expr)) as { fs: number; pcm: string };
    const [L, R] = decode(r.pcm);
    return { fs: r.fs, L, R };
  };
  {
    // race: 16 bars at full intensity
    const bar = (60 / 132) * 4;
    const m = await music(`AT.renderMusic('race', ${0.06 + bar * 16 + 0.5}, [[0, 1]])`);
    writeWav(`${WORK}music_race_full.wav`, [m.L, m.R], m.fs);
    const bpm = tempo(m.L, m.fs, 90, 180);
    check('race cue: clean, sane level', allFinite(m.L) && Math.max(peak(m.L), peak(m.R)) < 0.9 && dB(rms2(m.L, m.R)) > -26 && dB(rms2(m.L, m.R)) < -12, st(m.L, m.R));
    check('race cue: tempo 132 bpm', Math.abs(bpm - 132) < 2, `detected ${fmt(bpm, 1)} bpm`);
    const roots = [9, 5, 0, 7, 2, 9, 5, 4];
    const sets = [
      [9, 0, 4, 11],
      [5, 9, 0, 4],
      [0, 4, 7, 2],
      [7, 11, 2, 9],
      [2, 5, 9, 0],
      [9, 0, 4, 7],
      [5, 9, 0, 4],
      [4, 8, 11],
    ];
    let good = 0;
    const seen: string[] = [];
    for (let c = 0; c < 8; c++) {
      const pc = chroma(m.L, m.fs, 0.06 + c * 2 * bar + 0.15, 0.06 + (c + 1) * 2 * bar - 0.1);
      const top = topPcs(pc, 3);
      const inSet = top.filter((q) => sets[c].includes(q)).length;
      if (top.slice(0, 2).includes(roots[c]) && inSet >= 2) good++;
      seen.push(top.map((q) => NOTE[q]).join('/'));
    }
    check('race cue: harmony follows Am F C G | Dm Am F E', good >= 7, `${good}/8 chords — strongest pitch classes per chord: ${seen.join('  ')}`);
    const pw = welch(m.L, 0, m.L.length, 8192);
    const tot = bandPower(pw, m.fs, 30, 16000);
    const lo = bandPower(pw, m.fs, 30, 150) / tot;
    const mid = bandPower(pw, m.fs, 150, 2500) / tot;
    const hi = bandPower(pw, m.fs, 2500, 16000) / tot;
    check('race cue: full-range balance (bass / mids / highs)', lo > 0.15 && lo < 0.85 && mid > 0.08 && hi > 0.003, `bass ${fmt(lo * 100, 0)}% mids ${fmt(mid * 100, 0)}% highs ${fmt(hi * 100, 1)}%`);
  }
  {
    // race: intensity ramp — layers must open up
    const m = await music(`AT.renderMusic('race', 24, [[0, 0], [6, 0.35], [12, 0.7], [18, 1]])`);
    writeWav(`${WORK}music_race_ramp.wav`, [m.L, m.R], m.fs);
    const seg = (a: number, b: number) => {
      const s = Math.round(a * m.fs);
      const e = Math.round(b * m.fs);
      const pw = welch(m.L, s, e - s, 4096);
      return { r: rms(m.L, s, e), hi: bandPower(pw, m.fs, 2500, 16000) };
    };
    const s0 = seg(1.5, 5.5);
    const s1 = seg(7.5, 11.5);
    const s2 = seg(13.5, 17.5);
    const s3 = seg(19.5, 23.5);
    check(
      'race cue: setMusicIntensity opens layers (louder + brighter)',
      s1.r > s0.r * 1.15 && s3.r > s1.r && s3.hi > s0.hi * 20 && s2.hi > s1.hi,
      `rms ${[s0, s1, s2, s3].map((s) => fmt(dB(s.r), 1)).join(' → ')} dB; >2.5kHz ${[s0, s1, s2, s3].map((s) => fmt(10 * Math.log10(s.hi + 1e-20), 0)).join(' → ')} dB`,
    );
    check('race cue: no clicks while layers fade', maxStep(m.L) < 0.5, `max sample step ${fmt(maxStep(m.L), 3)}`);
  }
  {
    const bar = (60 / 84) * 4;
    const m = await music(`AT.renderMusic('menu', ${0.06 + bar * 16 + 1}, [[0, 0.5]])`);
    writeWav(`${WORK}music_menu.wav`, [m.L, m.R], m.fs);
    check('menu cue: clean, sane level', allFinite(m.L) && Math.max(peak(m.L), peak(m.R)) < 0.9 && dB(rms2(m.L, m.R)) > -30 && dB(rms2(m.L, m.R)) < -14, st(m.L, m.R));
    const top = topPcs(chroma(m.L, m.fs, 0.5, 0.06 + 2 * bar - 0.1), 4);
    check('menu cue: opens on D minor', top.includes(2) && top.filter((q) => [2, 5, 9, 0, 4].includes(q)).length >= 3, `strongest pitch classes ${top.map((q) => NOTE[q]).join('/')}`);
    // it should breathe: slow pads, far fewer onsets than the race cue
    let c = 0;
    for (let i = 0; i < m.L.length; i++) c += m.L[i] * m.R[i];
    check('menu cue: wide stereo (reverb / delay)', c / (m.L.length * rms(m.L) * rms(m.R)) < 0.97, `L/R correlation ${fmt(c / (m.L.length * rms(m.L) * rms(m.R)), 2)}`);
  }
  {
    const bar = (60 / 112) * 4;
    const m = await music(`AT.renderMusic('results', ${0.06 + bar * 8 + 0.5}, [[0, 0.5]])`);
    writeWav(`${WORK}music_results.wav`, [m.L, m.R], m.fs);
    const bpm = tempo(m.L, m.fs, 80, 160);
    check('results cue: clean, sane level', allFinite(m.L) && Math.max(peak(m.L), peak(m.R)) < 0.9 && dB(rms2(m.L, m.R)) > -26 && dB(rms2(m.L, m.R)) < -12, st(m.L, m.R));
    check('results cue: tempo 112 bpm', Math.abs(bpm - 112) < 2, `detected ${fmt(bpm, 1)} bpm`);
    const top = topPcs(chroma(m.L, m.fs, 0.1, 0.06 + bar - 0.05), 3);
    check('results cue: opens on C major', top.includes(0) && top.filter((q) => [0, 4, 7].includes(q)).length >= 2, `strongest pitch classes ${top.map((q) => NOTE[q]).join('/')}`);
  }
  {
    const m = await music(`AT.renderCrossfade(14, 6, 1.5)`);
    writeWav(`${WORK}music_crossfade.wav`, [m.L, m.R], m.fs);
    const a = Math.round(5.8 * m.fs);
    const b = Math.round(8.2 * m.fs);
    // the fade region must not contain a discontinuity larger than what the music itself produces elsewhere
    const stepFade = maxStep(m.L, a, b);
    const stepRef = maxStep(m.L, b, m.L.length);
    const gap = Math.min(...Array.from({ length: 20 }, (_, i) => rms(m.L, a + Math.round(i * 0.1 * m.fs), a + Math.round((i + 1) * 0.1 * m.fs))));
    check('cue crossfade (menu → race): no click, no silent gap', stepFade <= stepRef * 1.5 + 0.02 && dB(gap) > -50, `max step in fade ${fmt(stepFade, 3)} vs ${fmt(stepRef, 3)} later; quietest 100ms ${fmt(dB(gap), 1)}dB`);
  }

  // ── the complete mix through compressor + limiter, offline (deterministic) ──
  console.log('— full mix (offline, worklet + buses + compressor + limiter) —');
  {
    const r = (await p4.evaluate('AT.renderFullMix(10, 8)')) as { fs: number; pcm: string; info: RT['info']; errors: string[] };
    const [L, R] = decode(r.pcm);
    writeWav(`${WORK}fullmix_offline.wav`, [L, R], r.fs);
    const pk = Math.max(peak(L), peak(R));
    const a = Math.round(1 * r.fs);
    check('offline engine graph builds (worklet on OfflineAudioContext)', r.info.synth === 'worklet' && r.errors.length === 0, JSON.stringify(r.info) + (r.errors.length ? ' ' + r.errors.slice(0, 2).join('|') : ''));
    check('worst-case mix (8 cars, nitro, crashes, fanfare, cheer, music, rain) stays below 0 dBFS', allFinite(L) && pk < 0.999, st(L, R));
    check('worst-case mix loudness in range', dB(rms2(L.subarray(a), R.subarray(a))) > -22 && dB(rms2(L.subarray(a), R.subarray(a))) < -7, `rms after 1s ${fmt(dB(rms2(L.subarray(a), R.subarray(a))), 1)}dBFS`);
  }
  await p4.close();
  await browser.close();
  process.exit(summary('browser') ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
