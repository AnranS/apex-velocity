/**
 * In-page side of tools/audio/test-browser.ts (bundled to an IIFE exposing `AT`).
 * Everything here runs inside headless Chromium against the real Web Audio implementation.
 */
import { AudioEngine, carSynthWorkletSource, type CarAudioState, type CarVoice, type EngineType, type SfxName } from '../../src/audio/index';
import { CarSynthDSP } from '../../src/audio/engine-dsp';
import { engineConfig } from '../../src/audio/engine-configs';
import { MusicSequencer, buildMusicBank, type MusicCue } from '../../src/audio/music';

function b64(chans: Float32Array[]): string {
  const n = chans[0].length;
  const ch = chans.length;
  const i16 = new Int16Array(n * ch);
  for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) i16[i * ch + c] = Math.max(-32768, Math.min(32767, Math.round(chans[c][i] * 32767)));
  const u8 = new Uint8Array(i16.buffer);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  return btoa(s);
}
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

const REC_SRC = `class RecTap extends AudioWorkletProcessor {
  process(inputs) { const i = inputs[0]; if (i && i[0]) this.port.postMessage([i[0].slice(0), (i[1] || i[0]).slice(0)]); return true; }
}
registerProcessor('rec-tap', RecTap);`;

async function makeTap(ctx: AudioContext, node: AudioNode): Promise<{ start(): void; stop(): Float32Array[] }> {
  const url = URL.createObjectURL(new Blob([REC_SRC], { type: 'application/javascript' }));
  await ctx.audioWorklet.addModule(url);
  const rec = new AudioWorkletNode(ctx, 'rec-tap', { numberOfInputs: 1, numberOfOutputs: 0, channelCount: 2, channelCountMode: 'explicit' });
  node.connect(rec);
  let on = false;
  let L: Float32Array[] = [];
  let R: Float32Array[] = [];
  rec.port.onmessage = (e: MessageEvent) => {
    if (!on) return;
    L.push(e.data[0]);
    R.push(e.data[1]);
  };
  const cat = (xs: Float32Array[]): Float32Array => {
    const out = new Float32Array(xs.reduce((a, x) => a + x.length, 0));
    let o = 0;
    for (const x of xs) {
      out.set(x, o);
      o += x.length;
    }
    return out;
  };
  return {
    start: () => {
      L = [];
      R = [];
      on = true;
    },
    stop: () => {
      on = false;
      return [cat(L), cat(R)];
    },
  };
}

const TYPES: EngineType[] = ['v8', 'v12', 'i6t', 'v10'];
const RPM_MAX: Record<EngineType, number> = { v8: 7200, v12: 8800, i6t: 7600, v10: 8800 };

/** Deterministic test traffic: car 0 is the player, the others orbit / pass the listener. */
function carState(i: number, t: number, type: EngineType): CarAudioState {
  const rm = RPM_MAX[type];
  if (i === 0) {
    const g = (t % 2.2) / 2.2;
    return { rpm: rm * (0.45 + 0.52 * g), rpmMax: rm, throttle: 1, speed: 25 + 30 * g, nitro: t % 6 > 4.5 ? 1 : 0, slip: t % 3 > 2.5 ? 0.8 : 0, surface: t % 5 > 4.4 ? 1 : 0, scrape: 0, x: 0, y: 0.5, z: 0, vx: 0, vy: 0, vz: -40 };
  }
  const w = 0.5 + 0.13 * i;
  const r = 9 + 7 * i;
  const a = t * w + i * 1.7;
  const x = Math.sin(a) * r;
  const z = Math.cos(a) * r * 1.6 - 5;
  const vx = Math.cos(a) * r * w;
  const vz = -Math.sin(a) * r * 1.6 * w - 40;
  const g = ((t * (0.7 + 0.1 * i) + i * 0.37) % 2) / 2;
  return { rpm: rm * (0.4 + 0.55 * g), rpmMax: rm, throttle: g < 0.85 ? 0.95 : 0, speed: 45, nitro: 0, slip: i === 3 ? 0.5 : 0, surface: i === 5 ? 3 : 0, scrape: i === 2 && t % 4 > 3.6 ? 0.8 : 0, x, y: 0.5, z, vx, vy: 0, vz };
}

export interface RealtimeResult {
  info: ReturnType<AudioEngine['debugInfo']>;
  errors: string[];
  fs: number;
  pcm: string;
  pausedPcm: string;
  soloEnginePcm: string;
  load: { avg: number; peak: number; underrun: number; samples: number } | null;
  workletSrcLen: number;
  frames: number;
}

/** Full real-time path: init → 8 cars + music + ambience + SFX → record the output of the limiter. */
export async function testRealtime(opts: { noWorklet?: boolean; seconds: number; cars: number }): Promise<RealtimeResult> {
  const errors: string[] = [];
  const eng = new AudioEngine();
  const voices: { v: CarVoice; type: EngineType }[] = [];
  // every API call must be a safe no-op before init()
  try {
    eng.setVolumes({ master: 1, music: 0.7, sfx: 1, engine: 1 });
    eng.setListener({ x: 0, y: 1, z: 6, fx: 0, fy: 0, fz: -1, ux: 0, uy: 1, uz: 0, vx: 0, vy: 0, vz: -40 });
    eng.play('ui_move');
    eng.setAmbience({ wind: 0.5, rain: 0.4, crowd: 0.5, tunnel: 0.2 });
    eng.playMusic('race', 0.3);
    eng.setMusicIntensity(1);
    eng.setPaused(false);
    await eng.resume();
    const v0 = eng.createCar('v8', true);
    v0.update(carState(0, 0, 'v8'));
    v0.event('shift_up');
    v0.impact(0.5);
    v0.setActive(true);
    voices.push({ v: v0, type: 'v8' });
    if (eng.ready) errors.push('ready before init');
  } catch (e) {
    errors.push('pre-init: ' + String(e));
  }
  await eng.init({ noWorklet: !!opts.noWorklet });
  await eng.init(); // idempotent
  const ctx = eng.context as AudioContext;
  const out = eng.outputNode;
  if (!ctx || !out) {
    return { info: eng.debugInfo(), errors: [...errors, 'no context'], fs: 0, pcm: '', pausedPcm: '', soloEnginePcm: '', load: null, workletSrcLen: 0, frames: 0 };
  }
  const tap = await makeTap(ctx, out);
  for (let i = 1; i < opts.cars; i++) {
    const type = TYPES[i % 4];
    voices.push({ v: eng.createCar(type, false), type });
  }
  // audio-thread load (Chrome only)
  let load: RealtimeResult['load'] = null;
  const rc = (ctx as unknown as { renderCapacity?: { start(o: { updateInterval: number }): void; stop(): void; addEventListener(n: string, f: (e: { averageLoad: number; peakLoad: number; underrunRatio: number }) => void): void } }).renderCapacity;
  if (rc) {
    load = { avg: 0, peak: 0, underrun: 0, samples: 0 };
    rc.addEventListener('update', (e) => {
      if (!load) return;
      load.avg += e.averageLoad;
      load.peak = Math.max(load.peak, e.peakLoad);
      load.underrun = Math.max(load.underrun, e.underrunRatio);
      load.samples++;
    });
    rc.start({ updateInterval: 0.5 });
  }
  let frames = 0;
  const t0 = performance.now();
  const sfxPlan: [number, SfxName][] = [
    [0.4, 'countdown_beep'],
    [1.0, 'countdown_go'],
    [1.8, 'lap'],
    [2.4, 'crash_heavy'],
    [3.0, 'position_up'],
  ];
  let sfxI = 0;
  const drive = (): void => {
    const t = (performance.now() - t0) / 1000;
    try {
      eng.setListener({ x: 0, y: 1.2, z: 6, fx: 0, fy: 0, fz: -1, ux: 0, uy: 1, uz: 0, vx: 0, vy: 0, vz: -40 });
      voices.forEach((o, i) => o.v.update(carState(i, t, o.type)));
      if (t % 2.2 < 0.02 && frames > 5) voices[0].v.event('shift_up');
      while (sfxI < sfxPlan.length && t >= sfxPlan[sfxI][0]) eng.play(sfxPlan[sfxI++][1]);
      eng.setMusicIntensity(Math.min(1, t / 3));
      frames++;
    } catch (e) {
      errors.push('drive: ' + String(e));
    }
  };
  const iv = setInterval(drive, 16);
  await sleep(600); // let buffers settle
  tap.start();
  await sleep(opts.seconds * 1000);
  const main = tap.stop();
  // pause: world ducked + low-passed
  eng.setPaused(true);
  await sleep(500);
  tap.start();
  await sleep(700);
  const paused = tap.stop();
  eng.setPaused(false);
  // engine only (music/ambience off) for a clean look at the worklet output
  eng.playMusic(null, 0.05);
  eng.setAmbience({ wind: 0, rain: 0, crowd: 0, tunnel: 0 });
  await sleep(900);
  tap.start();
  await sleep(1500);
  const solo = tap.stop();
  clearInterval(iv);
  if (rc) rc.stop();
  if (load && load.samples) load.avg /= load.samples;
  const info = eng.debugInfo();
  // lifecycle
  try {
    voices.forEach((o) => o.v.dispose());
    await eng.suspend();
    await eng.resume();
    eng.dispose();
    eng.play('ui_move');
    eng.setAmbience({ wind: 1 });
  } catch (e) {
    errors.push('lifecycle: ' + String(e));
  }
  const src = carSynthWorkletSource();
  return { info, errors, fs: ctx.sampleRate, pcm: b64(main), pausedPcm: b64(paused), soloEnginePcm: b64(solo), load, workletSrcLen: src ? src.length : 0, frames };
}

/** Offline render of a music cue with an intensity plan [[time, intensity], ...]. */
export async function renderMusic(cue: MusicCue, dur: number, plan: [number, number][], fs = 48000): Promise<{ fs: number; pcm: string }> {
  const ctx = new OfflineAudioContext(2, Math.round(dur * fs), fs);
  const seq = new MusicSequencer(ctx, ctx.destination, buildMusicBank(ctx));
  seq.setIntensityAt(plan.length ? plan[0][1] : 0.5, 0);
  seq.play(cue, 0.05, 0);
  for (let i = 1; i < plan.length; i++) {
    seq.schedule(plan[i][0]);
    seq.setIntensityAt(plan[i][1], plan[i][0]);
  }
  seq.schedule(dur);
  const buf = await ctx.startRendering();
  return { fs, pcm: b64([buf.getChannelData(0), buf.getChannelData(1)]) };
}

/** Offline: menu → race crossfade at `at` seconds. */
export async function renderCrossfade(dur: number, at: number, fade: number, fs = 48000): Promise<{ fs: number; pcm: string }> {
  const ctx = new OfflineAudioContext(2, Math.round(dur * fs), fs);
  const seq = new MusicSequencer(ctx, ctx.destination, buildMusicBank(ctx));
  seq.setIntensityAt(0.8, 0);
  seq.play('menu', 0.05, 0);
  seq.schedule(at);
  seq.play('race', fade, at);
  seq.schedule(dur);
  const buf = await ctx.startRendering();
  return { fs, pcm: b64([buf.getChannelData(0), buf.getChannelData(1)]) };
}

/** Offline: the whole engine graph (worklet + buses + compressor + limiter) with everything playing at once. */
export async function renderFullMix(dur: number, cars: number, fs = 48000): Promise<{ fs: number; pcm: string; info: ReturnType<AudioEngine['debugInfo']>; errors: string[] }> {
  const errors: string[] = [];
  const ctx = new OfflineAudioContext(2, Math.round(dur * fs), fs);
  const eng = new AudioEngine();
  await eng.init({ context: ctx });
  eng.setAmbience({ wind: 0.6, rain: 0.6, crowd: 0.6, tunnel: 0.35 });
  eng.playMusic('race', 0.05);
  eng.setMusicIntensity(1);
  eng.scheduleMusic(dur);
  eng.setListener({ x: 0, y: 1.2, z: 6, fx: 0, fy: 0, fz: -1, ux: 0, uy: 1, uz: 0, vx: 0, vy: 0, vz: -40 });
  const voices: { v: CarVoice; type: EngineType }[] = [];
  for (let i = 0; i < cars; i++) {
    const type = i === 0 ? 'v8' : TYPES[i % 4];
    voices.push({ v: eng.createCar(type, i === 0), type });
  }
  const upd = (t: number): void => voices.forEach((o, i) => o.v.update(carState(i, t, o.type)));
  upd(0);
  const names: SfxName[] = ['countdown_go', 'crash_heavy', 'finish', 'cheer', 'best_lap'];
  let k = 0;
  for (let t = 0.1; t < dur - 0.05; t += 0.1) {
    const tt = t;
    void ctx
      .suspend(tt)
      .then(() => {
        try {
          upd(tt);
          if (Math.abs((tt % 1) - 0.5) < 0.04) eng.play(names[k++ % names.length]);
          if (Math.abs((tt % 2.2) - 0.1) < 0.04) voices[0].v.event('shift_up');
          if (Math.abs((tt % 3) - 1.5) < 0.04) voices[0].v.impact(0.9);
        } catch (e) {
          errors.push(String(e));
        }
        return ctx.resume();
      })
      .catch((e) => errors.push('suspend: ' + String(e)));
  }
  const buf = await ctx.startRendering();
  return { fs, pcm: b64([buf.getChannelData(0), buf.getChannelData(1)]), info: eng.debugInfo(), errors };
}

/** Small-signal gain of the master DynamicsCompressor settings (its automatic make-up gain). */
export async function measureCompressorMakeup(fs = 48000): Promise<number> {
  const ctx = new OfflineAudioContext(1, fs, fs);
  const o = ctx.createOscillator();
  o.frequency.value = 440;
  const g = ctx.createGain();
  g.gain.value = 0.01;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -14;
  comp.knee.value = 10;
  comp.ratio.value = 6;
  comp.attack.value = 0.004;
  comp.release.value = 0.18;
  o.connect(g);
  g.connect(comp);
  comp.connect(ctx.destination);
  o.start();
  const buf = await ctx.startRendering();
  const x = buf.getChannelData(0);
  let s = 0;
  for (let i = fs / 2; i < fs; i++) s += x[i] * x[i];
  return Math.sqrt(s / (fs / 2)) / (0.01 * Math.SQRT1_2);
}

/** CPU cost of the car synth in this browser's JS engine: ms per second of audio for 8 cars (typical / worst case). */
export function benchDsp(): { typical: number; worst: number } {
  const FS = 48000;
  const run = (worst: boolean): number => {
    const dsp = new CarSynthDSP(FS, 12);
    for (let i = 0; i < 8; i++) dsp.configure(i, engineConfig(TYPES[i % 4]), i === 0, 0);
    const q = [0, 1, 2, 3].map(() => new Float32Array(128));
    const st = new Float32Array(14 * 8);
    const upd = (t: number): void => {
      for (let i = 0; i < 8; i++) {
        const n = 0.5 + 0.45 * Math.abs(Math.sin(t * 0.7 + i));
        st.set([i, 1, RPM_MAX[TYPES[i % 4]] * n, n, 1, 60, worst && i === 0 ? 1 : 0, worst ? 0.7 : i % 4 === 1 ? 0.6 : 0, worst ? i % 4 : 0, worst && i % 3 === 0 ? 0.6 : 0, i === 0 ? 1 : 0.5, Math.sin(i), 1, 0.8], i * 14);
      }
      dsp.setStates(st);
    };
    let best = Infinity;
    for (let rep = 0; rep < 3; rep++) {
      const t0 = performance.now();
      for (let p = 0; p < FS * 5; p += 128) {
        if (p % 768 === 0) upd(p / FS);
        dsp.process(q[0], q[1], q[2], q[3], 128);
      }
      best = Math.min(best, (performance.now() - t0) / 5);
    }
    return best;
  };
  return { typical: run(false), worst: run(true) };
}
