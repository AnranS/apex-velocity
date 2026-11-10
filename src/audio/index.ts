/**
 * APEX VELOCITY — audio engine (public API).
 *
 * Fully procedural, dependency-free Web Audio module:
 *   • car voices (engine/exhaust/tyres) synthesised in one AudioWorklet (see engine-dsp.ts), with a
 *     ScriptProcessor fallback; manual distance / pan / Doppler spatialisation for non-player cars
 *   • one-shot SFX, ambience loops and impulse responses baked into AudioBuffers at init (sfx.ts, ambience.ts)
 *   • procedural music sequencer with three cues (music.ts)
 *   • buses: engine / tyres+surface / sfx / ambience / music / ui → master → compressor → soft limiter
 *
 * Every public method is a safe no-op before `init()` resolves and never throws; desired state
 * (volumes, ambience, music cue, pause) set before init is remembered and applied once audio is up.
 *
 * Per frame, the game should call:  audio.setListener(...)  and  voice.update(state) for every car.
 */
import { CarSynthDSP } from './engine-dsp';
import { engineConfig, NOMINAL_RPM_MAX, type EngineType } from './engine-configs';
import { renderSfx, SFX_NAMES, SFX_VARIANTS, SFX_UI, type SfxName } from './sfx';
import { renderWindLoop, renderRainLoop, renderCrowdLoop, renderTunnelIR } from './ambience';
import { MusicSequencer, buildMusicBank, toBuffer, type MusicCue } from './music';

export type { EngineType } from './engine-configs';
export type { SfxName } from './sfx';

export interface CarAudioState {
  rpm: number;
  rpmMax: number; // engine speed and redline
  throttle: number; // 0..1 effective engine load
  speed: number; // m/s
  nitro: number; // 0..1 boost currently active
  slip: number; // 0..1 tyre squeal intensity on tarmac
  surface: number; // dominant surface under the wheels: 0 asphalt, 1 kerb, 2 grass, 3 gravel
  scrape: number; // 0..1 wall-scrape intensity
  x: number;
  y: number;
  z: number; // world position, metres
  vx: number;
  vy: number;
  vz: number; // world velocity, m/s
}

export type CarEventName = 'shift_up' | 'shift_down' | 'backfire' | 'limiter' | 'nitro_start' | 'blowoff';

export interface CarVoice {
  update(state: CarAudioState): void; // called every rendered frame
  event(name: CarEventName, intensity?: number): void;
  impact(intensity: number): void; // collision thud/crunch at the car, 0..1
  setActive(active: boolean): void; // false = silent & cheap (far away / not racing)
  dispose(): void;
}

export interface ListenerState {
  x: number;
  y: number;
  z: number;
  fx: number;
  fy: number;
  fz: number;
  ux: number;
  uy: number;
  uz: number;
  vx: number;
  vy: number;
  vz: number;
}

/** Optional init parameters (testing / embedding). */
export interface AudioInitOptions {
  /** Use this context instead of creating an AudioContext (e.g. an OfflineAudioContext in tests). */
  context?: BaseAudioContext;
  /** Skip the AudioWorklet and run the car synth on a ScriptProcessorNode (fallback path). */
  noWorklet?: boolean;
}

export interface AudioDebugInfo {
  ready: boolean;
  state: string;
  sampleRate: number;
  synth: 'worklet' | 'script' | 'none';
  voices: number;
  sfxBuffers: number;
  music: MusicCue | null;
  initMs: number;
}

const MAX_VOICES = 12;
const STRIDE = 14;
const EVENT_CODE: Record<CarEventName, number> = { shift_up: 0, shift_down: 1, backfire: 2, limiter: 3, nitro_start: 4, blowoff: 5 };
const SPEED_OF_SOUND = 343;
/** static bus trims (before the user volumes) */
const TRIM = { engine: 0.8, tyres: 0.8, sfx: 0.85, amb: 0.8, music: 0.55, ui: 0.85, tunnel: 0.55 };
/** DynamicsCompressor auto make-up compensation (measured in tools/audio/test-browser.ts) */
export const MASTER_COMP_TRIM = 0.6;

const PROCESSOR_NAME = 'apex-car-synth';
const PROCESSOR_SRC = `
class ApexCarSynth extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.alive = true;
    this.dsp = null;
    try {
      const o = (options && options.processorOptions) || {};
      if (o.testCfg) {
        // self-test: one voice must produce finite, non-silent audio in this global scope
        const t = new DSP(sampleRate, 1);
        t.configure(0, o.testCfg, true, 0);
        const a = new Float32Array(128), b = new Float32Array(128), c = new Float32Array(128), d = new Float32Array(128);
        const st = new Float32Array(14);
        st[1] = 1; st[2] = 4000; st[3] = 0.5; st[4] = 1; st[5] = 30; st[6] = 1; st[7] = 0.6; st[9] = 0.5; st[10] = 1; st[12] = 1; st[13] = 1;
        t.setStates(st);
        t.event(0, 2, 1);
        let e = 0;
        for (let k = 0; k < 8; k++) {
          t.process(a, b, c, d, 128);
          for (let i = 0; i < 128; i++) e += a[i] * a[i] + c[i] * c[i];
        }
        if (!(e > 0) || e === Infinity) throw new Error('self-test failed');
      }
      this.dsp = new DSP(sampleRate, o.maxVoices || 12);
      this.port.postMessage({ t: 'ready', ok: true });
    } catch (err) {
      this.port.postMessage({ t: 'ready', ok: false, err: String(err) });
    }
    this.port.onmessage = (e) => {
      const m = e.data, d = this.dsp;
      if (!d || !m) return;
      try {
        if (m instanceof Float32Array) d.setStates(m);
        else if (m.t === 'cfg') d.configure(m.slot, m.cfg, m.player, m.debug | 0);
        else if (m.t === 'ev') d.event(m.slot, m.code, m.val);
        else if (m.t === 'stop') this.alive = false;
      } catch (err) {}
    };
  }
  process(inputs, outputs) {
    const d = this.dsp;
    if (d) {
      const a = outputs[0], b = outputs[1];
      if (a && b && a.length > 1 && b.length > 1) {
        try { d.process(a[0], a[1], b[0], b[1], a[0].length); } catch (err) {}
      }
    }
    return this.alive;
  }
}
registerProcessor(${JSON.stringify(PROCESSOR_NAME)}, ApexCarSynth);
`;

/**
 * The DSP class as a self-contained expression: `{ prelude, cls }` where `prelude` stubs transpiler helpers that some
 * build setups reference from inside class bodies (e.g. esbuild `keepNames` → `__name`). Unresolved identifiers are
 * found by evaluating + smoke-testing the text on this thread; if eval is forbidden (CSP) the text ships unmodified
 * and the worklet's own self-test decides (→ ScriptProcessor fallback on failure).
 */
export function carSynthClassSource(): { prelude: string; cls: string } | null {
  const cls = CarSynthDSP.toString();
  // If a build step down-levelled the class to a function it is no longer self-contained → use the fallback.
  if (!/^\s*class\b/.test(cls)) return null;
  let prelude = '';
  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      const C = new Function(`${prelude}return (${cls});`)() as typeof CarSynthDSP;
      const d = new C(48000, 2);
      d.configure(0, engineConfig('v8'), true, 0);
      d.configure(1, engineConfig('i6t'), false, 0);
      const q = [0, 1, 2, 3].map(() => new Float32Array(128));
      const st = new Float32Array(STRIDE * 2);
      st.set([0, 1, 4000, 0.5, 1, 30, 1, 0.6, 1, 0.5, 1, 0, 1, 1], 0);
      st.set([1, 1, 5000, 0.7, 0, 30, 0, 0.6, 3, 0.5, 1, 0.3, 1.1, 0.5], STRIDE);
      d.setStates(st);
      for (let code = 0; code < 6; code++) d.event(code & 1, code, 1);
      for (let k = 0; k < 6; k++) d.process(q[0], q[1], q[2], q[3], 128);
      break;
    } catch (e) {
      const m = e instanceof ReferenceError ? /^(?:Can't find variable: )?([A-Za-z_$][\w$]*)(?: is not defined)?$/.exec(e.message) : null;
      if (!m) break;
      prelude += `const ${m[1]} = (t) => t;\n`;
    }
  }
  return { prelude, cls };
}

/** Source text of the AudioWorklet module (DSP class + processor). Exported for the test tools. */
export function carSynthWorkletSource(): string | null {
  const c = carSynthClassSource();
  return c ? `${c.prelude}const DSP = (${c.cls});\n${PROCESSOR_SRC}` : null;
}

type SynthMsg = Float32Array | { t: 'cfg'; slot: number; cfg: unknown; player: boolean; debug: number } | { t: 'ev'; slot: number; code: number; val: number } | { t: 'stop' };

interface SynthHost {
  kind: 'worklet' | 'script';
  engineOut: AudioNode;
  tyreOut: AudioNode;
  post(msg: SynthMsg): void;
  dispose(): void;
}

const clamp = (v: number, a: number, b: number): number => (v < a ? a : v > b ? b : v);
const num = (v: number, d: number): number => (typeof v === 'number' && v === v && v !== Infinity && v !== -Infinity ? v : d);
const yieldTask = (): Promise<void> => new Promise((r) => setTimeout(r, 0));
const micro = (f: () => void): void => {
  if (typeof queueMicrotask === 'function') queueMicrotask(f);
  else void Promise.resolve().then(f);
};

async function createWorkletHost(ctx: BaseAudioContext): Promise<SynthHost | null> {
  try {
    if (!ctx.audioWorklet || typeof AudioWorkletNode === 'undefined' || typeof Blob === 'undefined') return null;
    const code = carSynthWorkletSource();
    if (!code) return null;
    const url = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const node = new AudioWorkletNode(ctx, PROCESSOR_NAME, {
      numberOfInputs: 0,
      numberOfOutputs: 2,
      outputChannelCount: [2, 2],
      processorOptions: { maxVoices: MAX_VOICES, testCfg: engineConfig('v8') },
    });
    const ok = await new Promise<boolean>((resolve) => {
      const to = setTimeout(() => resolve(false), 2000);
      node.port.onmessage = (e: MessageEvent) => {
        if (e.data && e.data.t === 'ready') {
          clearTimeout(to);
          resolve(!!e.data.ok);
        }
      };
      node.onprocessorerror = () => {
        clearTimeout(to);
        resolve(false);
      };
    });
    if (!ok) {
      try {
        node.port.postMessage({ t: 'stop' });
        node.disconnect();
      } catch {
        /* noop */
      }
      return null;
    }
    const eng = ctx.createGain();
    const tyr = ctx.createGain();
    node.connect(eng, 0);
    node.connect(tyr, 1);
    return {
      kind: 'worklet',
      engineOut: eng,
      tyreOut: tyr,
      post: (m) => {
        try {
          node.port.postMessage(m);
        } catch {
          /* noop */
        }
      },
      dispose: () => {
        try {
          node.port.postMessage({ t: 'stop' });
          node.disconnect();
          eng.disconnect();
          tyr.disconnect();
        } catch {
          /* noop */
        }
      },
    };
  } catch {
    return null;
  }
}

function createScriptHost(ctx: BaseAudioContext): SynthHost | null {
  try {
    if (typeof ctx.createScriptProcessor !== 'function') return null;
    const dsp = new CarSynthDSP(ctx.sampleRate, MAX_VOICES);
    const node = ctx.createScriptProcessor(1024, 0, 4);
    node.onaudioprocess = (e: AudioProcessingEvent) => {
      const ob = e.outputBuffer;
      try {
        dsp.process(ob.getChannelData(0), ob.getChannelData(1), ob.getChannelData(2), ob.getChannelData(3), ob.length);
      } catch {
        /* keep the audio callback alive */
      }
    };
    const split = ctx.createChannelSplitter(4);
    const eng = ctx.createChannelMerger(2);
    const tyr = ctx.createChannelMerger(2);
    node.connect(split);
    split.connect(eng, 0, 0);
    split.connect(eng, 1, 1);
    split.connect(tyr, 2, 0);
    split.connect(tyr, 3, 1);
    return {
      kind: 'script',
      engineOut: eng,
      tyreOut: tyr,
      post: (m) => {
        try {
          if (m instanceof Float32Array) dsp.setStates(m);
          else if (m.t === 'cfg') dsp.configure(m.slot, m.cfg as Parameters<CarSynthDSP['configure']>[1], m.player, m.debug | 0);
          else if (m.t === 'ev') dsp.event(m.slot, m.code, m.val);
        } catch {
          /* noop */
        }
      },
      dispose: () => {
        try {
          node.onaudioprocess = null;
          node.disconnect();
          split.disconnect();
          eng.disconnect();
          tyr.disconnect();
        } catch {
          /* noop */
        }
      },
    };
  } catch {
    return null;
  }
}

// ───────────────────────── car voice (main-thread proxy) ─────────────────────────
class Voice implements CarVoice {
  slot = -1;
  active = true;
  dead = false;
  hasState = false;
  lastImpact = -1e9;
  readonly s: CarAudioState = { rpm: 900, rpmMax: 8000, throttle: 0, speed: 0, nitro: 0, slip: 0, surface: 0, scrape: 0, x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
  constructor(
    private readonly eng: AudioEngine,
    readonly type: EngineType,
    readonly isPlayer: boolean,
  ) {}
  update(st: CarAudioState): void {
    if (this.dead || !st) return;
    const s = this.s;
    s.rpm = num(st.rpm, 900);
    s.rpmMax = num(st.rpmMax, 0);
    s.throttle = num(st.throttle, 0);
    s.speed = num(st.speed, 0);
    s.nitro = num(st.nitro, 0);
    s.slip = num(st.slip, 0);
    s.surface = num(st.surface, 0);
    s.scrape = num(st.scrape, 0);
    s.x = num(st.x, 0);
    s.y = num(st.y, 0);
    s.z = num(st.z, 0);
    s.vx = num(st.vx, 0);
    s.vy = num(st.vy, 0);
    s.vz = num(st.vz, 0);
    this.hasState = true;
    this.eng._dirty();
  }
  event(name: CarEventName, intensity = 1): void {
    if (!this.dead) this.eng._voiceEvent(this, name, intensity);
  }
  impact(intensity: number): void {
    if (!this.dead) this.eng._voiceImpact(this, intensity);
  }
  setActive(active: boolean): void {
    this.active = !!active;
    this.eng._dirty();
  }
  dispose(): void {
    if (this.dead) return;
    this.dead = true;
    this.eng._free(this);
  }
}

interface Buses {
  engine: GainNode;
  tyres: GainNode;
  sfx: GainNode;
  amb: GainNode;
  ui: GainNode;
  music: GainNode;
  musicDuck: GainNode;
  musicPause: GainNode;
  musicLP: BiquadFilterNode;
  worldDuck: GainNode;
  worldLP: BiquadFilterNode;
  tunnelSend: GainNode;
  master: GainNode;
}

export class AudioEngine {
  private _ready = false;
  private ctx: BaseAudioContext | null = null;
  private rt: AudioContext | null = null;
  private initPromise: Promise<void> | null = null;
  private host: SynthHost | null = null;
  private buses: Buses | null = null;
  private outNode: AudioNode | null = null;
  private music: MusicSequencer | null = null;
  private readonly sfx = new Map<SfxName, AudioBuffer[]>();
  private amb: { wind: GainNode; rain: GainNode; crowd: GainNode } | null = null;
  private readonly voices: Voice[] = [];
  private readonly slots: (Voice | null)[] = new Array<Voice | null>(MAX_VOICES).fill(null);
  private flushQueued = false;
  private timer: ReturnType<typeof setInterval> | null = null;
  private disposed = false;
  private userSuspended = false;
  private autoSuspended = false;
  private initMs = 0;
  private duckLast = 1;
  // desired state (kept so calls made before init() are honoured later)
  private readonly vol = { master: 1, music: 0.7, sfx: 1, engine: 1 };
  private readonly ambState = { wind: 0, rain: 0, crowd: 0, tunnel: 0 };
  private wantMusic: MusicCue | null = null;
  private wantIntensity = 0.5;
  private paused = false;
  private readonly lis: ListenerState = { x: 0, y: 0, z: 0, fx: 0, fy: 0, fz: -1, ux: 0, uy: 1, uz: 0, vx: 0, vy: 0, vz: 0 };

  get ready(): boolean {
    return this._ready;
  }

  /** Call from a user gesture. Idempotent. Resolves when everything (context, worklet, buffers, music) is up. */
  init(opts?: AudioInitOptions): Promise<void> {
    if (!this.initPromise) {
      this.disposed = false;
      this.initPromise = this.doInit(opts).catch(() => {
        /* audio unavailable: stay silent */
      });
    } else if (this.rt && this.rt.state === 'suspended' && !this.userSuspended) {
      void this.rt.resume().catch(() => {});
    }
    return this.initPromise;
  }

  private async doInit(opts?: AudioInitOptions): Promise<void> {
    const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
    let ctx: BaseAudioContext;
    if (opts && opts.context) ctx = opts.context;
    else {
      const w = globalThis as unknown as { AudioContext?: typeof AudioContext; webkitAudioContext?: typeof AudioContext };
      const AC = w.AudioContext || w.webkitAudioContext;
      if (!AC) return;
      const rt = new AC({ latencyHint: 'interactive' });
      this.rt = rt;
      ctx = rt;
      void rt.resume().catch(() => {});
      this.installUnlock();
      this.claimPlaybackSession();
    }
    this.ctx = ctx;
    const fs = ctx.sampleRate;

    // ── bus graph ──
    const gain = (v: number): GainNode => {
      const g = ctx.createGain();
      g.gain.value = v;
      return g;
    };
    const lowpass = (): BiquadFilterNode => {
      const f = ctx.createBiquadFilter();
      f.type = 'lowpass';
      f.frequency.value = 20000;
      f.Q.value = 0.7;
      return f;
    };
    const b: Buses = {
      engine: gain(0),
      tyres: gain(0),
      sfx: gain(0),
      amb: gain(0),
      ui: gain(0),
      music: gain(0),
      musicDuck: gain(1),
      musicPause: gain(1),
      musicLP: lowpass(),
      worldDuck: gain(1),
      worldLP: lowpass(),
      tunnelSend: gain(0),
      master: gain(1),
    };
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.knee.value = 10;
    comp.ratio.value = 6;
    comp.attack.value = 0.004;
    comp.release.value = 0.18;
    const post = gain(MASTER_COMP_TRIM * 0.5);
    const limiter = ctx.createWaveShaper();
    const curve = new Float32Array(4097);
    for (let i = 0; i < curve.length; i++) {
      const x = ((i / (curve.length - 1)) * 2 - 1) * 2; // shaper input ±1 ↔ signal ±2
      const a = Math.abs(x);
      const y = a < 0.8 ? a : 0.8 + 0.19 * Math.tanh((a - 0.8) / 0.19);
      curve[i] = x < 0 ? -y : y;
    }
    limiter.curve = curve;
    b.engine.connect(b.worldLP);
    b.tyres.connect(b.worldLP);
    b.sfx.connect(b.worldLP);
    b.amb.connect(b.worldLP);
    b.worldLP.connect(b.worldDuck);
    b.worldDuck.connect(b.master);
    b.music.connect(b.musicDuck);
    b.musicDuck.connect(b.musicPause);
    b.musicPause.connect(b.musicLP);
    b.musicLP.connect(b.master);
    b.ui.connect(b.master);
    b.master.connect(comp);
    comp.connect(post);
    post.connect(limiter);
    limiter.connect(ctx.destination);
    this.outNode = limiter;
    // tunnel / gallery reverb for engines and tyres
    const tun = renderTunnelIR(fs);
    const conv = ctx.createConvolver();
    conv.buffer = toBuffer(ctx, [tun.L, tun.R]);
    b.engine.connect(b.tunnelSend);
    b.tyres.connect(b.tunnelSend);
    b.tunnelSend.connect(conv);
    conv.connect(b.worldLP);
    this.buses = b;
    this.applyVolumes(true);

    // ── UI blips first (a few ms) so menus click immediately ──
    const bake = (name: SfxName): void => {
      const cnt = SFX_VARIANTS[name] ?? 1;
      const list: AudioBuffer[] = [];
      for (let k = 0; k < cnt; k++) {
        const s = renderSfx(name, fs, k);
        list.push(toBuffer(ctx, [s.L, s.R]));
      }
      this.sfx.set(name, list);
    };
    for (const name of SFX_NAMES) if (SFX_UI[name]) bake(name);

    // ── car synth ──
    let host: SynthHost | null = null;
    if (!(opts && opts.noWorklet)) host = await createWorkletHost(ctx);
    if (this.disposed) return;
    if (!host) host = createScriptHost(ctx);
    if (host) {
      host.engineOut.connect(b.engine);
      host.tyreOut.connect(b.tyres);
      this.host = host;
      for (const v of this.voices) this.configureVoice(v);
      this._dirty();
    }

    // ── baked sounds (yield between items so the page stays responsive) ──
    for (const name of SFX_NAMES) {
      if (SFX_UI[name]) continue;
      bake(name);
      await yieldTask();
      if (this.disposed) return;
    }

    // ── ambience ──
    const loop = (st: { L: Float32Array; R: Float32Array }, rate: number): GainNode => {
      const src = ctx.createBufferSource();
      src.buffer = toBuffer(ctx, [st.L, st.R], rate);
      src.loop = true;
      const g = gain(0);
      src.connect(g);
      g.connect(b.amb);
      src.start();
      return g;
    };
    const wind = loop(renderWindLoop(16000), 16000);
    await yieldTask();
    const rain = loop(renderRainLoop(Math.min(fs, 32000)), Math.min(fs, 32000));
    await yieldTask();
    const crowd = loop(renderCrowdLoop(16000), 16000);
    await yieldTask();
    if (this.disposed) return;
    this.amb = { wind, rain, crowd };
    this.applyAmbience(true);

    // ── music ──
    this.music = new MusicSequencer(ctx, b.music, buildMusicBank(ctx));
    this.music.setIntensity(this.wantIntensity);
    if (this.wantMusic) this.music.play(this.wantMusic, 1.2);
    if (this.rt) {
      this.timer = setInterval(() => {
        try {
          if (this.music && this.rt && this.rt.state === 'running') this.music.tick();
        } catch {
          /* noop */
        }
      }, 60);
      this.installVisibility();
    }
    if (this.paused) this.applyPause(true);
    this.initMs = (typeof performance !== 'undefined' ? performance.now() : 0) - t0;
    this._ready = true;
  }

  // ───────────── lifecycle helpers ─────────────
  private silentEl: HTMLAudioElement | null = null;

  /** iOS: route Web Audio as "playback" so the hardware silent switch does not mute the game. */
  private claimPlaybackSession(): void {
    try {
      if (typeof navigator === 'undefined') return;
      const nav = navigator as Navigator & { audioSession?: { type: string } };
      if (nav.audioSession) {
        nav.audioSession.type = 'playback';
        return;
      }
      const ios = /iPad|iPhone|iPod/.test(nav.userAgent) || (/Macintosh/.test(nav.userAgent) && nav.maxTouchPoints > 1);
      if (!ios || typeof Audio === 'undefined' || typeof Blob === 'undefined') return;
      // older iOS: a looping silent <audio> element switches the session category
      const n = 2205;
      const wav = new DataView(new ArrayBuffer(44 + n * 2));
      const put = (o: number, t: string): void => {
        for (let i = 0; i < t.length; i++) wav.setUint8(o + i, t.charCodeAt(i));
      };
      put(0, 'RIFF');
      wav.setUint32(4, 36 + n * 2, true);
      put(8, 'WAVEfmt ');
      wav.setUint32(16, 16, true);
      wav.setUint16(20, 1, true);
      wav.setUint16(22, 1, true);
      wav.setUint32(24, 22050, true);
      wav.setUint32(28, 44100, true);
      wav.setUint16(32, 2, true);
      wav.setUint16(34, 16, true);
      put(36, 'data');
      wav.setUint32(40, n * 2, true);
      const el = new Audio(URL.createObjectURL(new Blob([wav.buffer], { type: 'audio/wav' })));
      el.loop = true;
      el.setAttribute('playsinline', '');
      el.setAttribute('x-webkit-airplay', 'deny');
      void el.play().catch(() => {});
      this.silentEl = el;
    } catch {
      /* optional nicety */
    }
  }

  private unlock = (): void => {
    const rt = this.rt;
    if (!rt) return;
    if (rt.state === 'suspended' && !this.userSuspended && !this.autoSuspended) void rt.resume().catch(() => {});
    if (rt.state === 'running') this.removeUnlock();
  };
  private installUnlock(): void {
    if (typeof window === 'undefined') return;
    for (const ev of ['pointerdown', 'keydown', 'touchend', 'mousedown']) window.addEventListener(ev, this.unlock, { passive: true, capture: true });
  }
  private removeUnlock(): void {
    if (typeof window === 'undefined') return;
    for (const ev of ['pointerdown', 'keydown', 'touchend', 'mousedown']) window.removeEventListener(ev, this.unlock, { capture: true });
  }
  private onVisibility = (): void => {
    const rt = this.rt;
    if (!rt || typeof document === 'undefined') return;
    if (document.hidden) {
      if (rt.state === 'running') {
        this.autoSuspended = true;
        void rt.suspend().catch(() => {});
      }
    } else if (this.autoSuspended) {
      this.autoSuspended = false;
      if (!this.userSuspended) void rt.resume().catch(() => {});
    }
  };
  private installVisibility(): void {
    if (typeof document !== 'undefined') document.addEventListener('visibilitychange', this.onVisibility);
  }

  async resume(): Promise<void> {
    this.userSuspended = false;
    try {
      if (this.rt && this.rt.state !== 'closed' && !this.autoSuspended) await this.rt.resume();
    } catch {
      /* noop */
    }
  }
  async suspend(): Promise<void> {
    this.userSuspended = true;
    try {
      if (this.rt && this.rt.state === 'running') await this.rt.suspend();
    } catch {
      /* noop */
    }
  }

  // ───────────── volumes / mix ─────────────
  setVolumes(v: { master?: number; music?: number; sfx?: number; engine?: number }): void {
    try {
      if (!v) return;
      if (v.master !== undefined) this.vol.master = clamp(num(v.master, 1), 0, 1);
      if (v.music !== undefined) this.vol.music = clamp(num(v.music, 1), 0, 1);
      if (v.sfx !== undefined) this.vol.sfx = clamp(num(v.sfx, 1), 0, 1);
      if (v.engine !== undefined) this.vol.engine = clamp(num(v.engine, 1), 0, 1);
      this.applyVolumes(false);
    } catch {
      /* noop */
    }
  }
  private applyVolumes(instant: boolean): void {
    const b = this.buses;
    const ctx = this.ctx;
    if (!b || !ctx) return;
    const t = ctx.currentTime;
    const set = (g: GainNode, v: number): void => {
      if (instant) g.gain.setValueAtTime(v, t);
      else g.gain.setTargetAtTime(v, t, 0.03);
    };
    const curve = (v: number): number => Math.pow(v, 1.5);
    set(b.master, curve(this.vol.master));
    set(b.engine, curve(this.vol.engine) * TRIM.engine);
    set(b.tyres, curve(this.vol.sfx) * TRIM.tyres);
    set(b.sfx, curve(this.vol.sfx) * TRIM.sfx);
    set(b.amb, curve(this.vol.sfx) * TRIM.amb);
    set(b.ui, curve(this.vol.sfx) * TRIM.ui);
    set(b.music, curve(this.vol.music) * TRIM.music);
  }

  setPaused(paused: boolean): void {
    this.paused = !!paused;
    try {
      this.applyPause(false);
    } catch {
      /* noop */
    }
  }
  private applyPause(instant: boolean): void {
    const b = this.buses;
    const ctx = this.ctx;
    if (!b || !ctx) return;
    const t = ctx.currentTime;
    const tc = instant ? 0.001 : 0.07;
    const p = this.paused;
    b.worldDuck.gain.setTargetAtTime(p ? 0.28 : 1, t, tc);
    b.worldLP.frequency.setTargetAtTime(p ? 520 : 20000, t, tc);
    b.musicPause.gain.setTargetAtTime(p ? 0.55 : 1, t, tc);
    b.musicLP.frequency.setTargetAtTime(p ? 1000 : 20000, t, tc);
  }

  setAmbience(a: { wind?: number; rain?: number; crowd?: number; tunnel?: number }): void {
    try {
      if (!a) return;
      const s = this.ambState;
      let changed = false;
      for (const k of ['wind', 'rain', 'crowd', 'tunnel'] as const) {
        const v = a[k];
        if (v === undefined) continue;
        const c = clamp(num(v, 0), 0, 1);
        if (Math.abs(c - s[k]) > 0.004) {
          s[k] = c;
          changed = true;
        }
      }
      if (changed) this.applyAmbience(false);
    } catch {
      /* noop */
    }
  }
  private applyAmbience(instant: boolean): void {
    const ctx = this.ctx;
    const b = this.buses;
    if (!ctx || !b) return;
    const t = ctx.currentTime;
    const tc = instant ? 0.001 : 0.2;
    const s = this.ambState;
    b.tunnelSend.gain.setTargetAtTime(s.tunnel * TRIM.tunnel, t, instant ? 0.001 : 0.12);
    const a = this.amb;
    if (!a) return;
    a.wind.gain.setTargetAtTime(s.wind * 0.55, t, tc);
    a.rain.gain.setTargetAtTime(s.rain * 0.6, t, tc);
    a.crowd.gain.setTargetAtTime(s.crowd * 0.5, t, tc);
  }

  // ───────────── listener / spatialisation ─────────────
  setListener(l: ListenerState): void {
    if (!l) return;
    const s = this.lis;
    s.x = num(l.x, s.x);
    s.y = num(l.y, s.y);
    s.z = num(l.z, s.z);
    s.fx = num(l.fx, s.fx);
    s.fy = num(l.fy, s.fy);
    s.fz = num(l.fz, s.fz);
    s.ux = num(l.ux, s.ux);
    s.uy = num(l.uy, s.uy);
    s.uz = num(l.uz, s.uz);
    s.vx = num(l.vx, 0);
    s.vy = num(l.vy, 0);
    s.vz = num(l.vz, 0);
  }

  /** Distance gain, stereo pan, Doppler factor and air absorption for a source. */
  private spatial(x: number, y: number, z: number, vx: number, vy: number, vz: number, out: { gain: number; pan: number; pitch: number; air: number }): void {
    const l = this.lis;
    const dx = x - l.x;
    const dy = y - l.y;
    const dz = z - l.z;
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const ref = 7;
    let g = d > ref ? ref / d : 1;
    if (d > 110) {
      const k = clamp((220 - d) / 110, 0, 1);
      g *= k * k * (3 - 2 * k);
    }
    // listener right = forward × up
    let rx = l.fy * l.uz - l.fz * l.uy;
    let ry = l.fz * l.ux - l.fx * l.uz;
    let rz = l.fx * l.uy - l.fy * l.ux;
    const rl = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
    rx /= rl;
    ry /= rl;
    rz /= rl;
    const inv = d > 1e-4 ? 1 / d : 0;
    const nx = dx * inv;
    const ny = dy * inv;
    const nz = dz * inv;
    let pan = (nx * rx + ny * ry + nz * rz) * 0.9;
    if (d < 3) pan *= d / 3;
    const fl = Math.sqrt(l.fx * l.fx + l.fy * l.fy + l.fz * l.fz) || 1;
    const front = (nx * l.fx + ny * l.fy + nz * l.fz) / fl; // 1 ahead, -1 behind
    // Doppler: velocities projected on the line source → listener
    const vsToward = -(vx * nx + vy * ny + vz * nz);
    const vlToward = l.vx * nx + l.vy * ny + l.vz * nz;
    const pitch = clamp((SPEED_OF_SOUND + clamp(vlToward, -150, 150)) / (SPEED_OF_SOUND - clamp(vsToward, -150, 150)), 0.6, 1.6);
    let air = 1 - clamp((d - 10) / 170, 0, 1) * 0.8;
    if (front < 0) air *= 1 + 0.18 * front; // slightly duller behind the listener
    out.gain = g * 0.9;
    out.pan = clamp(pan, -1, 1);
    out.pitch = pitch;
    out.air = clamp(air, 0, 1);
  }

  // ───────────── car voices ─────────────
  createCar(type: EngineType, isPlayer: boolean): CarVoice {
    const v = new Voice(this, type, !!isPlayer);
    try {
      this.voices.push(v);
      const free = this.slots.indexOf(null);
      if (free >= 0) {
        this.slots[free] = v;
        v.slot = free;
        this.configureVoice(v);
      }
    } catch {
      /* noop */
    }
    return v;
  }

  private configureVoice(v: Voice): void {
    if (!this.host || v.slot < 0 || v.dead) return;
    this.host.post({ t: 'cfg', slot: v.slot, cfg: engineConfig(v.type), player: v.isPlayer, debug: 0 });
  }

  /** @internal */
  _free(v: Voice): void {
    try {
      const i = this.voices.indexOf(v);
      if (i >= 0) this.voices.splice(i, 1);
      if (v.slot >= 0) {
        if (this.host) this.host.post({ t: 'cfg', slot: v.slot, cfg: null, player: false, debug: 0 });
        this.slots[v.slot] = null;
        const s = v.slot;
        v.slot = -1;
        // hand the freed slot to a voice that was waiting for one
        const waiting = this.voices.find((o) => o.slot < 0 && !o.dead);
        if (waiting) {
          this.slots[s] = waiting;
          waiting.slot = s;
          this.configureVoice(waiting);
        }
      }
    } catch {
      /* noop */
    }
  }

  /** @internal */
  _dirty(): void {
    if (this.flushQueued || !this.host) return;
    this.flushQueued = true;
    micro(() => this.flush());
  }

  private readonly sp = { gain: 1, pan: 0, pitch: 1, air: 1 };

  private flush(): void {
    this.flushQueued = false;
    const host = this.host;
    if (!host) return;
    try {
      const vs = this.voices;
      let n = 0;
      for (const v of vs) if (v.slot >= 0) n++;
      if (!n) return;
      const d = new Float32Array(n * STRIDE);
      let o = 0;
      let duck = 1;
      const sp = this.sp;
      for (const v of vs) {
        if (v.slot < 0) continue;
        const s = v.s;
        const rpmMax = s.rpmMax > 0 ? s.rpmMax : NOMINAL_RPM_MAX[v.type];
        const norm = s.rpm / rpmMax;
        const load = clamp(s.throttle, 0, 1);
        if (v.isPlayer) {
          sp.gain = 1;
          sp.pan = 0;
          sp.pitch = 1;
          sp.air = 1;
          if (v.active && v.hasState) duck = Math.min(duck, 1 - 0.3 * clamp(load * (0.35 + norm), 0, 1));
        } else this.spatial(s.x, s.y, s.z, s.vx, s.vy, s.vz, sp);
        d[o] = v.slot;
        d[o + 1] = v.active && v.hasState ? 1 : 0;
        d[o + 2] = s.rpm;
        d[o + 3] = norm;
        d[o + 4] = load;
        d[o + 5] = Math.abs(s.speed);
        d[o + 6] = s.nitro;
        d[o + 7] = s.slip;
        d[o + 8] = s.surface;
        d[o + 9] = s.scrape;
        d[o + 10] = sp.gain;
        d[o + 11] = sp.pan;
        d[o + 12] = sp.pitch;
        d[o + 13] = sp.air;
        o += STRIDE;
      }
      host.post(d);
      // music sits slightly under a loaded engine
      const b = this.buses;
      if (b && this.ctx && Math.abs(duck - this.duckLast) > 0.03) {
        this.duckLast = duck;
        b.musicDuck.gain.setTargetAtTime(duck, this.ctx.currentTime, 0.25);
      }
    } catch {
      /* noop */
    }
  }

  /** @internal */
  _voiceEvent(v: Voice, name: CarEventName, intensity: number): void {
    try {
      const code = EVENT_CODE[name];
      if (code === undefined || !this.host || v.slot < 0) return;
      this.host.post({ t: 'ev', slot: v.slot, code, val: clamp(num(intensity, 1), 0, 1.5) });
    } catch {
      /* noop */
    }
  }

  /** @internal */
  _voiceImpact(v: Voice, intensity: number): void {
    try {
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      if (now - v.lastImpact < 70) return;
      v.lastImpact = now;
      const i = clamp(num(intensity, 0.5), 0, 1);
      if (i < 0.02) return;
      const heavy = i > 0.5;
      const o = { volume: heavy ? 0.45 + 0.55 * i : 0.35 + 1.1 * i, pitch: 0.9 + Math.random() * 0.25 };
      if (v.isPlayer) this.play(heavy ? 'crash_heavy' : 'crash_light', o);
      else this.play(heavy ? 'crash_heavy' : 'crash_light', { ...o, x: v.s.x, y: v.s.y, z: v.s.z });
    } catch {
      /* noop */
    }
  }

  // ───────────── one-shots ─────────────
  play(name: SfxName, opts?: { volume?: number; pitch?: number; x?: number; y?: number; z?: number }): void {
    try {
      const ctx = this.ctx;
      const b = this.buses;
      if (!ctx || !b) return;
      const list = this.sfx.get(name);
      if (!list || !list.length) return;
      const src = ctx.createBufferSource();
      src.buffer = list[(Math.random() * list.length) | 0];
      src.playbackRate.value = clamp(num(opts?.pitch ?? 1, 1), 0.25, 4);
      let vol = clamp(num(opts?.volume ?? 1, 1), 0, 4);
      let pan = 0;
      if (opts && opts.x !== undefined && opts.z !== undefined) {
        const sp = this.sp;
        this.spatial(num(opts.x, 0), num(opts.y ?? this.lis.y, 0), num(opts.z, 0), 0, 0, 0, sp);
        vol *= sp.gain;
        pan = sp.pan;
      }
      if (vol < 0.0005) return;
      const g = ctx.createGain();
      g.gain.value = vol;
      src.connect(g);
      let last: AudioNode = g;
      if (pan !== 0 && typeof ctx.createStereoPanner === 'function') {
        const p = ctx.createStereoPanner();
        p.pan.value = pan;
        g.connect(p);
        last = p;
      }
      last.connect(SFX_UI[name] ? b.ui : b.sfx);
      src.onended = () => {
        try {
          last.disconnect();
        } catch {
          /* noop */
        }
      };
      src.start();
    } catch {
      /* noop */
    }
  }

  // ───────────── music ─────────────
  playMusic(track: 'menu' | 'race' | 'results' | null, fadeSec = 1.5): void {
    this.wantMusic = track;
    try {
      if (this.music) this.music.play(track, clamp(num(fadeSec, 1.5), 0.02, 20));
    } catch {
      /* noop */
    }
  }
  setMusicIntensity(v: number): void {
    this.wantIntensity = clamp(num(v, 0.5), 0, 1);
    try {
      if (this.music) this.music.setIntensity(this.wantIntensity);
    } catch {
      /* noop */
    }
  }

  // ───────────── introspection / tests ─────────────
  debugInfo(): AudioDebugInfo {
    return {
      ready: this._ready,
      state: this.rt ? this.rt.state : this.ctx ? 'offline' : 'none',
      sampleRate: this.ctx ? this.ctx.sampleRate : 0,
      synth: this.host ? this.host.kind : 'none',
      voices: this.voices.length,
      sfxBuffers: this.sfx.size,
      music: this.music ? this.music.current : null,
      initMs: Math.round(this.initMs),
    };
  }
  /** The master bus (pre compressor/limiter) — lets tools tap or analyse the mix. */
  get masterNode(): AudioNode | null {
    return this.buses ? this.buses.master : null;
  }
  /** The final node feeding the destination (post compressor + limiter). */
  get outputNode(): AudioNode | null {
    return this.outNode;
  }
  get context(): BaseAudioContext | null {
    return this.ctx;
  }
  /** Offline rendering helper: schedule music events up to `untilTime` (context seconds). */
  scheduleMusic(untilTime: number): void {
    try {
      if (this.music) this.music.schedule(untilTime);
    } catch {
      /* noop */
    }
  }

  dispose(): void {
    this.disposed = true;
    try {
      if (this.timer) clearInterval(this.timer);
      this.timer = null;
      this.removeUnlock();
      if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', this.onVisibility);
      if (this.music) this.music.dispose();
      if (this.host) this.host.dispose();
      if (this.silentEl) {
        this.silentEl.pause();
        this.silentEl = null;
      }
      for (const v of this.voices) {
        v.dead = true;
        v.slot = -1;
      }
      this.voices.length = 0;
      this.slots.fill(null);
      if (this.rt && this.rt.state !== 'closed') void this.rt.close().catch(() => {});
    } catch {
      /* noop */
    }
    this.music = null;
    this.host = null;
    this.buses = null;
    this.outNode = null;
    this.amb = null;
    this.sfx.clear();
    this.ctx = null;
    this.rt = null;
    this.initPromise = null;
    this._ready = false;
    this.flushQueued = false;
  }
}

/** Shared singleton used by the game. */
export const audio = new AudioEngine();
