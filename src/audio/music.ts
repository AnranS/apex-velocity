/**
 * Procedural music: a look-ahead step sequencer on the Web Audio clock with three original cues.
 *   menu    — 84 bpm, D minor, slow cinematic pads / bells / sub
 *   race    — 132 bpm, A minor synthwave; layers open up with setIntensity(0..1)
 *   results — 112 bpm, C major, short triumphant loop
 * Works on any BaseAudioContext (AudioContext for the game, OfflineAudioContext for the tests: call
 * `schedule(untilTime)` up-front instead of `tick()`).
 */
import { renderDrum, DRUM_NAMES, type DrumName } from './sfx';
import { renderHallIR } from './ambience';
import { Rng, whiteLoop, midiHz, smooth } from './synth';

export type MusicCue = 'menu' | 'race' | 'results';

export interface MusicBank {
  drums: Record<DrumName, AudioBuffer>;
  noise: AudioBuffer;
  hallIR: AudioBuffer;
}

export function toBuffer(ctx: BaseAudioContext, chans: Float32Array[], rate = ctx.sampleRate): AudioBuffer {
  const b = ctx.createBuffer(chans.length, chans[0].length, rate);
  for (let c = 0; c < chans.length; c++) b.getChannelData(c).set(chans[c]);
  return b;
}

export function buildMusicBank(ctx: BaseAudioContext): MusicBank {
  const fs = ctx.sampleRate;
  const drums = {} as Record<DrumName, AudioBuffer>;
  for (const d of DRUM_NAMES) drums[d] = toBuffer(ctx, [renderDrum(d, fs)]);
  const ir = renderHallIR(fs);
  return {
    drums,
    noise: toBuffer(ctx, [whiteLoop(Math.round(fs * 2), new Rng(4242))]),
    hallIR: toBuffer(ctx, [ir.L, ir.R]),
  };
}

// ───────────────────────── score data ─────────────────────────
interface Chord {
  root: number;
  pad: number[];
  arp: number[];
}
type Note = [step: number, midi: number, len: number];

/** race: 16 bars, chords change every 2 bars. A: Am F C G — B: Dm Am F E */
const RACE_CHORDS: Chord[] = [
  { root: 33, pad: [57, 60, 64, 71], arp: [57, 60, 64] },
  { root: 29, pad: [53, 57, 60, 64], arp: [53, 57, 60] },
  { root: 36, pad: [55, 60, 64, 74], arp: [60, 64, 67] },
  { root: 31, pad: [55, 59, 62, 69], arp: [55, 59, 62] },
  { root: 38, pad: [53, 57, 60, 62], arp: [57, 62, 65] },
  { root: 33, pad: [57, 60, 64, 67], arp: [57, 60, 64] },
  { root: 29, pad: [53, 57, 60, 64], arp: [53, 57, 60] },
  { root: 28, pad: [52, 56, 59, 64], arp: [56, 59, 64] },
];
const RACE_ARP = [0, 2, 3, 2, 1, 3, 4, 3, 2, 4, 5, 4, 3, 2, 1, 2];
const RACE_LEAD: Note[][] = [
  [[0, 76, 6], [6, 74, 2], [8, 72, 4], [12, 69, 4]],
  [[0, 72, 6], [6, 74, 2], [8, 76, 8]],
  [[0, 77, 6], [6, 76, 2], [8, 72, 4], [12, 69, 4]],
  [[0, 72, 6], [6, 69, 2], [8, 69, 8]],
  [[0, 79, 6], [6, 76, 2], [8, 72, 4], [12, 76, 4]],
  [[0, 79, 4], [4, 81, 4], [8, 79, 8]],
  [[0, 74, 6], [6, 76, 2], [8, 74, 4], [12, 71, 4]],
  [[0, 74, 8], [8, 71, 4], [12, 67, 4]],
  [[0, 77, 4], [4, 76, 2], [6, 74, 2], [8, 69, 4], [12, 74, 4]],
  [[0, 77, 6], [6, 79, 2], [8, 81, 8]],
  [[0, 76, 6], [6, 72, 2], [8, 69, 8]],
  [[0, 72, 4], [4, 74, 4], [8, 76, 8]],
  [[0, 77, 6], [6, 76, 2], [8, 72, 4], [12, 69, 4]],
  [[0, 69, 4], [4, 72, 4], [8, 77, 8]],
  [[0, 76, 6], [6, 80, 2], [8, 83, 8]],
  [[0, 83, 4], [4, 80, 4], [8, 76, 4], [12, 71, 4]],
];

/** menu: 16 bars, chords change every 2 bars. Dm9 Bbmaj7 Fadd9 Cadd9 — Gm7 Dm/A Bbmaj7 Asus4 */
const MENU_CHORDS: Chord[] = [
  { root: 38, pad: [50, 53, 57, 60, 64], arp: [] },
  { root: 34, pad: [46, 53, 57, 62], arp: [] },
  { root: 41, pad: [53, 57, 60, 67], arp: [] },
  { root: 36, pad: [48, 55, 60, 62, 64], arp: [] },
  { root: 43, pad: [55, 58, 62, 65], arp: [] },
  { root: 33, pad: [57, 62, 65, 69], arp: [] },
  { root: 34, pad: [58, 62, 65, 69], arp: [] },
  { root: 33, pad: [57, 62, 64, 69], arp: [] },
];
const MENU_BELL: [number, number][][] = [
  [[0, 69], [8, 74]],
  [[0, 77], [8, 76], [12, 74]],
  [[0, 74], [8, 77]],
  [[0, 81]],
  [[0, 72], [8, 69]],
  [[0, 67], [8, 69], [12, 72]],
  [[0, 76], [8, 74]],
  [[0, 72], [8, 67]],
  [[0, 70], [8, 74]],
  [[0, 79], [8, 77], [12, 74]],
  [[0, 77], [8, 74]],
  [[0, 69], [10, 74]],
  [[0, 74], [8, 77]],
  [[0, 81], [8, 77]],
  [[0, 76], [8, 74]],
  [[0, 73], [8, 69]],
];

/** results: 8 bars, one chord per bar. C G/B Am F C/E F G Gsus4 */
const RESULT_CHORDS: Chord[] = [
  { root: 36, pad: [55, 60, 64, 67], arp: [60, 64, 67] },
  { root: 47, pad: [55, 59, 62, 67], arp: [59, 62, 67] },
  { root: 45, pad: [57, 60, 64, 69], arp: [57, 60, 64] },
  { root: 41, pad: [53, 57, 60, 65], arp: [53, 57, 60] },
  { root: 40, pad: [55, 60, 64, 67], arp: [55, 60, 64] },
  { root: 41, pad: [57, 60, 65, 69], arp: [57, 60, 65] },
  { root: 43, pad: [55, 59, 62, 67], arp: [55, 59, 62] },
  { root: 43, pad: [55, 60, 62, 67], arp: [55, 60, 62] },
];
const RESULT_LEAD: Note[][] = [
  [[0, 67, 2], [2, 72, 2], [4, 76, 4], [8, 79, 8]],
  [[0, 74, 4], [4, 71, 4], [8, 79, 8]],
  [[0, 76, 4], [4, 72, 4], [8, 81, 8]],
  [[0, 77, 4], [4, 81, 4], [8, 84, 8]],
  [[0, 79, 6], [6, 76, 2], [8, 72, 8]],
  [[0, 81, 6], [6, 77, 2], [8, 72, 8]],
  [[0, 83, 4], [4, 86, 4], [8, 79, 8]],
  [[0, 79, 4], [4, 81, 4], [8, 83, 8]],
];

const CUE_BPM: Record<MusicCue, number> = { menu: 84, race: 132, results: 112 };
/** overall cue trim so the three cues sit at a similar loudness */
const CUE_TRIM: Record<MusicCue, number> = { menu: 0.75, race: 0.62, results: 0.6 };

// ───────────────────────── one playing cue ─────────────────────────
class Cue {
  readonly name: MusicCue;
  readonly stepDur: number;
  readonly t0: number;
  step = 0;
  nextT: number;
  endTime = Infinity;
  private readonly ctx: BaseAudioContext;
  private readonly bank: MusicBank;
  private readonly gain: GainNode;
  private readonly tone: BiquadFilterNode;
  private readonly pump: GainNode;
  private readonly bDrum: GainNode;
  private readonly bKick: GainNode;
  private readonly bHat: GainNode;
  private readonly bBass: GainNode;
  private readonly bPad: GainNode;
  private readonly bArp: GainNode;
  private readonly bLead: GainNode;
  private readonly bFx: GainNode;
  private readonly padFilt: BiquadFilterNode;
  private readonly arpL: AudioNode;
  private readonly arpR: AudioNode;
  private readonly padL: AudioNode;
  private readonly padR: AudioNode;
  private readonly hatP: AudioNode;
  private arpFlip = false;
  private readonly nodes: AudioNode[] = [];
  private lastLead = 0;
  private I = 0.5;

  constructor(ctx: BaseAudioContext, bank: MusicBank, name: MusicCue, t0: number, out: AudioNode, rev: AudioNode, dly: AudioNode, intensity: number, fadeIn: number) {
    this.ctx = ctx;
    this.bank = bank;
    this.name = name;
    this.t0 = t0;
    this.nextT = t0;
    this.stepDur = 60 / CUE_BPM[name] / 4;
    this.I = intensity;
    const g = (v: number): GainNode => {
      const n = ctx.createGain();
      n.gain.value = v;
      this.nodes.push(n);
      return n;
    };
    this.gain = g(0);
    this.gain.gain.setValueAtTime(0, t0);
    this.gain.gain.linearRampToValueAtTime(CUE_TRIM[name], t0 + Math.max(0.02, fadeIn));
    this.gain.connect(out);
    this.tone = ctx.createBiquadFilter();
    this.tone.type = 'lowpass';
    this.tone.Q.value = 0.6;
    this.tone.frequency.value = 18000;
    this.nodes.push(this.tone);
    this.tone.connect(this.gain);
    this.pump = g(1);
    this.pump.connect(this.tone);
    this.bDrum = g(1);
    this.bKick = g(1);
    this.bHat = g(1);
    this.bBass = g(1);
    this.bPad = g(1);
    this.bArp = g(1);
    this.bLead = g(1);
    this.bFx = g(1);
    this.bKick.connect(this.tone);
    this.bDrum.connect(this.tone);
    this.bHat.connect(this.tone);
    this.bBass.connect(this.pump);
    this.padFilt = ctx.createBiquadFilter();
    this.padFilt.type = 'lowpass';
    this.padFilt.Q.value = 0.7;
    this.padFilt.frequency.value = 1500;
    this.nodes.push(this.padFilt);
    this.bPad.connect(this.padFilt);
    this.padFilt.connect(this.pump);
    this.bArp.connect(this.pump);
    this.bLead.connect(this.tone);
    this.bFx.connect(this.tone);
    // fixed pan positions (no per-note panner nodes): arp ping-pongs, pad voices are split left/right
    const panned = (dest: AudioNode, pan: number): AudioNode => {
      if (typeof ctx.createStereoPanner !== 'function') return dest;
      const p = ctx.createStereoPanner();
      p.pan.value = pan;
      p.connect(dest);
      this.nodes.push(p);
      return p;
    };
    this.arpL = panned(this.bArp, -0.55);
    this.arpR = panned(this.bArp, 0.55);
    this.padL = panned(this.bPad, -0.6);
    this.padR = panned(this.bPad, 0.6);
    this.hatP = panned(this.bHat, 0.25);
    // sends (through the cue gain so they fade with the cue)
    const send = (src: AudioNode, to: AudioNode, amt: number): void => {
      const s = g(amt);
      src.connect(s);
      const post = g(0);
      post.gain.setValueAtTime(0, t0);
      post.gain.linearRampToValueAtTime(CUE_TRIM[name], t0 + Math.max(0.02, fadeIn));
      s.connect(post);
      post.connect(to);
      this.sends.push(post);
    };
    send(this.padFilt, rev, name === 'menu' ? 0.5 : 0.28);
    send(this.bArp, rev, 0.22);
    send(this.bArp, dly, name === 'menu' ? 0.45 : 0.32);
    send(this.bLead, rev, name === 'menu' ? 0.7 : 0.3);
    send(this.bLead, dly, name === 'menu' ? 0.5 : 0.28);
    send(this.bDrum, rev, 0.08);
    send(this.bFx, rev, 0.4);
    this.applyIntensity(intensity, t0, true);
  }
  private readonly sends: GainNode[] = [];

  /** Fade out and stop scheduling. */
  stop(t: number, fade: number): void {
    const tc = Math.max(0.01, fade / 4);
    for (const n of [this.gain, ...this.sends]) {
      const p = n.gain as AudioParam & { cancelAndHoldAtTime?: (t: number) => AudioParam };
      if (typeof p.cancelAndHoldAtTime === 'function') p.cancelAndHoldAtTime(t);
      else p.cancelScheduledValues(t);
      p.setTargetAtTime(0, t, tc);
    }
    this.endTime = t + fade + 0.6;
    this.nextT = Infinity;
  }
  dispose(): void {
    for (const n of this.nodes) {
      try {
        n.disconnect();
      } catch {
        /* already gone */
      }
    }
  }

  applyIntensity(I: number, t: number, instant = false): void {
    this.I = I;
    const set = (p: AudioParam, v: number): void => {
      if (instant) p.setValueAtTime(v, t);
      else p.setTargetAtTime(v, t, 0.35);
    };
    if (this.name === 'race') {
      set(this.tone.frequency, Math.min(19000, 900 * Math.pow(2, I * 4.5)));
      set(this.bPad.gain, 0.9);
      set(this.bBass.gain, smooth(0.03, 0.22, I));
      set(this.bKick.gain, smooth(0.08, 0.28, I));
      set(this.bHat.gain, smooth(0.2, 0.45, I));
      set(this.bDrum.gain, smooth(0.3, 0.5, I));
      set(this.bArp.gain, smooth(0.4, 0.65, I));
      set(this.bLead.gain, smooth(0.7, 0.9, I));
      set(this.padFilt.frequency, 900 + 1800 * I);
    } else if (this.name === 'menu') {
      set(this.padFilt.frequency, 1900);
    } else {
      set(this.padFilt.frequency, 2400);
    }
  }

  // ───── instruments ─────
  private done(src: AudioScheduledSourceNode, last: AudioNode): void {
    src.onended = () => {
      try {
        last.disconnect();
      } catch {
        /* noop */
      }
    };
  }

  private hit(name: DrumName, t: number, vel: number, bus: AudioNode, rate = 1): void {
    const ctx = this.ctx;
    const s = ctx.createBufferSource();
    s.buffer = this.bank.drums[name];
    s.playbackRate.value = rate;
    const g = ctx.createGain();
    g.gain.value = vel;
    s.connect(g);
    g.connect(bus);
    s.start(t);
    this.done(s, g);
  }

  private bass(t: number, midi: number, dur: number, vel: number): void {
    const ctx = this.ctx;
    const f = midiHz(midi);
    const o1 = ctx.createOscillator();
    o1.type = 'sawtooth';
    o1.frequency.value = f;
    const o2 = ctx.createOscillator();
    o2.type = 'square';
    o2.frequency.value = f;
    o2.detune.value = -7;
    const o3 = ctx.createOscillator();
    o3.type = 'sine';
    o3.frequency.value = f;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.Q.value = 3.5;
    lp.frequency.setValueAtTime(260 + 1500 * vel, t);
    lp.frequency.exponentialRampToValueAtTime(240, t + Math.min(dur, 0.16));
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.07 * vel, t + 0.005);
    g.gain.setValueAtTime(0.07 * vel, t + Math.max(0.01, dur - 0.03));
    g.gain.linearRampToValueAtTime(0, t + dur);
    const sub = ctx.createGain();
    sub.gain.value = 0.7;
    o1.connect(lp);
    o2.connect(lp);
    lp.connect(g);
    o3.connect(sub);
    sub.connect(g);
    g.connect(this.bBass);
    for (const o of [o1, o2, o3]) {
      o.start(t);
      o.stop(t + dur + 0.02);
    }
    this.done(o1, g);
  }

  private pad(t: number, notes: number[], dur: number, vel: number, attack = 0.35, release = 0.9): void {
    const ctx = this.ctx;
    const hold = Math.max(attack, dur);
    const env = (dest: AudioNode): GainNode => {
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel, t + attack);
      g.gain.setValueAtTime(vel, t + hold);
      g.gain.linearRampToValueAtTime(0, t + hold + release);
      g.connect(dest);
      return g;
    };
    const gL = env(this.padL);
    const gR = env(this.padR);
    let first: OscillatorNode | null = null;
    let k = 0;
    for (const m of notes) {
      for (const det of [-8, 7]) {
        const o = ctx.createOscillator();
        o.type = 'sawtooth';
        o.frequency.value = midiHz(m);
        o.detune.value = det;
        const og = ctx.createGain();
        og.gain.value = 0.1;
        o.connect(og);
        og.connect((k++ & 1) === ((m >> 0) & 1) ? gL : gR);
        o.start(t);
        o.stop(t + hold + release + 0.05);
        if (!first) first = o;
      }
    }
    if (first) {
      first.onended = () => {
        try {
          gL.disconnect();
          gR.disconnect();
        } catch {
          /* noop */
        }
      };
    }
  }

  private pluck(t: number, midi: number, dur: number, vel: number, bright = 1): void {
    const ctx = this.ctx;
    const f = midiHz(midi);
    const o1 = ctx.createOscillator();
    o1.type = 'sawtooth';
    o1.frequency.value = f;
    const o2 = ctx.createOscillator();
    o2.type = 'square';
    o2.frequency.value = f;
    o2.detune.value = 6;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.Q.value = 2.5;
    lp.frequency.setValueAtTime(Math.min(12000, (1800 + 2600 * vel) * bright), t);
    lp.frequency.exponentialRampToValueAtTime(650, t + 0.11);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.16 * vel, t + 0.003);
    g.gain.exponentialRampToValueAtTime(0.0008, t + dur);
    o1.connect(lp);
    o2.connect(lp);
    lp.connect(g);
    g.connect(this.arpFlip ? this.arpL : this.arpR);
    this.arpFlip = !this.arpFlip;
    o1.start(t);
    o2.start(t);
    o1.stop(t + dur + 0.02);
    o2.stop(t + dur + 0.02);
    this.done(o1, g);
  }

  private lead(t: number, midi: number, dur: number, vel: number, brass = false): void {
    const ctx = this.ctx;
    const f = midiHz(midi);
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.Q.value = brass ? 0.9 : 1.2;
    if (brass) {
      lp.frequency.setValueAtTime(500, t);
      lp.frequency.linearRampToValueAtTime(3400, t + 0.05);
      lp.frequency.setTargetAtTime(1900, t + 0.05, 0.3);
    } else lp.frequency.value = 3000;
    const g = ctx.createGain();
    const a = brass ? 0.025 : 0.014;
    const lvl = (brass ? 0.15 : 0.13) * vel;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(lvl, t + a);
    g.gain.setValueAtTime(lvl, t + Math.max(a, dur - 0.02));
    g.gain.linearRampToValueAtTime(0, t + dur + 0.1);
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 5.6;
    const lg = ctx.createGain();
    lg.gain.setValueAtTime(0, t);
    lg.gain.setValueAtTime(0, t + 0.14);
    lg.gain.linearRampToValueAtTime(brass ? 5 : 9, t + 0.4);
    lfo.connect(lg);
    const oscs: OscillatorNode[] = [];
    for (const det of brass ? [-9, 0, 8] : [-7, 7]) {
      const o = ctx.createOscillator();
      o.type = 'sawtooth';
      if (!brass && this.lastLead > 0) {
        o.frequency.setValueAtTime(midiHz(this.lastLead), t);
        o.frequency.exponentialRampToValueAtTime(f, t + 0.045);
      } else o.frequency.value = f;
      o.detune.value = det;
      lg.connect(o.detune);
      o.connect(lp);
      oscs.push(o);
    }
    lp.connect(g);
    g.connect(this.bLead);
    for (const o of [...oscs, lfo]) {
      o.start(t);
      o.stop(t + dur + 0.14);
    }
    this.lastLead = midi;
    this.done(oscs[0], g);
  }

  private bell(t: number, midi: number, vel: number, decay: number): void {
    const ctx = this.ctx;
    const f = midiHz(midi);
    const g = ctx.createGain();
    g.gain.value = 1;
    g.connect(this.bLead);
    const parts: [number, number, number][] = [
      [1, 0.2, 1],
      [2, 0.07, 0.6],
      [3.01, 0.035, 0.4],
      [0.5, 0.06, 1.2],
    ];
    let first: OscillatorNode | null = null;
    for (const [r, a, d] of parts) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.value = f * r;
      const og = ctx.createGain();
      og.gain.setValueAtTime(0, t);
      og.gain.linearRampToValueAtTime(a * vel, t + 0.004);
      og.gain.setTargetAtTime(0, t + 0.004, (decay * d) / 3);
      o.connect(og);
      og.connect(g);
      o.start(t);
      o.stop(t + decay * d * 2.2 + 0.05);
      if (!first || d > 1) first = o;
    }
    if (first) this.done(first, g);
  }

  private sub(t: number, midi: number, dur: number, vel: number): void {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = midiHz(midi);
    const o2 = ctx.createOscillator();
    o2.type = 'triangle';
    o2.frequency.value = midiHz(midi + 12);
    const g2 = ctx.createGain();
    g2.gain.value = 0.18;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vel, t + 0.6);
    g.gain.setValueAtTime(vel, t + dur - 0.3);
    g.gain.linearRampToValueAtTime(0, t + dur + 0.5);
    o.connect(g);
    o2.connect(g2);
    g2.connect(g);
    g.connect(this.bBass);
    o.start(t);
    o2.start(t);
    o.stop(t + dur + 0.55);
    o2.stop(t + dur + 0.55);
    this.done(o, g);
  }

  private riser(t: number, dur: number, vel: number): void {
    const ctx = this.ctx;
    const s = ctx.createBufferSource();
    s.buffer = this.bank.noise;
    s.loop = true;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.Q.value = 1.4;
    bp.frequency.setValueAtTime(300, t);
    bp.frequency.exponentialRampToValueAtTime(7000, t + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vel, t + dur * 0.95);
    g.gain.linearRampToValueAtTime(0, t + dur);
    s.connect(bp);
    bp.connect(g);
    g.connect(this.bFx);
    s.start(t);
    s.stop(t + dur + 0.02);
    this.done(s, g);
  }

  private duck(t: number, depth: number): void {
    const p = this.pump.gain;
    p.setValueAtTime(1 - depth, t);
    p.setTargetAtTime(1, t + 0.015, this.stepDur * 0.9);
  }

  /** After a scheduling gap: restart the sustained parts (pad / sub) of the chord we landed in. */
  resync(step: number, t: number): void {
    const sd = this.stepDur;
    const s = step % 16;
    if (this.name === 'results') {
      const left = 16 - s;
      if (s > 0 && left > 3) this.pad(t, RESULT_CHORDS[Math.floor(step / 16) % 8].pad, sd * left, 0.8, 0.12, 0.5);
      return;
    }
    const bar = Math.floor(step / 16) % 16;
    const into = (bar & 1) * 16 + s;
    const left = 32 - into;
    if (into === 0 || left < 4) return;
    if (this.name === 'race') this.pad(t, RACE_CHORDS[bar >> 1].pad, sd * left, 0.85);
    else {
      this.pad(t, MENU_CHORDS[bar >> 1].pad, sd * left, 0.85, 1.0, 2.2);
      this.sub(t, MENU_CHORDS[bar >> 1].root, Math.max(1, sd * left), 0.13);
    }
  }

  // ───── the three scores ─────
  doStep(step: number, t: number): void {
    if (this.name === 'race') this.stepRace(step, t);
    else if (this.name === 'menu') this.stepMenu(step, t);
    else this.stepResults(step, t);
  }

  private stepRace(step: number, t: number): void {
    const sd = this.stepDur;
    const I = this.I;
    const bar = Math.floor(step / 16) % 16;
    const s = step % 16;
    const ch = RACE_CHORDS[bar >> 1];
    if (s === 0 && (bar & 1) === 0) this.pad(t, ch.pad, sd * 32, 0.85);
    // bass: 8ths with off-beat octaves, rolling 16ths when it gets intense
    if ((s & 1) === 0) this.bass(t, ch.root + 12 + (s === 6 || s === 14 ? 12 : 0), sd * 1.7, s % 4 === 0 ? 1 : 0.82);
    else if (I > 0.6) this.bass(t, ch.root + 12 + (s === 11 ? 12 : 0), sd * 0.85, 0.5);
    // drums
    if (s % 4 === 0) {
      this.hit('kick', t, 0.45, this.bKick);
      this.duck(t, 0.2 + 0.4 * I);
    }
    const fill = bar === 7 || bar === 15;
    if (s === 4 || s === 12) {
      this.hit('snare', t, 0.55, this.bDrum);
      this.hit('clap', t, 0.4, this.bDrum);
    }
    if (fill && s >= 12) {
      this.hit('snare', t, 0.3 + 0.08 * (s - 12), this.bDrum, 1.05);
      if (s >= 14) this.hit('tom', t, 0.5, this.bDrum, s === 14 ? 1.3 : 1);
    }
    if (I < 0.55) {
      if (s % 4 === 2) this.hit('hat', t, 0.45, this.hatP);
    } else {
      this.hit('hat', t, s % 4 === 2 ? 0.5 : (s & 1) === 0 ? 0.28 : 0.18, this.hatP, s % 4 === 2 ? 1 : 1.08);
    }
    if (s === 14 && (bar & 1) === 1) this.hit('hatOpen', t, 0.24, this.hatP);
    if (s === 0 && (bar === 0 || bar === 8)) this.hit('crash', t, 0.3, this.bDrum);
    // arp
    const a = ch.arp;
    const idx = RACE_ARP[s];
    this.pluck(t, a[idx % 3] + 12 * Math.floor(idx / 3) + 12, sd * 2.2, s % 4 === 0 ? 0.9 : 0.62);
    // lead
    for (const n of RACE_LEAD[bar]) if (n[0] === s) this.lead(t, n[1], sd * n[2] * 0.94, 0.9);
    if (s === 0 && bar === 15) this.riser(t, sd * 16, 0.12);
  }

  private stepMenu(step: number, t: number): void {
    const sd = this.stepDur;
    const bar = Math.floor(step / 16) % 16;
    const s = step % 16;
    const pass = Math.floor(step / 256);
    const ch = MENU_CHORDS[bar >> 1];
    if (s === 0 && (bar & 1) === 0) {
      this.pad(t, ch.pad, sd * 32, 0.85, 1.6, 2.2);
      this.sub(t, ch.root, sd * 32, 0.13);
    }
    for (const n of MENU_BELL[bar]) if (n[0] === s) this.bell(t, n[1], 0.9, 2.6);
    // soft pulse of chord tones
    if ((s & 1) === 0 && (pass > 0 || bar >= 4)) {
      const p = ch.pad;
      this.pluck(t, p[(s >> 1) % p.length] + 12, sd * 3, 0.36, 0.8);
    }
    if (s === 0 && (bar === 0 || bar === 8)) this.hit('boom', t, 0.28, this.bDrum);
    if ((pass > 0 || bar >= 4) && (s === 0 || s === 10)) this.hit('kick', t, s === 0 ? 0.22 : 0.14, this.bKick, 0.8);
    if ((pass > 0 || bar >= 8) && s % 4 === 2) this.hit('hat', t, 0.17, this.hatP, 0.9);
    if (s === 0 && (bar === 7 || bar === 15)) this.riser(t, sd * 16, 0.05);
  }

  private stepResults(step: number, t: number): void {
    const sd = this.stepDur;
    const bar = Math.floor(step / 16) % 8;
    const s = step % 16;
    const ch = RESULT_CHORDS[bar];
    if (s === 0) this.pad(t, ch.pad, sd * 16, 0.8, 0.12, 0.5);
    if (s === 0 || s === 6 || s === 10) this.pad(t, ch.pad.map((m) => m + 12), sd * 1.6, 0.5, 0.012, 0.16);
    if ((s & 1) === 0) this.bass(t, ch.root + (s % 4 === 2 ? 12 : 0), sd * 1.7, s % 4 === 0 ? 1 : 0.8);
    if (s % 4 === 0) {
      this.hit('kick', t, 0.45, this.bKick);
      this.duck(t, 0.4);
    }
    if (s === 4 || s === 12) this.hit('clap', t, 0.5, this.bDrum);
    if ((s & 1) === 0) this.hit('hat', t, s % 4 === 2 ? 0.48 : 0.26, this.hatP);
    if (s === 0 && bar === 0) this.hit('crash', t, 0.32, this.bDrum);
    if (bar === 7 && s >= 12) this.hit('snare', t, 0.3 + 0.08 * (s - 12), this.bDrum);
    const a = ch.arp;
    const idx = RACE_ARP[s];
    this.pluck(t, a[idx % 3] + 12 * Math.floor(idx / 3) + 12, sd * 2, s % 4 === 0 ? 0.75 : 0.5);
    for (const n of RESULT_LEAD[bar]) if (n[0] === s) this.lead(t, n[1], sd * n[2] * 0.92, 0.95, true);
  }
}

// ───────────────────────── sequencer ─────────────────────────
export class MusicSequencer {
  private readonly ctx: BaseAudioContext;
  private readonly bank: MusicBank;
  private readonly out: GainNode;
  private readonly rev: ConvolverNode;
  private readonly revOut: GainNode;
  private readonly dly: DelayNode;
  private readonly dlyFb: GainNode;
  private readonly dlyLp: BiquadFilterNode;
  private readonly dlyOut: GainNode;
  private cur: Cue | null = null;
  private old: Cue[] = [];
  private intensity = 0.5;
  private targetI = 0.5;
  /** seconds of look-ahead used by tick() */
  lookAhead = 0.45;

  constructor(ctx: BaseAudioContext, dest: AudioNode, bank: MusicBank) {
    this.ctx = ctx;
    this.bank = bank;
    this.out = ctx.createGain();
    this.out.gain.value = 1;
    this.out.connect(dest);
    this.rev = ctx.createConvolver();
    this.rev.buffer = bank.hallIR;
    this.revOut = ctx.createGain();
    this.revOut.gain.value = 0.9;
    this.rev.connect(this.revOut);
    this.revOut.connect(this.out);
    this.dly = ctx.createDelay(1.5);
    this.dly.delayTime.value = 0.34;
    this.dlyFb = ctx.createGain();
    this.dlyFb.gain.value = 0.36;
    this.dlyLp = ctx.createBiquadFilter();
    this.dlyLp.type = 'lowpass';
    this.dlyLp.frequency.value = 3200;
    this.dlyOut = ctx.createGain();
    this.dlyOut.gain.value = 0.8;
    this.dly.connect(this.dlyLp);
    this.dlyLp.connect(this.dlyFb);
    this.dlyFb.connect(this.dly);
    this.dlyLp.connect(this.dlyOut);
    this.dlyOut.connect(this.out);
  }

  get current(): MusicCue | null {
    return this.cur ? this.cur.name : null;
  }

  /** Switch cue (null = silence) with a crossfade. `when` defaults to "now". */
  play(name: MusicCue | null, fadeSec = 1.5, when?: number): void {
    const now = this.ctx.currentTime;
    const t = Math.max(when ?? 0, now + 0.06);
    if (this.cur && this.cur.name === name) return;
    if (this.cur) {
      this.cur.stop(t, Math.max(0.05, fadeSec));
      this.old.push(this.cur);
      this.cur = null;
    }
    if (name) {
      // dotted-eighth echo locked to the tempo
      const beat = 60 / CUE_BPM[name];
      this.dly.delayTime.setTargetAtTime(Math.min(1.4, beat * 0.75), t, 0.05);
      this.cur = new Cue(this.ctx, this.bank, name, t, this.out, this.rev, this.dly, this.intensity, Math.min(fadeSec, name === 'menu' ? 2.5 : 0.6));
    }
  }

  /** 0..1 — applied on the next tick()/schedule() call. */
  setIntensity(v: number): void {
    this.targetI = v !== v ? 0.5 : Math.min(1, Math.max(0, v));
  }

  /** Apply an intensity change at an explicit context time (offline rendering / tests). */
  setIntensityAt(v: number, t: number): void {
    this.targetI = this.intensity = Math.min(1, Math.max(0, v));
    if (this.cur) this.cur.applyIntensity(this.intensity, t);
  }

  private syncIntensity(): void {
    if (Math.abs(this.targetI - this.intensity) < 0.01) return;
    this.intensity = this.targetI;
    if (this.cur) this.cur.applyIntensity(this.intensity, this.ctx.currentTime + 0.01);
  }

  /** Schedule every step that starts before `until` (context time). */
  schedule(until: number): void {
    this.syncIntensity();
    const c = this.cur;
    if (!c) return;
    let guard = 0;
    while (c.nextT < until && guard++ < 20000) {
      c.doStep(c.step, c.nextT);
      c.step++;
      c.nextT = c.t0 + c.step * c.stepDur;
    }
  }

  /** Real-time pump: call every ~60 ms. */
  tick(): void {
    const now = this.ctx.currentTime;
    const c = this.cur;
    if (c && c.nextT < now + 0.01) {
      // the main thread stalled past our look-ahead: jump to the current position on the grid instead of
      // machine-gunning the backlog, and re-voice whatever should be sustaining right now
      const target = Math.ceil((now + 0.04 - c.t0) / c.stepDur);
      if (target > c.step) {
        c.step = target;
        c.nextT = c.t0 + c.step * c.stepDur;
        c.resync(c.step, c.nextT);
      }
    }
    this.schedule(now + this.lookAhead);
    if (this.old.length) {
      this.old = this.old.filter((o) => {
        if (now > o.endTime) {
          o.dispose();
          return false;
        }
        return true;
      });
    }
  }

  dispose(): void {
    if (this.cur) this.cur.dispose();
    for (const o of this.old) o.dispose();
    this.cur = null;
    this.old = [];
    for (const n of [this.out, this.rev, this.revOut, this.dly, this.dlyFb, this.dlyLp, this.dlyOut]) {
      try {
        n.disconnect();
      } catch {
        /* noop */
      }
    }
  }
}
