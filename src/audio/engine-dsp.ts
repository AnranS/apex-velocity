/**
 * CarSynthDSP — real-time synthesiser for car voices (engine + exhaust + tyres/surface layers).
 *
 * One instance renders up to `maxVoices` cars into two stereo buses:
 *   bus 0 = engine / exhaust / nitro,  bus 1 = tyres / surface / scrape / wind.
 *
 * Engine model (physically motivated source–filter):
 *   crank phase accumulator (one 720° cycle) → per-cylinder firing events (firing order, per-cylinder
 *   strength, cycle-to-cycle jitter) → sub-sample accurate impulses + pulse-synchronous turbulence noise
 *   → per-bank pulse shaper (order-locked low-pass whose bandwidth opens with load)
 *   → per-bank exhaust resonator bank (fixed formants = pipes/muffler) → bank mix
 *   + intake resonance (load²), mechanical noise (rpm), gear/turbo whine, overrun crackle, backfire pops,
 *   shift clunk, limiter ignition cut → level normalisation (calibration tables) → asymmetric soft clip
 *   → loudness curve (rpm, load) → DC blocker → air absorption → gain / equal-power pan.
 *
 * !!! SELF-CONTAINED BY DESIGN !!!
 * The class is shipped to the AudioWorklet as text (`CarSynthDSP.toString()`), so the class BODY must not
 * reference any module-scope identifier (no imports, helper functions, enums or constants), must not use
 * class-field initialisers, and must stick to syntax that needs no transpiler helpers.
 * Types/interfaces are fine (erased at build time).
 */

/** Plain-data description of an engine. Passed to the DSP by value (structured clone). */
export interface EngineCfg {
  /** number of cylinders */
  cyl: number;
  /** firing phase of every event inside one 720° cycle, ascending, 0..1 */
  fire: number[];
  /** exhaust bank (0/1) of every event */
  bank: number[];
  /** relative pulse strength of every event */
  amp: number[];
  /** number of exhaust banks (1 or 2) */
  banks: number;
  /** random cycle-to-cycle pulse strength variation (0..0.3) */
  jitter: number;
  /** probability of a weak pulse at idle (lumpy cam) */
  misfire: number;
  /** pulse shaper cutoff = lpBase + firingHz * lerp(hOff, hOn, load^0.7) */
  lpBase: number;
  hOff: number;
  hOn: number;
  /** upper limit of the pulse shaper cutoff (tail-pipe / muffler bandwidth, Hz) */
  lpMax: number;
  /** firing-time jitter as a fraction of the 720° cycle (blurs high orders like a real engine) */
  tjit: number;
  /** direct (un-resonated) pulse gain */
  dry: number;
  /** 0..1 order-locked pulse body (one-pole at 1.2x firing frequency) — keeps the firing fundamental dominant */
  body: number;
  /** acoustic path difference between the two banks (s) — keeps the half-order burble alive in mono */
  bankDelay: number;
  /** four exhaust formants */
  formF: number[];
  formQ: number[];
  formG: number[];
  /** formant detune ratio of bank 1 relative to bank 0 */
  detune: number;
  /** bank cross-feed: 0 = hard left/right, 0.5 = mono */
  cross: number;
  /** pulse-synchronous turbulence noise amount */
  pnoise: number;
  intakeF: number;
  intakeQ: number;
  intakeG: number;
  mechF: number;
  mechG: number;
  /** gear / supercharger whine amount and engine order */
  whine: number;
  whineOrder: number;
  /** turbo whistle + blow-off amount (0 = naturally aspirated) */
  turbo: number;
  /** overrun crackle amount */
  crackle: number;
  /** soft-clip drive */
  drive: number;
  /** exponent compensating level growth with firing rate */
  rateComp: number;
  /** backfire body resonance (Hz) */
  popF: number;
  /** calibration: rpmNorm grid [calR0..calR1] with calN rows x 3 loads (0, 0.5, 1), row-major */
  calR0: number;
  calR1: number;
  calN: number;
  /** measured raw RMS (before normalisation) */
  pre: number[];
  /** measured RMS after normalisation + clip */
  post: number[];
}

/** Internal per-voice state (plain object; all numbers). */
export interface SynthVoice {
  cfg: EngineCfg;
  player: boolean;
  debug: number;
  // targets (last state message)
  active: number;
  tRpm: number;
  tNorm: number;
  tLoad: number;
  tSpeed: number;
  tNitro: number;
  tSlip: number;
  tSurf: number;
  tScrape: number;
  tGain: number;
  tPan: number;
  tPitch: number;
  tAir: number;
  sinceUpd: number;
  // smoothed controls
  rpm: number;
  norm: number;
  load: number;
  speed: number;
  nitro: number;
  sq: number;
  scrape: number;
  gain: number;
  gl: number;
  gr: number;
  pan: number;
  pitch: number;
  air: number;
  wKerb: number;
  wGrass: number;
  wGravel: number;
  drift: number;
  // engine oscillator
  phase: number;
  inc: number;
  nextIdx: number;
  carry0: number;
  carry1: number;
  tj: number;
  pe0: number;
  pe1: number;
  lp0: number;
  lp1: number;
  aBase: number;
  gN: number;
  pG: number;
  // timers / envelopes
  cutT: number;
  blipT: number;
  limT: number;
  limPh: number;
  crackEnv: number;
  crackBoost: number;
  loadHi: number;
  blowT: number;
  blowCd: number;
  nBurst: number;
  turboSpd: number;
  popQ: Float32Array;
  popAct: number;
  popEnv: number;
  clunk: number;
  clunkAct: number;
  // filters
  coef: Float64Array;
  st: Float64Array;
  fcLast: number;
  pitchLast: number;
  // oscillators / misc
  whinePh: number;
  turboPh: number;
  sqPh1: number;
  sqPh2: number;
  kerbPh: number;
  lfo1: number;
  lfo2: number;
  lfo3: number;
  lfo4: number;
  seed: number;
  dcxL: number;
  dcyL: number;
  dcxR: number;
  dcyR: number;
  airL: number;
  airR: number;
  airT: number;
}

export class CarSynthDSP {
  declare fs: number;
  declare maxVoices: number;
  declare voices: (SynthVoice | null)[];
  declare e0: Float32Array;
  declare e1: Float32Array;
  declare a0: Float32Array;
  declare a1: Float32Array;
  declare xl: Float32Array;
  declare xr: Float32Array;
  declare pp: Float32Array;
  declare wh: Float32Array;
  declare nz: Float32Array;
  declare sc: Float32Array;
  declare ty: Float32Array;
  declare wl: Float32Array;
  declare wr: Float32Array;

  constructor(fs: number, maxVoices: number) {
    this.fs = fs;
    this.maxVoices = maxVoices;
    this.voices = [];
    for (let i = 0; i < maxVoices; i++) this.voices.push(null);
    const B = 130;
    this.e0 = new Float32Array(B);
    this.e1 = new Float32Array(B);
    this.a0 = new Float32Array(B);
    this.a1 = new Float32Array(B);
    this.xl = new Float32Array(B);
    this.xr = new Float32Array(B);
    this.pp = new Float32Array(B);
    this.wh = new Float32Array(B);
    this.nz = new Float32Array(B);
    this.sc = new Float32Array(B);
    this.ty = new Float32Array(B);
    this.wl = new Float32Array(B);
    this.wr = new Float32Array(B);
  }

  /** (Re)configure a voice slot. `cfg = null` frees the slot. debug: 0 normal, 1 raw core, 2 normalised+clipped. */
  configure(slot: number, cfg: EngineCfg | null, player: boolean, debug: number): void {
    if (slot < 0 || slot >= this.maxVoices) return;
    if (!cfg) {
      this.voices[slot] = null;
      return;
    }
    const fs = this.fs;
    const ph = ((slot * 0.6180339887) % 1 + 1) % 1;
    let idx = 0;
    while (idx < cfg.fire.length && cfg.fire[idx] <= ph) idx++;
    const v: SynthVoice = {
      cfg: cfg,
      player: !!player,
      debug: debug | 0,
      active: 1,
      tRpm: 900,
      tNorm: 0.1,
      tLoad: 0,
      tSpeed: 0,
      tNitro: 0,
      tSlip: 0,
      tSurf: 0,
      tScrape: 0,
      tGain: player ? 1 : 0,
      tPan: 0,
      tPitch: 1,
      tAir: 1,
      sinceUpd: 0,
      rpm: 900,
      norm: 0.1,
      load: 0,
      speed: 0,
      nitro: 0,
      sq: 0,
      scrape: 0,
      gain: 0,
      gl: 0,
      gr: 0,
      pan: 0,
      pitch: 1,
      air: 1,
      wKerb: 0,
      wGrass: 0,
      wGravel: 0,
      drift: 0,
      phase: ph,
      inc: 900 / 120 / fs,
      nextIdx: idx,
      carry0: 0,
      carry1: 0,
      tj: 0,
      pe0: 0,
      pe1: 0,
      lp0: 0,
      lp1: 0,
      aBase: 0,
      gN: 1,
      pG: 0,
      cutT: 0,
      blipT: 0,
      limT: 0,
      limPh: 0,
      crackEnv: 0,
      crackBoost: 0,
      loadHi: 0,
      blowT: 0,
      blowCd: 0,
      nBurst: 0,
      turboSpd: 0,
      popQ: new Float32Array(16),
      popAct: 0,
      popEnv: 0,
      clunk: 0,
      clunkAct: 0,
      coef: new Float64Array(36 * 5),
      st: new Float64Array(36 * 2),
      fcLast: 0,
      pitchLast: -1,
      whinePh: 0,
      turboPh: 0,
      sqPh1: 0,
      sqPh2: 0.37,
      kerbPh: 0,
      lfo1: 0,
      lfo2: 0,
      lfo3: 0,
      lfo4: 0,
      seed: (0x9e3779b9 ^ ((slot + 1) * 0x85ebca6b)) | 0 || 1,
      dcxL: 0,
      dcyL: 0,
      dcxR: 0,
      dcyR: 0,
      airL: 0,
      airR: 0,
      airT: 0,
    };
    this.retune(v);
    this.voices[slot] = v;
  }

  /** Packed per-frame states, stride 14: slot, active, rpm, rpmNorm, load, speed, nitro, slip, surface, scrape, gain, pan, pitch, air */
  setStates(d: Float32Array): void {
    const n = d.length;
    for (let o = 0; o + 14 <= n; o += 14) {
      const slot = d[o] | 0;
      if (slot < 0 || slot >= this.maxVoices) continue;
      const v = this.voices[slot];
      if (!v) continue;
      let x: number;
      v.active = d[o + 1] > 0.5 ? 1 : 0;
      x = d[o + 2];
      v.tRpm = x === x ? (x < 300 ? 300 : x > 22000 ? 22000 : x) : 900;
      x = d[o + 3];
      v.tNorm = x === x ? (x < 0 ? 0 : x > 1.15 ? 1.15 : x) : 0.1;
      x = d[o + 4];
      v.tLoad = x === x ? (x < 0 ? 0 : x > 1 ? 1 : x) : 0;
      x = d[o + 5];
      v.tSpeed = x === x ? (x < 0 ? -x : x) : 0;
      if (v.tSpeed > 200) v.tSpeed = 200;
      x = d[o + 6];
      v.tNitro = x === x ? (x < 0 ? 0 : x > 1 ? 1 : x) : 0;
      x = d[o + 7];
      v.tSlip = x === x ? (x < 0 ? 0 : x > 1 ? 1 : x) : 0;
      x = d[o + 8];
      v.tSurf = x === x ? x : 0;
      x = d[o + 9];
      v.tScrape = x === x ? (x < 0 ? 0 : x > 1 ? 1 : x) : 0;
      x = d[o + 10];
      v.tGain = x === x ? (x < 0 ? 0 : x > 2 ? 2 : x) : 0;
      x = d[o + 11];
      v.tPan = x === x ? (x < -1 ? -1 : x > 1 ? 1 : x) : 0;
      x = d[o + 12];
      v.tPitch = x === x ? (x < 0.5 ? 0.5 : x > 2 ? 2 : x) : 1;
      x = d[o + 13];
      v.tAir = x === x ? (x < 0 ? 0 : x > 1 ? 1 : x) : 1;
      v.sinceUpd = 0;
    }
  }

  /** code: 0 shift_up, 1 shift_down, 2 backfire, 3 limiter, 4 nitro_start, 5 blowoff */
  event(slot: number, code: number, val: number): void {
    if (slot < 0 || slot >= this.maxVoices) return;
    const v = this.voices[slot];
    if (!v) return;
    let a = val === val ? val : 1;
    if (a < 0) a = 0;
    if (a > 1.5) a = 1.5;
    if (code === 0) {
      v.cutT = 0.085;
      v.clunk += 0.9 * a;
      if (v.load > 0.55) this.pushPop(v, 0.03 + this.rnd(v) * 0.03, (0.3 + 0.35 * this.rnd(v)) * a * v.cfg.crackle);
    } else if (code === 1) {
      v.blipT = 0.17;
      v.clunk += 0.6 * a;
      v.crackBoost = 0.45;
    } else if (code === 2) {
      const k = 1 + ((this.rnd(v) * 2.6) | 0);
      let t = 0.004;
      for (let i = 0; i < k; i++) {
        this.pushPop(v, t, (0.75 + 0.5 * this.rnd(v)) * a);
        t += 0.035 + this.rnd(v) * 0.07;
      }
    } else if (code === 3) {
      if (v.limT < 0.14) v.limT = 0.14;
    } else if (code === 4) {
      v.nBurst = 1;
    } else if (code === 5) {
      if (v.blowCd <= 0 && v.cfg.turbo > 0) {
        v.blowT = 0.4;
        v.blowCd = 0.5;
      }
    }
  }

  rnd(v: SynthVoice): number {
    let s = v.seed;
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    v.seed = s;
    return (s >>> 0) / 4294967296;
  }

  pushPop(v: SynthVoice, delaySec: number, amp: number): void {
    const q = v.popQ;
    for (let i = 0; i < 16; i += 2) {
      if (q[i + 1] === 0) {
        q[i] = delaySec * this.fs;
        q[i + 1] = amp;
        return;
      }
    }
  }

  setBq(c: Float64Array, i: number, type: number, f: number, q: number): void {
    const fs = this.fs;
    if (!(f > 10)) f = 10;
    const fmax = fs * 0.45;
    if (f > fmax) f = fmax;
    const w = (6.283185307179586 * f) / fs;
    const cs = Math.cos(w);
    const al = Math.sin(w) / (2 * q);
    const a0 = 1 + al;
    let b0: number, b1: number, b2: number;
    if (type === 0) {
      b1 = 1 - cs;
      b0 = b1 * 0.5;
      b2 = b0;
    } else if (type === 1) {
      b1 = -(1 + cs);
      b0 = (1 + cs) * 0.5;
      b2 = b0;
    } else {
      b0 = al;
      b1 = 0;
      b2 = -al;
    }
    const o = i * 5;
    c[o] = b0 / a0;
    c[o + 1] = b1 / a0;
    c[o + 2] = b2 / a0;
    c[o + 3] = (-2 * cs) / a0;
    c[o + 4] = (1 - al) / a0;
  }

  /** Biquad (TDF-II) over a block. add=1 accumulates into y, add=0 overwrites (in-place allowed). */
  bq(v: SynthVoice, i: number, x: Float32Array, y: Float32Array, n: number, g: number, add: number): void {
    const c = v.coef;
    const s = v.st;
    const o = i * 5;
    const p = i * 2;
    const b0 = c[o];
    const b1 = c[o + 1];
    const b2 = c[o + 2];
    const a1 = c[o + 3];
    const a2 = c[o + 4];
    let z1 = s[p];
    let z2 = s[p + 1];
    if (add) {
      for (let k = 0; k < n; k++) {
        const xi = x[k];
        const yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        y[k] += yi * g;
      }
    } else {
      for (let k = 0; k < n; k++) {
        const xi = x[k];
        const yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        y[k] = yi * g;
      }
    }
    if (z1 < 1e-30 && z1 > -1e-30) z1 = 0;
    if (z2 < 1e-30 && z2 > -1e-30) z2 = 0;
    s[p] = z1;
    s[p + 1] = z2;
  }

  fillNoise(v: SynthVoice, buf: Float32Array, n: number): void {
    let s = v.seed;
    for (let k = 0; k < n; k++) {
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      buf[k] = (s | 0) / 2147483648;
    }
    v.seed = s;
  }

  /** Static filters (formants etc.); re-run when the Doppler pitch factor moves. */
  retune(v: SynthVoice): void {
    const c = v.coef;
    const cfg = v.cfg;
    const p = v.pitch;
    for (let k = 0; k < 4; k++) {
      this.setBq(c, 2 + k, 2, cfg.formF[k] * p, cfg.formQ[k]);
      this.setBq(c, 6 + k, 2, cfg.formF[k] * cfg.detune * p, cfg.formQ[k] * 0.9);
    }
    this.setBq(c, 10, 2, cfg.intakeF * p, cfg.intakeQ);
    this.setBq(c, 11, 2, cfg.mechF * p, 0.7);
    this.setBq(c, 12, 2, cfg.popF * p, 1.4);
    this.setBq(c, 13, 2, cfg.popF * 4.6 * p, 2.0);
    this.setBq(c, 14, 2, 1900 * p, 0.9);
    this.setBq(c, 34, 0, 105 * p, 1.1);
    this.setBq(c, 19, 2, 1700 * p, 1.0);
    this.setBq(c, 20, 0, 300 * p, 0.7);
    this.setBq(c, 21, 2, 880 * p, 1.1);
    this.setBq(c, 22, 2, 135 * p, 2.2);
    this.setBq(c, 23, 2, 470 * p, 3.0);
    this.setBq(c, 24, 0, 520 * p, 0.7);
    this.setBq(c, 25, 2, 2600 * p, 0.8);
    this.setBq(c, 26, 2, 1900 * p, 1.2);
    this.setBq(c, 27, 2, 4300 * p, 1.4);
    this.setBq(c, 28, 2, 740 * p, 14);
    this.setBq(c, 29, 2, 1330 * p, 18);
    this.setBq(c, 30, 2, 2210 * p, 20);
    this.setBq(c, 31, 2, 3480 * p, 22);
    this.setBq(c, 35, 1, 4200 * p, 0.7);
    v.pitchLast = p;
    v.fcLast = 0;
  }

  /** Render n samples (any n) for all voices. Output arrays are overwritten. */
  process(o0L: Float32Array, o0R: Float32Array, o1L: Float32Array, o1R: Float32Array, n: number): void {
    for (let k = 0; k < n; k++) {
      o0L[k] = 0;
      o0R[k] = 0;
      o1L[k] = 0;
      o1R[k] = 0;
    }
    const vs = this.voices;
    const nv = vs.length;
    let off = 0;
    while (off < n) {
      const m = n - off > 128 ? 128 : n - off;
      for (let i = 0; i < nv; i++) {
        const v = vs[i];
        if (v) this.voiceBlock(v, o0L, o0R, o1L, o1R, off, m);
      }
      off += m;
    }
  }

  voiceBlock(v: SynthVoice, oL: Float32Array, oR: Float32Array, tL: Float32Array, tR: Float32Array, off: number, n: number): void {
    const fs = this.fs;
    const cfg = v.cfg;
    const coef = v.coef;
    const dt = n / fs;
    const player = v.player;

    // ───────── control-rate smoothing ─────────
    v.sinceUpd += n;
    let gT = v.active > 0.5 ? v.tGain : 0;
    if (v.sinceUpd > fs * 0.45) gT = 0;
    const k15 = 1 - Math.exp(-dt / 0.015);
    const k30 = 1 - Math.exp(-dt / 0.03);
    const k60 = 1 - Math.exp(-dt / 0.06);
    const k250 = 1 - Math.exp(-dt / 0.25);

    v.pitch += (v.tPitch - v.pitch) * k60;
    v.rpm += (v.tRpm - v.rpm) * k30;
    v.norm += (v.tNorm - v.norm) * k30;
    v.speed += (v.tSpeed - v.speed) * k60;
    v.air += (v.tAir - v.air) * k60;
    v.pan += (v.tPan - v.pan) * k60;

    let loadT = v.tLoad;
    if (v.cutT > 0) {
      v.cutT -= dt;
      loadT *= 0.1;
    }
    if (v.blipT > 0) {
      v.blipT -= dt;
      if (v.blipT > 0.06 && loadT < 0.85) loadT = 0.85;
    }
    let limCut = 0;
    if (v.limT > 0) {
      v.limT -= dt;
      v.limPh += dt * 27;
      if (v.limPh - Math.floor(v.limPh) < 0.55) {
        limCut = 1;
        loadT *= 0.35;
      }
    } else if (v.debug === 0 && v.tNorm > 0.992 && v.tLoad > 0.6) {
      v.limT = 0.1;
    }
    v.load += (loadT - v.load) * (loadT > v.load ? k15 : k30);
    const L = v.load;
    const R = v.norm;
    const p = v.pitch;

    // lift-off detection → overrun crackle / turbo blow-off
    if (L > 0.55) v.loadHi = 0.25;
    else if (v.loadHi > 0) {
      v.loadHi -= dt;
      if (L < 0.18) {
        v.crackEnv = 1;
        if (cfg.turbo > 0 && v.turboSpd > 0.45 && v.blowCd <= 0) {
          v.blowT = 0.4;
          v.blowCd = 0.5;
        }
        v.loadHi = 0;
      }
    }
    if (v.blowCd > 0) v.blowCd -= dt;
    if (v.crackEnv > 0) v.crackEnv -= dt / 1.1;
    if (v.crackBoost > 0) v.crackBoost -= dt;

    // turbo spool
    if (cfg.turbo > 0) {
      let ts = (R - 0.22) / 0.45;
      ts = ts < 0 ? 0 : ts > 1 ? 1 : ts;
      ts *= Math.pow(L, 1.2);
      v.turboSpd += (ts - v.turboSpd) * (ts > v.turboSpd ? 1 - Math.exp(-dt / 0.45) : 1 - Math.exp(-dt / 0.8));
    }

    // slow random drift (keeps a held rpm alive)
    v.drift += (this.rnd(v) - 0.5) * 0.3 - v.drift * 0.04;
    if (v.drift > 1) v.drift = 1;
    if (v.drift < -1) v.drift = -1;

    const rpmS = v.rpm * p * (1 + 0.0035 * v.drift);
    const cycHz = (rpmS > 400 ? rpmS : 400) / 120;
    const incEnd = cycHz / fs;

    // gains / pan (equal power for mono voices, plain stereo for the player)
    const g0 = v.gain;
    const g1 = g0 + (gT - g0) * k60;
    let gl1: number, gr1: number;
    if (player) {
      gl1 = g1;
      gr1 = g1;
    } else {
      const th = (v.pan + 1) * 0.7853981633974483;
      gl1 = g1 * Math.cos(th);
      gr1 = g1 * Math.sin(th);
    }
    const gl0 = v.gl;
    const gr0 = v.gr;
    v.gain = g1;
    v.gl = gl1;
    v.gr = gr1;

    if (g0 < 1e-4 && g1 < 1e-4 && gT < 1e-4) {
      // inaudible: just keep the crank turning
      let ph = v.phase + incEnd * n;
      ph -= Math.floor(ph);
      let idx = 0;
      const fire = cfg.fire;
      while (idx < fire.length && fire[idx] <= ph) idx++;
      v.phase = ph;
      v.nextIdx = idx;
      v.inc = incEnd;
      v.carry0 = 0;
      v.carry1 = 0;
      v.gain = 0;
      v.gl = 0;
      v.gr = 0;
      return;
    }

    if (p - v.pitchLast > 0.008 || v.pitchLast - p > 0.008) this.retune(v);

    // ───────── engine excitation ─────────
    const two = player && cfg.banks > 1;
    const nb = two ? 2 : 1;
    const fireHz = cycHz * cfg.cyl;
    const Lb = Math.pow(L, 0.7);
    let fc = cfg.lpBase * p + fireHz * (cfg.hOff + (cfg.hOn - cfg.hOff) * Lb);
    if (fc > cfg.lpMax * p) fc = cfg.lpMax * p;
    if (fc - v.fcLast > v.fcLast * 0.015 || v.fcLast - fc > v.fcLast * 0.015) {
      this.setBq(coef, 0, 0, fc, 0.62);
      this.setBq(coef, 1, 0, fc * 1.04, 0.62);
      v.fcLast = fc;
    }
    const aBT = fs * 0.001 * (0.2 + 0.8 * Math.pow(L, 0.8)) * Math.pow(200 / (fireHz > 20 ? fireHz : 20), cfg.rateComp);
    const bankHz = fireHz / nb;
    const gPN = cfg.pnoise * 3.9 * Math.sqrt(bankHz / fs) * (0.5 + 0.5 * L);
    const pdec = Math.exp((-bankHz * 2.5) / fs);

    const e0 = this.e0;
    const e1 = this.e1;
    e0[0] = v.carry0;
    e1[0] = v.carry1;
    for (let i = 1; i <= n; i++) {
      e0[i] = 0;
      e1[i] = 0;
    }
    const fire = cfg.fire;
    const bankOf = cfg.bank;
    const ampOf = cfg.amp;
    const nEv = fire.length;
    let ph = v.phase;
    let idx = v.nextIdx;
    let inc = v.inc;
    const dInc = (incEnd - inc) / n;
    let aB = v.aBase;
    const dA = (aBT - aB) / n;
    let pe0 = v.pe0;
    let pe1 = v.pe1;
    let seed = v.seed;
    const jit = cfg.jitter;
    // single-chain (cheap) voices of two-bank engines keep their half-order character via unequal bank strength
    const b0s = !two && cfg.banks > 1 ? 1.2 : 1;
    const b1s = !two && cfg.banks > 1 ? 0.62 : 1;
    const mis = R < 0.2 && L < 0.3 ? cfg.misfire : 0;
    const gate = limCut ? 0.05 : 1;
    const tjit = cfg.tjit;
    let tj = v.tj;
    let bd = cfg.bankDelay * cycHz;
    if (bd > 0.55 / nEv) bd = 0.55 / nEv;
    for (let i = 0; i < n; i++) {
      inc += dInc;
      aB += dA;
      ph += inc;
      for (;;) {
        while (idx < nEv && ph >= fire[idx] + tj + (bankOf[idx] === 1 ? bd : 0)) {
          let fr = (ph - fire[idx] - tj - (bankOf[idx] === 1 ? bd : 0)) / inc;
          if (fr > 1) fr = 1;
          seed ^= seed << 13;
          seed ^= seed >>> 17;
          seed ^= seed << 5;
          let a = aB * ampOf[idx] * (bankOf[idx] === 1 ? b1s : b0s) * gate * (1 + (jit * (seed | 0)) / 2147483648);
          if (mis > 0) {
            seed ^= seed << 13;
            seed ^= seed >>> 17;
            seed ^= seed << 5;
            if ((seed >>> 0) / 4294967296 < mis) a *= 0.3;
          }
          const w0 = a * fr;
          const w1 = a - w0;
          if (two && bankOf[idx] === 1) {
            e1[i] += w0;
            e1[i + 1] += w1;
            pe1 = 1;
          } else {
            e0[i] += w0;
            e0[i + 1] += w1;
            pe0 = 1;
          }
          idx++;
          seed ^= seed << 13;
          seed ^= seed >>> 17;
          seed ^= seed << 5;
          tj = (tjit * (seed >>> 0)) / 4294967296;
        }
        if (ph >= 1) {
          ph -= 1;
          idx = 0;
          continue;
        }
        break;
      }
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      e0[i] += (aB * gate * gPN * pe0 * (seed | 0)) / 2147483648;
      pe0 *= pdec;
      if (two) {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        e1[i] += (aB * gate * gPN * pe1 * (seed | 0)) / 2147483648;
        pe1 *= pdec;
      }
    }
    v.carry0 = e0[n];
    v.carry1 = e1[n];
    v.tj = tj;
    v.phase = ph;
    v.nextIdx = idx;
    v.inc = inc;
    v.aBase = aB;
    v.pe0 = pe0;
    v.pe1 = pe1;
    v.seed = seed;

    // ───────── exhaust: pulse shaper + resonators per bank ─────────
    const A0 = this.a0;
    const A1 = this.a1;
    const XL = this.xl;
    const XR = this.xr;
    const NZ = this.nz;
    const SC = this.sc;
    const fg = cfg.formG;
    const dry = cfg.dry;
    const g4 = fg[3] * (0.15 + 0.85 * L) * (0.45 + 0.55 * (R > 1 ? 1 : R));
    this.bq(v, 0, e0, e0, n, 1, 0);
    // order-locked pulse body: an exhaust pulse lasts a fixed crank angle, so its spectrum tilts around the firing rate
    const bm = cfg.body;
    const bodyK = 1 - Math.exp((-6.283185307179586 * 1.2 * fireHz) / fs);
    const bodyD = 1 - bm;
    const bodyG = bm * 1.3;
    if (bm > 0) {
      let l0 = v.lp0;
      for (let i = 0; i < n; i++) {
        const x = e0[i];
        l0 += (x - l0) * bodyK;
        e0[i] = x * bodyD + l0 * bodyG;
      }
      v.lp0 = l0 < 1e-30 && l0 > -1e-30 ? 0 : l0;
    }
    for (let i = 0; i < n; i++) A0[i] = e0[i] * dry;
    this.bq(v, 2, e0, A0, n, fg[0], 1);
    this.bq(v, 3, e0, A0, n, fg[1], 1);
    this.bq(v, 4, e0, A0, n, fg[2], 1);
    this.bq(v, 5, e0, A0, n, g4, 1);
    if (two) {
      this.bq(v, 1, e1, e1, n, 1, 0);
      if (bm > 0) {
        let l1 = v.lp1;
        for (let i = 0; i < n; i++) {
          const x = e1[i];
          l1 += (x - l1) * bodyK;
          e1[i] = x * bodyD + l1 * bodyG;
        }
        v.lp1 = l1 < 1e-30 && l1 > -1e-30 ? 0 : l1;
      }
      for (let i = 0; i < n; i++) A1[i] = e1[i] * dry;
      this.bq(v, 6, e1, A1, n, fg[0], 1);
      this.bq(v, 7, e1, A1, n, fg[1], 1);
      this.bq(v, 8, e1, A1, n, fg[2], 1);
      this.bq(v, 9, e1, A1, n, g4, 1);
      const cx = cfg.cross;
      const cd = 1 - cx;
      for (let i = 0; i < n; i++) {
        const b0 = A0[i];
        const b1 = A1[i];
        XL[i] = b0 * cd + b1 * cx;
        XR[i] = b1 * cd + b0 * cx;
        SC[i] = e0[i] + e1[i];
      }
    } else if (player) {
      // single-pipe engine, player voice: a second, slightly detuned resonator set gives the two ears different colour
      for (let i = 0; i < n; i++) A1[i] = e0[i] * dry;
      this.bq(v, 6, e0, A1, n, fg[0], 1);
      this.bq(v, 7, e0, A1, n, fg[1], 1);
      this.bq(v, 8, e0, A1, n, fg[2], 1);
      this.bq(v, 9, e0, A1, n, g4, 1);
      for (let i = 0; i < n; i++) {
        const b0 = A0[i];
        const b1 = A1[i];
        XL[i] = b0 * 0.92 + b1 * 0.08;
        XR[i] = b1 * 0.92 + b0 * 0.08;
        SC[i] = e0[i];
      }
    } else {
      for (let i = 0; i < n; i++) {
        XL[i] = A0[i];
        SC[i] = e0[i];
      }
    }

    // intake roar (load²) + mechanical noise (rpm) + whine / turbo whistle → centre
    const Rc = R > 1 ? 1 : R;
    if (player) {
      for (let i = 0; i < n; i++) A0[i] = 0;
      this.bq(v, 10, SC, A0, n, cfg.intakeG * Math.pow(L, 1.4) * (0.6 + 0.4 * Rc), 1);
      this.fillNoise(v, NZ, n);
      this.bq(v, 11, NZ, A0, n, cfg.mechG * (0.15 + 0.85 * Math.pow(Rc, 1.4)), 1);
      if (cfg.whine > 0) {
        const wi = (cfg.whineOrder * rpmS) / 60 / fs;
        const wa = cfg.whine * Rc * (0.35 + 0.65 * L);
        let wp = v.whinePh;
        if (wi < 0.45) {
          for (let i = 0; i < n; i++) {
            wp += wi;
            if (wp >= 1) wp -= 1;
            const ang = 6.283185307179586 * wp;
            A0[i] += wa * (Math.sin(ang) + 0.3 * Math.sin(2 * ang));
          }
        }
        v.whinePh = wp;
      }
    } else {
      for (let i = 0; i < n; i++) A0[i] = 0;
    }
    if (cfg.turbo > 0 && v.turboSpd > 0.02) {
      const tsp = v.turboSpd;
      const ti = ((2300 + 6400 * Math.pow(tsp, 0.8)) * p) / fs;
      const ta = cfg.turbo * 0.035 * Math.pow(tsp, 1.6);
      let tp = v.turboPh;
      if (ti < 0.45) {
        for (let i = 0; i < n; i++) {
          tp += ti;
          if (tp >= 1) tp -= 1;
          A0[i] += ta * Math.sin(6.283185307179586 * tp);
        }
      }
      v.turboPh = tp;
    }
    if (player) {
      for (let i = 0; i < n; i++) {
        const c = A0[i];
        XL[i] += c;
        XR[i] += c;
      }
    } else {
      for (let i = 0; i < n; i++) XL[i] += A0[i];
    }

    // ───────── pops / crackle / clunk (added after normalisation, before the clipper) ─────────
    const PP = this.pp;
    let popOn = v.popAct > 0;
    // overrun crackle
    const crk = (v.crackEnv > 0 ? v.crackEnv : 0) + (v.crackBoost > 0 ? 0.8 : 0);
    if (crk > 0 && L < 0.25 && R > 0.3 && cfg.crackle > 0) {
      const lam = cfg.crackle * 26 * R * crk * dt;
      if (this.rnd(v) < lam) this.pushPop(v, this.rnd(v) * dt, (0.12 + 0.4 * this.rnd(v) * this.rnd(v)) * cfg.crackle);
    }
    // limiter: small pop when the ignition comes back
    if (v.limT > 0 && !limCut && this.rnd(v) < dt * 14) this.pushPop(v, 0, 0.18 + 0.2 * this.rnd(v));
    const q = v.popQ;
    let anyPop = false;
    for (let k = 0; k < 16; k += 2) {
      if (q[k + 1] !== 0) {
        anyPop = true;
        break;
      }
    }
    const clunkNow = v.clunk;
    if (anyPop || popOn || clunkNow > 0 || v.clunkAct > 0) {
      for (let i = 0; i < n; i++) {
        PP[i] = 0;
        SC[i] = 0;
      }
      let pEnv = v.popEnv;
      if (anyPop) {
        for (let k = 0; k < 16; k += 2) {
          const am = q[k + 1];
          if (am === 0) continue;
          if (q[k] < n) {
            const at = q[k] | 0;
            SC[at < 0 ? 0 : at] += am;
            pEnv += am;
            q[k + 1] = 0;
            v.popAct = 0.3;
          } else {
            q[k] -= n;
          }
        }
      }
      if (v.popAct > 0) {
        // body + bark resonances and a short noise burst
        const scl = fs / (6.283185307179586 * ((cfg.popF * p) / 1.4));
        for (let i = 0; i < n; i++) A1[i] = SC[i] * scl;
        this.bq(v, 12, A1, PP, n, 1.25, 1);
        this.bq(v, 13, A1, PP, n, 0.55, 1);
        const nd = Math.exp(-1 / (0.016 * fs));
        let s2 = v.seed;
        for (let i = 0; i < n; i++) {
          s2 ^= s2 << 13;
          s2 ^= s2 >>> 17;
          s2 ^= s2 << 5;
          A1[i] = (pEnv * (s2 | 0)) / 2147483648;
          pEnv *= nd;
        }
        v.seed = s2;
        this.bq(v, 14, A1, PP, n, 1.1, 1);
        v.popAct -= dt;
      }
      v.popEnv = pEnv < 1e-5 ? 0 : pEnv;
      if (clunkNow > 0 || v.clunkAct > 0) {
        for (let i = 0; i < n; i++) A1[i] = 0;
        if (clunkNow > 0) {
          A1[0] = (clunkNow * fs) / (6.283185307179586 * 105 * p);
          v.clunk = 0;
          v.clunkAct = 0.25;
        }
        this.bq(v, 34, A1, PP, n, 0.55, 1);
        v.clunkAct -= dt;
      }
      popOn = true;
    }

    // ───────── nitro whoosh / blow-off (added after the clipper) ─────────
    const WH = this.wh;
    const nzT = v.tNitro;
    v.nitro += (nzT - v.nitro) * (nzT > v.nitro ? k30 : k250);
    if (v.nBurst > 0) v.nBurst -= dt * 3.2;
    const nz = v.nitro;
    const nbu = v.nBurst > 0 ? v.nBurst : 0;
    let whOn = false;
    if (nz > 0.003 || nbu > 0 || v.blowT > 0) {
      whOn = true;
      for (let i = 0; i < n; i++) WH[i] = 0;
      this.fillNoise(v, NZ, n);
      if (nz > 0.003 || nbu > 0) {
        this.setBq(coef, 15, 2, (650 + 1500 * nz) * p, 0.8);
        this.setBq(coef, 16, 2, (380 + 420 * nz) * p, 5);
        const na = 0.27 * Math.pow(nz, 0.7) + 0.3 * nbu * nbu;
        this.bq(v, 15, NZ, WH, n, na, 1);
        this.bq(v, 16, NZ, WH, n, na * 0.9, 1);
        this.bq(v, 35, NZ, WH, n, na * 0.22, 1);
      }
      if (v.blowT > 0) {
        const bt = v.blowT / 0.4;
        this.setBq(coef, 17, 2, (700 + 2700 * bt) * p, 1.3);
        const fl = v.blowT < 0.27 ? 0.55 + 0.45 * Math.sin(v.blowT * 150) : 1;
        this.bq(v, 17, NZ, WH, n, 0.34 * cfg.turbo * Math.pow(bt, 1.3) * fl, 1);
        v.blowT -= dt;
      }
    }

    // ───────── normalise → clip → loudness → DC → air → out ─────────
    const rIdx = cfg.calN > 1 ? ((Rc - cfg.calR0) / (cfg.calR1 - cfg.calR0)) * (cfg.calN - 1) : 0;
    const off2 = player ? 0 : cfg.calN * 3;
    let gNT = 0.25 / this.lutO(cfg.pre, off2, cfg.calN, rIdx, L);
    const loud = (0.3 + 0.7 * Math.pow(Rc, 1.2)) * (0.5 + 0.5 * Math.pow(L, 0.75));
    let pGT = (loud * 0.2) / this.lutO(cfg.post, off2, cfg.calN, rIdx, L);
    const dbg = v.debug;
    if (dbg === 1) {
      gNT = 1;
      pGT = 1;
    } else if (dbg === 2) {
      pGT = 1;
    }
    let gN = v.gN;
    const dgN = ((gNT - gN) * k30) / n;
    let pG = v.pG;
    const dpG = ((pGT - pG) * k30) / n;
    const drv = cfg.drive * (0.7 + 0.6 * L);
    const airK = player ? 1 : 1 - Math.exp((-6.283185307179586 * (700 + 17300 * v.air * v.air)) / fs);
    if (popOn) {
      // pops / clunks keep their punch on the overrun: own level (rpm only) and own soft knee
      const popLvl = dbg === 0 ? 0.42 * (0.4 + 0.6 * Rc) : 0;
      for (let i = 0; i < n; i++) {
        const pv = PP[i] * popLvl;
        PP[i] = pv / (1 + 0.6 * (pv < 0 ? -pv : pv));
      }
    }
    const dcR = 1 - 131.9 / fs;
    let gl = gl0;
    let gr = gr0;
    const dgl = (gl1 - gl0) / n;
    const dgr = (gr1 - gr0) / n;
    let dxL = v.dcxL;
    let dyL = v.dcyL;
    let aL = v.airL;
    if (player) {
      let dxR = v.dcxR;
      let dyR = v.dcyR;
      for (let i = 0; i < n; i++) {
        gN += dgN;
        pG += dpG;
        gl += dgl;
        gr += dgr;
        let xl = XL[i] * gN;
        let xr = XR[i] * gN;
        if (dbg !== 1) {
          xl = xl * drv + 0.1;
          xl = xl / (1 + (xl < 0 ? -xl : xl)) - 0.09090909;
          xr = xr * drv + 0.1;
          xr = xr / (1 + (xr < 0 ? -xr : xr)) - 0.09090909;
        }
        xl *= pG;
        xr *= pG;
        if (popOn) {
          const pv = PP[i];
          xl += pv;
          xr += pv;
        }
        if (whOn) {
          const w = WH[i];
          xl += w;
          xr += w;
        }
        const yl = xl - dxL + dcR * dyL;
        dxL = xl;
        dyL = yl;
        const yr = xr - dxR + dcR * dyR;
        dxR = xr;
        dyR = yr;
        oL[off + i] += yl * gl;
        oR[off + i] += yr * gr;
      }
      v.dcxR = dxR;
      v.dcyR = dyR;
    } else {
      for (let i = 0; i < n; i++) {
        gN += dgN;
        pG += dpG;
        gl += dgl;
        gr += dgr;
        let xl = XL[i] * gN;
        if (dbg !== 1) {
          xl = xl * drv + 0.1;
          xl = xl / (1 + (xl < 0 ? -xl : xl)) - 0.09090909;
        }
        xl *= pG;
        if (popOn) xl += PP[i];
        if (whOn) xl += WH[i];
        let yl = xl - dxL + dcR * dyL;
        dxL = xl;
        dyL = yl;
        if (airK < 1) {
          aL += (yl - aL) * airK;
          yl = aL;
        }
        oL[off + i] += yl * gl;
        oR[off + i] += yl * gr;
      }
    }
    v.gN = gN;
    v.pG = pG;
    v.dcxL = dxL;
    v.dcyL = dyL;
    v.airL = aL < 1e-30 && aL > -1e-30 ? 0 : aL;

    // ───────── tyres & surface ─────────
    const TY = this.ty;
    let tyOn = false;
    const spd = v.speed;
    // LFOs (band-limited random walks)
    v.lfo1 += (this.rnd(v) - 0.5) * 0.5 - v.lfo1 * 0.08;
    v.lfo2 += (this.rnd(v) - 0.5) * 0.5 - v.lfo2 * 0.08;
    v.lfo3 += (this.rnd(v) - 0.5) * 0.6 - v.lfo3 * 0.1;
    v.lfo4 += (this.rnd(v) - 0.5) * 0.25 - v.lfo4 * 0.03;

    const slipT = v.tSlip > 0.06 ? Math.pow((v.tSlip - 0.06) / 0.94, 1.3) : 0;
    v.sq += (slipT - v.sq) * (slipT > v.sq ? k30 : k60);
    const sq = v.sq;
    const scrT = v.tScrape;
    v.scrape += (scrT - v.scrape) * (scrT > v.scrape ? k15 : k60);
    const scr = v.scrape;
    const sf = Math.round(v.tSurf);
    v.wKerb += ((sf === 1 ? 1 : 0) - v.wKerb) * k30;
    v.wGrass += ((sf === 2 ? 1 : 0) - v.wGrass) * k30;
    v.wGravel += ((sf === 3 ? 1 : 0) - v.wGravel) * k30;
    const roadLvl = Math.pow(spd / 70 > 1.4 ? 1.4 : spd / 70, 1.6) * (1 - v.wGrass) * (1 - v.wGravel);
    const needNoise = roadLvl > 0.004 || v.wGrass > 0.004 || v.wGravel > 0.004 || scr > 0.003 || (sq > 0.002 && player);
    for (let i = 0; i < n; i++) TY[i] = 0;
    if (needNoise) this.fillNoise(v, NZ, n);

    if (sq > 0.002) {
      tyOn = true;
      const f1 = (620 + 360 * sq + 1.6 * spd) * (1 + 0.03 * v.lfo1) * p;
      const f2 = f1 * 1.013 * (1 + 0.006 * v.lfo2);
      const i1 = f1 / fs;
      const i2 = f2 / fs;
      let p1 = v.sqPh1;
      let p2 = v.sqPh2;
      let s3 = v.seed;
      for (let i = 0; i < n; i++) {
        p1 += i1;
        if (p1 >= 1) p1 -= 1;
        p2 += i2;
        if (p2 >= 1) p2 -= 1;
        const a1 = 6.283185307179586 * p1;
        const a2 = 6.283185307179586 * p2;
        s3 ^= s3 << 13;
        s3 ^= s3 >>> 17;
        s3 ^= s3 << 5;
        SC[i] = Math.sin(a1 + 0.7 * Math.sin(a1)) * 0.5 + Math.sin(a2 + 0.9 * Math.sin(2 * a2)) * 0.4 + (0.4 * (s3 | 0)) / 2147483648;
      }
      v.seed = s3;
      v.sqPh1 = p1;
      v.sqPh2 = p2;
      this.setBq(coef, 18, 2, f1 * 1.35, 0.9);
      let fl3 = 0.8 + 0.25 * v.lfo3;
      if (fl3 < 0.4) fl3 = 0.4;
      this.bq(v, 18, SC, TY, n, sq * fl3 * 0.36, 1);
      if (player) this.bq(v, 19, NZ, TY, n, Math.pow(sq, 1.5) * 0.3, 1);
    }
    if (roadLvl > 0.004) {
      tyOn = true;
      this.bq(v, 20, NZ, TY, n, roadLvl * 0.6, 1);
      if (player) this.bq(v, 21, NZ, TY, n, roadLvl * 0.11, 1);
    }
    if (v.wKerb > 0.004 && spd > 0.5) {
      tyOn = true;
      const ki = spd / 0.55 / fs;
      let kp = v.kerbPh;
      const ka = v.wKerb * (spd / 12 > 1 ? 1 : spd / 12) * 26;
      for (let i = 0; i < n; i++) {
        kp += ki;
        if (kp >= 1) {
          kp -= 1;
          SC[i] = ka * (0.8 + 0.4 * this.rnd(v));
        } else SC[i] = 0;
      }
      v.kerbPh = kp;
      this.bq(v, 22, SC, TY, n, 1.0, 1);
      this.bq(v, 23, SC, TY, n, 0.45, 1);
    }
    if (v.wGrass > 0.004) {
      tyOn = true;
      const ga = v.wGrass * (spd / 22 > 1 ? 1 : spd / 22);
      let am = 0.75 + 0.5 * v.lfo3;
      if (am < 0.25) am = 0.25;
      this.bq(v, 24, NZ, TY, n, ga * 0.75, 1);
      this.bq(v, 25, NZ, TY, n, ga * 0.2 * am, 1);
    }
    if (v.wGravel > 0.004) {
      tyOn = true;
      const ga = v.wGravel * (spd / 18 > 1 ? 1 : spd / 18);
      const lam = (30 + 14 * spd) / fs;
      let s4 = v.seed;
      for (let i = 0; i < n; i++) {
        s4 ^= s4 << 13;
        s4 ^= s4 >>> 17;
        s4 ^= s4 << 5;
        const u = (s4 >>> 0) / 4294967296;
        SC[i] = NZ[i] * 0.6 + (u < lam ? (u / lam) * 3.2 : 0);
      }
      v.seed = s4;
      this.bq(v, 26, SC, TY, n, ga * 0.42, 1);
      this.bq(v, 27, SC, TY, n, ga * 0.25, 1);
      this.bq(v, 24, NZ, TY, n, ga * 0.4, 1);
    }
    if (scr > 0.003) {
      tyOn = true;
      let fl = 0.75 + 0.4 * v.lfo2;
      if (fl < 0.3) fl = 0.3;
      const sa = Math.pow(scr, 0.8) * (0.45 + 0.55 * (spd / 25 > 1 ? 1 : spd / 25)) * fl;
      this.bq(v, 28, NZ, TY, n, sa * 1.7, 1);
      this.bq(v, 29, NZ, TY, n, sa * 1.55, 1);
      this.bq(v, 30, NZ, TY, n, sa * 1.25, 1);
      this.bq(v, 31, NZ, TY, n, sa * 1.0, 1);
      this.bq(v, 35, NZ, TY, n, sa * 0.2, 1);
    }

    gl = gl0;
    gr = gr0;
    if (tyOn) {
      if (airK < 1) {
        let aT = v.airT;
        for (let i = 0; i < n; i++) {
          gl += dgl;
          gr += dgr;
          aT += (TY[i] - aT) * airK;
          tL[off + i] += aT * gl;
          tR[off + i] += aT * gr;
        }
        v.airT = aT < 1e-30 && aT > -1e-30 ? 0 : aT;
      } else {
        for (let i = 0; i < n; i++) {
          gl += dgl;
          gr += dgr;
          const t = TY[i];
          tL[off + i] += t * gl;
          tR[off + i] += t * gr;
        }
      }
    }

    // wind rush (player only, decorrelated stereo)
    if (player) {
      const wv = spd / 85;
      const wLvl = (wv > 1.3 ? 1.3 : wv) * (wv > 1.3 ? 1.3 : wv);
      if (wLvl > 0.002) {
        const WL = this.wl;
        const WR = this.wr;
        const wf = (300 + 8 * spd) * (1 + 0.12 * v.lfo4);
        this.setBq(coef, 32, 2, wf, 0.55);
        this.setBq(coef, 33, 2, wf * 1.13, 0.55);
        let gu = 0.8 + 0.35 * v.lfo4;
        if (gu < 0.4) gu = 0.4;
        this.fillNoise(v, NZ, n);
        this.bq(v, 32, NZ, WL, n, wLvl * gu * 0.34, 0);
        this.fillNoise(v, NZ, n);
        this.bq(v, 33, NZ, WR, n, wLvl * gu * 0.34, 0);
        gl = gl0;
        gr = gr0;
        for (let i = 0; i < n; i++) {
          gl += dgl;
          gr += dgr;
          tL[off + i] += WL[i] * gl;
          tR[off + i] += WR[i] * gr;
        }
      }
    }
  }

  /** lut() with a row offset into the table (player rows first, then cheap-voice rows). */
  lutO(tab: number[], offset: number, rows: number, r: number, l: number): number {
    if (!tab || tab.length < offset + rows * 3 || rows < 1) return 0.1;
    if (r < 0) r = 0;
    if (r > rows - 1) r = rows - 1;
    let i0 = r | 0;
    if (i0 > rows - 2) i0 = rows - 2;
    if (i0 < 0) i0 = 0;
    const fr = rows > 1 ? r - i0 : 0;
    const i1 = rows > 1 ? i0 + 1 : i0;
    let j0 = 0;
    let fl = l * 2;
    if (fl >= 1) {
      j0 = 1;
      fl -= 1;
    }
    if (fl < 0) fl = 0;
    if (fl > 1) fl = 1;
    const a = tab[offset + i0 * 3 + j0] + (tab[offset + i0 * 3 + j0 + 1] - tab[offset + i0 * 3 + j0]) * fl;
    const b = tab[offset + i1 * 3 + j0] + (tab[offset + i1 * 3 + j0 + 1] - tab[offset + i1 * 3 + j0]) * fl;
    const y = a + (b - a) * fr;
    return y > 1e-6 ? y : 1e-6;
  }
}
