/**
 * Engine voicings. Pure data — `engineConfig()` output is structured-cloned into the audio worklet.
 * After changing anything here, regenerate the level tables:  npx tsx tools/audio/calibrate.ts
 */
import type { EngineCfg } from './engine-dsp';
import { ENGINE_CALIB } from './engine-calib';

export type EngineType = 'v8' | 'v12' | 'i6t' | 'v10';

/** rpmNorm grid of the calibration tables (rows) — 3 load columns (0, 0.5, 1). */
export const CAL_R0 = 0.06;
export const CAL_R1 = 1.05;
export const CAL_N = 12;

/** Typical redline per engine type; used for calibration and as a fallback when the game passes rpmMax <= 0. */
export const NOMINAL_RPM_MAX: Record<EngineType, number> = { v8: 7200, v12: 8800, i6t: 7600, v10: 8800 };

type Voicing = Omit<EngineCfg, 'calR0' | 'calR1' | 'calN' | 'pre' | 'post'>;

const even = (n: number): number[] => Array.from({ length: n }, (_, i) => i / n);
const alt = (n: number): number[] => Array.from({ length: n }, (_, i) => i & 1);

export const ENGINE_VOICINGS: Record<EngineType, Voicing> = {
  // Cross-plane V8: banks fire L R R L R L L R → each pipe gets an uneven 270/180/90/180° pulse rhythm (the burble).
  v8: {
    cyl: 8,
    fire: even(8),
    bank: [0, 1, 1, 0, 1, 0, 0, 1],
    amp: [1.0, 0.9, 0.68, 1.1, 0.85, 1.08, 0.78, 0.88],
    banks: 2,
    jitter: 0.14,
    misfire: 0.1,
    lpBase: 140,
    hOff: 1.6,
    hOn: 6.5,
    lpMax: 3400,
    tjit: 0.0035,
    dry: 0.55,
    body: 0.5,
    bankDelay: 0.00025,
    formF: [92, 235, 610, 1650],
    formQ: [3.2, 3.0, 3.5, 3.0],
    formG: [1.4, 1.0, 0.7, 0.42],
    detune: 1.07,
    cross: 0.24,
    pnoise: 0.5,
    intakeF: 420,
    intakeQ: 2.2,
    intakeG: 0.5,
    mechF: 4800,
    mechG: 0.012,
    whine: 0.012,
    whineOrder: 9.37,
    turbo: 0,
    crackle: 0.9,
    drive: 1.5,
    rateComp: 0.55,
    popF: 140,
  },
  // V12: two even-firing sixes, 60° apart — dense, smooth, wailing top end.
  v12: {
    cyl: 12,
    fire: even(12),
    bank: alt(12),
    amp: [1.0, 0.97, 1.03, 1.0, 0.96, 1.02, 1.01, 0.98, 1.04, 0.97, 1.0, 1.02],
    banks: 2,
    jitter: 0.05,
    misfire: 0,
    lpBase: 180,
    hOff: 1.4,
    hOn: 7.5,
    lpMax: 5200,
    tjit: 0.0016,
    dry: 0.45,
    body: 0.4,
    bankDelay: 0.00015,
    formF: [150, 430, 1150, 2900],
    formQ: [2.5, 3.2, 4.0, 4.5],
    formG: [0.9, 1.0, 0.8, 0.55],
    detune: 1.05,
    cross: 0.35,
    pnoise: 0.3,
    intakeF: 620,
    intakeQ: 2.5,
    intakeG: 0.6,
    mechF: 6000,
    mechG: 0.015,
    whine: 0.02,
    whineOrder: 12.6,
    turbo: 0,
    crackle: 0.35,
    drive: 1.3,
    rateComp: 0.6,
    popF: 190,
  },
  // Turbo straight six: single pipe through the turbine (muffled pulses), raspy turbulence, whistle + blow-off.
  i6t: {
    cyl: 6,
    fire: even(6),
    bank: [0, 0, 0, 0, 0, 0],
    amp: [1.0, 0.96, 1.03, 0.98, 1.02, 0.95],
    banks: 1,
    jitter: 0.08,
    misfire: 0,
    lpBase: 150,
    hOff: 1.3,
    hOn: 6.0,
    lpMax: 3000,
    tjit: 0.0035,
    dry: 0.5,
    body: 0.55,
    bankDelay: 0,
    formF: [115, 370, 950, 2300],
    formQ: [2.6, 3.0, 4.0, 4.0],
    formG: [1.1, 1.0, 0.85, 0.7],
    detune: 1.09,
    cross: 0.5,
    pnoise: 0.6,
    intakeF: 520,
    intakeQ: 2.0,
    intakeG: 0.45,
    mechF: 5000,
    mechG: 0.012,
    whine: 0,
    whineOrder: 0,
    turbo: 1.0,
    crackle: 0.7,
    drive: 1.7,
    rateComp: 0.55,
    popF: 150,
  },
  // V10: two even-firing fives (order 2.5 per bank) — the howl.
  v10: {
    cyl: 10,
    fire: even(10),
    bank: alt(10),
    amp: [1.0, 0.95, 1.05, 0.98, 1.03, 0.94, 1.02, 1.0, 0.96, 1.06],
    banks: 2,
    jitter: 0.07,
    misfire: 0,
    lpBase: 170,
    hOff: 1.5,
    hOn: 8.0,
    lpMax: 5600,
    tjit: 0.002,
    dry: 0.4,
    body: 0.55,
    bankDelay: 0.0002,
    formF: [135, 540, 1350, 3200],
    formQ: [2.8, 4.0, 4.5, 5.0],
    formG: [1.0, 1.1, 0.75, 0.7],
    detune: 1.09,
    cross: 0.25,
    pnoise: 0.35,
    intakeF: 700,
    intakeQ: 3.0,
    intakeG: 0.7,
    mechF: 5500,
    mechG: 0.014,
    whine: 0.015,
    whineOrder: 11.2,
    turbo: 0,
    crackle: 0.6,
    drive: 1.5,
    rateComp: 0.6,
    popF: 170,
  },
};

/** Full DSP config (voicing + level calibration tables: player rows first, then cheap-voice rows). */
export function engineConfig(type: EngineType): EngineCfg {
  const base = ENGINE_VOICINGS[type] ?? ENGINE_VOICINGS.v8;
  const cal = ENGINE_CALIB[type];
  const cells = CAL_N * 3 * 2;
  const ok = !!cal && cal.pre.length === cells && cal.post.length === cells;
  return {
    ...base,
    calR0: CAL_R0,
    calR1: CAL_R1,
    calN: CAL_N,
    pre: ok ? cal.pre.slice() : new Array<number>(cells).fill(0.25),
    post: ok ? cal.post.slice() : new Array<number>(cells).fill(0.2),
  };
}
