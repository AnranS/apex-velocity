// Car physics specifications. Geometry defaults match the model contract
// (origin at ground level under the wheelbase midpoint, +Z forward, +X left, +Y up)
// and are overridden by the model metadata JSON when it is available.

export type EngineType = 'v8' | 'v12' | 'i6t' | 'v10';

export interface CarSpec {
  id: string;
  name: string;
  nameCn: string;
  tagline: string;
  engineType: EngineType;
  engineLabel: string;
  powerHp: number;

  // geometry (model space)
  wheelbase: number;
  axleFrontZ: number;
  axleRearZ: number;
  trackFront: number;
  trackRear: number;
  wheelRadiusF: number;
  wheelRadiusR: number;
  bodyHalfWidth: number;
  bodyFrontZ: number;
  bodyRearZ: number;
  bodyHeight: number;
  rideHeight: number;

  // mass properties
  mass: number;
  comHeight: number;
  comZ: number;
  inertiaScale: [number, number, number]; // pitch (X), yaw (Y), roll (Z) multipliers on the box estimate

  // suspension
  springHzF: number;
  springHzR: number;
  dampRatio: number;
  bumpTravel: number;
  arbF: number;
  arbR: number;

  // tyres
  muLong: number;
  muLat: number;
  rearGripBias: number; // multiplies rear muLat (>1 = more stable)
  kappaPeak: number;
  alphaPeak: number;
  slideFrac: number;
  loadSens: number;

  // drivetrain
  drive: 'RWD' | 'AWD';
  awdFront: number;
  gears: number[];
  reverseRatio: number;
  finalDrive: number;
  torqueCurve: [number, number][]; // [rpm, N·m]
  idleRpm: number;
  redline: number;
  engineInertia: number;
  shiftTime: number;
  lsd: number;
  efficiency: number;

  // brakes
  brakeTorque: number;
  brakeBias: number;
  handbrakeTorque: number;

  // aero
  dragK: number; // N per (m/s)^2
  downforceK: number; // N per (m/s)^2
  aeroBalance: number; // front share

  // steering
  steerLock: number;

  // boost
  nitroBoost: number; // extra torque fraction
  nitroDuration: number; // seconds for a full tank

  stats: { speed: number; accel: number; handling: number; braking: number };
  paints: string[]; // suggested paint colours (hex)
}

const AURION: CarSpec = {
  id: 'aurion',
  name: 'AURION S8',
  nameCn: '曜影 S8',
  tagline: '中置 V8 · 均衡之王',
  engineType: 'v8',
  engineLabel: '4.0L V8 双涡轮',
  powerHp: 720,
  wheelbase: 2.7,
  axleFrontZ: 1.35,
  axleRearZ: -1.35,
  trackFront: 1.68,
  trackRear: 1.66,
  wheelRadiusF: 0.335,
  wheelRadiusR: 0.35,
  bodyHalfWidth: 0.99,
  bodyFrontZ: 2.3,
  bodyRearZ: -2.3,
  bodyHeight: 1.19,
  rideHeight: 0.11,
  mass: 1420,
  comHeight: 0.36,
  comZ: -0.1,
  inertiaScale: [0.9, 0.82, 0.9],
  springHzF: 2.5,
  springHzR: 2.35,
  dampRatio: 0.72,
  bumpTravel: 0.085,
  arbF: 26000,
  arbR: 12000,
  muLong: 1.55,
  muLat: 1.58,
  rearGripBias: 1.06,
  kappaPeak: 0.11,
  alphaPeak: 0.135,
  slideFrac: 0.82,
  loadSens: 0.1,
  drive: 'RWD',
  awdFront: 0,
  gears: [3.31, 2.26, 1.68, 1.31, 1.07, 0.89, 0.75],
  reverseRatio: 3.0,
  finalDrive: 3.85,
  torqueCurve: [
    [800, 330],
    [2000, 520],
    [3200, 640],
    [4500, 690],
    [6200, 690],
    [7400, 655],
    [8200, 600],
    [8600, 540],
  ],
  idleRpm: 950,
  redline: 8600,
  engineInertia: 0.14,
  shiftTime: 0.11,
  lsd: 60,
  efficiency: 0.9,
  brakeTorque: 7800,
  brakeBias: 0.62,
  handbrakeTorque: 4200,
  dragK: 0.46,
  downforceK: 1.5,
  aeroBalance: 0.44,
  steerLock: 0.6,
  nitroBoost: 0.45,
  nitroDuration: 3.6,
  stats: { speed: 9.1, accel: 9.0, handling: 8.8, braking: 8.6 },
  paints: ['#b3121d', '#0d1b3d', '#e9e9ee', '#f2a900', '#0e3b2e', '#16171a'],
};

const VALKEN: CarSpec = {
  ...AURION,
  id: 'valken',
  name: 'VALKEN GT',
  nameCn: '雷隼 GT',
  tagline: '前置 V12 · 直线猛兽',
  engineType: 'v12',
  engineLabel: '6.2L V12 自然吸气',
  powerHp: 780,
  wheelbase: 2.75,
  axleFrontZ: 1.375,
  axleRearZ: -1.375,
  trackFront: 1.66,
  trackRear: 1.66,
  wheelRadiusF: 0.345,
  wheelRadiusR: 0.355,
  bodyHalfWidth: 0.975,
  bodyFrontZ: 2.4,
  bodyRearZ: -2.38,
  bodyHeight: 1.3,
  rideHeight: 0.125,
  mass: 1570,
  comHeight: 0.4,
  comZ: 0.06,
  inertiaScale: [0.95, 0.9, 0.9],
  springHzF: 2.2,
  springHzR: 2.1,
  dampRatio: 0.68,
  arbF: 24000,
  arbR: 13000,
  muLong: 1.57,
  muLat: 1.585,
  rearGripBias: 1.07,
  gears: [2.9, 2.0, 1.5, 1.2, 1.0, 0.85, 0.73],
  finalDrive: 4.2,
  torqueCurve: [
    [800, 440],
    [2000, 630],
    [3500, 760],
    [5200, 815],
    [6800, 805],
    [7800, 745],
    [8400, 660],
    [8800, 570],
  ],
  idleRpm: 900,
  redline: 8800,
  engineInertia: 0.2,
  shiftTime: 0.13,
  lsd: 80,
  brakeTorque: 8200,
  brakeBias: 0.64,
  dragK: 0.47,
  downforceK: 1.38,
  aeroBalance: 0.42,
  steerLock: 0.58,
  nitroBoost: 0.5,
  stats: { speed: 9.6, accel: 8.6, handling: 7.6, braking: 8.0 },
  paints: ['#1d2b53', '#7a0f14', '#c9ccd1', '#101113', '#3b2a1a', '#e8e3d4'],
};

const ZEPHYR: CarSpec = {
  ...AURION,
  id: 'zephyr',
  name: 'ZEPHYR RX',
  nameCn: '疾风 RX',
  tagline: '轻量赛道利器 · 弯道之刃',
  engineType: 'i6t',
  engineLabel: '3.0L 直六涡轮',
  powerHp: 560,
  wheelbase: 2.52,
  axleFrontZ: 1.26,
  axleRearZ: -1.26,
  trackFront: 1.6,
  trackRear: 1.58,
  wheelRadiusF: 0.325,
  wheelRadiusR: 0.335,
  bodyHalfWidth: 0.93,
  bodyFrontZ: 2.12,
  bodyRearZ: -2.13,
  bodyHeight: 1.25,
  rideHeight: 0.1,
  mass: 1180,
  comHeight: 0.35,
  comZ: -0.04,
  inertiaScale: [0.9, 0.84, 0.9],
  springHzF: 2.8,
  springHzR: 2.6,
  dampRatio: 0.75,
  arbF: 24000,
  arbR: 12000,
  muLong: 1.57,
  muLat: 1.6,
  rearGripBias: 1.07,
  gears: [3.2, 2.2, 1.66, 1.3, 1.06, 0.88],
  finalDrive: 3.7,
  torqueCurve: [
    [900, 260],
    [2200, 430],
    [3200, 560],
    [4500, 590],
    [6200, 585],
    [7200, 540],
    [7800, 480],
    [8200, 420],
  ],
  idleRpm: 1000,
  redline: 8200,
  engineInertia: 0.11,
  shiftTime: 0.09,
  lsd: 55,
  brakeTorque: 7000,
  brakeBias: 0.6,
  handbrakeTorque: 3600,
  dragK: 0.53,
  downforceK: 1.75,
  aeroBalance: 0.43,
  steerLock: 0.62,
  nitroBoost: 0.42,
  stats: { speed: 8.2, accel: 8.8, handling: 9.7, braking: 9.3 },
  paints: ['#f4f4f6', '#00a6d6', '#ff5a1f', '#c8e21a', '#16171a', '#6a1b9a'],
};

export const CAR_SPECS: Record<string, CarSpec> = {
  aurion: AURION,
  valken: VALKEN,
  zephyr: ZEPHYR,
};
export const CAR_IDS = ['aurion', 'valken', 'zephyr'];

/** Geometry block of the model metadata JSON (public/assets/cars/<id>.json). */
export interface CarModelMeta {
  id: string;
  length: number;
  width: number;
  height: number;
  wheelbase: number;
  axleFrontZ: number;
  axleRearZ: number;
  trackFront: number;
  trackRear: number;
  wheelRadiusFront: number;
  wheelRadiusRear: number;
  bodyWidth?: number; // body shell without mirrors
  rideHeight?: number;
  bodyBox?: { min: number[]; max: number[] };
  points?: Record<string, number[]>;
}

/** Returns a copy of `spec` with geometry taken from the model metadata. */
export function applyModelMeta(spec: CarSpec, meta: CarModelMeta): CarSpec {
  const s: CarSpec = { ...spec };
  s.wheelbase = meta.wheelbase;
  s.axleFrontZ = meta.axleFrontZ;
  s.axleRearZ = meta.axleRearZ;
  s.trackFront = meta.trackFront;
  s.trackRear = meta.trackRear;
  s.wheelRadiusF = meta.wheelRadiusFront;
  s.wheelRadiusR = meta.wheelRadiusRear;
  if (meta.bodyBox) {
    s.bodyHalfWidth = meta.bodyWidth ? meta.bodyWidth * 0.5 : Math.max(Math.abs(meta.bodyBox.min[0]), Math.abs(meta.bodyBox.max[0]));
    s.bodyFrontZ = meta.bodyBox.max[2];
    s.bodyRearZ = meta.bodyBox.min[2];
    s.bodyHeight = meta.bodyBox.max[1];
  } else {
    s.bodyHalfWidth = meta.width * 0.5;
    s.bodyFrontZ = meta.length * 0.5;
    s.bodyRearZ = -meta.length * 0.5;
    s.bodyHeight = meta.height;
  }
  if (meta.rideHeight) s.rideHeight = meta.rideHeight;
  return s;
}
