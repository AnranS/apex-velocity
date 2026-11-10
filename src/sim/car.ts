// Vehicle dynamics: a custom 6-DOF rigid body with ray/sphere suspension, combined-slip tyres with
// implicitly integrated wheel spin, engine + gearbox + clutch, aero, and impulse-based contacts
// against the analytic track (walls + ground). Everything is allocation-free in the hot path.
//
// Body axes (world columns of the rotation matrix): X = left, Y = up, Z = forward.
import { clamp, clamp01, damp, lerp, moveToward, smoothstep } from './math';
import type { CarSpec } from './car_specs';
import { SURF_ASPHALT, SURF_KERB, Track } from './track';

export interface CarInput {
  throttle: number; // 0..1
  brake: number; // 0..1
  steer: number; // -1..1, + = left
  handbrake: number; // 0..1
  nitro: boolean;
  shiftUp: boolean; // edge-triggered by the caller
  shiftDown: boolean;
  /** fire small boosts automatically when their window opens (touch / drivers without a throttle key) */
  autoBoost?: boolean;
}
export function makeInput(): CarInput {
  return { throttle: 0, brake: 0, steer: 0, handbrake: 0, nitro: false, shiftUp: false, shiftDown: false };
}

export const EV_SHIFT_UP = 1;
export const EV_SHIFT_DOWN = 2;
export const EV_LIMITER = 4;
export const EV_BACKFIRE = 8;
export const EV_NITRO_START = 16;
export const EV_BLOWOFF = 32;
export const EV_LAND = 64;
export const EV_NITRO_END = 128;
export const EV_DRIFT_START = 256;
export const EV_DRIFT_END = 512;
export const EV_MINI_BOOST = 1024;
export const EV_NITRO_READY = 2048;

const G = 9.81;
const V0 = 2.5; // slip reference floor speed (m/s)
const RPM2RAD = Math.PI / 30;
const SURF_MU = [1.0, 0.95, 0.6, 0.5];
const SURF_ROLL = [0.011, 0.016, 0.065, 0.17];
const WET_MU = 0.8;

function tireF(s: number, slide: number): number {
  if (s <= 1) return s * (2 - s);
  return slide + (1 - slide) * Math.exp(-(s - 1) * 1.1);
}

export class Wheel {
  // configuration
  lx = 0; // attach point (top of travel) relative to the centre of mass, body frame
  ly = 0;
  lz = 0;
  radius = 0.34;
  inertia = 1.5;
  k = 80000;
  cBump = 5000;
  cRebound = 7500;
  freeLen = 0.13; // suspension length (attach -> wheel centre) at zero spring force
  maxLen = 0.16;
  bump = 0.085;
  staticLoad = 3500;
  front = false;
  left = false;
  driven = false;
  driveShare = 0; // share of total drive torque
  // state
  omega = 0;
  angle = 0;
  steer = 0;
  len = 0.085;
  fz = 0;
  contact = false;
  surface = SURF_ASPHALT;
  hint = 0;
  cpx = 0;
  cpy = 0;
  cpz = 0;
  nx = 0;
  ny = 1;
  nz = 0;
  slipRatio = 0;
  slipAngle = 0;
  slipNorm = 0;
  skid = 0;
  absFactor = 1;
  driveTorque = 0;
  brakeTorque = 0;
  // render interpolation
  prevAngle = 0;
  prevLen = 0.085;
  prevSteer = 0;
}

export class Car {
  readonly spec: CarSpec;
  index = 0;
  // rigid body state (position = centre of mass)
  px = 0;
  py = 0;
  pz = 0;
  qx = 0;
  qy = 0;
  qz = 0;
  qw = 1;
  vx = 0;
  vy = 0;
  vz = 0;
  wx = 0;
  wy = 0;
  wz = 0;
  // previous state for render interpolation
  ppx = 0;
  ppy = 0;
  ppz = 0;
  pqx = 0;
  pqy = 0;
  pqz = 0;
  pqw = 1;
  // rotation matrix (body -> world)
  m00 = 1;
  m01 = 0;
  m02 = 0;
  m10 = 0;
  m11 = 1;
  m12 = 0;
  m20 = 0;
  m21 = 0;
  m22 = 1;
  // mass properties
  invMass = 1;
  invIx = 1;
  invIy = 1;
  invIz = 1;
  Ix = 1;
  Iy = 1;
  Iz = 1;
  /** centre of mass in model space */
  comY = 0.36;
  comZ = 0;

  readonly wheels: Wheel[] = [new Wheel(), new Wheel(), new Wheel(), new Wheel()];
  private torqueTable: Float32Array;

  // controls / drivetrain state
  steer = 0; // smoothed normalised steering
  steerAngle = 0; // rad at the front wheels
  thr = 0;
  brk = 0;
  gear = 1; // -1 reverse, 0 neutral, 1..n
  shiftTimer = 0;
  sinceShift = 1;
  engineW = 100; // rad/s
  rpm = 1000;
  clutchSlip = false;
  limiterTimer = 0;
  reverseHold = 0;
  /** nitro gauge 0..1: drifting fills it; a full gauge is stored as a bottle */
  nitro = 0;
  /** stored nitro bottles (0..2); one press of the nitro key burns one */
  nitroBottles = 0;
  nitroOn = false;
  /** seconds of nitro burn remaining */
  nitroTime = 0;
  /** kind of the last small boost: 1 small, 2 double, 3 super start, 4 landing */
  boostKind = 0;
  /** seconds left to fire a small boost (a fresh press of the accelerator) */
  boostWindow = 0;
  private windowKind = 1;
  private chain = 0;
  private chainTimer = 0;
  private prevNitro = false;
  private prevThr = false;
  private prevDriftKey = false;
  private sinceGo = 99;
  private wasHold = true;
  private startUsed = false;
  tcsScale = 1;
  hold = true; // held on the grid (neutral + brakes)
  private liftArmed = false;
  private hbGrip = 1;

  // ---- arcade handling (direct path control + scripted drift); `false` = full tyre simulation
  arcade = true;
  drifting = false;
  driftDir = 0; // +1 left, -1 right
  driftTime = 0;
  driftAngle = 0; // current target body slip angle (rad)
  miniBoost = 0; // seconds of post-drift boost remaining
  /** tank refill per second while not drifting (AI drivers, who do not drift, get a higher rate) */
  passiveNitro = 0;
  /** body sideslip: + when the nose points left of the direction of travel */
  beta = 0;
  private arcW = 0;
  private flipTimer = 0;
  private arcYawAcc = 0;

  // assists / modifiers
  abs = true;
  tcs = 1; // 0 off, 1 sport, 2 full
  steerAssist = 1;
  autoGear = true;
  gripScale = 1;
  powerScale = 1;
  wet = 0;
  draft = 0;
  ghost = 0; // seconds of collision immunity after a reset

  // track coordinates
  trkIdx = 0;
  trkT = 0;
  s = 0;
  d = 0;

  // telemetry
  speed = 0;
  fwdSpeed = 0;
  latG = 0;
  longG = 0;
  yawRate = 0;
  rearSlipAngle = 0;
  scrape = 0;
  wheelsOnGround = 4;
  airTime = 0;
  upDot = 1;
  events = 0;
  impact = 0; // strongest impact impulse (N·s) since last consumed
  impactX = 0;
  impactY = 0;
  impactZ = 0;
  driftScore = 0;
  surfaceMix = 0; // dominant surface under the wheels

  constructor(spec: CarSpec) {
    this.spec = spec;
    this.torqueTable = new Float32Array(101);
    this.configure();
  }

  private configure(): void {
    const sp = this.spec;
    const m = sp.mass;
    this.invMass = 1 / m;
    this.comY = sp.comHeight;
    this.comZ = sp.comZ;
    const L = sp.bodyFrontZ - sp.bodyRearZ;
    const W = sp.bodyHalfWidth * 2;
    const H = sp.bodyHeight;
    this.Ix = (m / 12) * (L * L + H * H) * sp.inertiaScale[0];
    this.Iy = (m / 12) * (L * L + W * W) * sp.inertiaScale[1];
    this.Iz = (m / 12) * (W * W + H * H) * sp.inertiaScale[2] * 1.6;
    this.invIx = 1 / this.Ix;
    this.invIy = 1 / this.Iy;
    this.invIz = 1 / this.Iz;

    const a = sp.axleFrontZ - sp.comZ; // com -> front axle
    const b = sp.comZ - sp.axleRearZ; // com -> rear axle
    const frontShare = b / (a + b);
    for (let i = 0; i < 4; i++) {
      const w = this.wheels[i];
      const front = i < 2;
      const left = i % 2 === 0;
      w.front = front;
      w.left = left;
      w.radius = front ? sp.wheelRadiusF : sp.wheelRadiusR;
      w.inertia = front ? 1.3 : 1.6;
      const trackW = front ? sp.trackFront : sp.trackRear;
      const load = (m * G * (front ? frontShare : 1 - frontShare)) / 2;
      const hz = front ? sp.springHzF : sp.springHzR;
      const mc = load / G;
      w.k = Math.pow(2 * Math.PI * hz, 2) * mc;
      const cc = 2 * Math.sqrt(w.k * mc) * sp.dampRatio;
      w.cBump = cc * 0.8;
      w.cRebound = cc * 1.3;
      w.staticLoad = load;
      w.bump = sp.bumpTravel;
      w.freeLen = sp.bumpTravel + load / w.k;
      w.maxLen = w.freeLen + 0.035;
      w.len = sp.bumpTravel;
      w.prevLen = w.len;
      w.lx = (left ? 1 : -1) * trackW * 0.5;
      w.ly = w.radius + sp.bumpTravel - sp.comHeight;
      w.lz = (front ? sp.axleFrontZ : sp.axleRearZ) - sp.comZ;
      if (sp.drive === 'RWD') {
        w.driven = !front;
        w.driveShare = front ? 0 : 0.5;
      } else {
        w.driven = true;
        w.driveShare = front ? sp.awdFront * 0.5 : (1 - sp.awdFront) * 0.5;
      }
    }
    // torque lookup: index = rpm / (redline*1.05) * 100
    const tc = sp.torqueCurve;
    for (let i = 0; i <= 100; i++) {
      const rpm = (i / 100) * sp.redline * 1.05;
      let t = tc[tc.length - 1][1];
      if (rpm <= tc[0][0]) t = tc[0][1] * clamp01(rpm / tc[0][0] + 0.3);
      else {
        for (let k = 0; k < tc.length - 1; k++) {
          if (rpm >= tc[k][0] && rpm <= tc[k + 1][0]) {
            t = lerp(tc[k][1], tc[k + 1][1], (rpm - tc[k][0]) / (tc[k + 1][0] - tc[k][0]));
            break;
          }
        }
        if (rpm > tc[tc.length - 1][0]) t = tc[tc.length - 1][1] * clamp01(1 - (rpm - tc[tc.length - 1][0]) / (sp.redline * 0.06));
      }
      this.torqueTable[i] = t;
    }
  }

  torqueAt(rpm: number): number {
    const f = clamp((rpm / (this.spec.redline * 1.05)) * 100, 0, 99.999);
    const i = Math.floor(f);
    return this.torqueTable[i] + (this.torqueTable[i + 1] - this.torqueTable[i]) * (f - i);
  }

  /** Peak wheel power (W), used for AI speed profiles. */
  peakPower(): number {
    let best = 0;
    for (let rpm = 2000; rpm <= this.spec.redline; rpm += 100) best = Math.max(best, this.torqueAt(rpm) * rpm * RPM2RAD);
    return best * this.spec.efficiency;
  }

  updateMatrix(): void {
    const x = this.qx;
    const y = this.qy;
    const z = this.qz;
    const w = this.qw;
    this.m00 = 1 - 2 * (y * y + z * z);
    this.m01 = 2 * (x * y - z * w);
    this.m02 = 2 * (x * z + y * w);
    this.m10 = 2 * (x * y + z * w);
    this.m11 = 1 - 2 * (x * x + z * z);
    this.m12 = 2 * (y * z - x * w);
    this.m20 = 2 * (x * z - y * w);
    this.m21 = 2 * (y * z + x * w);
    this.m22 = 1 - 2 * (x * x + y * y);
  }

  /** Heading angle psi (forward = (sin psi, 0, cos psi)). */
  heading(): number {
    return Math.atan2(this.m02, this.m22);
  }

  /** Max front wheel angle available at the current speed. */
  maxSteer(): number {
    const sp = this.spec;
    const v = this.fwdSpeed;
    const aLat = sp.muLat * this.gripScale * (G + (sp.downforceK * v * v) / sp.mass);
    return Math.min(sp.steerLock, Math.atan((sp.wheelbase * aLat) / Math.max(v * v, 1)) + sp.alphaPeak * 1.25);
  }

  /** Lateral grip coefficient of the arcade model. */
  arcadeMu(): number {
    return this.spec.muLat * 1.12 * this.gripScale * (1 - this.wet * 0.16);
  }

  /** Max path yaw rate (rad/s) the arcade model allows at the current speed. */
  maxYawRate(): number {
    const sp = this.spec;
    const v = Math.max(Math.abs(this.fwdSpeed), 3);
    const aMax = this.arcadeMu() * (G + (sp.downforceK * v * v) / sp.mass);
    return Math.min(aMax / v, (v * Math.tan(sp.steerLock * 0.8)) / sp.wheelbase);
  }

  /** Normalised steering input (-1..1) that makes the car follow path curvature `kappa` (1/m, + left). */
  steerForCurvature(kappa: number): number {
    if (this.arcade && this.fwdSpeed > 6) return clamp((Math.max(this.fwdSpeed, 3) * kappa) / this.maxYawRate(), -1, 1);
    return clamp(Math.atan(this.spec.wheelbase * kappa) / Math.max(this.maxSteer(), 0.02), -1, 1);
  }

  /** Place the car at rest on the track at lap distance s, lateral offset d. */
  placeOnTrack(track: Track, s: number, d: number, extraYaw = 0): void {
    const o = { x: 0, y: 0, z: 0, psi: 0 };
    track.posAt(s, d, o);
    const psi = o.psi + extraYaw;
    // orientation: yaw only, then tilt to the ground normal
    const nx = track.onx;
    const ny = track.ony;
    const nz = track.onz;
    let fx = Math.sin(psi);
    let fy = 0;
    let fz = Math.cos(psi);
    const fd = fx * nx + fy * ny + fz * nz;
    fx -= nx * fd;
    fy -= ny * fd;
    fz -= nz * fd;
    const fl = 1 / Math.hypot(fx, fy, fz);
    fx *= fl;
    fy *= fl;
    fz *= fl;
    // left = up x forward
    const lx = ny * fz - nz * fy;
    const ly = nz * fx - nx * fz;
    const lz = nx * fy - ny * fx;
    this.setBasis(lx, ly, lz, nx, ny, nz, fx, fy, fz);
    this.updateMatrix();
    // model origin sits on the ground; centre of mass offset in body frame = (0, comY, comZ)
    this.px = o.x + this.m01 * this.comY + this.m02 * this.comZ;
    this.py = o.y + this.m11 * this.comY + this.m12 * this.comZ + 0.005;
    this.pz = o.z + this.m21 * this.comY + this.m22 * this.comZ;
    this.vx = this.vy = this.vz = 0;
    this.wx = this.wy = this.wz = 0;
    this.steer = 0;
    this.thr = 0;
    this.gear = 1;
    this.shiftTimer = 0;
    this.drifting = false;
    this.driftAngle = 0;
    this.driftTime = 0;
    this.miniBoost = 0;
    this.boostWindow = 0;
    this.chain = 0;
    this.beta = 0;
    this.engineW = this.spec.idleRpm * RPM2RAD;
    this.rpm = this.spec.idleRpm;
    track.locateGlobal(this.px, this.pz);
    this.trkIdx = track.oi;
    this.trkT = track.ot;
    this.d = track.od;
    this.s = (track.oi + track.ot) * track.ds;
    for (const w of this.wheels) {
      w.omega = 0;
      w.len = w.bump;
      w.prevLen = w.len;
      w.hint = this.trkIdx;
      w.contact = true;
      w.fz = w.staticLoad;
      w.skid = 0;
      w.slipRatio = 0;
      w.slipAngle = 0;
    }
    this.savePrev();
  }

  private setBasis(lx: number, ly: number, lz: number, ux: number, uy: number, uz: number, fx: number, fy: number, fz: number): void {
    // rotation matrix columns (left, up, forward) -> quaternion
    const m00 = lx;
    const m10 = ly;
    const m20 = lz;
    const m01 = ux;
    const m11 = uy;
    const m21 = uz;
    const m02 = fx;
    const m12 = fy;
    const m22 = fz;
    const tr = m00 + m11 + m22;
    if (tr > 0) {
      const s = Math.sqrt(tr + 1) * 2;
      this.qw = 0.25 * s;
      this.qx = (m21 - m12) / s;
      this.qy = (m02 - m20) / s;
      this.qz = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
      const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
      this.qw = (m21 - m12) / s;
      this.qx = 0.25 * s;
      this.qy = (m01 + m10) / s;
      this.qz = (m02 + m20) / s;
    } else if (m11 > m22) {
      const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
      this.qw = (m02 - m20) / s;
      this.qx = (m01 + m10) / s;
      this.qy = 0.25 * s;
      this.qz = (m12 + m21) / s;
    } else {
      const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
      this.qw = (m10 - m01) / s;
      this.qx = (m02 + m20) / s;
      this.qy = (m12 + m21) / s;
      this.qz = 0.25 * s;
    }
  }

  savePrev(): void {
    this.ppx = this.px;
    this.ppy = this.py;
    this.ppz = this.pz;
    this.pqx = this.qx;
    this.pqy = this.qy;
    this.pqz = this.qz;
    this.pqw = this.qw;
    for (let i = 0; i < 4; i++) {
      const w = this.wheels[i];
      w.prevAngle = w.angle;
      w.prevLen = w.len;
      w.prevSteer = w.steer;
    }
  }

  // ------------------------------------------------------------------ main force step
  step(dt: number, inp: CarInput, track: Track): void {
    const sp = this.spec;
    this.updateMatrix();
    const lx = this.m00;
    const ly = this.m10;
    const lz = this.m20;
    const ux = this.m01;
    const uy = this.m11;
    const uz = this.m21;
    const fx = this.m02;
    const fy = this.m12;
    const fz = this.m22;
    const px = this.px;
    const py = this.py;
    const pz = this.pz;
    const vx0 = this.vx;
    const vy0 = this.vy;
    const vz0 = this.vz;
    const wx = this.wx;
    const wy = this.wy;
    const wz = this.wz;
    const wheels = this.wheels;

    const fwdSpeed = vx0 * fx + vy0 * fy + vz0 * fz;
    const speed = Math.sqrt(vx0 * vx0 + vy0 * vy0 + vz0 * vz0);
    this.fwdSpeed = fwdSpeed;
    this.speed = speed;
    if (this.ghost > 0) this.ghost -= dt;

    // ------------------------------------------------ driver controls
    let thrCmd = clamp01(inp.throttle);
    let brkCmd = clamp01(inp.brake);
    let hbCmd = clamp01(inp.handbrake);
    if (this.hold) {
      brkCmd = 1;
      hbCmd = 1;
    } else if (this.gear >= 1) {
      if (inp.brake > 0.2 && inp.throttle < 0.1 && fwdSpeed < 1.2) {
        this.reverseHold += dt;
        if (this.reverseHold > 0.22) {
          this.gear = -1;
          this.reverseHold = 0;
          this.sinceShift = 0;
        }
      } else this.reverseHold = 0;
    } else if (this.gear === -1) {
      thrCmd = clamp01(inp.brake);
      brkCmd = clamp01(inp.throttle);
      if (inp.throttle > 0.2 && fwdSpeed > -1.2) {
        this.gear = 1;
        this.sinceShift = 0;
      }
    } else if (this.gear === 0 && !this.hold) {
      this.gear = 1;
    }
    if (this.thr > 0.85) this.liftArmed = true;
    else if (this.liftArmed && thrCmd < 0.15) {
      this.liftArmed = false;
      if (!this.hold && this.rpm > sp.redline * 0.55) {
        this.events |= EV_BLOWOFF;
        if (this.rpm > sp.redline * 0.72) this.events |= EV_BACKFIRE;
      }
    }
    this.thr = moveToward(this.thr, thrCmd, dt * (thrCmd > this.thr ? 9 : 14));
    this.brk = moveToward(this.brk, brkCmd, dt * 12);
    const thr = this.thr;
    const brk = this.brk;

    // steering (speed-sensitive lock, rate limited, optional countersteer assist)
    const steerMax = this.maxSteer();
    const steerTarget = clamp(inp.steer, -1, 1);
    const returning = Math.abs(steerTarget) < Math.abs(this.steer) || steerTarget * this.steer < 0;
    const steerRate = this.arcade ? (returning ? 12 : lerp(7.5, 3.4, clamp01(speed / 75))) : returning ? 8.5 : lerp(5.2, 2.8, clamp01(speed / 75));
    this.steer = moveToward(this.steer, steerTarget, steerRate * dt);
    let delta = this.steer * steerMax;
    // arcade weight: fades in with speed so parking-speed behaviour stays physical
    // (uses planar speed, so a car knocked sideways stays under arcade control and straightens itself)
    const vLat0 = vx0 * lx + vy0 * ly + vz0 * lz;
    const arcW = this.arcade && this.gear >= 1 ? smoothstep(3.5, 9, Math.sqrt(fwdSpeed * fwdSpeed + vLat0 * vLat0)) : 0;
    this.arcW = arcW;
    if (arcW > 0) delta = clamp(delta - this.beta * 0.75 * arcW, -sp.steerLock, sp.steerLock);
    if (this.steerAssist > 0 && speed > 6 && arcW < 0.5) {
      const beta = this.rearSlipAngle;
      const ab = Math.abs(beta);
      const dz = 0.065;
      if (ab > dz) {
        const ex = (ab - dz) * (beta > 0 ? 1 : -1);
        delta += clamp(ex * 0.9 * this.steerAssist, -0.42, 0.42) * smoothstep(6, 14, speed);
      }
    }
    delta = clamp(delta, -sp.steerLock, sp.steerLock);
    this.steerAngle = delta;
    wheels[0].steer = delta;
    wheels[1].steer = delta;

    // ------------------------------------------------ suspension
    let FX = 0;
    let FY = 0;
    let FZ = 0;
    let TX = 0;
    let TY = 0;
    let TZ = 0;
    let onGround = 0;
    for (let i = 0; i < 4; i++) {
      const w = wheels[i];
      const ox = px + lx * w.lx + ux * w.ly + fx * w.lz;
      const oy = py + ly * w.lx + uy * w.ly + fy * w.lz;
      const oz = pz + lz * w.lx + uz * w.ly + fz * w.lz;
      const reach = w.len + w.radius;
      const gx = ox - ux * reach;
      const gz = oz - uz * reach;
      track.ground(gx, gz, w.hint);
      w.hint = track.oi;
      const nx = track.onx;
      const ny = track.ony;
      const nz = track.onz;
      const un = Math.max(ux * nx + uy * ny + uz * nz, 0.3);
      const distN = ux * reach * nx + (oy - track.oy) * ny + uz * reach * nz; // (O - G)·n, G = ground point under the estimate
      let L = (distN - w.radius) / un;
      let bumpStop = 0;
      if (L < 0) {
        bumpStop = -L;
        L = 0;
      }
      w.nx = nx;
      w.ny = ny;
      w.nz = nz;
      w.surface = track.osurf;
      if (L >= w.freeLen) {
        w.contact = false;
        w.fz = 0;
        w.len = Math.min(L, w.maxLen);
        w.cpx = ox - ux * (w.len + w.radius);
        w.cpy = oy - uy * (w.len + w.radius);
        w.cpz = oz - uz * (w.len + w.radius);
        continue;
      }
      // wheel centre and its velocity
      const cx = ox - ux * L;
      const cy = oy - uy * L;
      const cz = oz - uz * L;
      const rx = cx - px;
      const ry = cy - py;
      const rz = cz - pz;
      const vcx = vx0 + (wy * rz - wz * ry);
      const vcy = vy0 + (wz * rx - wx * rz);
      const vcz = vz0 + (wx * ry - wy * rx);
      const xdot = -(vcx * nx + vcy * ny + vcz * nz) / un;
      const x = w.freeLen - L;
      let f = w.k * x + (xdot > 0 ? w.cBump : w.cRebound) * xdot;
      if (bumpStop > 0) f += w.k * 6 * bumpStop + w.cBump * 2 * Math.max(xdot, 0);
      const fMax = w.staticLoad * 6;
      w.fz = f < 0 ? 0 : f > fMax ? fMax : f;
      w.contact = true;
      w.len = L;
      w.cpx = cx - nx * w.radius;
      w.cpy = cy - ny * w.radius;
      w.cpz = cz - nz * w.radius;
      onGround++;
    }
    // anti-roll bars
    for (let a = 0; a < 2; a++) {
      const wl = wheels[a * 2];
      const wr = wheels[a * 2 + 1];
      if (!wl.contact && !wr.contact) continue;
      const karb = a === 0 ? sp.arbF : sp.arbR;
      const df = karb * (wr.len - wl.len); // left more compressed (shorter) -> + on left
      if (wl.contact) wl.fz = Math.max(0, wl.fz + df);
      if (wr.contact) wr.fz = Math.max(0, wr.fz - df);
    }
    if (onGround === 0) this.airTime += dt;
    else {
      if (this.airTime > 0.25) {
        this.events |= EV_LAND;
        // landing boost window
        this.boostWindow = 0.4;
        this.windowKind = 4;
      }
      this.airTime = 0;
    }
    this.wheelsOnGround = onGround;

    // ------------------------------------------------ drivetrain
    const nGears = sp.gears.length;
    this.sinceShift += dt;
    if (this.shiftTimer > 0) this.shiftTimer -= dt;
    const gearRatio = this.gear > 0 ? sp.gears[this.gear - 1] : this.gear < 0 ? -sp.reverseRatio : 0;
    const Gr = gearRatio * sp.finalDrive;
    let wDriven: number;
    if (sp.drive === 'RWD') wDriven = (wheels[2].omega + wheels[3].omega) * 0.5;
    else wDriven = sp.awdFront * (wheels[0].omega + wheels[1].omega) * 0.5 + (1 - sp.awdFront) * (wheels[2].omega + wheels[3].omega) * 0.5;
    const wIdle = sp.idleRpm * RPM2RAD;
    const wRed = sp.redline * RPM2RAD;
    let wheelTorque = 0;
    let slipping = false;
    if (this.limiterTimer > 0) this.limiterTimer -= dt;

    // ---- nitro + small boosts (same rules in both handling modes)
    const nitroDown = inp.nitro && !this.hold;
    const nitroEdge = nitroDown && !this.prevNitro;
    this.prevNitro = nitroDown;
    const thrDown = inp.throttle > 0.5 && !this.hold;
    const thrEdge = thrDown && !this.prevThr;
    this.prevThr = thrDown;
    // (the raw key is tracked through the countdown, so holding it across the green light is not a "tap")
    const rawDrift = inp.handbrake > 0.5;
    const driftKey = rawDrift && !this.hold;
    const driftEdge = driftKey && !this.prevDriftKey;
    this.prevDriftKey = rawDrift;
    if (this.wasHold && !this.hold) {
      this.sinceGo = 0;
      this.startUsed = false;
    }
    this.wasHold = this.hold;
    if (this.sinceGo < 10) this.sinceGo += dt;
    // one press burns one stored bottle
    if (this.nitroOn) {
      this.nitroTime -= dt;
      if (this.nitroTime <= 0 || this.hold || this.gear < 1) {
        this.nitroOn = false;
        this.events |= EV_NITRO_END;
      }
    } else if (nitroEdge && this.nitroBottles > 0 && this.gear >= 1) {
      this.nitroBottles--;
      this.nitroOn = true;
      this.nitroTime = sp.nitroDuration;
      this.events |= EV_NITRO_START;
    }
    // drifting fills the gauge; a full gauge becomes a bottle (two at most)
    if (!this.hold) {
      const charge = this.drifting ? 0.3 * (0.45 + 0.55 * clamp01(Math.abs(this.beta) / 0.6)) : this.passiveNitro + this.draft * 0.03;
      if (charge > 0) {
        this.nitro += dt * charge;
        if (this.nitro >= 1) {
          if (this.nitroBottles < 2) {
            this.nitroBottles++;
            this.nitro -= 1;
            this.events |= EV_NITRO_READY;
          } else this.nitro = 1;
        }
      }
    }
    // small boost: a fresh press of the accelerator inside the window after a drift ends or a landing
    if (this.chainTimer > 0) {
      this.chainTimer -= dt;
      if (this.chainTimer <= 0) this.chain = 0;
    }
    if (this.boostWindow > 0) {
      this.boostWindow -= dt;
      if (thrEdge || inp.autoBoost) {
        this.boostWindow = 0;
        this.fireBoost(this.windowKind === 4 ? 4 : this.chain === 0 ? 1 : 2);
      }
    }
    // super start: tap the drift key right as the lights go out
    if (driftEdge && this.sinceGo < 0.45 && this.gear >= 1 && !this.startUsed) {
      this.startUsed = true;
      this.fireBoost(3);
    }
    if (this.miniBoost > 0) this.miniBoost -= dt;

    // traction control (uses last tick's slip)
    let tcsTarget = 1;
    if (this.tcs > 0) {
      const lim = this.tcs === 2 ? 1.05 : 1.75;
      let worst = 0;
      const dir = Gr < 0 ? -1 : 1;
      for (let i = 0; i < 4; i++) {
        const w = wheels[i];
        if (!w.driven || !w.contact) continue;
        const kx = (w.slipRatio * dir) / sp.kappaPeak;
        // combined slip: drive slip only counts once it uses a real share of the friction circle
        if (kx > 0.3) worst = Math.max(worst, this.tcs === 2 ? w.slipNorm : kx);
      }
      if (worst > lim) tcsTarget = clamp(1 - (worst - lim) * (this.tcs === 2 ? 1.6 : 0.8), 0.12, 1);
    }
    this.tcsScale = damp(this.tcsScale, tcsTarget, tcsTarget < this.tcsScale ? 60 : 14, dt);

    if (this.gear === 0 || this.hold) {
      const target = wIdle + thr * (wRed * 0.86 - wIdle);
      this.engineW = damp(this.engineW, target, target > this.engineW ? 7 : 4, dt);
      if (this.engineW > wRed * 0.85 && thr > 0.9 && this.limiterTimer <= 0) {
        this.events |= EV_LIMITER;
        this.limiterTimer = 0.09;
      }
    } else {
      const wFromWheels = wDriven * Gr;
      const first = this.gear === 1 || this.gear === -1;
      const launch = first ? lerp(wIdle * 1.25, wRed * 0.58, thr) : wIdle * 1.05;
      let wEng = wFromWheels;
      if (wFromWheels < launch) {
        slipping = true;
        wEng = launch;
      }
      this.engineW = damp(this.engineW, wEng, 28, dt);
      const rpmEff = wEng / RPM2RAD;
      const boost = 1 + (this.nitroOn ? sp.nitroBoost + 0.2 : 0) + (this.miniBoost > 0 ? 0.55 : 0);
      let T = thr * this.torqueAt(rpmEff) * boost * this.powerScale * this.tcsScale;
      const fric = 30 + 0.012 * rpmEff;
      if (!slipping) T -= (1 - thr) * fric;
      else if (thr < 0.04) T = 0;
      if (wEng > wRed) {
        T = -fric * 1.5;
        if (this.limiterTimer <= 0) {
          this.events |= EV_LIMITER;
          this.limiterTimer = 0.08;
        }
      }
      if (this.shiftTimer > 0) T = Math.min(T, 0) * 0.3;
      wheelTorque = T * Gr * sp.efficiency;

      // automatic gearbox / manual shifting
      if (this.gear >= 1 && this.shiftTimer <= 0) {
        const rpmNow = wFromWheels / RPM2RAD;
        let up = false;
        let down = false;
        if (this.autoGear) {
          if (this.sinceShift > 0.32) {
            const upRpm = sp.redline * lerp(0.74, 0.975, smoothstep(0.3, 0.85, thr));
            if (this.gear < nGears && rpmNow > upRpm && (!slipping || rpmNow > sp.redline)) up = true;
            if (this.gear > 1 && !up) {
              const lower = (rpmNow * sp.gears[this.gear - 2]) / sp.gears[this.gear - 1];
              const lim = sp.redline * (brk > 0.2 ? 0.9 : thr > 0.85 ? 0.8 : 0.52);
              if (lower < lim && this.sinceShift > 0.42) down = true;
            }
          }
        } else {
          if (inp.shiftUp && this.gear < nGears) up = true;
          if (inp.shiftDown && this.gear > 1) {
            const lower = (rpmNow * sp.gears[this.gear - 2]) / sp.gears[this.gear - 1];
            if (lower < sp.redline * 1.02) down = true;
          }
        }
        if (up) {
          this.gear++;
          this.shiftTimer = sp.shiftTime;
          this.sinceShift = 0;
          this.events |= EV_SHIFT_UP;
          if (thr > 0.7 && rpmNow > sp.redline * 0.8) this.events |= EV_BACKFIRE;
        } else if (down) {
          this.gear--;
          this.shiftTimer = sp.shiftTime * 0.8;
          this.sinceShift = 0;
          this.events |= EV_SHIFT_DOWN;
        }
      }
    }
    this.clutchSlip = slipping;
    this.rpm = this.engineW / RPM2RAD;

    // distribute torques
    const reflected = slipping || this.gear === 0 || this.hold ? 0 : sp.engineInertia * Gr * Gr;
    for (let i = 0; i < 4; i++) {
      const w = wheels[i];
      w.driveTorque = wheelTorque * w.driveShare;
      let bt = brk * sp.brakeTorque * (w.front ? sp.brakeBias : 1 - sp.brakeBias) * 0.5;
      if (this.abs && brk > 0.05 && w.contact) {
        const k = -w.slipRatio / sp.kappaPeak;
        const target = k > 1.15 && Math.abs(this.fwdSpeed) > 3 ? clamp(1 - (k - 1.15) * 1.8, 0.12, 1) : 1;
        w.absFactor = damp(w.absFactor, target, target < w.absFactor ? 80 : 20, dt);
        bt *= w.absFactor;
      } else w.absFactor = 1;
      if (!w.front) bt += hbCmd * (1 - arcW) * (this.hold || Math.abs(fwdSpeed) < 4 ? sp.handbrakeTorque : Math.min(sp.handbrakeTorque, 1050));
      w.brakeTorque = bt;
    }
    if (sp.lsd > 0) {
      const wl = wheels[2];
      const wr = wheels[3];
      const lim = Math.abs(wheelTorque) * 0.25 + 120;
      const dT = clamp(sp.lsd * (wr.omega - wl.omega), -lim, lim);
      wl.driveTorque += dT;
      wr.driveTorque -= dT;
    }

    // ------------------------------------------------ tyres
    const wetMu = 1 - this.wet * (1 - WET_MU);
    this.hbGrip = damp(this.hbGrip, 1 - 0.52 * hbCmd * (1 - arcW), hbCmd > 0 ? 14 : 5, dt);
    const hbGrip = this.hbGrip;
    let rearBeta = 0;
    let rearCount = 0;
    let driftAcc = 0;
    let surfAcc = 0;
    for (let i = 0; i < 4; i++) {
      const w = wheels[i];
      const I = w.inertia + reflected * w.driveShare;
      if (!w.contact) {
        // free-spinning wheel
        let om = w.omega + (w.driveTorque / I) * dt;
        const db = (w.brakeTorque / I) * dt;
        om = Math.abs(om) <= db ? 0 : om - Math.sign(om) * db;
        w.omega = om;
        w.angle += om * dt;
        w.slipRatio = 0;
        w.slipAngle = 0;
        w.slipNorm = 0;
        w.skid = 0;
        continue;
      }
      const nx = w.nx;
      const ny = w.ny;
      const nz = w.nz;
      // wheel heading projected on the ground plane
      const cs = Math.cos(w.steer);
      const sn = Math.sin(w.steer);
      let hx = fx * cs + lx * sn;
      let hy = fy * cs + ly * sn;
      let hz = fz * cs + lz * sn;
      const hd = hx * nx + hy * ny + hz * nz;
      hx -= nx * hd;
      hy -= ny * hd;
      hz -= nz * hd;
      const hl = 1 / Math.sqrt(hx * hx + hy * hy + hz * hz);
      hx *= hl;
      hy *= hl;
      hz *= hl;
      // side (left) = n x heading
      const sx = ny * hz - nz * hy;
      const sy = nz * hx - nx * hz;
      const sz = nx * hy - ny * hx;
      // contact velocity
      const rx = w.cpx - px;
      const ry = w.cpy - py;
      const rz = w.cpz - pz;
      const vcx = vx0 + (wy * rz - wz * ry);
      const vcy = vy0 + (wz * rx - wx * rz);
      const vcz = vz0 + (wx * ry - wy * rx);
      const vlong = vcx * hx + vcy * hy + vcz * hz;
      const vlat = vcx * sx + vcy * sy + vcz * sz;
      const vref = Math.max(Math.abs(vlong), V0);

      const surf = w.surface;
      surfAcc += surf;
      const muS = SURF_MU[surf] * (surf <= SURF_KERB ? wetMu : 1 - this.wet * 0.08) * this.gripScale;
      const ls = clamp(1 - sp.loadSens * (w.fz / w.staticLoad - 1), 0.7, 1.12);
      const muX = sp.muLong * muS * ls;
      const muY = sp.muLat * muS * ls * (w.front ? 1 : sp.rearGripBias * hbGrip);
      const Fz = w.fz;

      const slipVel = w.omega * w.radius - vlong;
      const sxn = slipVel / vref / sp.kappaPeak;
      const alpha = Math.atan2(vlat, vref);
      const syn = (alpha / sp.alphaPeak) * (1 - arcW);
      const s = Math.sqrt(sxn * sxn + syn * syn);
      const g = s < 1e-3 ? 2 : tireF(s, sp.slideFrac) / s;

      // implicit wheel-spin update with secant longitudinal stiffness
      const Kx = (muX * Fz * g) / (sp.kappaPeak * vref);
      const r = w.radius;
      const denom = 1 + (dt * Kx * r * r) / I;
      let om = (w.omega + (dt / I) * (w.driveTorque + Kx * vlong * r)) / denom;
      const db = (dt * w.brakeTorque) / (I * denom);
      om = Math.abs(om) <= db ? 0 : om - Math.sign(om) * db;
      let Fx = Kx * (om * r - vlong);
      const FxMax = muX * Fz;
      if (Fx > FxMax) Fx = FxMax;
      else if (Fx < -FxMax) Fx = -FxMax;
      const Fy = -muY * Fz * g * syn;
      // rolling resistance
      Fx -= SURF_ROLL[surf] * Fz * clamp(vlong * 0.8, -1, 1);

      w.omega = om;
      w.angle += om * dt;
      w.slipRatio = (om * r - vlong) / vref;
      w.slipAngle = alpha;
      w.slipNorm = s;
      const slideSpeed = Math.sqrt(slipVel * slipVel + vlat * vlat);
      w.skid = clamp01((s - 0.95) / 1.1) * smoothstep(1.2, 5.5, slideSpeed) * clamp01(Fz / w.staticLoad);
      if (!w.front) {
        rearBeta += Math.atan2(vlat, Math.abs(vlong) + 1.5);
        rearCount++;
        driftAcc += w.skid;
      }

      // total force at the contact point: normal + friction
      const Fwx = nx * Fz + hx * Fx + sx * Fy;
      const Fwy = ny * Fz + hy * Fx + sy * Fy;
      const Fwz = nz * Fz + hz * Fx + sz * Fy;
      FX += Fwx;
      FY += Fwy;
      FZ += Fwz;
      TX += ry * Fwz - rz * Fwy;
      TY += rz * Fwx - rx * Fwz;
      TZ += rx * Fwy - ry * Fwx;
    }
    this.rearSlipAngle = rearCount > 0 ? damp(this.rearSlipAngle, rearBeta / rearCount, 25, dt) : damp(this.rearSlipAngle, 0, 10, dt);
    this.driftScore = damp(this.driftScore, speed > 12 ? driftAcc * 0.5 * clamp01(Math.abs(this.rearSlipAngle) * 4) : 0, 4, dt);
    this.surfaceMix = onGround > 0 ? surfAcc / onGround : 0;

    // ------------------------------------------------ arcade path control + drift
    this.arcYawAcc = 0;
    const vLongB = vx0 * fx + vy0 * fy + vz0 * fz;
    const vLatB = vx0 * lx + vy0 * ly + vz0 * lz;
    const vPlane = Math.sqrt(vLongB * vLongB + vLatB * vLatB);
    const betaNow = vPlane > 2 ? Math.atan2(-vLatB, vLongB) : 0;
    this.beta = betaNow;
    if (arcW > 0 && onGround >= 2) {
      // surface under the car scales the available grip
      const surfMu = 1 - 0.42 * clamp01(this.surfaceMix - 1);
      const aMax = this.arcadeMu() * surfMu * (G + (sp.downforceK * vPlane * vPlane) / sp.mass);
      const wMax = Math.min(aMax / Math.max(vPlane, 3), (Math.max(vPlane, 3) * Math.tan(sp.steerLock * 0.8)) / sp.wheelbase);
      const st = this.steer;

      // drift state machine: the drift key plus a direction throws the car into a slide. It keeps sliding
      // after the key is released and ends once the nose has been pulled straight (counter-steer is quickest).
      const steerIn = inp.steer;
      if (!this.drifting) {
        if (driftKey && Math.abs(steerIn) > 0.2 && vPlane > 12 && arcW > 0.6) {
          this.drifting = true;
          this.driftDir = steerIn > 0 ? 1 : -1;
          this.driftTime = 0;
          this.driftAngle = 0.2;
          this.flipTimer = 0;
          this.events |= EV_DRIFT_START;
        }
      } else {
        this.driftTime += dt;
        const into = st * this.driftDir;
        if (driftKey && into < -0.55) this.flipTimer += dt;
        else this.flipTimer = 0;
        if (vPlane < 8) {
          this.drifting = false;
          this.events |= EV_DRIFT_END;
        } else if (this.flipTimer > 0.26) {
          // key still held, steering thrown the other way: link straight into the opposite drift
          this.driftDir = -this.driftDir;
          this.driftAngle = 0.12;
          this.flipTimer = 0;
        } else if (!driftKey && this.driftAngle < 0.085 && Math.abs(betaNow) < 0.2) {
          this.drifting = false;
          this.events |= EV_DRIFT_END;
          // the small-boost window: a real drift, or a quick tap chained right after a boost (double boost)
          if ((this.driftTime >= 0.3 || this.chain > 0) && this.chain < 2) {
            this.boostWindow = 0.5;
            this.windowKind = 1;
          }
        }
      }

      let wPath: number;
      let betaT = 0;
      let scrub: number;
      if (this.drifting) {
        const into = clamp(st * this.driftDir, -1, 1);
        let factor: number;
        if (driftKey) {
          // key held: the slide deepens and the car rotates hard into the corner
          const want = into >= 0 ? lerp(0.44, 0.86, into) : lerp(0.44, 0.22, -into);
          factor = into >= 0 ? lerp(0.66, 1.2, into) : lerp(0.66, 0.25, -into);
          this.driftAngle = moveToward(this.driftAngle, want, (want > this.driftAngle ? 2.3 : 3.0) * dt);
        } else {
          // key released: steering into the slide holds it, counter-steer pulls the nose straight
          const rate = into > 0.3 ? 0.35 : into < -0.2 ? lerp(1.6, 3.8, clamp01(-into)) : 1.0;
          this.driftAngle = moveToward(this.driftAngle, 0, rate * dt);
          factor = (into >= 0 ? lerp(0.5, 1.0, into) : lerp(0.5, 0.12, -into)) * clamp01(this.driftAngle / 0.3 + 0.25);
        }
        wPath = this.driftDir * factor * wMax;
        betaT = this.driftDir * this.driftAngle;
        scrub = 0.1;
      } else {
        this.driftAngle = moveToward(this.driftAngle, 0, 3.5 * dt);
        wPath = st * wMax;
        scrub = 0.03;
      }
      // centripetal force perpendicular to the direction of travel (applied low, so the body still rolls)
      const aLat = vPlane * wPath;
      const ivp = 1 / Math.max(vPlane, 0.5);
      const dvx = (fx * vLongB + lx * vLatB) * ivp; // unit planar velocity
      const dvy = (fy * vLongB + ly * vLatB) * ivp;
      const dvz = (fz * vLongB + lz * vLatB) * ivp;
      // left-perpendicular in the body plane: up x v
      const pxx = uy * dvz - uz * dvy;
      const pyy = uz * dvx - ux * dvz;
      const pzz = ux * dvy - uy * dvx;
      const Fc = sp.mass * aLat * arcW;
      const Fd = sp.mass * Math.abs(aLat) * scrub * arcW;
      const Fax = pxx * Fc - dvx * Fd;
      const Fay = pyy * Fc - dvy * Fd;
      const Faz = pzz * Fc - dvz * Fd;
      const arm = -sp.comHeight * 0.65;
      const rax = ux * arm;
      const ray = uy * arm;
      const raz = uz * arm;
      FX += Fax;
      FY += Fay;
      FZ += Faz;
      TX += ray * Faz - raz * Fay;
      TY += raz * Fax - rax * Faz;
      TZ += rax * Fay - ray * Fax;
      // heading: follow the path, hold the requested slip angle
      const wDes = wPath + clamp((this.drifting ? 5.5 : 7.5) * (betaT - betaNow), -3.6, 3.6);
      const wYaw = ux * wx + uy * wy + uz * wz;
      this.arcYawAcc = clamp(10 * (wDes - wYaw), -45, 45) * arcW;
      // tyre effects for audio / smoke / skid marks
      const squeal = clamp01((Math.abs(aLat) / Math.max(aMax, 1) - 0.72) * 2.2);
      for (let i = 0; i < 4; i++) {
        const w = wheels[i];
        if (!w.contact) continue;
        const sk = this.drifting && !w.front ? clamp01(Math.abs(betaNow) / 0.3) * smoothstep(9, 18, vPlane) : squeal * (w.front ? 0.35 : 0.25);
        if (sk > w.skid) w.skid = sk;
      }
      this.driftScore = damp(this.driftScore, this.drifting ? 1 : 0, 5, dt);
    } else if (this.drifting) {
      this.drifting = false;
      this.events |= EV_DRIFT_END;
    }

    // ------------------------------------------------ aero + gravity
    const dragK = sp.dragK * (1 - this.draft * 0.32) * (this.nitroOn && this.arcade ? 0.72 : 1);
    FX -= dragK * speed * vx0;
    FY -= dragK * speed * vy0;
    FZ -= dragK * speed * vz0;
    const df = sp.downforceK * fwdSpeed * fwdSpeed * (1 - this.draft * 0.12) * (onGround > 0 ? 1 : 0.35);
    if (df > 0) {
      const dfF = df * sp.aeroBalance;
      const dfR = df - dfF;
      // applied at the axle centres, along body -up
      const zf = sp.axleFrontZ - sp.comZ;
      const zr = sp.axleRearZ - sp.comZ;
      const ay = 0.15 - sp.comHeight;
      for (let k = 0; k < 2; k++) {
        const fmag = k === 0 ? dfF : dfR;
        const zz = k === 0 ? zf : zr;
        const rx = ux * ay + fx * zz;
        const ry = uy * ay + fy * zz;
        const rz = uz * ay + fz * zz;
        const Fx_ = -ux * fmag;
        const Fy_ = -uy * fmag;
        const Fz_ = -uz * fmag;
        FX += Fx_;
        FY += Fy_;
        FZ += Fz_;
        TX += ry * Fz_ - rz * Fy_;
        TY += rz * Fx_ - rx * Fz_;
        TZ += rx * Fy_ - ry * Fx_;
      }
    }
    FY -= sp.mass * G;

    // ------------------------------------------------ integrate velocities
    const im = this.invMass;
    this.vx = vx0 + FX * im * dt;
    this.vy = vy0 + FY * im * dt;
    this.vz = vz0 + FZ * im * dt;
    // angular: body frame
    const tbx = lx * TX + ly * TY + lz * TZ;
    const tby = ux * TX + uy * TY + uz * TZ;
    const tbz = fx * TX + fy * TY + fz * TZ;
    let wbx = lx * wx + ly * wy + lz * wz;
    let wby = ux * wx + uy * wy + uz * wz;
    let wbz = fx * wx + fy * wy + fz * wz;
    const gxb = wby * (this.Iz * wbz) - wbz * (this.Iy * wby);
    const gyb = wbz * (this.Ix * wbx) - wbx * (this.Iz * wbz);
    const gzb = wbx * (this.Iy * wby) - wby * (this.Ix * wbx);
    let assistAcc = 0;
    if (this.steerAssist > 0 && onGround >= 2 && speed > 8 && arcW < 0.5) {
      const beta = this.rearSlipAngle;
      const ab = Math.abs(beta);
      const soft = 0.3;
      if (ab > soft) {
        const ex = Math.min(ab - soft, 0.7);
        assistAcc = ((beta > 0 ? 1 : -1) * 24 * ex - 5 * wby * smoothstep(0, 0.22, ex)) * this.steerAssist;
      }
    }
    wbx += (tbx - gxb) * this.invIx * dt;
    wby += ((tby - gyb) * this.invIy + assistAcc + this.arcYawAcc) * dt;
    wbz += (tbz - gzb) * this.invIz * dt;
    // light angular damping; stronger in the air so cars do not tumble forever
    const ad = Math.exp(-(onGround > 0 ? 0.12 : 0.9) * dt);
    wbx *= ad;
    wby *= ad;
    wbz *= ad;
    this.wx = lx * wbx + ux * wby + fx * wbz;
    this.wy = ly * wbx + uy * wby + fy * wbz;
    this.wz = lz * wbx + uz * wby + fz * wbz;

    // telemetry
    const ax = (this.vx - vx0) / dt;
    const ay2 = (this.vy - vy0) / dt + G;
    const az = (this.vz - vz0) / dt;
    this.longG = damp(this.longG, (ax * fx + ay2 * fy + az * fz) / G, 12, dt);
    this.latG = damp(this.latG, (ax * lx + ay2 * ly + az * lz) / G, 12, dt);
    this.yawRate = wby;
    this.scrape = damp(this.scrape, 0, 9, dt);
  }

  // ------------------------------------------------------------------ contacts
  /**
   * Inelastic contact impulse at world point (cx,cy,cz), normal (nx,ny,nz) pointing from the obstacle
   * into free space. Returns the normal impulse magnitude (0 if separating).
   */
  contactImpulse(cx: number, cy: number, cz: number, nx: number, ny: number, nz: number, restitution: number, friction: number): number {
    const rx = cx - this.px;
    const ry = cy - this.py;
    const rz = cz - this.pz;
    let vx = this.vx + (this.wy * rz - this.wz * ry);
    let vy = this.vy + (this.wz * rx - this.wx * rz);
    let vz = this.vz + (this.wx * ry - this.wy * rx);
    const vn = vx * nx + vy * ny + vz * nz;
    if (vn >= 0) return 0;
    const kn = this.invMass + this.angularFactor(rx, ry, rz, nx, ny, nz);
    const jn = (-(1 + restitution) * vn) / kn;
    this.applyImpulse(nx * jn, ny * jn, nz * jn, rx, ry, rz);
    if (friction > 0) {
      vx = this.vx + (this.wy * rz - this.wz * ry);
      vy = this.vy + (this.wz * rx - this.wx * rz);
      vz = this.vz + (this.wx * ry - this.wy * rx);
      const vn2 = vx * nx + vy * ny + vz * nz;
      let tx = vx - nx * vn2;
      let ty = vy - ny * vn2;
      let tz = vz - nz * vn2;
      const vt = Math.sqrt(tx * tx + ty * ty + tz * tz);
      if (vt > 1e-3) {
        tx /= vt;
        ty /= vt;
        tz /= vt;
        const kt = this.invMass + this.angularFactor(rx, ry, rz, tx, ty, tz);
        const jt = Math.min(vt / kt, friction * jn);
        this.applyImpulse(-tx * jt, -ty * jt, -tz * jt, rx, ry, rz);
      }
    }
    return jn;
  }

  /** n · ((I^-1 (r × n)) × r) */
  private angularFactor(rx: number, ry: number, rz: number, nx: number, ny: number, nz: number): number {
    // c = r × n (world) -> body, scale by inverse inertia, back to world
    const cx = ry * nz - rz * ny;
    const cy = rz * nx - rx * nz;
    const cz = rx * ny - ry * nx;
    const bx = (this.m00 * cx + this.m10 * cy + this.m20 * cz) * this.invIx;
    const by = (this.m01 * cx + this.m11 * cy + this.m21 * cz) * this.invIy;
    const bz = (this.m02 * cx + this.m12 * cy + this.m22 * cz) * this.invIz;
    const ax = this.m00 * bx + this.m01 * by + this.m02 * bz;
    const ay = this.m10 * bx + this.m11 * by + this.m12 * bz;
    const az = this.m20 * bx + this.m21 * by + this.m22 * bz;
    // (a × r) · n
    return (ay * rz - az * ry) * nx + (az * rx - ax * rz) * ny + (ax * ry - ay * rx) * nz;
  }

  applyImpulse(jx: number, jy: number, jz: number, rx: number, ry: number, rz: number): void {
    this.vx += jx * this.invMass;
    this.vy += jy * this.invMass;
    this.vz += jz * this.invMass;
    const cx = ry * jz - rz * jy;
    const cy = rz * jx - rx * jz;
    const cz = rx * jy - ry * jx;
    const bx = (this.m00 * cx + this.m10 * cy + this.m20 * cz) * this.invIx;
    const by = (this.m01 * cx + this.m11 * cy + this.m21 * cz) * this.invIy;
    const bz = (this.m02 * cx + this.m12 * cy + this.m22 * cz) * this.invIz;
    this.wx += this.m00 * bx + this.m01 * by + this.m02 * bz;
    this.wy += this.m10 * bx + this.m11 * by + this.m12 * bz;
    this.wz += this.m20 * bx + this.m21 * by + this.m22 * bz;
  }

  /** Small boost. kind: 1 small, 2 double, 3 super start, 4 landing. Stacks with nitro. */
  private fireBoost(kind: number): void {
    this.miniBoost = Math.max(this.miniBoost, kind === 3 ? 1.0 : kind === 2 ? 0.7 : 0.55);
    this.boostKind = kind;
    this.events |= EV_MINI_BOOST;
    // an instant shove along the nose
    const kick = kind === 3 ? 2.2 : 1.4;
    this.vx += this.m02 * kick;
    this.vy += this.m12 * kick;
    this.vz += this.m22 * kick;
    if (kind === 1 || kind === 2) {
      this.chain = Math.min(2, this.chain + 1);
      this.chainTimer = 0.9;
    }
  }

  noteImpact(j: number, x: number, y: number, z: number): void {
    if (j > this.impact) {
      this.impact = j;
      this.impactX = x;
      this.impactY = y;
      this.impactZ = z;
    }
  }

  /** Walls (track-space lateral limits) and body-vs-ground contacts. */
  collideWorld(track: Track): void {
    const sp = this.spec;
    const hw = sp.bodyHalfWidth;
    const zf = sp.bodyFrontZ - sp.comZ;
    const zr = sp.bodyRearZ - sp.comZ;
    const n = track.n;
    // ---- walls: four corners at centre-of-mass height
    for (let c = 0; c < 4; c++) {
      const lx = c % 2 === 0 ? hw : -hw;
      const lz = c < 2 ? zf : zr;
      const x = this.px + this.m00 * lx + this.m02 * lz;
      const y = this.py + this.m10 * lx + this.m12 * lz;
      const z = this.pz + this.m20 * lx + this.m22 * lz;
      track.locate(x, z, this.trkIdx);
      const i = track.oi;
      const j = i + 1 === n ? 0 : i + 1;
      const t = track.ot;
      const d = track.od;
      let pen = 0;
      let sgn = 0;
      if (d > 0) {
        const wl = track.wallL[i] + (track.wallL[j] - track.wallL[i]) * t;
        if (d > wl) {
          pen = d - wl;
          sgn = -1;
        }
      } else {
        const wr = track.wallR[i] + (track.wallR[j] - track.wallR[i]) * t;
        if (-d > wr) {
          pen = -d - wr;
          sgn = 1;
        }
      }
      if (pen > 0) {
        let nx = track.nx[i] + (track.nx[j] - track.nx[i]) * t;
        let nz = track.nz[i] + (track.nz[j] - track.nz[i]) * t;
        const il = sgn / Math.hypot(nx, nz);
        nx *= il;
        nz *= il;
        const vBefore = Math.hypot(this.vx, this.vz);
        const jn = this.contactImpulse(x, y, z, nx, 0, nz, 0.14, 0.2);
        if (jn > 0) {
          this.noteImpact(jn, x, y, z);
          this.scrape = Math.max(this.scrape, clamp01(vBefore / 45));
        } else if (vBefore > 4) {
          this.scrape = Math.max(this.scrape, clamp01(vBefore / 60) * 0.7);
        }
        const push = Math.min(pen, 0.6);
        this.px += nx * push;
        this.pz += nz * push;
      }
    }
    // ---- body vs ground (bottom corners + roof), only matters on crests/landings/rollovers
    const yb = Math.min(0.03, sp.rideHeight * 0.3) - sp.comHeight;
    const yt = sp.bodyHeight - sp.comHeight;
    let corr = 0;
    for (let c = 0; c < 8; c++) {
      const top = c >= 4;
      const lx = (c % 2 === 0 ? 1 : -1) * hw * (top ? 0.55 : 0.85);
      const lz = (c % 4 < 2 ? zf : zr) * (top ? 0.3 : 0.94);
      const ly = top ? yt : yb;
      const x = this.px + this.m00 * lx + this.m01 * ly + this.m02 * lz;
      const y = this.py + this.m10 * lx + this.m11 * ly + this.m12 * lz;
      const z = this.pz + this.m20 * lx + this.m21 * ly + this.m22 * lz;
      track.ground(x, z, this.trkIdx);
      const pen = track.oy - y;
      if (pen > 0) {
        const jn = this.contactImpulse(x, y, z, track.onx, track.ony, track.onz, 0.05, top ? 0.7 : 0.45);
        if (jn > 800) this.noteImpact(jn * 0.5, x, y, z);
        if (pen > corr) corr = pen;
      }
    }
    if (corr > 0) this.py += Math.min(corr, 0.5) * 0.8;
  }

  /** Integrate pose and refresh track coordinates. */
  integrate(dt: number, track: Track): void {
    this.px += this.vx * dt;
    this.py += this.vy * dt;
    this.pz += this.vz * dt;
    const wx = this.wx;
    const wy = this.wy;
    const wz = this.wz;
    const qx = this.qx;
    const qy = this.qy;
    const qz = this.qz;
    const qw = this.qw;
    const h = 0.5 * dt;
    let nx = qx + h * (wx * qw + wy * qz - wz * qy);
    let ny = qy + h * (wy * qw + wz * qx - wx * qz);
    let nz = qz + h * (wz * qw + wx * qy - wy * qx);
    let nw = qw + h * (-wx * qx - wy * qy - wz * qz);
    const il = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
    nx *= il;
    ny *= il;
    nz *= il;
    nw *= il;
    this.qx = nx;
    this.qy = ny;
    this.qz = nz;
    this.qw = nw;
    this.updateMatrix();
    this.upDot = this.m11;
    track.locate(this.px, this.pz, this.trkIdx);
    this.trkIdx = track.oi;
    this.trkT = track.ot;
    this.d = track.od;
    this.s = (track.oi + track.ot) * track.ds;
  }

  /** Model-space origin (ground point under the wheelbase midpoint) in world space. */
  originX(): number {
    return this.px - this.m01 * this.comY - this.m02 * this.comZ;
  }
  originY(): number {
    return this.py - this.m11 * this.comY - this.m12 * this.comZ;
  }
  originZ(): number {
    return this.pz - this.m21 * this.comY - this.m22 * this.comZ;
  }
}
