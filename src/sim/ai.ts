// AI driver: pure-pursuit steering along the racing line (plus a lateral lane offset for
// overtaking/avoidance) and a throttle/brake controller tracking a per-car reference speed profile.
import { Car, CarInput, makeInput } from './car';
import { clamp, clamp01, moveToward, ringDiff, Rng } from './math';
import type { SpeedProfileParams, Track } from './track';

export function profileParams(car: Car, skill: number): SpeedProfileParams {
  const sp = car.spec;
  const a = sp.axleFrontZ - sp.comZ;
  const b = sp.comZ - sp.axleRearZ;
  const rearShare = sp.drive === 'RWD' ? a / (a + b) + 0.1 : 1;
  return {
    mass: sp.mass,
    muLat: (car.arcade ? car.arcadeMu() * 0.9 : sp.muLat * 0.86 * car.gripScale) * skill,
    muBrake: sp.muLong * 0.74 * skill * car.gripScale,
    downforceK: sp.downforceK * 0.9,
    dragK: sp.dragK,
    powerW: car.peakPower() * 0.93 * car.powerScale,
    tractionG: sp.muLong * 9.81 * Math.min(1, rearShare) * 0.9,
    vTop: 100,
  };
}

const tmp = { x: 0, y: 0, z: 0, psi: 0 };

export class AIDriver {
  readonly car: Car;
  readonly track: Track;
  skill: number;
  vref: Float32Array;
  readonly input: CarInput = makeInput();
  laneOffset = 0;
  private laneTarget = 0;
  /** speed multiplier set by the race (rubber banding) */
  pace = 1;
  reaction = 0;
  wantsReset = false;
  private stuck = 0;
  private stuckTotal = 0;
  private reversing = 0;
  private nitroHold = 0;
  private rng: Rng;
  private wobblePhase: number;

  constructor(car: Car, track: Track, skill: number, seed: number) {
    this.car = car;
    this.track = track;
    this.skill = skill;
    this.rng = new Rng(seed);
    this.wobblePhase = this.rng.range(0, 6.28);
    this.reaction = this.rng.range(0.08, 0.42);
    this.vref = track.buildSpeedProfile(profileParams(car, skill));
  }

  setSkill(skill: number): void {
    this.skill = skill;
    this.vref = this.track.buildSpeedProfile(profileParams(this.car, skill));
  }

  update(dt: number, cars: readonly Car[], racing: boolean): CarInput {
    const car = this.car;
    const tr = this.track;
    const inp = this.input;
    const v = car.speed;
    const L = tr.length;
    inp.handbrake = 0;
    inp.nitro = false;
    inp.shiftUp = false;
    inp.shiftDown = false;

    if (!racing) {
      // blip the throttle on the grid
      this.wobblePhase += dt * (2.2 + (car.index % 3) * 0.6);
      inp.throttle = Math.max(0, Math.sin(this.wobblePhase)) * 0.55;
      inp.brake = 1;
      inp.steer = 0;
      return inp;
    }
    if (this.reaction > 0) {
      this.reaction -= dt;
      inp.throttle = 0.4;
      inp.brake = 0;
      inp.steer = 0;
      return inp;
    }

    // ---- traffic: pick a lane offset, find a blocking speed
    const hw = tr.hw[car.trkIdx];
    let desired = 0;
    let block = Infinity;
    const lookTime = 0.9;
    for (let k = 0; k < cars.length; k++) {
      const o = cars[k];
      if (o === car || o.ghost > 0) continue;
      const ds = ringDiff(o.s, car.s, L);
      if (ds < -6 || ds > 9 + v * lookTime) continue;
      const dd = o.d - car.d; // + = other is to my left
      const add = Math.abs(dd);
      if (ds > 4.6) {
        const closing = v - o.speed;
        if (add < 2.7 && closing > -2) {
          const roomL = hw - 1.4 - (o.d + 2.5);
          const roomR = o.d - 2.5 + hw - 1.4;
          let side: number;
          if (roomL < 0.2 && roomR < 0.2) side = 0;
          else if (roomL < 0.2) side = -1;
          else if (roomR < 0.2) side = 1;
          else side = dd > 0.3 ? -1 : dd < -0.3 ? 1 : roomL > roomR ? 1 : -1;
          if (side === 0) block = Math.min(block, o.speed + (ds - 6) * 0.9);
          else {
            const targetD = o.d + side * 2.75;
            const lineHere = tr.line[car.trkIdx];
            const want = targetD - lineHere;
            if (Math.abs(want) > Math.abs(desired)) desired = want;
            // not yet clear of the car ahead: do not ram it
            if (add < 2.0 && ds < 9 + closing * 0.5) block = Math.min(block, o.speed + (ds - 5.5) * 1.2);
          }
        }
      } else if (add < 3.2) {
        // alongside: open a gap; if it is getting tight, the car that is behind lifts
        const push = (3.2 - add) * 1.5 * (dd > 0 ? -1 : 1);
        desired += push;
        if (add < 2.35 && ds > 0.3) block = Math.min(block, o.speed - 1.5);
      }
    }
    this.laneTarget = desired;
    this.laneOffset = moveToward(this.laneOffset, this.laneTarget, (Math.abs(this.laneTarget) > Math.abs(this.laneOffset) ? 4.2 : 1.4) * dt);

    // ---- steering: pure pursuit toward a point on (racing line + lane offset)
    const Ld = clamp(5 + 0.36 * v, 9, 40);
    const sT = car.s + Ld;
    const iT = tr.indexAt(sT);
    const lim = tr.hw[iT] - 1.25;
    this.wobblePhase += dt * 0.7;
    const wobble = Math.sin(this.wobblePhase) * 0.35 * (1.05 - this.skill) * 4;
    const off = clamp(tr.line[iT] + this.laneOffset + wobble, -lim, lim);
    tr.posAt(sT, off, tmp);
    const fl = Math.hypot(car.m02, car.m22) || 1;
    const fx = car.m02 / fl;
    const fz = car.m22 / fl;
    // measure from the front axle
    const ax = car.px + fx * (car.spec.axleFrontZ - car.spec.comZ);
    const az = car.pz + fz * (car.spec.axleFrontZ - car.spec.comZ);
    const dx = tmp.x - ax;
    const dz = tmp.z - az;
    const tl = dx * fz - dz * fx; // lateral, + left
    const d2 = dx * dx + dz * dz;
    const kappa = (2 * tl) / Math.max(d2, 1);
    inp.steer = car.steerForCurvature(kappa);

    // ---- speed
    const lead = v * 0.1 + 2;
    let vT = Math.min(tr.sample(this.vref, car.s + lead), tr.sample(this.vref, car.s + lead + v * 0.25) + 3) * this.pace;
    if (block < vT) vT = Math.max(block, 4);
    const err = vT - v;
    let thr = 0;
    let brk = 0;
    if (err > 0) thr = clamp01(err * 0.55 + 0.12);
    else if (err < -0.4) brk = clamp01(-err * 0.28);
    // ease off when sliding or when steering is saturated
    const slide = Math.abs(car.rearSlipAngle);
    if (slide > 0.11) thr *= clamp01(1 - (slide - 0.11) * 6);
    if (Math.abs(inp.steer) > 0.92 && v > 14) thr *= 0.55;
    inp.throttle = thr;
    inp.brake = brk;

    // ---- nitro on straights
    const farK = Math.abs(tr.kappa[tr.indexAt(car.s + v * 2.2 + 30)]);
    const straight = Math.abs(tr.kappa[iT]) < 0.002 && Math.abs(tr.kappa[car.trkIdx]) < 0.002 && Math.abs(inp.steer) < 0.25;
    if (this.nitroHold > 0) {
      this.nitroHold -= dt;
      if (!straight || farK > 0.004) this.nitroHold = 0;
      inp.nitro = this.nitroHold > 0 && thr > 0.9;
    } else if (car.nitroBottles > 0 && !car.nitroOn && thr > 0.95 && err > 6 && straight && farK < 0.003) {
      if (this.rng.next() < dt * 0.7) this.nitroHold = this.rng.range(1.2, 2.6);
    }

    // ---- recovery: back up once, then ask the race for a reset
    if (v < 1.6) {
      this.stuck += dt;
      this.stuckTotal += dt;
    } else {
      this.stuck = Math.max(0, this.stuck - dt * 2);
      this.stuckTotal = Math.max(0, this.stuckTotal - dt * 0.5);
    }
    if (this.reversing > 0) {
      this.reversing -= dt;
      inp.throttle = 0;
      inp.brake = 1;
      inp.steer = -inp.steer;
      inp.nitro = false;
      if (this.reversing <= 0) this.stuck = 0;
    } else if (this.stuckTotal > 4.2) {
      this.wantsReset = true;
      this.stuck = 0;
      this.stuckTotal = 0;
    } else if (this.stuck > 1.6) {
      this.reversing = 1.1;
    }
    return inp;
  }
}
