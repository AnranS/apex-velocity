// Race simulation: owns the track, cars and AI drivers; runs the fixed-step physics, race rules
// (grid, countdown, laps, standings, wrong-way, resets, drafting) and emits events for UI/audio/FX.
import { AIDriver } from './ai';
import { Car, CarInput, makeInput, EV_BACKFIRE, EV_BLOWOFF, EV_DRIFT_END, EV_DRIFT_START, EV_LAND, EV_LIMITER, EV_MINI_BOOST, EV_NITRO_END, EV_NITRO_READY, EV_NITRO_START, EV_SHIFT_DOWN, EV_SHIFT_UP } from './car';
import { CAR_IDS, CAR_SPECS, CarSpec } from './car_specs';
import { collideCars } from './collide';
import { clamp, damp, ringDiff, Rng } from './math';
import { Track } from './track';

export type RaceMode = 'race' | 'timetrial';
export type RaceState = 'grid' | 'countdown' | 'racing' | 'finished';

export interface RaceConfig {
  mode: RaceMode;
  laps: number;
  playerCar: string;
  aiCount: number;
  difficulty: number; // 0 easy, 1 normal, 2 hard
  wet: number; // 0..1
  seed: number;
  /** arcade (drift-boost) handling for every car; false = tyre simulation */
  arcade?: boolean;
  /** player assists */
  abs: boolean;
  tcs: number;
  steerAssist: number;
  autoGear: boolean;
  /** overrides for car specs (geometry from model metadata) */
  specs?: Record<string, CarSpec>;
  /** explicit AI car ids (otherwise chosen from the seed) */
  aiCars?: string[];
}

export type RaceEventType =
  | 'count'
  | 'go'
  | 'lap'
  | 'final_lap'
  | 'finish'
  | 'position'
  | 'impact'
  | 'shift_up'
  | 'shift_down'
  | 'limiter'
  | 'backfire'
  | 'blowoff'
  | 'nitro_start'
  | 'nitro_end'
  | 'land'
  | 'reset'
  | 'wrong_way'
  | 'sector'
  | 'drift_start'
  | 'drift_end'
  | 'mini_boost'
  | 'nitro_ready';

export interface RaceEvent {
  type: RaceEventType;
  car: number;
  a: number;
  b: number;
  x: number;
  y: number;
  z: number;
}

export interface CarRaceInfo {
  name: string;
  carId: string;
  isPlayer: boolean;
  paint: number; // index into spec.paints
  progress: number;
  maxProgress: number;
  lastS: number;
  lapsDone: number;
  lapStart: number;
  lastLap: number;
  bestLap: number;
  lapTimes: number[];
  finished: boolean;
  finishTime: number;
  position: number;
  wrongWay: number;
  flipped: number;
  sector: number;
  sectorTimes: number[]; // current lap split times
  bestSectors: number[]; // splits of the best lap
  lastDelta: number; // last split delta vs best (s)
  gap: number; // metres behind the leader
}

const AI_NAMES = [
  'K. Aldrin',
  'M. Sorvino',
  'T. Hayashi',
  'L. Brandt',
  'R. Okafor',
  'S. Vance',
  'E. Moreau',
  'D. Castell',
  'J. Lindqvist',
  'A. Petrov',
  'N. Zhao',
  'C. Ferreira',
];

const COUNT_FROM = 3.0;

export class RaceSim {
  readonly track: Track;
  readonly cfg: RaceConfig;
  readonly cars: Car[] = [];
  readonly ai: (AIDriver | null)[] = [];
  readonly info: CarRaceInfo[] = [];
  readonly order: number[] = [];
  readonly events: RaceEvent[] = [];
  playerIdx = 0;
  state: RaceState = 'grid';
  time = 0; // race clock (s), starts at GO
  countdown = COUNT_FROM + 1.2;
  /** drive the player's car with an AI (attract mode / tests) */
  autopilot = false;
  private playerBot: AIDriver | null = null;
  private tick = 0;
  private readonly hit = { x: 0, y: 0, z: 0 };
  private readonly idle: CarInput = makeInput();
  private finishClock = 0;
  private lastImpact: number[] = [];
  private rng: Rng;

  constructor(track: Track, cfg: RaceConfig) {
    this.track = track;
    this.cfg = cfg;
    this.rng = new Rng(cfg.seed);
    const specs = cfg.specs ?? CAR_SPECS;
    const n = cfg.mode === 'timetrial' ? 1 : cfg.aiCount + 1;
    // skills: front of the grid is fastest
    const base = [0.865, 0.925, 0.975][cfg.difficulty] ?? 0.925;
    const spread = [0.07, 0.065, 0.06][cfg.difficulty] ?? 0.065;
    const names = AI_NAMES.slice();
    for (let i = 0; i < n; i++) {
      const isPlayer = i === n - 1; // player starts at the back of the grid
      let carId = cfg.playerCar;
      if (!isPlayer) carId = cfg.aiCars?.[i] ?? CAR_IDS[(i + (cfg.seed % 3)) % CAR_IDS.length];
      const spec = specs[carId] ?? CAR_SPECS[carId];
      const car = new Car(spec);
      car.index = i;
      car.wet = cfg.wet;
      car.arcade = cfg.arcade !== false;
      const slot = i; // 0 = pole
      const s = track.length - 9 - slot * 8.5;
      const d = (slot % 2 === 0 ? 1 : -1) * 3.3;
      car.placeOnTrack(track, s, d);
      car.hold = true;
      if (isPlayer) {
        this.playerIdx = i;
        car.abs = cfg.abs;
        car.tcs = cfg.tcs;
        car.steerAssist = cfg.steerAssist;
        car.autoGear = cfg.autoGear;
        this.ai.push(null);
      } else {
        const skill = base + spread * (1 - i / Math.max(1, n - 2)) + this.rng.range(-0.008, 0.008);
        car.tcs = car.arcade ? 1 : 2;
        // AI drivers do not drift: their gauge fills slowly on its own instead
        car.passiveNitro = 0.045 + cfg.difficulty * 0.015;
        car.abs = true;
        car.steerAssist = 1;
        car.gripScale = 1.0 + cfg.difficulty * 0.015;
        this.ai.push(new AIDriver(car, track, skill, cfg.seed * 31 + i * 977));
      }
      this.cars.push(car);
      const ni = isPlayer ? -1 : Math.floor(this.rng.next() * names.length);
      this.info.push({
        name: isPlayer ? 'YOU' : names.splice(ni, 1)[0],
        carId,
        isPlayer,
        paint: isPlayer ? 0 : Math.floor(this.rng.next() * spec.paints.length),
        progress: s - track.length,
        maxProgress: s - track.length,
        lastS: car.s,
        lapsDone: 0,
        lapStart: 0,
        lastLap: 0,
        bestLap: 0,
        lapTimes: [],
        finished: false,
        finishTime: 0,
        position: i + 1,
        wrongWay: 0,
        flipped: 0,
        sector: 0,
        sectorTimes: [],
        bestSectors: [],
        lastDelta: 0,
        gap: 0,
      });
      this.order.push(i);
      this.lastImpact.push(-1);
    }
  }

  get player(): Car {
    return this.cars[this.playerIdx];
  }
  get playerInfo(): CarRaceInfo {
    return this.info[this.playerIdx];
  }

  private emit(type: RaceEventType, car: number, a = 0, b = 0, x = 0, y = 0, z = 0): void {
    if (this.events.length < 256) this.events.push({ type, car, a, b, x, y, z });
  }

  /** Begin the start sequence (lights). */
  startCountdown(): void {
    if (this.state === 'grid') {
      this.state = 'countdown';
      this.countdown = COUNT_FROM + 0.9;
    }
  }

  resetCar(i: number): void {
    const car = this.cars[i];
    const tr = this.track;
    // back up a little so the car rejoins before the trouble spot, on the racing line side of centre
    const s = car.s - 6;
    const idx = tr.indexAt(s);
    const d = clamp(tr.line[idx] * 0.5, -tr.hw[idx] + 2.2, tr.hw[idx] - 2.2);
    const hold = car.hold;
    car.placeOnTrack(tr, s, d);
    car.hold = hold;
    car.ghost = 2.2;
    this.info[i].lastS = car.s;
    this.info[i].wrongWay = 0;
    this.info[i].flipped = 0;
    this.emit('reset', i);
  }

  step(dt: number, playerInput: CarInput): void {
    const cars = this.cars;
    const n = cars.length;
    const tr = this.track;
    this.tick++;
    for (let i = 0; i < n; i++) cars[i].savePrev();

    // ---- start sequence
    if (this.state === 'countdown') {
      const before = Math.ceil(this.countdown);
      this.countdown -= dt;
      const after = Math.ceil(this.countdown);
      if (after !== before && after >= 1 && after <= COUNT_FROM) this.emit('count', -1, after);
      if (this.countdown <= 0) {
        this.state = 'racing';
        this.time = 0;
        for (let i = 0; i < n; i++) cars[i].hold = false;
        this.emit('go', -1);
      }
    }
    const racing = this.state === 'racing' || this.state === 'finished';
    if (this.state === 'racing' || this.state === 'finished') this.time += dt;

    // ---- drive
    for (let i = 0; i < n; i++) {
      const car = cars[i];
      let inp: CarInput;
      const bot = this.ai[i];
      if (bot) inp = bot.update(dt, cars, racing && !this.info[i].finished ? true : racing);
      else if (this.autopilot || this.info[i].finished) {
        if (!this.playerBot) {
          this.playerBot = new AIDriver(car, tr, 0.93, 4242);
          this.playerBot.reaction = 0.15;
        }
        inp = this.playerBot.update(dt, cars, racing);
        if (this.info[i].finished) this.playerBot.pace = 0.6;
      } else inp = racing ? playerInput : this.gridInput(playerInput);
      car.step(dt, inp, tr);
    }

    // ---- collisions
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const jn = collideCars(cars[i], cars[j], this.hit);
        if (jn > 0) {
          cars[i].noteImpact(jn, this.hit.x, this.hit.y, this.hit.z);
          cars[j].noteImpact(jn, this.hit.x, this.hit.y, this.hit.z);
        }
      }
    }
    for (let i = 0; i < n; i++) {
      const car = cars[i];
      car.collideWorld(tr);
      car.integrate(dt, tr);
      this.forwardCarEvents(i);
    }

    // ---- rules (every 4th tick is plenty)
    if ((this.tick & 3) === 0) this.updateRules(dt * 4);
  }

  private gridInput(p: CarInput): CarInput {
    const g = this.idle;
    g.throttle = p.throttle;
    g.brake = 1;
    g.steer = p.steer;
    g.handbrake = 1;
    g.nitro = false;
    g.shiftUp = false;
    g.shiftDown = false;
    return g;
  }

  private forwardCarEvents(i: number): void {
    const car = this.cars[i];
    const ev = car.events;
    if (ev !== 0) {
      if (ev & EV_SHIFT_UP) this.emit('shift_up', i, car.gear);
      if (ev & EV_SHIFT_DOWN) this.emit('shift_down', i, car.gear);
      if (ev & EV_LIMITER) this.emit('limiter', i);
      if (ev & EV_BACKFIRE) this.emit('backfire', i);
      if (ev & EV_BLOWOFF) this.emit('blowoff', i);
      if (ev & EV_NITRO_START) this.emit('nitro_start', i);
      if (ev & EV_NITRO_END) this.emit('nitro_end', i);
      if (ev & EV_LAND) this.emit('land', i);
      if (ev & EV_DRIFT_START) this.emit('drift_start', i);
      if (ev & EV_DRIFT_END) this.emit('drift_end', i, car.driftTime);
      // a = kind: 1 small boost, 2 double boost, 3 super start, 4 landing boost
      if (ev & EV_MINI_BOOST) this.emit('mini_boost', i, car.boostKind);
      if (ev & EV_NITRO_READY) this.emit('nitro_ready', i, car.nitroBottles);
      car.events = 0;
    }
    if (car.impact > 0) {
      // debounce: sustained rubbing should not flood listeners
      if (car.impact > 500 && this.time - this.lastImpact[i] > 0.12) {
        this.lastImpact[i] = this.time;
        this.emit('impact', i, car.impact, 0, car.impactX, car.impactY, car.impactZ);
      }
      car.impact = 0;
    }
  }

  private updateRules(dt: number): void {
    const cars = this.cars;
    const n = cars.length;
    const tr = this.track;
    const L = tr.length;
    const racing = this.state === 'racing' || this.state === 'finished';
    const player = this.info[this.playerIdx];

    for (let i = 0; i < n; i++) {
      const car = cars[i];
      const inf = this.info[i];
      // progress & laps
      const ds = ringDiff(car.s, inf.lastS, L);
      inf.lastS = car.s;
      inf.progress += ds;
      if (inf.progress > inf.maxProgress) inf.maxProgress = inf.progress;
      if (racing && !inf.finished) {
        const laps = Math.floor(inf.maxProgress / L);
        // sectors (thirds of the lap)
        const frac = (inf.maxProgress - Math.max(0, laps) * L) / L;
        const sec = frac < 0 ? 0 : Math.min(2, Math.floor(frac * 3));
        if (sec !== inf.sector && inf.maxProgress > 0) {
          if (sec > inf.sector) {
            const split = this.time - inf.lapStart;
            inf.sectorTimes[inf.sector] = split;
            if (inf.bestSectors.length > inf.sector) {
              inf.lastDelta = split - inf.bestSectors[inf.sector];
              if (inf.isPlayer) this.emit('sector', i, inf.sector, inf.lastDelta);
            }
          }
          inf.sector = sec;
        }
        if (laps > inf.lapsDone) {
          const lapTime = this.time - inf.lapStart;
          inf.lapsDone = laps;
          inf.lapStart = this.time;
          inf.lastLap = lapTime;
          inf.lapTimes.push(lapTime);
          inf.sectorTimes[2] = lapTime;
          let best = false;
          if (inf.bestLap === 0 || lapTime < inf.bestLap) {
            inf.bestLap = lapTime;
            inf.bestSectors = inf.sectorTimes.slice(0, 3);
            best = true;
          }
          inf.sectorTimes = [];
          inf.sector = 0;
          this.emit('lap', i, lapTime, best ? 1 : 0);
          if (this.cfg.mode === 'race') {
            if (laps >= this.cfg.laps) {
              inf.finished = true;
              inf.finishTime = this.time;
              this.emit('finish', i, this.time);
              if (inf.isPlayer) {
                this.state = 'finished';
                this.finishClock = 0;
              }
            } else if (laps === this.cfg.laps - 1 && inf.isPlayer) this.emit('final_lap', i);
          }
        }
      }
      // wrong way
      if (racing && !inf.finished) {
        const idx = car.trkIdx;
        const along = car.vx * tr.tx[idx] + car.vz * tr.tz[idx];
        if (along < -3) inf.wrongWay += dt;
        else inf.wrongWay = Math.max(0, inf.wrongWay - dt * 2);
        if (inf.isPlayer && inf.wrongWay > 1.4 && inf.wrongWay - dt <= 1.4) this.emit('wrong_way', i);
      }
      // rollover / out-of-bounds recovery
      if (car.upDot < 0.25) inf.flipped += dt;
      else inf.flipped = 0;
      const wall = car.d > 0 ? tr.wallL[car.trkIdx] : tr.wallR[car.trkIdx];
      const bot = this.ai[i];
      if (inf.flipped > 1.6 || Math.abs(car.d) > wall + 3 || (bot && bot.wantsReset) || !Number.isFinite(car.px + car.py + car.pz + car.vx)) {
        if (bot) bot.wantsReset = false;
        this.resetCar(i);
      }
    }

    // standings
    const order = this.order;
    order.sort((a, b) => {
      const ia = this.info[a];
      const ib = this.info[b];
      if (ia.finished && ib.finished) return ia.finishTime - ib.finishTime;
      if (ia.finished) return -1;
      if (ib.finished) return 1;
      return ib.progress - ia.progress;
    });
    const lead = this.info[order[0]].progress;
    for (let k = 0; k < n; k++) {
      const inf = this.info[order[k]];
      if (inf.position !== k + 1) {
        if (inf.isPlayer && racing) this.emit('position', order[k], k + 1, inf.position);
        inf.position = k + 1;
      }
      inf.gap = lead - inf.progress;
    }

    // drafting + AI pacing
    for (let i = 0; i < n; i++) {
      const car = cars[i];
      let best = 0;
      if (car.speed > 28) {
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const o = cars[j];
          const ds = ringDiff(o.s, car.s, L);
          if (ds < 3 || ds > 48) continue;
          const dd = Math.abs(o.d - car.d);
          if (dd > 2.0) continue;
          const q = (1 - (ds - 3) / 45) * (1 - dd / 2.0);
          if (q > best) best = q;
        }
      }
      car.draft = damp(car.draft, best, 2.5, dt);
      const bot = this.ai[i];
      if (bot && this.cfg.mode === 'race') {
        const delta = this.info[i].progress - player.progress; // + = AI ahead of the player
        const hard = this.cfg.difficulty;
        let pace = 1;
        if (delta > 50) pace = 1 - Math.min([0.085, 0.055, 0.03][hard] ?? 0.055, (delta - 50) / 2600);
        else if (delta < -50) pace = 1 + Math.min([0.02, 0.035, 0.05][hard] ?? 0.035, (-delta - 50) / 3000);
        bot.pace = damp(bot.pace, pace, 0.8, dt);
      }
    }

    if (this.state === 'finished') this.finishClock += dt;
  }

  /** Seconds since the player took the flag. */
  get sinceFinish(): number {
    return this.state === 'finished' ? this.finishClock : 0;
  }
}
