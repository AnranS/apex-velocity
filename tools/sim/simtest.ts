// Headless physics/race checks:  npx tsx tools/sim/simtest.ts [statics|accel|skidpad|lap|race|drift|all] [carId]
import { AIDriver } from '../../src/sim/ai';
import { Car, makeInput } from '../../src/sim/car';
import { CAR_SPECS } from '../../src/sim/car_specs';
import { RaceSim, RaceConfig } from '../../src/sim/race';
import { Track, TrackDef } from '../../src/sim/track';
import { CRIMSON_BAY } from '../../src/sim/tracks';

const DT = 1 / 240;
const which = process.argv[2] ?? 'all';
const carId = process.argv[3] ?? 'aurion';
const spec = CAR_SPECS[carId];

const OVAL: TrackDef = {
  id: 'oval', name: 'TEST OVAL', nameCn: '', blurb: '', heading0: 90, y0: 0, halfWidth: 9, bankGain: 0, bankMax: 0, seed: 1, zones: [],
  segs: [
    { kind: 'S', len: 3000, adjust: 'a' },
    { kind: 'A', r: 300, deg: -90 },
    { kind: 'S', len: 60, adjust: 'b' },
    { kind: 'A', r: 300, deg: -90 },
    { kind: 'S', len: 3000 },
    { kind: 'A', r: 300, deg: -90 },
    { kind: 'S', len: 60 },
    { kind: 'A', r: 300, deg: -90 },
  ],
};
const circle = (r: number): TrackDef => ({
  id: 'circle', name: 'SKIDPAD', nameCn: '', blurb: '', heading0: 0, y0: 0, halfWidth: 8, bankGain: 0, bankMax: 0, seed: 1, zones: [],
  segs: [
    { kind: 'A', r, deg: 90 }, { kind: 'A', r, deg: 90 }, { kind: 'A', r, deg: 90 }, { kind: 'A', r, deg: 90 },
  ],
});

function fmt(n: number, d = 2): string {
  return Number.isFinite(n) ? n.toFixed(d) : String(n);
}
function stepCar(car: Car, tr: Track, inp: ReturnType<typeof makeInput>): void {
  car.step(DT, inp, tr);
  car.collideWorld(tr);
  car.integrate(DT, tr);
}

function statics(): void {
  const tr = new Track(CRIMSON_BAY);
  const car = new Car(spec);
  car.placeOnTrack(tr, tr.length - 20, 3);
  car.hold = true;
  const inp = makeInput();
  const y0 = car.originY();
  for (let i = 0; i < 240 * 3; i++) stepCar(car, tr, inp);
  tr.ground(car.originX(), car.originZ(), car.trkIdx);
  console.log(
    `[statics ${carId}] origin above ground ${fmt((car.originY() - tr.oy) * 1000, 1)} mm (start drift ${fmt((car.originY() - y0) * 1000, 1)} mm), speed ${fmt(car.speed, 4)} m/s, ` +
      `loads ${car.wheels.map((w) => fmt(w.fz, 0)).join('/')} N (static ${car.wheels.map((w) => fmt(w.staticLoad, 0)).join('/')}), susp ${car.wheels.map((w) => fmt(w.len * 1000, 1)).join('/')} mm, up ${fmt(car.upDot, 4)}`,
  );
}

function accel(): void {
  const tr = new Track(OVAL);
  const car = new Car(spec);
  car.placeOnTrack(tr, 20, 0);
  car.hold = false;
  car.tcs = 1;
  const inp = makeInput();
  inp.throttle = 1;
  let t = 0;
  const marks: Record<string, number> = {};
  let dist = 0;
  let maxSlip = 0;
  let q = 0;
  for (let i = 0; i < 240 * 45; i++) {
    stepCar(car, tr, inp);
    t += DT;
    dist += car.speed * DT;
    const kmh = car.speed * 3.6;
    for (const m of [60, 100, 160, 200, 250, 300]) if (kmh >= m && marks[m] === undefined) marks[m] = t;
    if (dist >= 402.3 && !q) q = t;
    maxSlip = Math.max(maxSlip, car.wheels[2].slipRatio);
    if (car.s > 2700) break;
  }
  const vTop = car.speed * 3.6;
  const topGear = car.gear;
  const topRpm = car.rpm;
  // nitro check
  inp.nitro = true;
  const car2 = new Car(spec);
  car2.placeOnTrack(tr, 20, 0);
  car2.hold = false;
  let t2 = 0;
  let n200 = 0;
  for (let i = 0; i < 240 * 20; i++) {
    stepCar(car2, tr, inp);
    t2 += DT;
    if (car2.speed * 3.6 >= 200 && !n200) n200 = t2;
  }
  // braking from ~200 km/h
  const car3 = new Car(spec);
  car3.placeOnTrack(tr, 20, 0);
  car3.hold = false;
  inp.nitro = false;
  let phase = 0;
  let bd = 0;
  let bt = 0;
  let peakG = 0;
  for (let i = 0; i < 240 * 40; i++) {
    if (phase === 0 && car3.speed * 3.6 >= 200) {
      phase = 1;
      inp.throttle = 0;
      inp.brake = 1;
    }
    stepCar(car3, tr, inp);
    if (phase === 1) {
      bd += car3.speed * DT;
      bt += DT;
      peakG = Math.min(peakG, car3.longG);
      if (car3.speed < 0.5) break;
    }
  }
  console.log(
    `[accel ${carId}] 0-60 ${fmt(marks[60])}s 0-100 ${fmt(marks[100])}s 0-160 ${fmt(marks[160])}s 0-200 ${fmt(marks[200])}s 0-250 ${fmt(marks[250])}s 0-300 ${fmt(marks[300])}s | 1/4 mile ${fmt(q)}s | ` +
      `v@2.7km ${fmt(vTop, 0)} km/h gear ${topGear} rpm ${fmt(topRpm, 0)} | max rear slip ${fmt(maxSlip, 2)} | nitro 0-200 ${fmt(n200)}s | brake 200-0: ${fmt(bd, 1)} m ${fmt(bt)} s peak ${fmt(peakG)} g`,
  );
}

function skidpad(): void {
  for (const r of [40, 100, 250]) {
    const tr = new Track(circle(r));
    const car = new Car(spec);
    car.placeOnTrack(tr, 5, 0);
    car.hold = false;
    car.tcs = 2;
    const bot = new AIDriver(car, tr, 1.0, 7);
    bot.reaction = 0;
    // ramp the reference speed up until the car can no longer hold the line
    let best = 0;
    let bestV = 0;
    for (let scale = 0.8; scale <= 1.6; scale += 0.04) {
      bot.pace = scale;
      let ok = true;
      let accG = 0;
      let cnt = 0;
      for (let i = 0; i < 240 * 8; i++) {
        const inp = bot.update(DT, [car], true);
        stepCar(car, tr, inp);
        if (i > 240 * 4) {
          accG += Math.abs(car.latG);
          cnt++;
          if (Math.abs(car.d) > 7.6) ok = false;
        }
      }
      if (!ok) break;
      const g = accG / cnt;
      if (g > best) {
        best = g;
        bestV = car.speed;
      }
    }
    console.log(`[skidpad ${carId}] R=${r} m: max sustained ${fmt(best)} g at ${fmt(bestV * 3.6, 0)} km/h (v²/R = ${fmt((bestV * bestV) / r / 9.81)} g)`);
  }
}

function baseCfg(over: Partial<RaceConfig> = {}): RaceConfig {
  return { mode: 'race', laps: 3, playerCar: carId, aiCount: 7, difficulty: 1, wet: 0, seed: 7, abs: true, tcs: 1, steerAssist: 1, autoGear: true, ...over };
}

function lap(): void {
  const tr = new Track(CRIMSON_BAY);
  for (const skill of [0.9, 1.0, 1.06]) {
    const race = new RaceSim(tr, baseCfg({ aiCount: 0, laps: 3 }));
    race.autopilot = true;
    race.startCountdown();
    const inp = makeInput();
    let impacts = 0;
    let resets = 0;
    let maxOff = -Infinity;
    let maxSpeed = 0;
    let minSpeed = Infinity;
    let offTime = 0;
    for (let i = 0; i < 240 * 400 && race.state !== 'finished'; i++) {
      race.step(DT, inp);
      // patch the autopilot skill once it exists
      const pb = (race as unknown as { playerBot: AIDriver | null }).playerBot;
      if (pb && pb.skill !== skill) pb.setSkill(skill);
      const car = race.player;
      for (const e of race.events) {
        if (e.type === 'impact') impacts++;
        if (e.type === 'reset') resets++;
      }
      race.events.length = 0;
      if (race.state === 'racing' && race.time > 6) {
        const off = Math.abs(car.d) - tr.hw[car.trkIdx];
        maxOff = Math.max(maxOff, off);
        if (off > 0) offTime += DT;
        maxSpeed = Math.max(maxSpeed, car.speed);
        minSpeed = Math.min(minSpeed, car.speed);
      }
    }
    const inf = race.playerInfo;
    console.log(
      `[lap ${carId} skill ${skill}] laps ${inf.lapTimes.map((t) => fmt(t)).join(' / ')} s | v ${fmt(minSpeed * 3.6, 0)}-${fmt(maxSpeed * 3.6, 0)} km/h | max off-road ${fmt(maxOff)} m, off time ${fmt(offTime, 1)} s | impacts ${impacts} resets ${resets}`,
    );
  }
}

function raceTest(): void {
  const tr = new Track(CRIMSON_BAY);
  for (const difficulty of [0, 1, 2]) {
    const race = new RaceSim(tr, baseCfg({ difficulty, laps: 3 }));
    race.autopilot = true;
    race.startCountdown();
    const inp = makeInput();
    let impacts = 0;
    let bigImpacts = 0;
    let resets = 0;
    let nan = false;
    const t0 = performance.now();
    let simT = 0;
    let done = 0;
    for (let i = 0; i < 240 * 500; i++) {
      race.step(DT, inp);
      simT += DT;
      for (const e of race.events) {
        if (e.type === 'impact') {
          impacts++;
          if (e.a > 4000) bigImpacts++;
        }
        if (e.type === 'reset') resets++;
      }
      race.events.length = 0;
      for (const c of race.cars) if (!Number.isFinite(c.px + c.vx + c.qw)) nan = true;
      done = race.info.filter((x) => x.finished).length;
      if (done === race.cars.length || nan) break;
    }
    const ms = performance.now() - t0;
    const rows = race.order.map((k) => {
      const x = race.info[k];
      return `${x.position}.${x.isPlayer ? 'P' : 'ai'}(${x.carId[0]}) ${x.finished ? fmt(x.finishTime, 1) : 'DNF@' + fmt(x.progress, 0)} best ${fmt(x.bestLap, 1)}`;
    });
    console.log(
      `[race diff ${difficulty}] ${rows.join(' | ')}\n   impacts ${impacts} (big ${bigImpacts}) resets ${resets} nan ${nan} | sim ${fmt(simT, 0)} s in ${fmt(ms, 0)} ms (${fmt(ms / simT, 2)} ms per sim-second)`,
    );
  }
}

function drift(): void {
  const tr = new Track(OVAL);
  const car = new Car(spec);
  car.placeOnTrack(tr, 20, -6);
  car.hold = false;
  car.nitro = 0;
  const inp = makeInput();
  let t = 0;
  const log: string[] = [];
  const h0 = car.heading();
  let yaw = 0;
  let hPrev = h0;
  let maxBeta = 0;
  for (let i = 0; i < 240 * 14; i++) {
    t += DT;
    inp.throttle = 1;
    inp.steer = 0;
    inp.handbrake = 0;
    if (t > 5 && t < 7.2) {
      inp.steer = t < 6.4 ? 1 : 0.3;
      inp.handbrake = 1;
    } else if (t >= 7.2 && t < 7.6) {
      inp.steer = 0;
    }
    stepCar(car, tr, inp);
    let dh = car.heading() - hPrev;
    if (dh > Math.PI) dh -= 2 * Math.PI;
    if (dh < -Math.PI) dh += 2 * Math.PI;
    yaw += dh;
    hPrev = car.heading();
    maxBeta = Math.max(maxBeta, Math.abs(car.beta));
    if (i % 96 === 0 && t > 4.4 && t < 9.5)
      log.push(`t${fmt(t, 1)} v${fmt(car.speed * 3.6, 0)} beta${fmt((car.beta * 180) / Math.PI, 0)} yaw${fmt((yaw * 180) / Math.PI, 0)} d${fmt(car.d, 1)} drift${car.drifting ? 1 : 0} nos${fmt(car.nitro, 2)} mb${fmt(car.miniBoost, 2)} latG${fmt(car.latG, 2)} roll${fmt(Math.asin(car.m10) * 57.3, 1)}`);
  }
  console.log(`[drift ${carId}] max rear slip ${fmt((maxBeta * 180) / Math.PI, 0)} deg, total yaw ${fmt((yaw * 180) / Math.PI, 0)} deg, up ${fmt(car.upDot, 3)}\n   ${log.join('\n   ')}`);
}

const all: Record<string, () => void> = { statics, accel, skidpad, lap, race: raceTest, drift };
if (which === 'all') for (const k of Object.keys(all)) all[k]();
else all[which]?.();
