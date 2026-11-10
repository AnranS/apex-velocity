// Checks the QQ-Speed-style control logic: drift -> pull the nose straight -> small boost -> double boost,
// nitro bottles from drifting, super start.   npx tsx tools/sim/qqtest.ts
import { makeInput } from '../../src/sim/car';
import { RaceSim } from '../../src/sim/race';
import { Track } from '../../src/sim/track';
import { CRIMSON_BAY } from '../../src/sim/tracks';

const DT = 1 / 240;
const tr = new Track(CRIMSON_BAY);
const race = new RaceSim(tr, { mode: 'race', laps: 3, playerCar: 'aurion', aiCount: 0, difficulty: 1, wet: 0, seed: 7, abs: true, tcs: 1, steerAssist: 1, autoGear: true });
race.startCountdown();
const car = race.player;
const inp = makeInput();
const log: string[] = [];
const events: string[] = [];
let t = 0;
const step = (n: number, set: () => void, tag = '') => {
  for (let i = 0; i < n; i++) {
    set();
    race.step(DT, inp);
    if (race.state === 'racing') t += DT;
    for (const e of race.events) {
      if (e.type === 'mini_boost') events.push(`${t.toFixed(2)}s boost kind ${e.a}`);
      if (e.type === 'drift_start') events.push(`${t.toFixed(2)}s drift start`);
      if (e.type === 'drift_end') events.push(`${t.toFixed(2)}s drift end (${car.driftTime.toFixed(2)}s)`);
      if (e.type === 'nitro_ready') events.push(`${t.toFixed(2)}s nitro bottle -> ${e.a}`);
      if (e.type === 'nitro_start') events.push(`${t.toFixed(2)}s nitro start`);
      if (e.type === 'nitro_end') events.push(`${t.toFixed(2)}s nitro end`);
      if (e.type === 'impact') events.push(`${t.toFixed(2)}s impact`);
    }
    race.events.length = 0;
  }
  if (tag) log.push(`${tag}: t=${t.toFixed(2)} v=${(car.speed * 3.6).toFixed(0)} s=${car.s.toFixed(0)} d=${car.d.toFixed(1)} beta=${((car.beta * 180) / Math.PI).toFixed(0)} drift=${car.drifting} gauge=${car.nitro.toFixed(2)} bottles=${car.nitroBottles} window=${car.boostWindow.toFixed(2)}`);
};
const keys = (o: Partial<typeof inp>) => () => Object.assign(inp, { throttle: 0, brake: 0, steer: 0, handbrake: 0, nitro: false }, o);

// countdown with the accelerator held, then tap the drift key right at the lights
while (race.state !== 'racing') step(1, keys({ throttle: 1 }));
step(20, keys({ throttle: 1 }));
step(12, keys({ throttle: 1, handbrake: 1 }));
step(1, keys({ throttle: 1 }), 'after super start');
// run to T1 braking zone
while (car.s > 3000 || car.s < 315) step(1, keys({ throttle: 1 }));
while (car.speed * 3.6 > 118) step(1, keys({ brake: 1 }));
while (car.s < 372) step(1, keys({ throttle: 1 }));
log.push('--- T1 (right-hander)');
// drift: right + Shift held
step(Math.round(0.85 / DT), keys({ throttle: 1, steer: -1, handbrake: 1 }), 'drift held 0.85s');
// release Shift, keep steering in briefly
step(Math.round(0.15 / DT), keys({ throttle: 1, steer: -1 }), 'released');
// pull the nose straight with opposite lock until the drift ends
let guard = 0;
while (car.drifting && guard++ < 600) step(1, keys({ throttle: 1, steer: 1 }));
step(1, keys({ throttle: 1 }), 'drift ended');
// small boost: release and re-press the accelerator
step(14, keys({ throttle: 0 }));
step(8, keys({ throttle: 1 }), 'after small boost');
// double boost: quick tap of drift + direction, straighten, re-press accelerator
step(Math.round(0.12 / DT), keys({ throttle: 1, steer: -1, handbrake: 1 }));
guard = 0;
while (car.drifting && guard++ < 400) step(1, keys({ throttle: 1, steer: 1 }));
step(10, keys({ throttle: 0 }));
step(10, keys({ throttle: 1 }), 'after double boost');
step(Math.round(1.5 / DT), keys({ throttle: 1 }), 'exit');
// fill a bottle with repeated drifts through the next corners, then fire it on the straight
log.push('--- collecting nitro');
guard = 0;
while (car.nitroBottles < 1 && guard++ < 240 * 60) {
  const k = tr.kappa[tr.indexAt(car.s + 10)];
  const turning = Math.abs(k) > 1 / 180 && car.speed > 16;
  // crude driver: follow the centreline, drift whenever the road bends
  const want = Math.max(-1, Math.min(1, -car.d * 0.12 + k * 40));
  step(1, keys({ throttle: car.speed * 3.6 > 150 && Math.abs(k) > 0.008 ? 0 : 1, steer: want, handbrake: turning ? 1 : 0 }));
}
step(1, keys({ throttle: 1 }), 'bottle collected');
const v0 = car.speed;
step(6, keys({ throttle: 1, nitro: true }));
step(Math.round(1.5 / DT), keys({ throttle: 1 }), 'nitro +1.5s');
log.push(`nitro on=${car.nitroOn} time left ${car.nitroTime.toFixed(2)} dv=${((car.speed - v0) * 3.6).toFixed(0)} km/h`);
console.log(log.join('\n'));
console.log('events:\n  ' + events.join('\n  '));
