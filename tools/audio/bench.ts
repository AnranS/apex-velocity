/**
 * DSP cost breakdown (ms of CPU per second of audio).   npx tsx tools/audio/bench.ts
 */
import { CarSynthDSP } from '../../src/audio/engine-dsp';
import { engineConfig, NOMINAL_RPM_MAX, type EngineType } from '../../src/audio/engine-configs';
import { packState, fmt, type DriveState } from './lib';

const FS = 48000;
const TY: EngineType[] = ['v8', 'v12', 'i6t', 'v10', 'v8', 'v12', 'i6t', 'v10'];
function run(label: string, voices: { type: EngineType; player: boolean }[], st: (i: number, t: number) => Partial<DriveState>): void {
  const dsp = new CarSynthDSP(FS, 12);
  voices.forEach((v, i) => dsp.configure(i, engineConfig(v.type), v.player, 0));
  const q = [0, 1, 2, 3].map(() => new Float32Array(128));
  const upd = (t: number): void =>
    voices.forEach((v, i) => {
      const rm = NOMINAL_RPM_MAX[v.type];
      const n = 0.5 + 0.45 * Math.abs(Math.sin(t * 0.7 + i));
      dsp.setStates(packState(i, { rpm: rm * n, norm: n, load: 1, gain: v.player ? 1 : 0.5, pan: Math.sin(i), ...st(i, t) }, rm));
    });
  const secs = 10;
  let best = Infinity;
  for (let rep = 0; rep < 4; rep++) {
    const t0 = performance.now();
    for (let p = 0; p < FS * secs; p += 128) {
      if (p % 768 === 0) upd(p / FS);
      dsp.process(q[0], q[1], q[2], q[3], 128);
    }
    best = Math.min(best, (performance.now() - t0) / secs);
  }
  console.log(`${label.padEnd(58)} ${fmt(best, 2).padStart(6)} ms/s  (${fmt(best / 10, 2)}% of a core)`);
}
const one = (player: boolean) => [{ type: 'v8' as EngineType, player }];
const eight = TY.map((type, i) => ({ type, player: i === 0 }));
run('player V8, engine only', one(true), () => ({ speed: 0 }));
run('player V8, engine + road/wind @60m/s', one(true), () => ({ speed: 60 }));
run('player V8, + squeal + scrape + nitro', one(true), () => ({ speed: 60, slip: 0.8, scrape: 0.6, nitro: 1 }));
run('cheap V8, engine only', one(false), () => ({ speed: 0 }));
run('cheap V8, engine + road @60m/s', one(false), () => ({ speed: 60 }));
run('cheap V8, + squeal + scrape', one(false), () => ({ speed: 60, slip: 0.8, scrape: 0.6 }));
run('8 cars typical race (road noise, 2 cars sliding)', eight, (i) => ({ speed: 60, slip: i % 4 === 1 ? 0.6 : 0 }));
run('8 cars worst case (all sliding/scraping/off-road, nitro)', eight, (i) => ({ speed: 60, slip: 0.7, scrape: i % 3 === 0 ? 0.6 : 0, surface: i % 4, nitro: i === 0 ? 1 : 0 }));
