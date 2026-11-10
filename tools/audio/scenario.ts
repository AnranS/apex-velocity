/**
 * Renders a short "drive" (launch → upshifts → redline limiter → lift-off overrun → downshifts → nitro)
 * for one engine type to tools/audio/_work/drive_<type>.wav so it can be inspected (spectrogram) or listened to.
 *   npx tsx tools/audio/scenario.ts [type] [player 0|1]
 */
import { renderVoice, writeWav, WORK, rms, peak, dB, fmt, type DriveState } from './lib';
import { NOMINAL_RPM_MAX, type EngineType } from '../../src/audio/engine-configs';

export interface Scenario {
  dur: number;
  state: (t: number) => DriveState;
  events: { t: number; code: number; val?: number }[];
}

/** A scripted drive: returns the state function and the event list. */
export function driveScenario(type: EngineType): Scenario {
  const rm = NOMINAL_RPM_MAX[type];
  const events: { t: number; code: number; val?: number }[] = [];
  // piecewise segments: [t0, t1, rpm0, rpm1, load, speed0, speed1]
  const seg: number[][] = [];
  let t = 0;
  const add = (d: number, r0: number, r1: number, load: number, s0: number, s1: number) => {
    seg.push([t, t + d, r0 * rm, r1 * rm, load, s0, s1]);
    t += d;
  };
  add(1.0, 0.12, 0.12, 0, 0, 0); // idle
  add(0.5, 0.12, 0.55, 0.9, 0, 0); // rev
  add(0.5, 0.55, 0.14, 0, 0, 0);
  add(1.3, 0.45, 0.97, 1, 2, 18); // 1st
  events.push({ t, code: 0 });
  add(1.5, 0.62, 0.97, 1, 18, 32); // 2nd
  events.push({ t, code: 0 });
  add(1.8, 0.7, 0.97, 1, 32, 47); // 3rd
  events.push({ t, code: 0 });
  add(2.0, 0.76, 1.0, 1, 47, 60); // 4th
  add(0.7, 1.0, 1.0, 1, 60, 61); // on the limiter
  add(1.6, 0.98, 0.7, 0, 61, 50); // lift-off: overrun
  events.push({ t, code: 1 });
  add(0.9, 0.86, 0.66, 0, 50, 40); // downshift
  events.push({ t, code: 1 });
  add(0.9, 0.84, 0.6, 0.05, 40, 30);
  events.push({ t, code: 2, val: 1 }); // backfire
  add(0.4, 0.6, 0.55, 0, 30, 28);
  events.push({ t, code: 4 }); // nitro
  add(2.0, 0.55, 0.95, 1, 28, 55);
  const nitroFrom = t - 2.0;
  const nitroTo = t - 0.4;
  events.push({ t, code: 0 });
  add(1.2, 0.72, 0.85, 0.6, 55, 60);
  add(1.0, 0.85, 0.2, 0, 60, 30);
  const dur = t;
  const state = (tt: number): DriveState => {
    let s = seg[seg.length - 1];
    for (const q of seg) {
      if (tt >= q[0] && tt < q[1]) {
        s = q;
        break;
      }
    }
    const k = Math.min(1, Math.max(0, (tt - s[0]) / (s[1] - s[0])));
    const rpm = s[2] + (s[3] - s[2]) * k;
    return {
      rpm,
      norm: rpm / rm,
      load: s[4],
      speed: s[5] + (s[6] - s[5]) * k,
      nitro: tt >= nitroFrom && tt < nitroTo ? 1 : 0,
      gain: 1,
      pan: 0,
    };
  };
  return { dur, state, events };
}

const isMain = process.argv[1]?.endsWith('scenario.ts');
if (isMain) {
  const type = (process.argv[2] as EngineType) || 'v8';
  const player = process.argv[3] !== '0';
  const sc = driveScenario(type);
  const r = renderVoice({ type, player, dur: sc.dur, state: sc.state, events: sc.events });
  const mix = (a: Float32Array, b: Float32Array) => a.map((v, i) => v + b[i]);
  writeWav(`${WORK}drive_${type}${player ? '' : '_cheap'}.wav`, [mix(r.eL, r.tL), mix(r.eR, r.tR)], r.fs);
  console.log(`drive_${type}: ${fmt(sc.dur, 1)}s  engine rms ${fmt(dB(rms(r.eL)), 1)}dB peak ${fmt(dB(peak(r.eL)), 1)}dB  tyres rms ${fmt(dB(rms(r.tL)), 1)}dB`);
}
