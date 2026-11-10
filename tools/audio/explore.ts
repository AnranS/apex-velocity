/**
 * Exploratory spectral dump of the engine voices (not a pass/fail test).
 *   npx tsx tools/audio/explore.ts [type] [player 0|1]
 */
import { renderVoice, rms, peak, dB, welch, bandPower, centroid, harmonicPower, fmt, writeWav, WORK } from './lib';
import { ENGINE_VOICINGS, NOMINAL_RPM_MAX, type EngineType } from '../../src/audio/engine-configs';

const only = process.argv[2] as EngineType | undefined;
const player = process.argv[3] !== '0';
const TYPES: EngineType[] = only ? [only] : ['v8', 'v12', 'i6t', 'v10'];
const N = 32768;

for (const type of TYPES) {
  const rpmMax = NOMINAL_RPM_MAX[type];
  const cyl = ENGINE_VOICINGS[type].cyl;
  console.log(`\n=== ${type} (${player ? 'player' : 'cheap'}) redline ${rpmMax} ===`);
  for (const [norm, load] of [
    [0.12, 0],
    [0.3, 0.3],
    [0.5, 1],
    [0.5, 0],
    [0.75, 1],
    [0.98, 1],
  ]) {
    const rpm = norm * rpmMax;
    const r = renderVoice({ type, player, dur: 2.6, state: () => ({ rpm, norm, load, gain: 1, pan: 0, speed: 0 }) });
    const a = Math.round(0.5 * r.fs);
    const pw = welch(r.eL, a, r.eL.length - a, N);
    const fCyc = rpm / 120;
    const fFire = fCyc * cyl;
    const tot = bandPower(pw, r.fs, 20, 6000);
    const harm = harmonicPower(pw, r.fs, fCyc, 6000);
    const fire = harmonicPower(pw, r.fs, fFire, 6000);
    // top peaks (by engine order)
    const binHz = r.fs / N;
    const peaks: { f: number; p: number }[] = [];
    for (let i = 3; i < pw.length - 3 && i * binHz < 8000; i++) {
      if (pw[i] > pw[i - 1] && pw[i] >= pw[i + 1] && pw[i] > pw[i - 2] && pw[i] > pw[i + 2]) peaks.push({ f: i * binHz, p: pw[i] });
    }
    peaks.sort((x, y) => y.p - x.p);
    const top = peaks
      .slice(0, 9)
      .map((q) => `${fmt(q.f, 0)}Hz(o${fmt(q.f / (rpm / 60), 2)}):${fmt(10 * Math.log10(q.p), 0)}`)
      .join(' ');
    const lo = bandPower(pw, r.fs, 20, 250);
    const mid = bandPower(pw, r.fs, 250, 2000);
    const hi = bandPower(pw, r.fs, 2000, 12000);
    console.log(
      `rpm ${fmt(rpm, 0).padStart(5)} load ${load}  rms ${fmt(dB(rms(r.eL, a)), 1)}dB peak ${fmt(dB(peak(r.eL, a)), 1)}dB crest ${fmt(peak(r.eL, a) / rms(r.eL, a), 1)}` +
        `  cent ${fmt(centroid(pw, r.fs), 0)}Hz  harm ${fmt(harm / tot, 2)} fireOrd ${fmt(fire / harm, 2)}  lo/mid/hi ${fmt(10 * Math.log10(lo / tot), 0)}/${fmt(10 * Math.log10(mid / tot), 0)}/${fmt(10 * Math.log10(hi / tot), 0)}dB`,
    );
    console.log(`      fire ${fmt(fFire, 0)}Hz  top: ${top}`);
    if (process.argv[4] === 'wav') writeWav(`${WORK}explore_${type}_${Math.round(rpm)}_${load}.wav`, [r.eL, r.eR], r.fs);
  }
}
