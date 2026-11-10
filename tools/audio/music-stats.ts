/**
 * Quick level / balance report for the music cues (offline renders in headless Chromium).
 *   npx tsx tools/audio/music-stats.ts
 */
import { rms, peak, dB, fmt, welch, bandPower, writeWav, WORK } from './lib';
import { launch, openPage, decode, bundle } from './browser-lib';

const browser = await launch();
const page = await openPage(browser, await bundle(false));
const plans: [string, string, number, string][] = [
  ['race I=1.0', 'race', 30, '[[0,1]]'],
  ['race I=0.5', 'race', 16, '[[0,0.5]]'],
  ['race I=0.0', 'race', 16, '[[0,0]]'],
  ['menu', 'menu', 47, '[[0,0.5]]'],
  ['results', 'results', 18, '[[0,0.5]]'],
];
for (const [label, cue, dur, plan] of plans) {
  const r = (await page.evaluate(`AT.renderMusic('${cue}', ${dur}, ${plan})`)) as { fs: number; pcm: string };
  const [L, R] = decode(r.pcm);
  const a = Math.round(1.0 * r.fs);
  const pw = welch(L, a, L.length - a, 8192);
  const tot = bandPower(pw, r.fs, 25, 16000);
  const sh = (f0: number, f1: number) => fmt((bandPower(pw, r.fs, f0, f1) / tot) * 100, 1);
  let c = 0;
  for (let i = 0; i < L.length; i++) c += L[i] * R[i];
  const corr = c / (L.length * rms(L) * rms(R));
  // short-term loudness spread (1 s windows)
  const win: number[] = [];
  for (let s = a; s + r.fs <= L.length; s += r.fs) win.push(dB(rms(L, s, s + r.fs)));
  console.log(
    `${label.padEnd(12)} peak ${fmt(dB(Math.max(peak(L), peak(R))), 1)} rms ${fmt(dB(rms(L, a)), 1)} dBFS | 1s-window ${fmt(Math.min(...win), 1)}..${fmt(Math.max(...win), 1)} | sub<60 ${sh(25, 60)}% bass60-150 ${sh(60, 150)}% lowmid150-500 ${sh(150, 500)}% mid500-2.5k ${sh(500, 2500)}% hi2.5-8k ${sh(2500, 8000)}% air>8k ${sh(8000, 16000)}% | L/R corr ${fmt(corr, 2)}`,
  );
  writeWav(`${WORK}music_${label.replace(/[^a-z0-9]+/gi, '_')}.wav`, [L, R], r.fs);
}
await browser.close();
