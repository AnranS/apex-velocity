// Dumps the sampled track to JSON for plotting / sanity checks:  npx tsx tools/sim/dump_track.ts [out.json]
import { writeFileSync } from 'node:fs';
import { Track } from '../../src/sim/track';
import { CRIMSON_BAY } from '../../src/sim/tracks';

const t0 = performance.now();
const tr = new Track(CRIMSON_BAY);
const buildMs = performance.now() - t0;
const out = process.argv[2] ?? '/tmp/apex_track.json';
const vref = tr.buildSpeedProfile({
  mass: 1450, muLat: 1.35, muBrake: 1.25, downforceK: 1.9, dragK: 0.42, powerW: 480000, tractionG: 9.5, vTop: 96,
});

// minimum distance between non-adjacent parts of the corridor
let minGap = Infinity;
let minAt = [0, 0];
const skip = Math.round(160 / tr.ds);
for (let i = 0; i < tr.n; i += 3) {
  for (let j = i + skip; j < tr.n; j += 3) {
    if (tr.n - j + i < skip) continue;
    const d = Math.hypot(tr.cx[i] - tr.cx[j], tr.cz[i] - tr.cz[j]);
    const need = Math.max(tr.wallL[i], tr.wallR[i]) + Math.max(tr.wallL[j], tr.wallR[j]);
    if (d - need < minGap) {
      minGap = d - need;
      minAt = [i, j];
    }
  }
}
let maxGrade = 0;
let maxBank = 0;
let maxK = 0;
let lapTime = 0;
for (let i = 0; i < tr.n; i++) {
  maxGrade = Math.max(maxGrade, Math.abs(tr.grade[i]));
  maxBank = Math.max(maxBank, Math.abs(tr.slopeL[i]));
  maxK = Math.max(maxK, Math.abs(tr.kappa[i]));
  lapTime += tr.ds / vref[i];
}
console.log(
  JSON.stringify({
    buildMs: +buildMs.toFixed(1), n: tr.n, ds: +tr.ds.toFixed(4), length: +tr.length.toFixed(1),
    bbox: [tr.minX, tr.maxX, tr.minZ, tr.maxZ].map((v) => +v.toFixed(1)),
    minCorridorGap: +minGap.toFixed(1), minAt,
    maxGradePct: +(maxGrade * 100).toFixed(1), maxBankDeg: +((Math.atan(maxBank) * 180) / Math.PI).toFixed(1),
    minRadius: +(1 / maxK).toFixed(1), refLap: +lapTime.toFixed(2),
    vmin: +(Math.min(...vref) * 3.6).toFixed(0), vmax: +(Math.max(...vref) * 3.6).toFixed(0),
    segS: tr.segS.map((s) => Math.round(s)),
  }),
);
writeFileSync(
  out,
  JSON.stringify({
    n: tr.n, ds: tr.ds,
    cx: Array.from(tr.cx), cy: Array.from(tr.cy), cz: Array.from(tr.cz), nx: Array.from(tr.nx), nz: Array.from(tr.nz),
    hw: Array.from(tr.hw), wallL: Array.from(tr.wallL), wallR: Array.from(tr.wallR), kerbL: Array.from(tr.kerbL), kerbR: Array.from(tr.kerbR),
    gravelL: Array.from(tr.gravelL), gravelR: Array.from(tr.gravelR), line: Array.from(tr.line), kappa: Array.from(tr.kappa),
    vref: Array.from(vref), segS: tr.segS, segTag: tr.segTag,
  }),
);
