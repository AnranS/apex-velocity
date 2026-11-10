// Track geometry: built from straights/arcs ("turtle" path) with automatic loop closure,
// smoothed into transition curves, resampled uniformly, then decorated with banking,
// walls, kerbs, run-off zones, a racing line and reference speed profiles.
//
// Conventions (shared by the whole game):
//   world: +Y up. heading psi: forward = (sin psi, 0, cos psi), positive psi turns LEFT.
//   track frame: s = distance along centreline, d = lateral offset, POSITIVE = LEFT.
//   left normal N = (tz, 0, -tx). curvature kappa > 0 = left turn.
import { clamp, lerp, smoothstep, mod } from './math';

export const SURF_ASPHALT = 0;
export const SURF_KERB = 1;
export const SURF_GRASS = 2;
export const SURF_GRAVEL = 3;

export type Seg =
  | { kind: 'S'; len: number; y?: number; hw?: number; adjust?: 'a' | 'b'; tag?: string }
  | { kind: 'A'; r: number; deg: number; y?: number; hw?: number; tag?: string };

export interface ZoneDef {
  kind: 'grandstand' | 'pit' | 'gallery' | 'billboards' | 'lights' | 'forest_gap' | 'tyrewall';
  /** distances in metres along the lap (start line = 0); negative values wrap from the end */
  s0: number;
  s1: number;
  side: 'L' | 'R' | 'both';
  /** lateral distance from the wall (m) where the feature sits */
  offset?: number;
}

export interface TrackDef {
  id: string;
  name: string;
  nameCn: string;
  blurb: string;
  heading0: number; // degrees
  y0: number;
  halfWidth: number;
  segs: Seg[];
  bankGain: number;
  bankMax: number;
  zones: ZoneDef[];
  seed: number;
}

const KERB_W = 1.35;
const EDGE_BLEND = 0.5;
const VERGE_SLOPE = -0.025;
const KERB_H = 0.035;

function smoothRing(a: Float64Array, radius: number, passes: number): void {
  const n = a.length;
  const tmp = new Float64Array(n);
  const w = 2 * radius + 1;
  for (let p = 0; p < passes; p++) {
    let acc = 0;
    for (let k = -radius; k <= radius; k++) acc += a[mod(k, n)];
    for (let i = 0; i < n; i++) {
      tmp[i] = acc / w;
      acc += a[mod(i + radius + 1, n)] - a[mod(i - radius, n)];
    }
    a.set(tmp);
  }
}

function smoothRingF32(a: Float32Array, radius: number, passes: number): void {
  const b = new Float64Array(a);
  smoothRing(b, radius, passes);
  a.set(b);
}

/** Resample closed polyline channels to `m` uniformly spaced points (by xz arc length). */
function resampleClosed(xs: Float64Array, zs: Float64Array, chans: Float64Array[], m: number) {
  const n = xs.length;
  const cum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    cum[i + 1] = cum[i] + Math.hypot(xs[j] - xs[i], zs[j] - zs[i]);
  }
  const total = cum[n];
  const ox = new Float64Array(m);
  const oz = new Float64Array(m);
  const oc = chans.map(() => new Float64Array(m));
  let seg = 0;
  for (let k = 0; k < m; k++) {
    const target = (k / m) * total;
    while (seg < n - 1 && cum[seg + 1] < target) seg++;
    const span = cum[seg + 1] - cum[seg];
    const t = span > 1e-9 ? (target - cum[seg]) / span : 0;
    const j = (seg + 1) % n;
    ox[k] = xs[seg] + (xs[j] - xs[seg]) * t;
    oz[k] = zs[seg] + (zs[j] - zs[seg]) * t;
    for (let c = 0; c < chans.length; c++) oc[c][k] = chans[c][seg] + (chans[c][j] - chans[c][seg]) * t;
  }
  return { xs: ox, zs: oz, chans: oc, total };
}

export interface SpeedProfileParams {
  mass: number;
  muLat: number; // lateral grip coefficient used for cornering speed
  muBrake: number;
  downforceK: number; // N per (m/s)^2
  dragK: number; // N per (m/s)^2
  powerW: number; // peak power at the wheels (W)
  tractionG: number; // max traction-limited acceleration (m/s^2)
  vTop: number; // m/s cap
}

export class Track {
  readonly def: TrackDef;
  n = 0;
  ds = 0;
  length = 0;
  cx!: Float32Array;
  cy!: Float32Array;
  cz!: Float32Array;
  tx!: Float32Array;
  tz!: Float32Array;
  nx!: Float32Array;
  nz!: Float32Array;
  psi!: Float32Array;
  kappa!: Float32Array;
  grade!: Float32Array;
  slopeL!: Float32Array;
  hw!: Float32Array;
  wallL!: Float32Array;
  wallR!: Float32Array;
  kerbL!: Float32Array;
  kerbR!: Float32Array;
  /** distance from centreline where gravel starts on each side (0 = no gravel trap) */
  gravelL!: Float32Array;
  gravelR!: Float32Array;
  /** racing line lateral offset (m, + left) and its curvature */
  line!: Float32Array;
  lineKappa!: Float32Array;
  /** start distance (m) of each definition segment in final lap coordinates */
  segS: number[] = [];
  segTag: (string | undefined)[] = [];
  minX = 0;
  maxX = 0;
  minZ = 0;
  maxZ = 0;

  // scratch outputs of locate()/ground() (avoid allocations in the hot path)
  oi = 0;
  ot = 0;
  od = 0;
  oy = 0;
  onx = 0;
  ony = 1;
  onz = 0;
  osurf = 0;

  constructor(def: TrackDef) {
    this.def = def;
    this.build();
  }

  // ---------------------------------------------------------------- build
  private build(): void {
    const def = this.def;
    const H = 0.5; // dense step (m)

    // -- 1. loop closure: solve the two adjustable straights
    const segs = def.segs.map((s) => ({ ...s }));
    const run = (emit: ((x: number, z: number, segIdx: number, dist: number) => void) | null) => {
      let x = 0;
      let z = 0;
      let psi = (def.heading0 * Math.PI) / 180;
      let dist = 0;
      const dirs: Record<string, [number, number]> = {};
      const starts: number[] = [];
      for (let k = 0; k < segs.length; k++) {
        const s = segs[k];
        starts.push(dist);
        if (s.kind === 'S') {
          const fx = Math.sin(psi);
          const fz = Math.cos(psi);
          if (s.adjust) dirs[s.adjust] = [fx, fz];
          if (emit) {
            const steps = Math.max(1, Math.round(s.len / H));
            for (let i = 0; i < steps; i++) {
              const u = (i / steps) * s.len;
              emit(x + fx * u, z + fz * u, k, dist + u);
            }
          }
          x += fx * s.len;
          z += fz * s.len;
          dist += s.len;
        } else {
          const ang = (s.deg * Math.PI) / 180;
          const sgn = Math.sign(ang);
          const arcLen = Math.abs(ang) * s.r;
          const lx0 = Math.cos(psi);
          const lz0 = -Math.sin(psi);
          if (emit) {
            const steps = Math.max(1, Math.round(arcLen / H));
            for (let i = 0; i < steps; i++) {
              const phi = (i / steps) * ang;
              const lx = Math.cos(psi + phi);
              const lz = -Math.sin(psi + phi);
              emit(x + sgn * s.r * (lx0 - lx), z + sgn * s.r * (lz0 - lz), k, dist + (i / steps) * arcLen);
            }
          }
          const lx1 = Math.cos(psi + ang);
          const lz1 = -Math.sin(psi + ang);
          x += sgn * s.r * (lx0 - lx1);
          z += sgn * s.r * (lz0 - lz1);
          psi += ang;
          dist += arcLen;
        }
      }
      return { x, z, psi, dist, dirs, starts };
    };
    const first = run(null);
    const da = first.dirs['a'];
    const db = first.dirs['b'];
    if (da && db) {
      const det = da[0] * db[1] - da[1] * db[0];
      if (Math.abs(det) > 1e-6) {
        const ex = -first.x;
        const ez = -first.z;
        const dA = (ex * db[1] - ez * db[0]) / det;
        const dB = (da[0] * ez - da[1] * ex) / det;
        for (const s of segs) {
          if (s.kind === 'S' && s.adjust === 'a') s.len += dA;
          if (s.kind === 'S' && s.adjust === 'b') s.len += dB;
          if (s.kind === 'S' && s.len < 20) throw new Error(`track closure produced a degenerate straight (${s.len.toFixed(1)} m)`);
        }
      }
    }

    // -- 2. dense emission
    const dx: number[] = [];
    const dz: number[] = [];
    const dseg: number[] = [];
    const ddist: number[] = [];
    const res = run((x, z, k, dist) => {
      dx.push(x);
      dz.push(z);
      dseg.push(k);
      ddist.push(dist);
    });
    const closeErr = Math.hypot(res.x, res.z);
    if (closeErr > 0.5) throw new Error(`track does not close (error ${closeErr.toFixed(2)} m)`);
    const rawLen = res.dist;

    // elevation + width knots (piecewise linear over raw distance)
    const yKnots: [number, number][] = [[0, def.y0]];
    const wKnots: [number, number][] = [];
    for (let k = 0; k < segs.length; k++) {
      const s = segs[k];
      const end = k + 1 < segs.length ? res.starts[k + 1] : rawLen;
      if (s.y !== undefined) yKnots.push([end, s.y]);
      wKnots.push([(res.starts[k] + end) * 0.5, s.hw ?? def.halfWidth]);
    }
    if (yKnots[yKnots.length - 1][0] < rawLen - 1e-6) yKnots.push([rawLen, def.y0]);
    else yKnots[yKnots.length - 1][1] = def.y0;
    const evalKnots = (kn: [number, number][], d: number, ring: boolean): number => {
      if (ring) {
        // wrap-around interpolation between the last and first knot
        const firstK = kn[0];
        const lastK = kn[kn.length - 1];
        if (d < firstK[0]) {
          const span = firstK[0] + rawLen - lastK[0];
          return lerp(lastK[1], firstK[1], (d + rawLen - lastK[0]) / span);
        }
        if (d >= lastK[0]) {
          const span = firstK[0] + rawLen - lastK[0];
          return lerp(lastK[1], firstK[1], (d - lastK[0]) / span);
        }
      }
      for (let i = 0; i < kn.length - 1; i++) {
        if (d >= kn[i][0] && d <= kn[i + 1][0]) {
          const span = kn[i + 1][0] - kn[i][0];
          return span < 1e-9 ? kn[i + 1][1] : lerp(kn[i][1], kn[i + 1][1], (d - kn[i][0]) / span);
        }
      }
      return kn[kn.length - 1][1];
    };

    const nd = dx.length;
    const X = new Float64Array(dx);
    const Z = new Float64Array(dz);
    const Y = new Float64Array(nd);
    const W = new Float64Array(nd);
    const D = new Float64Array(ddist);
    for (let i = 0; i < nd; i++) {
      Y[i] = evalKnots(yKnots, ddist[i], false);
      W[i] = evalKnots(wKnots, ddist[i], true);
    }

    // -- 3. uniform dense resample, then smooth into transition curves
    const m0 = Math.round(rawLen / H);
    const r0 = resampleClosed(X, Z, [Y, W, D], m0);
    smoothRing(r0.xs, 14, 3);
    smoothRing(r0.zs, 14, 3);
    smoothRing(r0.chans[0], 56, 3);
    smoothRing(r0.chans[1], 40, 3);

    // -- 4. final resample
    const cumLen = (() => {
      let L = 0;
      for (let i = 0; i < m0; i++) {
        const j = (i + 1) % m0;
        L += Math.hypot(r0.xs[j] - r0.xs[i], r0.zs[j] - r0.zs[i]);
      }
      return L;
    })();
    const n = Math.round(cumLen / 1.5);
    const r1 = resampleClosed(r0.xs, r0.zs, [r0.chans[0], r0.chans[1], r0.chans[2]], n);
    this.n = n;
    this.length = r1.total;
    this.ds = r1.total / n;

    // centre the layout on the origin
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      minX = Math.min(minX, r1.xs[i]);
      maxX = Math.max(maxX, r1.xs[i]);
      minZ = Math.min(minZ, r1.zs[i]);
      maxZ = Math.max(maxZ, r1.zs[i]);
    }
    const ox = (minX + maxX) * 0.5;
    const oz = (minZ + maxZ) * 0.5;
    this.minX = minX - ox;
    this.maxX = maxX - ox;
    this.minZ = minZ - oz;
    this.maxZ = maxZ - oz;

    this.cx = new Float32Array(n);
    this.cy = new Float32Array(n);
    this.cz = new Float32Array(n);
    this.hw = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this.cx[i] = r1.xs[i] - ox;
      this.cz[i] = r1.zs[i] - oz;
      this.cy[i] = r1.chans[0][i];
      this.hw[i] = r1.chans[1][i];
    }
    // segment start distances in final lap coordinates
    const scale = this.length / rawLen;
    this.segS = res.starts.map((s) => s * scale);
    this.segTag = segs.map((s) => s.tag);

    // -- 5. frames
    this.tx = new Float32Array(n);
    this.tz = new Float32Array(n);
    this.nx = new Float32Array(n);
    this.nz = new Float32Array(n);
    this.psi = new Float32Array(n);
    this.kappa = new Float32Array(n);
    this.grade = new Float32Array(n);
    this.slopeL = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = mod(i - 1, n);
      const b = (i + 1) % n;
      let tx = this.cx[b] - this.cx[a];
      let tz = this.cz[b] - this.cz[a];
      const l = Math.hypot(tx, tz);
      tx /= l;
      tz /= l;
      this.tx[i] = tx;
      this.tz[i] = tz;
      this.nx[i] = tz;
      this.nz[i] = -tx;
      this.psi[i] = Math.atan2(tx, tz);
      this.grade[i] = (this.cy[b] - this.cy[a]) / l;
    }
    for (let i = 0; i < n; i++) {
      const a = mod(i - 1, n);
      const b = (i + 1) % n;
      let dpsi = this.psi[b] - this.psi[a];
      while (dpsi > Math.PI) dpsi -= 2 * Math.PI;
      while (dpsi < -Math.PI) dpsi += 2 * Math.PI;
      this.kappa[i] = dpsi / (2 * this.ds);
    }
    smoothRingF32(this.kappa, 2, 2);
    // banking follows a smoothed curvature
    const bank = new Float32Array(this.kappa);
    smoothRingF32(bank, 8, 2);
    for (let i = 0; i < n; i++) {
      const phi = clamp(bank[i] * def.bankGain, -def.bankMax, def.bankMax);
      this.slopeL[i] = -Math.tan(phi);
    }

    this.buildEdges();
    this.buildRacingLine();
  }

  /** Walls, kerbs, gravel traps derived from curvature. */
  private buildEdges(): void {
    const n = this.n;
    const ds = this.ds;
    this.wallL = new Float32Array(n);
    this.wallR = new Float32Array(n);
    this.kerbL = new Float32Array(n);
    this.kerbR = new Float32Array(n);
    this.gravelL = new Float32Array(n);
    this.gravelR = new Float32Array(n);

    const ahead = Math.round(70 / ds);
    const behind = Math.round(45 / ds);
    const outL = new Float32Array(n); // run-off demand on the left (outside of right-handers)
    const outR = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let mL = 0;
      let mR = 0;
      for (let k = -behind; k <= ahead; k += 2) {
        const kk = this.kappa[mod(i + k, n)];
        // weight corners ahead more than corners behind
        const wgt = k >= 0 ? 1 : 0.75;
        const sev = smoothstep(1 / 170, 1 / 42, Math.abs(kk)) * wgt;
        if (kk > 0) mR = Math.max(mR, sev);
        else mL = Math.max(mL, sev);
      }
      outL[i] = mL;
      outR[i] = mR;
    }
    smoothRingF32(outL, Math.round(14 / ds), 3);
    smoothRingF32(outR, Math.round(14 / ds), 3);

    const kSm = new Float32Array(this.kappa);
    smoothRingF32(kSm, Math.round(9 / ds), 2);
    for (let i = 0; i < n; i++) {
      const hw = this.hw[i];
      const k = kSm[i];
      const inner = 3.4;
      const base = 3.6;
      const runL = base + outL[i] * 12.5;
      const runR = base + outR[i] * 12.5;
      let wl = hw + (k > 0.004 ? lerp(runL, inner, smoothstep(0.004, 0.012, k)) : runL);
      let wr = hw + (k < -0.004 ? lerp(runR, inner, smoothstep(0.004, 0.012, -k)) : runR);
      // never let the inside wall cross the centre of curvature
      if (k > 1e-4) wl = Math.min(wl, 1 / k - 5);
      if (k < -1e-4) wr = Math.min(wr, -1 / k - 5);
      this.wallL[i] = wl;
      this.wallR[i] = wr;
    }
    smoothRingF32(this.wallL, Math.round(8 / ds), 2);
    smoothRingF32(this.wallR, Math.round(8 / ds), 2);
    for (let i = 0; i < n; i++) {
      this.wallL[i] = Math.max(this.wallL[i], this.hw[i] + 2.4);
      this.wallR[i] = Math.max(this.wallR[i], this.hw[i] + 2.4);
    }

    // kerbs: inside of corners + outside on entry/exit
    const exitSpan = Math.round(38 / ds);
    for (let i = 0; i < n; i++) {
      const k = kSm[i];
      const inside = smoothstep(0.0045, 0.008, Math.abs(k));
      let outside = 0;
      let outSign = k >= 0 ? 1 : -1;
      for (let q = -exitSpan; q <= exitSpan; q += 3) {
        const kk = kSm[mod(i + q, n)];
        const o = smoothstep(0.009, 0.016, Math.abs(kk)) * (1 - Math.abs(q) / (exitSpan + 1));
        if (o > outside) {
          outside = o;
          outSign = kk >= 0 ? 1 : -1;
        }
      }
      const kOut = Math.abs(k) < 0.02 ? smoothstep(0.25, 0.6, outside) : 0;
      // inside kerb on the side of the local turn, outside kerb opposite the dominant nearby corner
      if (k > 0) this.kerbL[i] = inside;
      else this.kerbR[i] = inside;
      if (outSign > 0) this.kerbR[i] = Math.max(this.kerbR[i], kOut);
      else this.kerbL[i] = Math.max(this.kerbL[i], kOut);
    }
    smoothRingF32(this.kerbL, 2, 1);
    smoothRingF32(this.kerbR, 2, 1);
    // gravel traps on generous run-offs
    for (let i = 0; i < n; i++) {
      this.gravelL[i] = this.wallL[i] - this.hw[i] > 8.5 ? this.hw[i] + 3.2 : 0;
      this.gravelR[i] = this.wallR[i] - this.hw[i] > 8.5 ? this.hw[i] + 3.2 : 0;
    }
  }

  /** Minimum-curvature racing line (coarse relaxation, then upsample). */
  private buildRacingLine(): void {
    const n = this.n;
    const step = 4;
    const m = Math.floor(n / step);
    const off = new Float64Array(m);
    const px = new Float64Array(m);
    const pz = new Float64Array(m);
    const margin = 1.75;
    const idx = (k: number) => mod(k, m) * step;
    for (let iter = 0; iter < 900; iter++) {
      for (let k = 0; k < m; k++) {
        const i = idx(k);
        px[k] = this.cx[i] + this.nx[i] * off[k];
        pz[k] = this.cz[i] + this.nz[i] * off[k];
      }
      for (let k = 0; k < m; k++) {
        const a = mod(k - 1, m);
        const b = (k + 1) % m;
        const i = idx(k);
        const mx = (px[a] + px[b]) * 0.5 - px[k];
        const mz = (pz[a] + pz[b]) * 0.5 - pz[k];
        const lim = this.hw[i] - margin;
        off[k] = clamp(off[k] + (mx * this.nx[i] + mz * this.nz[i]) * 0.9, -lim, lim);
      }
    }
    this.line = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const f = i / step;
      const k0 = Math.floor(f);
      const t = f - k0;
      this.line[i] = lerp(off[mod(k0, m)], off[mod(k0 + 1, m)], t);
    }
    smoothRingF32(this.line, 4, 2);
    // curvature of the line (three-point, spaced to suppress noise)
    this.lineKappa = new Float32Array(n);
    const sp = 4;
    for (let i = 0; i < n; i++) {
      const a = mod(i - sp, n);
      const b = (i + sp) % n;
      const ax = this.cx[a] + this.nx[a] * this.line[a];
      const az = this.cz[a] + this.nz[a] * this.line[a];
      const mx = this.cx[i] + this.nx[i] * this.line[i];
      const mz = this.cz[i] + this.nz[i] * this.line[i];
      const bx = this.cx[b] + this.nx[b] * this.line[b];
      const bz = this.cz[b] + this.nz[b] * this.line[b];
      const ux = mx - ax;
      const uz = mz - az;
      const vx = bx - mx;
      const vz = bz - mz;
      const cross = uz * vx - ux * vz; // + = left turn (psi increasing)
      const lu = Math.hypot(ux, uz);
      const lv = Math.hypot(vx, vz);
      const lw = Math.hypot(bx - ax, bz - az);
      this.lineKappa[i] = (2 * cross) / (lu * lv * lw + 1e-9);
    }
    smoothRingF32(this.lineKappa, 3, 2);
  }

  /** Reference speed (m/s) along the racing line for a car with the given capabilities. */
  buildSpeedProfile(p: SpeedProfileParams): Float32Array {
    const n = this.n;
    const ds = this.ds;
    const g = 9.81;
    const v = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const k = Math.abs(this.lineKappa[i]);
      const denom = k - (p.muLat * p.downforceK) / p.mass;
      let vv = p.vTop;
      if (denom > 1e-5) vv = Math.min(p.vTop, Math.sqrt((p.muLat * g) / denom));
      v[i] = vv;
    }
    for (let pass = 0; pass < 2; pass++) {
      // braking (backward)
      for (let q = 2 * n - 1; q >= 0; q--) {
        const i = q % n;
        const j = (i + 1) % n;
        const vj = v[j];
        const aBrake = p.muBrake * (g + (p.downforceK * vj * vj) / p.mass) + (p.dragK * vj * vj) / p.mass + g * this.grade[i];
        const lim = Math.sqrt(vj * vj + 2 * Math.max(2, aBrake) * ds);
        if (v[i] > lim) v[i] = lim;
      }
      // acceleration (forward)
      for (let q = 0; q < 2 * n; q++) {
        const i = q % n;
        const j = (i + 1) % n;
        const vi = Math.max(v[i], 3);
        const aAcc = Math.min(p.tractionG, p.powerW / (p.mass * vi)) - (p.dragK * vi * vi) / p.mass - g * this.grade[i];
        const lim = Math.sqrt(vi * vi + 2 * Math.max(0.2, aAcc) * ds);
        if (v[j] > lim) v[j] = lim;
      }
    }
    return v;
  }

  // ---------------------------------------------------------------- queries
  /** Nearest centreline segment near `hint`. Results in oi/ot/od. */
  locate(x: number, z: number, hint: number, window = 10): void {
    const n = this.n;
    const cx = this.cx;
    const cz = this.cz;
    let best = Infinity;
    let bi = 0;
    let bt = 0;
    let bk = 0;
    for (let k = -window; k <= window; k++) {
      let i = hint + k;
      if (i < 0) i += n;
      else if (i >= n) i -= n;
      const j = i + 1 === n ? 0 : i + 1;
      const ax = cx[i];
      const az = cz[i];
      const sx = cx[j] - ax;
      const sz = cz[j] - az;
      let t = ((x - ax) * sx + (z - az) * sz) / (sx * sx + sz * sz);
      if (t < 0) t = 0;
      else if (t > 1) t = 1;
      const ex = x - (ax + sx * t);
      const ez = z - (az + sz * t);
      const d2 = ex * ex + ez * ez;
      if (d2 < best) {
        best = d2;
        bi = i;
        bt = t;
        bk = k;
      }
    }
    if ((bk === -window || bk === window) && window < n / 2) {
      // hint was stale; widen the search
      this.locate(x, z, bi, Math.min(window * 4, Math.floor(n / 2)));
      return;
    }
    const j = bi + 1 === n ? 0 : bi + 1;
    const sx = cx[j] - cx[bi];
    const sz = cz[j] - cz[bi];
    const side = (x - cx[bi]) * sz - (z - cz[bi]) * sx; // dot with left normal (sz, -sx)
    this.oi = bi;
    this.ot = bt;
    this.od = side >= 0 ? Math.sqrt(best) : -Math.sqrt(best);
  }

  locateGlobal(x: number, z: number): void {
    this.locate(x, z, 0, Math.floor(this.n / 2));
  }

  /** Ground height, normal and surface at world (x, z). Results in oy/onx/ony/onz/osurf (+ oi/ot/od). */
  ground(x: number, z: number, hint: number): void {
    this.locate(x, z, hint);
    this.groundAt(this.oi, this.ot, this.od);
  }

  /** Same as ground() but from track coordinates. */
  groundAt(i: number, t: number, d: number): void {
    const n = this.n;
    const j = i + 1 === n ? 0 : i + 1;
    const hw = this.hw[i] + (this.hw[j] - this.hw[i]) * t;
    const yc = this.cy[i] + (this.cy[j] - this.cy[i]) * t;
    const sl = this.slopeL[i] + (this.slopeL[j] - this.slopeL[i]) * t;
    const gr = this.grade[i] + (this.grade[j] - this.grade[i]) * t;
    const ad = d < 0 ? -d : d;
    const sgn = d < 0 ? -1 : 1;
    const edge = hw + EDGE_BLEND;
    let y: number;
    let slopeD: number;
    if (ad <= edge) {
      y = yc + sl * d;
      slopeD = sl;
    } else {
      y = yc + sl * edge * sgn + (ad - edge) * VERGE_SLOPE;
      slopeD = VERGE_SLOPE * sgn;
    }
    let surf = SURF_ASPHALT;
    if (ad > hw) {
      const kerb = d > 0 ? this.kerbL[i] + (this.kerbL[j] - this.kerbL[i]) * t : this.kerbR[i] + (this.kerbR[j] - this.kerbR[i]) * t;
      const grav = d > 0 ? this.gravelL[i] : this.gravelR[i];
      const wall = d > 0 ? this.wallL[i] : this.wallR[i];
      const over = ad - hw;
      if (kerb > 0.5 && over <= KERB_W) {
        surf = SURF_KERB;
        const ramp = Math.min(over / 0.3, (KERB_W - over) / 0.25, 1);
        y += KERB_H * (ramp < 0 ? 0 : ramp);
      } else if (grav > 0 && ad >= grav && ad <= wall - 0.5) {
        surf = SURF_GRAVEL;
      } else {
        surf = SURF_GRASS;
      }
    }
    const tx = this.tx[i] + (this.tx[j] - this.tx[i]) * t;
    const tz = this.tz[i] + (this.tz[j] - this.tz[i]) * t;
    // left normal (tz, -tx)
    let nx = -gr * tx - slopeD * tz;
    let nz = -gr * tz + slopeD * tx;
    const inv = 1 / Math.sqrt(nx * nx + 1 + nz * nz);
    nx *= inv;
    nz *= inv;
    this.oi = i;
    this.ot = t;
    this.od = d;
    this.oy = y;
    this.onx = nx;
    this.ony = inv;
    this.onz = nz;
    this.osurf = surf;
  }

  /** World position on the surface at lap distance s (m) and lateral offset d (m, + left). */
  posAt(s: number, d: number, out: { x: number; y: number; z: number; psi: number }): void {
    const n = this.n;
    const f = mod(s, this.length) / this.ds;
    let i = Math.floor(f);
    if (i >= n) i = n - 1;
    const t = f - i;
    const j = i + 1 === n ? 0 : i + 1;
    const nx = this.nx[i] + (this.nx[j] - this.nx[i]) * t;
    const nz = this.nz[i] + (this.nz[j] - this.nz[i]) * t;
    const il = 1 / Math.hypot(nx, nz);
    out.x = this.cx[i] + (this.cx[j] - this.cx[i]) * t + nx * il * d;
    out.z = this.cz[i] + (this.cz[j] - this.cz[i]) * t + nz * il * d;
    this.groundAt(i, t, d);
    out.y = this.oy;
    const tx = this.tx[i] + (this.tx[j] - this.tx[i]) * t;
    const tz = this.tz[i] + (this.tz[j] - this.tz[i]) * t;
    out.psi = Math.atan2(tx, tz);
  }

  /** Linear sample of any per-sample array at lap distance s. */
  sample(arr: Float32Array, s: number): number {
    const n = this.n;
    const f = mod(s, this.length) / this.ds;
    let i = Math.floor(f);
    if (i >= n) i = n - 1;
    const j = i + 1 === n ? 0 : i + 1;
    return arr[i] + (arr[j] - arr[i]) * (f - i);
  }

  indexAt(s: number): number {
    return Math.min(this.n - 1, Math.floor(mod(s, this.length) / this.ds));
  }
}
