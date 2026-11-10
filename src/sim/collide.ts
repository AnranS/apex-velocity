// Car vs car collision: 2D oriented boxes on the ground plane (cars are upright while racing),
// separating-axis test, single contact point, impulse with friction acting on linear velocity and yaw.
import type { Car } from './car';

const RESTITUTION = 0.22;
const FRICTION = 0.28;

/** Resolves a collision between two cars. Returns the normal impulse (N·s), 0 if none. */
export function collideCars(a: Car, b: Car, out: { x: number; y: number; z: number }): number {
  const ddx0 = b.px - a.px;
  const ddz0 = b.pz - a.pz;
  if (ddx0 * ddx0 + ddz0 * ddz0 > 42) return 0;
  if (Math.abs(b.py - a.py) > 1.2) return 0;
  if (a.ghost > 0 || b.ghost > 0) return 0;

  let afx = a.m02;
  let afz = a.m22;
  let l = Math.hypot(afx, afz);
  if (l < 0.3) return 0;
  afx /= l;
  afz /= l;
  let bfx = b.m02;
  let bfz = b.m22;
  l = Math.hypot(bfx, bfz);
  if (l < 0.3) return 0;
  bfx /= l;
  bfz /= l;
  const alx = afz;
  const alz = -afx;
  const blx = bfz;
  const blz = -bfx;
  const as = a.spec;
  const bs = b.spec;
  const aoff = (as.bodyFrontZ + as.bodyRearZ) * 0.5 - as.comZ;
  const boff = (bs.bodyFrontZ + bs.bodyRearZ) * 0.5 - bs.comZ;
  const acx = a.px + afx * aoff;
  const acz = a.pz + afz * aoff;
  const bcx = b.px + bfx * boff;
  const bcz = b.pz + bfz * boff;
  const ahl = (as.bodyFrontZ - as.bodyRearZ) * 0.5;
  const bhl = (bs.bodyFrontZ - bs.bodyRearZ) * 0.5;
  const ahw = as.bodyHalfWidth;
  const bhw = bs.bodyHalfWidth;
  const ddx = bcx - acx;
  const ddz = bcz - acz;

  let minOv = Infinity;
  let nx = 0;
  let nz = 0;
  for (let k = 0; k < 4; k++) {
    const ex = k === 0 ? afx : k === 1 ? alx : k === 2 ? bfx : blx;
    const ez = k === 0 ? afz : k === 1 ? alz : k === 2 ? bfz : blz;
    const ra = ahl * Math.abs(ex * afx + ez * afz) + ahw * Math.abs(ex * alx + ez * alz);
    const rb = bhl * Math.abs(ex * bfx + ez * bfz) + bhw * Math.abs(ex * blx + ez * blz);
    const dist = ddx * ex + ddz * ez;
    const ov = ra + rb - Math.abs(dist);
    if (ov <= 0) return 0;
    if (ov < minOv) {
      minOv = ov;
      const sg = dist >= 0 ? 1 : -1;
      nx = ex * sg;
      nz = ez * sg;
    }
  }

  // contact point: average of the corners of each box that lie inside the other
  let cx = 0;
  let cz = 0;
  let cnt = 0;
  for (let k = 0; k < 4; k++) {
    const su = k < 2 ? 1 : -1;
    const sv = k % 2 === 0 ? 1 : -1;
    // corner of B in A
    let x = bcx + bfx * bhl * su + blx * bhw * sv;
    let z = bcz + bfz * bhl * su + blz * bhw * sv;
    let u = (x - acx) * afx + (z - acz) * afz;
    let v = (x - acx) * alx + (z - acz) * alz;
    if (Math.abs(u) <= ahl + 0.02 && Math.abs(v) <= ahw + 0.02) {
      cx += x;
      cz += z;
      cnt++;
    }
    // corner of A in B
    x = acx + afx * ahl * su + alx * ahw * sv;
    z = acz + afz * ahl * su + alz * ahw * sv;
    u = (x - bcx) * bfx + (z - bcz) * bfz;
    v = (x - bcx) * blx + (z - bcz) * blz;
    if (Math.abs(u) <= bhl + 0.02 && Math.abs(v) <= bhw + 0.02) {
      cx += x;
      cz += z;
      cnt++;
    }
  }
  if (cnt > 0) {
    cx /= cnt;
    cz /= cnt;
  } else {
    cx = (acx + bcx) * 0.5;
    cz = (acz + bcz) * 0.5;
  }
  out.x = cx;
  out.y = (a.py + b.py) * 0.5;
  out.z = cz;

  // positional correction (split by mass)
  const ma = as.mass;
  const mb = bs.mass;
  const corr = Math.min(minOv, 0.5) * 0.85;
  const wa = mb / (ma + mb);
  const wb = ma / (ma + mb);
  a.px -= nx * corr * wa;
  a.pz -= nz * corr * wa;
  b.px += nx * corr * wb;
  b.pz += nz * corr * wb;

  // impulse (linear + yaw)
  const rax = cx - a.px;
  const raz = cz - a.pz;
  const rbx = cx - b.px;
  const rbz = cz - b.pz;
  const vax = a.vx + a.wy * raz;
  const vaz = a.vz - a.wy * rax;
  const vbx = b.vx + b.wy * rbz;
  const vbz = b.vz - b.wy * rbx;
  const rvx = vax - vbx;
  const rvz = vaz - vbz;
  const vn = rvx * nx + rvz * nz; // > 0 = closing (n points from a to b)
  if (vn <= 0) return 0;
  const ran = raz * nx - rax * nz;
  const rbn = rbz * nx - rbx * nz;
  const kn = a.invMass + b.invMass + ran * ran * a.invIy + rbn * rbn * b.invIy;
  const jn = ((1 + RESTITUTION) * vn) / kn;
  a.vx -= nx * jn * a.invMass;
  a.vz -= nz * jn * a.invMass;
  a.wy -= ran * jn * a.invIy;
  b.vx += nx * jn * b.invMass;
  b.vz += nz * jn * b.invMass;
  b.wy += rbn * jn * b.invIy;

  // friction along the tangent
  const tx = -nz;
  const tz = nx;
  const vt = rvx * tx + rvz * tz;
  const rat = raz * tx - rax * tz;
  const rbt = rbz * tx - rbx * tz;
  const kt = a.invMass + b.invMass + rat * rat * a.invIy + rbt * rbt * b.invIy;
  let jt = vt / kt;
  const lim = FRICTION * jn;
  if (jt > lim) jt = lim;
  else if (jt < -lim) jt = -lim;
  a.vx -= tx * jt * a.invMass;
  a.vz -= tz * jt * a.invMass;
  a.wy -= rat * jt * a.invIy;
  b.vx += tx * jt * b.invMass;
  b.vz += tz * jt * b.invMass;
  b.wy += rbt * jt * b.invIy;
  return jn;
}
