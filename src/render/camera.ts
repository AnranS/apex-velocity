// Race cameras: spring-damped chase cams, hood/bumper cams, speed-sensitive FOV and shake.
import * as THREE from 'three';
import type { Car } from '../sim/car';
import { clamp, damp, lerp, smoothstep } from '../sim/math';
import type { Track } from '../sim/track';
import type { CarVisual } from './car_visual';

export type CamMode = 'chase' | 'far' | 'hood' | 'bumper';
export const CAM_MODES: CamMode[] = ['chase', 'far', 'hood', 'bumper'];

const _fwd = new THREE.Vector3();
const _up = new THREE.Vector3();
const _pos = new THREE.Vector3();
const _look = new THREE.Vector3();
const _tmp = new THREE.Vector3();
const _q = new THREE.Quaternion();

export class RaceCamera {
  readonly camera: THREE.PerspectiveCamera;
  mode: CamMode = 'chase';
  private yaw = 0;
  private pitch = 0;
  private dist = 6;
  private height = 1.6;
  private fov = 62;
  private shake = 0;
  private shakeT = 0;
  private lookSide = 0;
  private inited = false;
  /** 0..1 punch added by boost/impacts */
  kick = 0;
  /** set when the camera jumped (teleport/mode change) so motion blur can skip a frame */
  cut = false;

  constructor(camera: THREE.PerspectiveCamera) {
    this.camera = camera;
  }

  nextMode(): void {
    this.mode = CAM_MODES[(CAM_MODES.indexOf(this.mode) + 1) % CAM_MODES.length];
    this.cut = true;
  }

  addShake(amount: number): void {
    this.shake = Math.min(1.4, this.shake + amount);
  }

  snap(): void {
    this.inited = false;
    this.cut = true;
  }

  update(dt: number, car: Car, vis: CarVisual, track: Track, lookBack: boolean): boolean {
    const cam = this.camera;
    const root = vis.root;
    _q.copy(root.quaternion);
    _fwd.set(0, 0, 1).applyQuaternion(_q);
    _up.set(0, 1, 0).applyQuaternion(_q);
    const speed = car.speed;
    const sp01 = clamp(speed / 85, 0, 1);

    // heading the camera wants to sit behind: blend of body heading and velocity direction (shows drift angle)
    const headYaw = Math.atan2(_fwd.x, _fwd.z);
    let wantYaw = headYaw;
    if (speed > 4) {
      const velYaw = Math.atan2(car.vx, car.vz);
      let dv = velYaw - headYaw;
      while (dv > Math.PI) dv -= 2 * Math.PI;
      while (dv < -Math.PI) dv += 2 * Math.PI;
      if (Math.abs(dv) < 1.6) wantYaw = headYaw + dv * 0.55 * smoothstep(4, 16, speed);
    }
    const wantPitch = Math.asin(clamp(_fwd.y, -0.5, 0.5));
    if (!this.inited) {
      this.yaw = wantYaw;
      this.pitch = wantPitch;
      this.inited = true;
    }
    let dy = wantYaw - this.yaw;
    while (dy > Math.PI) dy -= 2 * Math.PI;
    while (dy < -Math.PI) dy += 2 * Math.PI;
    this.yaw += dy * (1 - Math.exp(-dt * lerp(5.5, 8.5, sp01)));
    this.pitch = damp(this.pitch, wantPitch, 4, dt);
    this.kick = damp(this.kick, 0, 2.2, dt);
    this.lookSide = damp(this.lookSide, lookBack ? 1 : 0, 14, dt);

    let targetFov = 60;
    const shakeAmt = this.shake + sp01 * sp01 * 0.1 + (car.surfaceMix > 0.5 ? 0.25 * sp01 : 0);
    this.shake = damp(this.shake, 0, 5, dt);
    this.shakeT += dt * (22 + speed * 0.5);

    if (this.mode === 'chase' || this.mode === 'far') {
      const far = this.mode === 'far';
      const baseDist = far ? 8.2 : 5.5;
      const baseH = far ? 2.5 : 1.55;
      // the car pulls away slightly under acceleration and with speed
      const wantDist = baseDist + sp01 * 1.5 + clamp(car.longG, -1.2, 1.2) * 0.55 + this.kick * 1.2;
      this.dist = damp(this.dist, wantDist, 5, dt);
      this.height = damp(this.height, baseH - sp01 * 0.22, 4, dt);
      const yaw = this.yaw + this.lookSide * Math.PI;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const tx = root.position.x;
      const ty = root.position.y;
      const tz = root.position.z;
      const cp = Math.cos(this.pitch * 0.6);
      const spch = Math.sin(this.pitch * 0.6);
      _pos.set(tx - sy * this.dist * cp, ty + this.height - spch * this.dist, tz - cy * this.dist * cp);
      // keep above the ground
      track.ground(_pos.x, _pos.z, car.trkIdx);
      if (_pos.y < track.oy + 0.5) _pos.y = track.oy + 0.5;
      _look.set(tx + sy * (3.2 + sp01 * 4), ty + 0.95 + spch * 2, tz + cy * (3.2 + sp01 * 4));
      targetFov = (far ? 58 : 62) + sp01 * 15 + this.kick * 7;
      cam.position.copy(_pos);
      cam.up.set(0, 1, 0);
      cam.lookAt(_look);
      // slight roll into corners
      cam.rotateZ(clamp(-car.latG * 0.012, -0.05, 0.05));
    } else {
      const p = this.mode === 'hood' ? vis.points.camHood ?? [0, 1.02, 0.35] : vis.points.camBumper ?? [0, 0.55, car.spec.bodyFrontZ - 0.1];
      _tmp.set(p[0], p[1] + (this.mode === 'hood' ? 0.06 : 0.05), p[2]).applyQuaternion(_q).add(root.position);
      cam.position.copy(_tmp);
      cam.quaternion.copy(_q);
      // model forward is +Z, cameras look down -Z
      cam.rotateY(Math.PI + this.lookSide * Math.PI);
      targetFov = (this.mode === 'hood' ? 68 : 74) + sp01 * 10 + this.kick * 6;
    }
    if (shakeAmt > 0.001) {
      const t = this.shakeT;
      cam.position.x += (Math.sin(t * 1.31) + Math.sin(t * 2.73 + 1.3)) * 0.012 * shakeAmt;
      cam.position.y += (Math.sin(t * 1.77 + 0.6) + Math.sin(t * 3.1)) * 0.014 * shakeAmt;
      cam.rotateZ(Math.sin(t * 0.93) * 0.0025 * shakeAmt);
    }
    this.fov = damp(this.fov, targetFov, 4.5, dt);
    let changed = false;
    if (Math.abs(cam.fov - this.fov) > 0.02) {
      cam.fov = this.fov;
      changed = true;
    }
    return changed;
  }
}
