// Game orchestration: engine + persistent track world, menu showroom, race lifecycle,
// per-frame simulation/render loop, and the glue to audio and effects.
//
// UI code talks to the game through:
//   game.mode, game.race, settings-driven methods (enterMenu / startRace / restartRace / quitToMenu / setPaused)
//   game.onFrame(fn), game.onRaceEvent(fn), game.onMode(fn), game.onLoading(fn)
import * as THREE from 'three';
import { audio, CarVoice } from '../audio';
import { Car, makeInput } from '../sim/car';
import { CAR_SPECS, CarSpec, applyModelMeta } from '../sim/car_specs';
import { clamp, damp } from '../sim/math';
import { RaceConfig, RaceEvent, RaceSim } from '../sim/race';
import { Track } from '../sim/track';
import { TRACKS } from '../sim/tracks';
import { RaceCamera } from '../render/camera';
import { CarVisual, loadCarMeta } from '../render/car_visual';
import { Engine, QualityName } from '../render/engine';
import { Fx } from '../render/fx';
import { Scenery } from '../render/scenery';
import { Sky, WEATHER, WeatherId } from '../render/sky';
import { World } from '../render/world';
import { Input } from './input';
import { recordLap, saveSettings, settings } from './settings';

export const SIM_DT = 1 / 240;

export type GameMode = 'boot' | 'menu' | 'race';

export interface RaceOptions {
  trackId: string;
  weather: WeatherId;
  car: string;
  paint: string;
  laps: number;
  aiCount: number;
  difficulty: number;
  mode: 'race' | 'timetrial';
  autopilot?: boolean;
}

export interface LoadingState {
  active: boolean;
  label: string;
  progress: number;
}

export class Game {
  readonly engine: Engine;
  readonly sky: Sky;
  readonly input: Input;
  readonly cam: RaceCamera;
  mode: GameMode = 'boot';
  track: Track | null = null;
  world: World | null = null;
  scenery: Scenery | null = null;
  fx: Fx | null = null;
  race: RaceSim | null = null;
  visuals: CarVisual[] = [];
  voices: CarVoice[] = [];
  options: RaceOptions | null = null;
  paused = false;
  /** smoothed frames per second / frame time */
  fps = 60;
  frameMs = 16;
  /** seconds since the race scene became visible (intro camera) */
  raceClock = 0;
  /** menu orbit camera (UI may write yaw/pitch/dist; `auto` resumes the slow spin) */
  readonly menuCam = { yaw: 0.6, pitch: 0.16, dist: 6.6, auto: true, height: 0.7 };
  readonly loading: LoadingState = { active: false, label: '', progress: 0 };

  private acc = 0;
  private last = 0;
  private specs: Record<string, CarSpec> | null = null;
  private showCar: Car | null = null;
  private showVis: CarVisual | null = null;
  private showToken = 0;
  private frameFns: ((dt: number) => void)[] = [];
  private eventFns: ((e: RaceEvent) => void)[] = [];
  private modeFns: ((m: GameMode) => void)[] = [];
  private loadFns: ((s: LoadingState) => void)[] = [];
  private half = new THREE.Vector3();
  private inv = new THREE.Matrix4();
  private shift = new THREE.Matrix4();
  private spheres: { x: number; y: number; z: number; r: number }[] = [];
  private fwdTmp = new THREE.Vector3();
  private upTmp = new THREE.Vector3();
  private sunColor = new THREE.Color();
  private musicIntensity = 0;
  /** max milliseconds spent per frame section since last reset */
  readonly prof = { game: 0, scenery: 0, ui: 0, render: 0, sim: 0, fx: 0, audio: 0 };
  private audioDead = false;
  private drsClock = 0;
  private drsCheck = 0;
  private drsStrikes = 0;
  private drsFrames = 0;
  private finishOrbit = 0;
  private audioState = { rpm: 0, rpmMax: 8000, throttle: 0, speed: 0, nitro: 0, slip: 0, surface: 0, scrape: 0, x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };

  constructor(canvas: HTMLCanvasElement, quality: QualityName) {
    this.engine = new Engine(canvas, quality);
    this.sky = new Sky(this.engine);
    this.input = new Input();
    this.cam = new RaceCamera(this.engine.camera);
    this.cam.mode = settings.camera;
    window.addEventListener('resize', () => this.engine.resize());
    // Ctrl is the nitro key; with W held that is "close tab" on Windows/Linux, so ask before leaving mid-race
    window.addEventListener('beforeunload', (e) => {
      if (this.mode === 'race' && this.race && this.race.state === 'racing' && !this.paused && !this.race.autopilot) {
        e.preventDefault();
        e.returnValue = '';
      }
    });
    // audio needs a user gesture
    let watchdog = false;
    const unlock = () => {
      if (this.audioDead) return;
      audio
        .init()
        .then(() => {
          this.applyAudioSettings();
          if (watchdog) return;
          watchdog = true;
          // no working output device (the audio clock never starts): park the context now, otherwise the
          // browser tears it down ~10 s later with a visible stall in the middle of the race
          setTimeout(() => {
            const ctx = audio.context as AudioContext | null;
            if (ctx && ctx.state === 'running' && ctx.currentTime < 0.05) {
              this.audioDead = true;
              audio.suspend().catch(() => {});
            }
          }, 1600);
        })
        .catch(() => {});
      audio.resume().catch(() => {});
    };
    window.addEventListener('pointerdown', unlock, { passive: true });
    window.addEventListener('keydown', unlock);
    window.addEventListener('touchend', unlock, { passive: true });
    this.applyAudioSettings();
  }

  // ---------------------------------------------------------------- subscriptions
  onFrame(fn: (dt: number) => void): void {
    this.frameFns.push(fn);
  }
  onRaceEvent(fn: (e: RaceEvent) => void): void {
    this.eventFns.push(fn);
  }
  onMode(fn: (m: GameMode) => void): void {
    this.modeFns.push(fn);
  }
  onLoading(fn: (s: LoadingState) => void): void {
    this.loadFns.push(fn);
  }
  private setMode(m: GameMode): void {
    this.mode = m;
    for (const f of this.modeFns) f(m);
  }
  private setLoading(active: boolean, label = '', progress = 0): void {
    this.loading.active = active;
    this.loading.label = label;
    this.loading.progress = progress;
    for (const f of this.loadFns) f(this.loading);
  }
  /** Let the browser paint (loading screen) before a blocking step. */
  private nextPaint(): Promise<void> {
    return new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  }

  // ---------------------------------------------------------------- settings
  applyAudioSettings(): void {
    audio.setVolumes({ master: settings.master, music: settings.music, sfx: settings.sfx, engine: settings.engine });
  }
  /** Apply + persist the current `settings` object (volumes, motion blur, camera). */
  applySettings(): void {
    this.applyAudioSettings();
    if (this.engine.motion) this.engine.motion.shutter = settings.motionBlur ? 0.5 : 0;
    this.cam.mode = settings.camera;
    saveSettings();
  }

  // ---------------------------------------------------------------- world
  private async ensureWorld(trackId: string, weather: WeatherId): Promise<void> {
    if (!this.specs) {
      const specs: Record<string, CarSpec> = {};
      for (const id of Object.keys(CAR_SPECS)) {
        const meta = await loadCarMeta(id);
        specs[id] = meta ? applyModelMeta(CAR_SPECS[id], meta) : CAR_SPECS[id];
      }
      this.specs = specs;
    }
    this.setLoading(true, '加载天空与光照', 0.1);
    await this.sky.load(weather);
    if (this.track && this.track.def.id === trackId) return;
    this.disposeWorld();
    this.setLoading(true, '生成赛道', 0.25);
    await this.nextPaint();
    const track = new Track(TRACKS[trackId]);
    this.track = track;
    const world = new World(this.engine, track);
    world.build();
    this.world = world;
    this.setLoading(true, '布置场景', 0.5);
    await this.nextPaint();
    const scenery = new Scenery(this.engine, world);
    this.scenery = scenery;
    await scenery.build();
    this.fx = new Fx(this.engine);
    this.setLoading(true, '编译着色器', 0.7);
    await this.nextPaint();
    await this.engine.prewarm();
    this.warmupTour();
    this.setLoading(true, '准备车辆', 0.85);
  }

  /**
   * Render a handful of hidden frames from points around the lap while the loading screen is up:
   * this uploads geometry and compiles the shadow/LOD shader variants that would otherwise hitch
   * the first time each part of the circuit comes into view.
   */
  private warmupTour(): void {
    const tr = this.track;
    if (!tr) return;
    const cam = this.engine.camera;
    const pos = cam.position.clone();
    const quat = cam.quaternion.clone();
    const o = { x: 0, y: 0, z: 0, psi: 0 };
    const stops = 12;
    for (let k = 0; k < stops; k++) {
      tr.posAt((k / stops) * tr.length, 0, o);
      cam.position.set(o.x, o.y + 2.4, o.z);
      cam.up.set(0, 1, 0);
      cam.lookAt(o.x + Math.sin(o.psi) * 40, o.y + 1.5, o.z + Math.cos(o.psi) * 40);
      cam.updateMatrixWorld();
      this.scenery?.forceLod(cam.position);
      this.sky.update(cam);
      this.engine.render(1 / 60);
    }
    cam.position.copy(pos);
    cam.quaternion.copy(quat);
    cam.updateMatrixWorld();
    this.scenery?.forceLod(cam.position);
    this.engine.motion?.reset();
  }

  private disposeWorld(): void {
    this.disposeRace();
    this.disposeShowroom();
    this.fx?.dispose();
    this.fx = null;
    this.scenery?.dispose();
    this.scenery = null;
    this.world?.dispose();
    this.world = null;
    this.track = null;
  }

  /** Car spec with geometry from the model metadata (falls back to defaults before the world is loaded). */
  specFor(id: string): CarSpec {
    return this.specs?.[id] ?? CAR_SPECS[id];
  }

  // ---------------------------------------------------------------- menu / showroom
  async enterMenu(): Promise<void> {
    this.disposeRace();
    this.fx?.clear();
    await this.ensureWorld(settings.track, settings.weather);
    await this.setShowroomCar(settings.car, settings.paint);
    this.paused = false;
    audio.setPaused(false);
    this.setLoading(false);
    this.setMode('menu');
    audio.playMusic('menu', 1.5);
    audio.setAmbience({ wind: 0.15, rain: WEATHER[settings.weather].rain, crowd: 0.25, tunnel: 0 });
  }

  /** Swap the car on display (menu). Safe to call repeatedly; the latest request wins. */
  async setShowroomCar(carId: string, paint: string): Promise<void> {
    if (!this.track) return;
    const token = ++this.showToken;
    const spec = this.specFor(carId);
    const car = new Car(spec);
    // parked on the grid, turned toward the camera side
    car.placeOnTrack(this.track, this.track.length - 5.5, 0.6, 0.42);
    const settle = makeInput();
    for (let i = 0; i < 120; i++) {
      car.step(SIM_DT, settle, this.track);
      car.collideWorld(this.track);
      car.integrate(SIM_DT, this.track);
    }
    car.savePrev();
    const vis = new CarVisual(spec, { color: paint });
    await vis.build({ color: paint }, 0);
    if (token !== this.showToken) {
      vis.dispose();
      return;
    }
    this.disposeShowroom();
    this.showCar = car;
    this.showVis = vis;
    this.engine.scene.add(vis.root);
    vis.update(car, 1);
    // studio-style key light from the front three-quarter, high up
    const kl = this.engine.keyLight;
    const r = vis.root;
    const off = this.fwdTmp.set(2.6, 4.2, 4.6).applyQuaternion(r.quaternion);
    kl.position.copy(r.position).add(off);
    kl.target.position.set(r.position.x, r.position.y + 0.5, r.position.z);
    kl.target.updateMatrixWorld();
    kl.intensity = 260 * (0.45 + 0.55 * (1 - this.sky.weather.night * 0.5));
    kl.visible = true;
    await this.engine.prewarm();
  }

  setShowroomPaint(paint: string): void {
    this.showVis?.setPaint(paint);
  }

  /** Change the weather preset shown in the menu (also stored in settings). */
  async setMenuWeather(weather: WeatherId): Promise<void> {
    settings.weather = weather;
    await this.sky.load(weather);
    audio.setAmbience({ rain: WEATHER[weather].rain });
  }

  private disposeShowroom(): void {
    this.engine.keyLight.visible = false;
    this.showVis?.dispose();
    this.showVis = null;
    this.showCar = null;
  }

  // ---------------------------------------------------------------- race lifecycle
  async startRace(opts: RaceOptions): Promise<void> {
    this.setLoading(true, '加载中', 0.05);
    await this.nextPaint();
    this.disposeRace();
    this.disposeShowroom();
    this.options = opts;
    await this.ensureWorld(opts.trackId, opts.weather);
    const track = this.track!;
    const cfg: RaceConfig = {
      mode: opts.mode,
      laps: opts.laps,
      playerCar: opts.car,
      aiCount: opts.aiCount,
      difficulty: opts.difficulty,
      wet: WEATHER[opts.weather].grip,
      seed: (Date.now() & 0xffff) | 1,
      arcade: settings.handling === 'arcade',
      abs: true,
      tcs: 1,
      steerAssist: 1,
      autoGear: settings.autoGear,
      specs: this.specs ?? undefined,
    };
    const race = new RaceSim(track, cfg);
    race.autopilot = !!opts.autopilot;
    this.race = race;
    const builds: Promise<void>[] = [];
    for (let i = 0; i < race.cars.length; i++) {
      const car = race.cars[i];
      const inf = race.info[i];
      const color = inf.isPlayer ? opts.paint : car.spec.paints[inf.paint % car.spec.paints.length];
      const vis = new CarVisual(car.spec, { color });
      builds.push(vis.build({ color }, inf.isPlayer ? 0 : 1));
      this.engine.scene.add(vis.root);
      this.visuals.push(vis);
      this.voices.push(audio.createCar(car.spec.engineType, inf.isPlayer));
    }
    await Promise.all(builds);
    for (let i = 0; i < race.cars.length; i++) this.visuals[i].update(race.cars[i], 1);
    await this.engine.prewarm();
    this.fx?.reset(race.cars.length);
    this.cam.snap();
    this.engine.setDof(null);
    this.acc = 0;
    this.raceClock = 0;
    this.finishOrbit = 0;
    this.paused = false;
    this.musicIntensity = 0;
    audio.setPaused(false);
    audio.playMusic(null, 0.8);
    audio.setAmbience({ wind: 0, rain: WEATHER[opts.weather].rain, crowd: 0.6, tunnel: 0 });
    this.setLoading(false);
    this.setMode('race');
  }

  async restartRace(): Promise<void> {
    if (this.options) await this.startRace(this.options);
  }

  async quitToMenu(): Promise<void> {
    this.setLoading(true, '返回', 0.3);
    await this.nextPaint();
    await this.enterMenu();
  }

  setPaused(p: boolean): void {
    if (this.mode !== 'race') return;
    this.paused = p;
    audio.setPaused(p);
  }

  private disposeRace(): void {
    for (const v of this.visuals) v.dispose();
    this.visuals = [];
    for (const v of this.voices) v.dispose();
    this.voices = [];
    this.race = null;
  }

  // ---------------------------------------------------------------- loop
  start(): void {
    this.last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min(0.1, Math.max(0.0005, (now - this.last) / 1000));
      this.last = now;
      this.frame(dt);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  /** Advance one rendered frame. Exposed so tests can drive the game deterministically. */
  frame(dt: number): void {
    this.fps += (1 / dt - this.fps) * 0.05;
    this.frameMs += (dt * 1000 - this.frameMs) * 0.05;
    const inp = this.input.poll(dt);
    const eng = this.engine;
    const pr = this.prof;
    const t0 = performance.now();
    if (this.mode === 'race' && this.race && this.track) this.frameRace(dt, inp);
    else if (this.mode === 'menu') this.frameMenu(dt);
    const t1 = performance.now();
    this.sky.update(eng.camera);
    this.updateHeadlight();
    if (this.scenery) this.scenery.update(dt, eng.camera.position);
    const t2 = performance.now();
    for (const f of this.frameFns) f(dt);
    const t3 = performance.now();
    this.adaptResolution(dt);
    eng.render(dt);
    const t4 = performance.now();
    // worst-case section times since the last read (diagnostics; see tools/drive.ts)
    pr.game = Math.max(pr.game, t1 - t0);
    pr.scenery = Math.max(pr.scenery, t2 - t1);
    pr.ui = Math.max(pr.ui, t3 - t2);
    pr.render = Math.max(pr.render, t4 - t3);
  }

  /** Dynamic resolution safety net: if the frame rate stays low, render fewer pixels. */
  private adaptResolution(dt: number): void {
    if (this.loading.active || this.paused || this.mode === 'boot') {
      this.drsClock = 0;
      return;
    }
    this.drsClock += dt;
    if (this.drsClock < 9) {
      // let load-time hitches settle first
      this.drsCheck = 0;
      this.drsFrames = 0;
      return;
    }
    this.drsCheck += dt;
    this.drsFrames++;
    if (this.drsCheck < 2) return;
    // average over the window (a single long hitch barely moves it, sustained slowness does)
    const avg = this.drsFrames / this.drsCheck;
    this.drsCheck = 0;
    this.drsFrames = 0;
    const eng = this.engine;
    if (avg < 45 && eng.renderScale > 0.56) {
      if (++this.drsStrikes >= 3) {
        eng.renderScale = Math.max(0.55, eng.renderScale - 0.12);
        eng.resize();
        this.drsStrikes = 0;
        this.drsClock = 6;
      }
    } else this.drsStrikes = 0;
  }

  private frameMenu(dt: number): void {
    const eng = this.engine;
    const mc = this.menuCam;
    const vis = this.showVis;
    if (mc.auto) mc.yaw += dt * 0.11;
    if (vis) {
      const t = vis.root.position;
      const cp = Math.cos(mc.pitch);
      const cam = eng.camera;
      cam.position.set(t.x + Math.sin(mc.yaw) * mc.dist * cp, t.y + mc.height + Math.sin(mc.pitch) * mc.dist, t.z + Math.cos(mc.yaw) * mc.dist * cp);
      cam.up.set(0, 1, 0);
      cam.lookAt(t.x, t.y + mc.height, t.z);
      if (Math.abs(cam.fov - 42) > 0.01) {
        cam.fov = 42;
        eng.cameraChanged();
      }
      eng.setDof(mc.dist, 3.2, 3.5);
      if (eng.motion) eng.motion.speedScale = 0;
      eng.grade.set('uSpeedFx', 0);
      if (this.showCar) vis.update(this.showCar, 1);
    }
    this.scenery?.setStartLights(0);
    this.setListener(0, 0, 0);
  }

  private frameRace(dt: number, inp: ReturnType<Input['poll']>): void {
    const race = this.race!;
    const track = this.track!;
    const eng = this.engine;
    this.raceClock += this.paused ? 0 : dt;
    if (race.state === 'grid' && this.raceClock > 1.4) race.startCountdown();
    const ts0 = performance.now();
    if (!this.paused) {
      this.acc += dt;
      let steps = 0;
      while (this.acc >= SIM_DT && steps < 48) {
        race.step(SIM_DT, inp);
        this.acc -= SIM_DT;
        steps++;
      }
      if (steps === 48) this.acc = 0;
    }
    this.prof.sim = Math.max(this.prof.sim, performance.now() - ts0);
    const alpha = this.paused ? 1 : this.acc / SIM_DT;
    for (let i = 0; i < race.cars.length; i++) this.visuals[i].update(race.cars[i], alpha);
    const p = race.player;
    const pv = this.visuals[race.playerIdx];

    // ---- events -> audio / camera / fx / listeners
    for (const e of race.events) this.handleEvent(e);
    race.events.length = 0;

    // ---- camera
    if (race.state === 'finished' && race.sinceFinish > 1.2) {
      // slow orbit around the player's car after the flag
      this.finishOrbit += dt * 0.35;
      const t = pv.root.position;
      const cam = eng.camera;
      cam.position.set(t.x + Math.sin(this.finishOrbit) * 7.5, t.y + 1.9, t.z + Math.cos(this.finishOrbit) * 7.5);
      cam.up.set(0, 1, 0);
      cam.lookAt(t.x, t.y + 0.7, t.z);
      if (Math.abs(cam.fov - 50) > 0.01) {
        cam.fov = 50;
        eng.cameraChanged();
      }
    } else if (this.cam.update(dt, p, pv, track, this.input.lookBack)) eng.cameraChanged();

    if (eng.motion) {
      if (this.cam.cut) {
        eng.motion.reset();
        this.cam.cut = false;
      }
      pv.root.updateMatrixWorld();
      this.inv.copy(pv.root.matrixWorld).invert();
      const sp = p.spec;
      const lo = 0.13;
      const hi = sp.bodyHeight + 0.25;
      this.half.set(sp.bodyHalfWidth + 0.14, (hi - lo) * 0.5, (sp.bodyFrontZ - sp.bodyRearZ) * 0.5 + 0.15);
      this.shift.makeTranslation(0, -(hi + lo) * 0.5, -(sp.bodyFrontZ + sp.bodyRearZ) * 0.5);
      eng.motion.setProtectBox(this.shift.multiply(this.inv), this.half);
      this.spheres.length = 0;
      for (let i = 0; i < race.cars.length; i++) {
        if (i === race.playerIdx) continue;
        const vr = this.visuals[i].root;
        const r = vr.position;
        if (this.spheres.length >= 8 || r.distanceToSquared(eng.camera.position) > 90 * 90) continue;
        const f = this.fwdTmp.set(0, 0, 1).applyQuaternion(vr.quaternion);
        this.spheres.push({ x: r.x + f.x * 1.05, y: r.y + 0.6, z: r.z + f.z * 1.05, r: 1.55 });
        this.spheres.push({ x: r.x - f.x * 1.05, y: r.y + 0.6, z: r.z - f.z * 1.05, r: 1.55 });
      }
      eng.motion.setSpheres(this.spheres);
      eng.motion.speedScale = this.paused ? 0 : 0.12 + 0.88 * clamp((p.speed - 14) / 46, 0, 1);
    }
    eng.grade.set('uSpeedFx', clamp(p.speed / 90, 0, 1) * (p.nitroOn ? 1.7 : 1));
    if (eng.chroma) eng.chroma.offset.setScalar(0.0004 + (p.nitroOn ? 0.0011 : 0) + clamp(p.speed / 95, 0, 1) * 0.0006);

    // ---- start lights
    if (this.scenery) {
      if (race.state === 'countdown') this.scenery.setStartLights(clamp(Math.floor((3.95 - race.countdown) * 1.7), 0, 5));
      else this.scenery.setStartLights(0, race.state === 'racing' && race.time < 2.5);
    }

    // ---- effects
    const tf0 = performance.now();
    if (this.fx && !this.paused) {
      const w = this.sky.weather;
      this.sunColor.setRGB(w.sunColor[0] * w.sunIntensity, w.sunColor[1] * w.sunIntensity, w.sunColor[2] * w.sunIntensity);
      this.fx.update(dt, race, this.visuals, eng.camera.position, this.sunColor, w.envIntensity * (1 - w.night * 0.75) + 0.08);
    }

    const tf1 = performance.now();
    this.prof.fx = Math.max(this.prof.fx, tf1 - tf0);
    // ---- audio
    this.setListener(p.vx, p.vy, p.vz);
    if (!this.paused) {
      const st = this.audioState;
      for (let i = 0; i < race.cars.length; i++) {
        const c = race.cars[i];
        let slip = 0;
        for (let k = 0; k < 4; k++) slip = Math.max(slip, c.wheels[k].skid);
        st.rpm = c.rpm;
        st.rpmMax = c.spec.redline;
        st.throttle = c.thr;
        st.speed = c.speed;
        st.nitro = c.nitroOn ? 1 : c.miniBoost > 0 ? 0.5 : 0;
        st.slip = slip;
        st.surface = Math.round(c.surfaceMix);
        st.scrape = c.scrape;
        st.x = c.px;
        st.y = c.py;
        st.z = c.pz;
        st.vx = c.vx;
        st.vy = c.vy;
        st.vz = c.vz;
        this.voices[i]?.update(st);
      }
      // music builds with speed and racing position
      const inf = race.playerInfo;
      const target = race.state === 'racing' ? clamp(0.35 + p.speed / 160 + (race.cars.length - inf.position) * 0.03 + (p.nitroOn ? 0.25 : 0), 0, 1) : 0.2;
      this.musicIntensity = damp(this.musicIntensity, target, 0.8, dt);
      audio.setMusicIntensity(this.musicIntensity);
      // crowd swells along the pit straight
      const s = p.s;
      const nearStands = s > track.length - 260 || s < 180 ? 1 : 0.15;
      audio.setAmbience({ wind: clamp(p.speed / 80, 0, 1), crowd: nearStands * 0.7 });
    }
    this.prof.audio = Math.max(this.prof.audio, performance.now() - tf1);
  }

  /** Real spot light for the hero car (player in a race, show car in the menu) once it gets dark. */
  private updateHeadlight(): void {
    const hl = this.engine.headlight;
    const vis = this.mode === 'race' && this.race ? this.visuals[this.race.playerIdx] : this.showVis;
    const night = this.sky.weather.night;
    if (!vis || night < 0.35) {
      // invisible lights are dropped from the shader light count, so daytime pays nothing for them
      hl.visible = false;
      return;
    }
    const root = vis.root;
    const f = this.fwdTmp.set(0, 0, 1).applyQuaternion(root.quaternion);
    const z = vis.spec.bodyFrontZ - 0.2;
    hl.position.set(root.position.x + f.x * z, root.position.y + 0.62, root.position.z + f.z * z);
    hl.target.position.set(hl.position.x + f.x * 30, hl.position.y + f.y * 30 - 2.6, hl.position.z + f.z * 30);
    hl.target.updateMatrixWorld();
    hl.visible = true;
    hl.intensity = 9000 * Math.min(1, (night - 0.35) * 2.5);
  }

  private setListener(vx: number, vy: number, vz: number): void {
    const cam = this.engine.camera;
    cam.getWorldDirection(this.fwdTmp);
    this.upTmp.set(0, 1, 0).applyQuaternion(cam.quaternion);
    audio.setListener({
      x: cam.position.x, y: cam.position.y, z: cam.position.z,
      fx: this.fwdTmp.x, fy: this.fwdTmp.y, fz: this.fwdTmp.z,
      ux: this.upTmp.x, uy: this.upTmp.y, uz: this.upTmp.z,
      vx, vy, vz,
    });
  }

  private handleEvent(e: RaceEvent): void {
    const race = this.race!;
    const isPlayer = e.car === race.playerIdx;
    const voice = e.car >= 0 ? this.voices[e.car] : null;
    switch (e.type) {
      case 'count':
        audio.play('countdown_beep');
        break;
      case 'go':
        audio.play('countdown_go');
        audio.playMusic('race', 0.4);
        break;
      case 'shift_up':
      case 'shift_down':
      case 'limiter':
      case 'backfire':
      case 'blowoff':
      case 'nitro_start':
        voice?.event(e.type);
        if (e.type === 'nitro_start' && isPlayer) {
          this.cam.kick = 1;
          this.input.rumble(0.5, 0.8, 300);
        }
        break;
      case 'mini_boost':
        voice?.event('backfire', 0.8);
        if (isPlayer) this.cam.kick = Math.max(this.cam.kick, e.a === 3 ? 0.8 : 0.5);
        break;
      case 'nitro_ready':
        if (isPlayer) audio.play('nitro_ready');
        break;
      case 'impact': {
        const k = clamp(e.a / 9000, 0, 1);
        voice?.impact(k);
        if (isPlayer) {
          this.cam.addShake(k * 1.1);
          this.input.rumble(k, k * 0.6, 160 + k * 200);
        }
        break;
      }
      case 'land':
        voice?.impact(0.25);
        if (isPlayer) this.cam.addShake(0.4);
        break;
      case 'lap':
        if (isPlayer) {
          const best = recordLap(this.track!.def.id, race.player.spec.id, e.a);
          audio.play(best || e.b > 0 ? 'best_lap' : 'lap');
        }
        break;
      case 'final_lap':
        audio.play('final_lap');
        break;
      case 'finish':
        if (isPlayer) {
          audio.play('finish');
          audio.play('cheer');
          audio.playMusic('results', 1.2);
          settings.races++;
          if (race.playerInfo.position === 1) settings.wins++;
          saveSettings();
        }
        break;
      case 'position':
        audio.play(e.a < e.b ? 'position_up' : 'position_down');
        break;
      case 'wrong_way':
        audio.play('wrong_way');
        break;
      default:
        break;
    }
    this.fx?.onEvent(e, race, this.visuals);
    for (const f of this.eventFns) f(e);
  }

  /** Cycle the race camera and remember the choice. */
  nextCamera(): void {
    this.cam.nextMode();
    settings.camera = this.cam.mode;
    saveSettings();
  }

  resetPlayer(): void {
    if (this.race && this.race.state === 'racing') this.race.resetCar(this.race.playerIdx);
  }
}
