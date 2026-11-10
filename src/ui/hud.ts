// Race HUD: canvas speed/rev/nitro cluster, minimap, leaderboard tower, timers, start lights,
// drift / boost / slipstream feedback and toasts. Text is refreshed at 20 Hz, gauges every frame.
import type { Game } from '../core/game';
import { settings } from '../core/settings';
import type { Car } from '../sim/car';
import type { RaceEvent, RaceSim } from '../sim/race';
import type { Track } from '../sim/track';
import { clamp, fmtDelta, fmtTime, h, setText, toggle } from './dom';

const DEG = Math.PI / 180;
const A0 = 150 * DEG;
const A1 = 390 * DEG;
const ROW_H = 2.02; // em

interface TowerRow {
  el: HTMLDivElement;
  p: HTMLElement;
  n: HTMLElement;
  g: HTMLElement;
  slot: number;
}

export class Hud {
  readonly root: HTMLDivElement;
  private readonly game: Game;
  private posWrap: HTMLElement;
  private posN: HTMLElement;
  private posOf: HTMLElement;
  private lapV: HTMLElement;
  private timer: HTMLElement;
  private lastB: HTMLElement;
  private bestB: HTMLElement;
  private delta: HTMLElement;
  private lightsWrap: HTMLElement;
  private lamps: HTMLElement[] = [];
  private go: HTMLElement;
  private toasts: HTMLElement;
  private wrong: HTMLElement;
  private tower: HTMLElement;
  private rows: TowerRow[] = [];
  private mapWrap: HTMLElement;
  private map: HTMLCanvasElement;
  private mapCtx: CanvasRenderingContext2D;
  private mapBase: HTMLCanvasElement | null = null;
  private mapT = { ox: 0, oy: 0, sc: 1 };
  private cluster: HTMLElement;
  private cv: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private spd: HTMLElement;
  private unit: HTMLElement;
  private gear: HTMLElement;
  private drift: HTMLElement;
  private driftV: HTMLElement;
  private driftC: HTMLElement;
  private boost: HTMLElement;
  private slip: HTMLElement;
  private keys: HTMLElement;
  private fps: HTMLElement;

  private visible = false;
  private txtT = 0;
  private mapTimer = 0;
  private clock = 0;
  private rpmSm = 0;
  private nitroSm = 0;
  private spdSm = 0;
  private deltaT = 0;
  private driftHold = 0;
  private driftStartNitro = 0;
  private lastLit = -1;
  private keysT = 0;
  private dpr = 1;
  private sizeKey = '';
  private showFps: boolean;

  constructor(game: Game, parent: HTMLElement, showFps: boolean) {
    this.game = game;
    this.showFps = showFps;
    const root = h('div', 'hud');
    this.root = root;

    // top-left: position + lap
    const tl = h('div', 'tl');
    this.posWrap = h('div', 'pos');
    this.posN = h('div', 'n num', '1');
    const of = h('div', 'of');
    this.posOf = h('b', '', '/8');
    of.append(this.posOf, h('span', 'micro', 'POS 名次'));
    this.posWrap.append(this.posN, of);
    const lap = h('div', 'lap');
    this.lapV = h('div', 'v num', '1<small>/3</small>');
    lap.append(h('span', 'micro', 'LAP 圈数'), this.lapV);
    tl.append(this.posWrap, lap);

    // top-centre: timers
    const tc = h('div', 'tc');
    this.timer = h('div', 'timer num', '0:00.000');
    const sub = h('div', 'sub');
    const last = h('span', '', 'LAST');
    this.lastB = h('b', '', '-:--.---');
    last.append(this.lastB);
    const best = h('span', '', 'BEST');
    this.bestB = h('b', '', '-:--.---');
    best.append(this.bestB);
    sub.append(last, best);
    this.delta = h('div', 'delta num');
    tc.append(this.timer, sub, this.delta);

    // start lights / GO
    this.lightsWrap = h('div', 'lights');
    for (let i = 0; i < 5; i++) {
      const l = h('i');
      this.lamps.push(l);
      this.lightsWrap.append(l);
    }
    this.go = h('div', 'go num', 'GO!');
    this.toasts = h('div', 'toasts');
    this.wrong = h('div', 'wrong', '<b>逆行</b><span class="micro" style="display:block;color:#fff">WRONG WAY</span>');

    this.tower = h('div', 'tower');

    this.mapWrap = h('div', 'map');
    this.map = h('canvas');
    this.mapCtx = this.map.getContext('2d')!;
    this.mapWrap.append(this.map);

    // speed cluster
    this.cluster = h('div', 'cluster');
    this.cv = h('canvas');
    this.ctx = this.cv.getContext('2d')!;
    this.spd = h('div', 'spd num', '0');
    this.unit = h('div', 'unit micro', 'KM/H');
    this.gear = h('div', 'gear num', 'N');
    this.cluster.append(this.cv, this.spd, this.unit, this.gear);

    // bottom-centre feedback
    const bc = h('div', 'bc');
    this.slip = h('div', 'slip', '<i></i><b>尾流 SLIPSTREAM</b><i></i>');
    this.drift = h('div', 'drift');
    this.driftV = h('span', 'v num', '0.0');
    this.driftC = h('span', 'c', '');
    this.drift.append(h('span', 't', '漂移'), this.driftV, this.driftC);
    this.boost = h('div', 'boost num', '小喷!');
    bc.append(this.slip, this.drift, this.boost);

    this.keys = h('div', 'keys');
    this.fps = h('div', 'fps');
    if (!showFps) this.fps.classList.add('hidden');

    root.append(tl, tc, this.lightsWrap, this.go, this.toasts, this.wrong, this.tower, this.mapWrap, this.cluster, bc, this.keys, this.fps);
    parent.append(root);
    window.addEventListener('resize', () => (this.sizeKey = ''));
  }

  setVisible(on: boolean): void {
    this.visible = on;
    toggle(this.root, 'on', on);
  }

  /** Prepare for a new race. */
  reset(): void {
    const race = this.game.race;
    if (!race) return;
    this.toasts.innerHTML = '';
    this.delta.classList.remove('show');
    this.wrong.classList.remove('on');
    this.go.classList.remove('show');
    this.lightsWrap.classList.remove('on');
    this.drift.classList.remove('on');
    this.boost.classList.remove('show');
    this.lastLit = -1;
    this.rpmSm = 0;
    this.nitroSm = race.player.nitro;
    this.spdSm = 0;
    this.keysT = 0;
    this.clock = 0;
    this.keys.classList.remove('fade');
    this.buildKeys();
    // leaderboard rows
    this.tower.innerHTML = '';
    this.rows = [];
    const n = race.cars.length;
    this.tower.style.height = `${n * ROW_H}em`;
    this.tower.classList.toggle('hidden', n < 2);
    for (let i = 0; i < n; i++) {
      const el = h('div', 'trow');
      if (race.info[i].isPlayer) el.classList.add('me');
      const p = h('span', 'p', String(i + 1));
      const nm = h('span', 'n', race.info[i].isPlayer ? '你 YOU' : race.info[i].name);
      const g = h('span', 'g', '');
      el.append(p, nm, g);
      el.style.position = 'absolute';
      el.style.left = '0';
      el.style.right = '0';
      el.style.top = '0';
      this.tower.append(el);
      this.rows.push({ el, p, n: nm, g, slot: -1 });
    }
    this.posWrap.classList.toggle('hidden', n < 2);
    this.sizeKey = '';
    this.mapBase = null;
    setText(this.unit, settings.units === 'mph' ? 'MPH' : 'KM/H');
  }

  private buildKeys(): void {
    const dev = this.game.input.lastDevice;
    const k = (keys: string[], label: string) => `<span>${keys.map((x) => `<kbd>${x}</kbd>`).join('')}${label}</span>`;
    this.keys.innerHTML =
      dev === 'gamepad'
        ? k(['A'], '漂移') + k(['X'], '氮气') + k(['Y'], '视角') + k(['START'], '暂停')
        : k(['↑', '↓', '←', '→'], '驾驶') + k(['Shift'], '+方向 漂移') + k(['Ctrl'], '氮气') + k(['R'], '复位') + k(['C'], '视角') + k(['Esc'], '暂停');
  }

  toast(main: string, sub = '', cls = '', dur = 2.6): void {
    const el = h('div', `toast ${cls}`, `<b>${main}</b>${sub ? `<span class="micro">${sub}</span>` : ''}`);
    this.toasts.append(el);
    while (this.toasts.children.length > 3) this.toasts.firstElementChild?.remove();
    window.setTimeout(() => {
      el.classList.add('out');
      window.setTimeout(() => el.remove(), 380);
    }, dur * 1000);
  }

  private flashBoost(text: string, nitro: boolean): void {
    this.boost.textContent = text;
    this.boost.classList.toggle('nitro', nitro);
    this.boost.classList.remove('show');
    void this.boost.offsetWidth; // restart the animation
    this.boost.classList.add('show');
  }

  onEvent(e: RaceEvent): void {
    const race = this.game.race;
    if (!race) return;
    const me = e.car === race.playerIdx;
    switch (e.type) {
      case 'go':
        this.go.classList.remove('show');
        void this.go.offsetWidth;
        this.go.classList.add('show');
        break;
      case 'lap':
        if (me && race.playerInfo.lapsDone < race.cfg.laps) {
          if (e.b > 0 && race.playerInfo.lapTimes.length > 1) this.toast('最快圈速', `BEST LAP ${fmtTime(e.a)}`, 'good', 3);
          else this.toast(`第 ${race.playerInfo.lapsDone} 圈`, `LAP TIME ${fmtTime(e.a)}`, '', 2.4);
        }
        break;
      case 'final_lap':
        this.toast('最后一圈', 'FINAL LAP', 'hot', 3.2);
        break;
      case 'position':
        if (me) {
          this.posWrap.classList.remove('bump');
          void this.posWrap.offsetWidth;
          this.posWrap.classList.add('bump');
        }
        break;
      case 'sector':
        if (me) {
          this.delta.textContent = fmtDelta(e.b);
          toggle(this.delta, 'good', e.b <= 0);
          toggle(this.delta, 'bad', e.b > 0);
          this.delta.classList.add('show');
          this.deltaT = 3.2;
        }
        break;
      case 'drift_start':
        if (me) this.driftStartNitro = race.player.nitroBottles + race.player.nitro;
        break;
      case 'mini_boost':
        // a = 1 small boost, 2 double boost, 3 super start, 4 landing boost
        if (me) this.flashBoost(e.a === 2 ? '双喷 DOUBLE!!' : e.a === 3 ? '超级起步 START!' : e.a === 4 ? '落地喷 BOOST!' : '小喷 BOOST!', false);
        break;
      case 'nitro_ready':
        if (me) this.toast('氮气 +1', 'N₂O READY', 'good', 1.4);
        break;
      case 'nitro_start':
        if (me) this.flashBoost('氮气 NITRO!', true);
        break;
      case 'reset':
        if (me) this.toast('已复位', 'RESET TO TRACK', '', 1.6);
        break;
      default:
        break;
    }
  }

  // ------------------------------------------------------------------ frame
  update(dt: number): void {
    const race = this.game.race;
    const track = this.game.track;
    if (!this.visible || !race || !track) return;
    this.clock += dt;
    const p = race.player;
    const inf = race.playerInfo;
    this.ensureSizes(track);
    if (!this.mapBase) return;

    // smoothed gauge values
    const k = 1 - Math.exp(-dt * 18);
    this.rpmSm += (p.rpm - this.rpmSm) * k;
    // gauge fill; snaps when it wraps into a stored bottle
    if (p.nitro < this.nitroSm - 0.5) this.nitroSm = p.nitro;
    else this.nitroSm += (p.nitro - this.nitroSm) * (1 - Math.exp(-dt * 10));
    this.drawCluster(p);

    this.mapTimer -= dt;
    if (this.mapTimer <= 0) {
      this.mapTimer = 1 / 30;
      this.drawMap(race);
    }

    // start lights follow the simulation clock
    const counting = race.state === 'countdown';
    const lit = counting ? clamp(Math.floor((3.95 - race.countdown) * 1.7), 0, 5) : 0;
    if (lit !== this.lastLit) {
      this.lastLit = lit;
      for (let i = 0; i < 5; i++) toggle(this.lamps[i], 'lit', i < lit);
    }
    toggle(this.lightsWrap, 'on', counting || (race.state === 'grid' && this.game.raceClock > 0.5));

    // drift popup (linger briefly after the slide ends)
    if (p.drifting) this.driftHold = 0.7;
    else this.driftHold -= dt;
    toggle(this.drift, 'on', this.driftHold > 0 && race.state === 'racing');
    toggle(this.slip, 'on', p.draft > 0.3 && race.state === 'racing');
    toggle(this.wrong, 'on', inf.wrongWay > 1.2 && race.state === 'racing');

    if (this.deltaT > 0) {
      this.deltaT -= dt;
      if (this.deltaT <= 0) this.delta.classList.remove('show');
    }
    if (race.state === 'racing' && this.keysT < 14) {
      this.keysT += dt;
      if (this.keysT >= 14) this.keys.classList.add('fade');
    }

    // ---- text at 20 Hz
    this.txtT -= dt;
    if (this.txtT > 0) return;
    this.txtT = 0.05;
    const mph = settings.units === 'mph';
    const v = Math.abs(p.speed) * (mph ? 2.23694 : 3.6);
    this.spdSm = v;
    setText(this.spd, String(Math.round(v)));
    setText(this.gear, p.hold || p.gear === 0 ? 'N' : p.gear < 0 ? 'R' : String(p.gear));
    toggle(this.gear, 'shift', p.rpm > p.spec.redline * 0.955 && p.gear >= 1 && p.gear < p.spec.gears.length && !p.hold);
    setText(this.posN, String(inf.position));
    setText(this.posOf, `/${race.cars.length}`);
    const lapNow = Math.min(inf.lapsDone + 1, race.cfg.laps);
    const lapHtml = `${lapNow}<small>/${race.cfg.laps}</small>`;
    if (this.lapV.innerHTML !== lapHtml) this.lapV.innerHTML = lapHtml;
    const racing = race.state === 'racing' || race.state === 'finished';
    setText(this.timer, inf.finished ? fmtTime(inf.finishTime) : racing ? fmtTime(Math.max(0.001, race.time - inf.lapStart)) : '0:00.000');
    setText(this.lastB, fmtTime(inf.lastLap));
    setText(this.bestB, fmtTime(inf.bestLap));
    if (this.driftHold > 0) {
      setText(this.driftV, `${p.driftTime.toFixed(1)}s`);
      const gain = Math.max(0, p.nitroBottles + p.nitro - this.driftStartNitro);
      setText(this.driftC, gain > 0.005 ? `N₂O +${Math.round(gain * 100)}%` : '');
    }
    this.updateTower(race);
    if (this.showFps) setText(this.fps, `${this.game.fps.toFixed(0)} FPS · ${this.game.engine.q.name.toUpperCase()}`);
  }

  private updateTower(race: RaceSim): void {
    const n = race.cars.length;
    if (n < 2) return;
    const leader = race.cars[race.order[0]];
    for (let k = 0; k < n; k++) {
      const ci = race.order[k];
      const row = this.rows[ci];
      const inf = race.info[ci];
      if (row.slot !== k) {
        row.slot = k;
        row.el.style.transform = `translateY(${k * ROW_H}em)`;
        setText(row.p, String(k + 1));
      }
      let g = '';
      if (inf.finished) g = '完赛';
      else if (k === 0) g = race.state === 'racing' || race.state === 'finished' ? 'LEAD' : '';
      else if (race.state === 'racing' || race.state === 'finished') {
        const sp = Math.max(24, (race.cars[ci].speed + leader.speed) * 0.5);
        const sec = inf.gap / sp;
        g = sec < 99 ? `+${sec.toFixed(1)}` : '+99';
      }
      setText(row.g, g);
      toggle(row.el, 'fin', inf.finished);
    }
  }

  // ------------------------------------------------------------------ canvases
  private ensureSizes(track: Track): void {
    // measure only when flagged dirty (reset / resize): no layout reads in the steady state
    if (this.sizeKey !== '' && this.mapBase) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = this.cluster.clientWidth;
    const mw = this.mapWrap.clientWidth;
    const mh = this.mapWrap.clientHeight;
    if (cw < 2 || mw < 2) return;
    this.sizeKey = `${cw}|${mw}|${mh}|${dpr}`;
    this.dpr = dpr;
    this.cv.width = this.cv.height = Math.max(64, Math.round(cw * dpr));
    this.map.width = Math.max(32, Math.round(mw * dpr));
    this.map.height = Math.max(32, Math.round(mh * dpr));
    this.buildMapBase(track);
  }

  private buildMapBase(track: Track): void {
    const W = this.map.width;
    const H = this.map.height;
    const base = document.createElement('canvas');
    base.width = W;
    base.height = H;
    const c = base.getContext('2d')!;
    const pad = Math.round(W * 0.07);
    const sx = (W - pad * 2) / (track.maxX - track.minX);
    const sy = (H - pad * 2) / (track.maxZ - track.minZ);
    const sc = Math.min(sx, sy);
    const ox = (W - (track.maxX - track.minX) * sc) * 0.5 - track.minX * sc;
    const oy = (H - (track.maxZ - track.minZ) * sc) * 0.5 - track.minZ * sc;
    this.mapT = { ox, oy, sc };
    const path = () => {
      c.beginPath();
      for (let i = 0; i < track.n; i += 3) {
        const x = track.cx[i] * sc + ox;
        const y = track.cz[i] * sc + oy;
        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
      }
      c.closePath();
    };
    c.lineJoin = 'round';
    c.lineCap = 'round';
    path();
    c.strokeStyle = 'rgba(4,6,9,0.78)';
    c.lineWidth = Math.max(5, W * 0.05);
    c.stroke();
    path();
    c.strokeStyle = 'rgba(244,246,249,0.92)';
    c.lineWidth = Math.max(2, W * 0.017);
    c.stroke();
    // start / finish tick
    const x0 = track.cx[0] * sc + ox;
    const y0 = track.cz[0] * sc + oy;
    const nx = track.nx[0];
    const nz = track.nz[0];
    const l = W * 0.035;
    c.strokeStyle = '#ff3d2e';
    c.lineWidth = Math.max(2, W * 0.016);
    c.beginPath();
    c.moveTo(x0 - nx * l, y0 - nz * l);
    c.lineTo(x0 + nx * l, y0 + nz * l);
    c.stroke();
    this.mapBase = base;
  }

  private drawMap(race: RaceSim): void {
    const c = this.mapCtx;
    const W = this.map.width;
    const H = this.map.height;
    c.clearRect(0, 0, W, H);
    if (this.mapBase) c.drawImage(this.mapBase, 0, 0);
    const { ox, oy, sc } = this.mapT;
    const r = Math.max(2.5, W * 0.016);
    for (let k = race.cars.length - 1; k >= 0; k--) {
      const ci = race.order[k];
      if (ci === race.playerIdx) continue;
      const car = race.cars[ci];
      c.beginPath();
      c.arc(car.px * sc + ox, car.pz * sc + oy, r, 0, Math.PI * 2);
      c.fillStyle = k === 0 ? '#ffc83d' : '#aab3bf';
      c.fill();
      c.lineWidth = Math.max(1, r * 0.4);
      c.strokeStyle = 'rgba(4,6,9,0.85)';
      c.stroke();
    }
    // player: heading wedge
    const p = race.player;
    const x = p.px * sc + ox;
    const y = p.pz * sc + oy;
    const fl = Math.hypot(p.m02, p.m22) || 1;
    const fx = p.m02 / fl;
    const fz = p.m22 / fl;
    const s = r * 2.3;
    c.beginPath();
    c.moveTo(x + fx * s, y + fz * s);
    c.lineTo(x - fx * s * 0.7 + fz * s * 0.7, y - fz * s * 0.7 - fx * s * 0.7);
    c.lineTo(x - fx * s * 0.25, y - fz * s * 0.25);
    c.lineTo(x - fx * s * 0.7 - fz * s * 0.7, y - fz * s * 0.7 + fx * s * 0.7);
    c.closePath();
    c.fillStyle = '#ff3d2e';
    c.fill();
    c.lineWidth = Math.max(1.2, r * 0.5);
    c.strokeStyle = '#fff';
    c.stroke();
  }

  private drawCluster(p: Car): void {
    const ctx = this.ctx;
    const S = this.cv.width;
    ctx.clearRect(0, 0, S, S);
    const cx = S * 0.5;
    const cy = S * 0.5;
    const R = S * 0.43;
    const red = p.spec.redline;
    const maxRpm = Math.ceil((red + 150) / 1000) * 1000;
    const f = clamp(this.rpmSm / maxRpm, 0, 1);
    const ang = A0 + (A1 - A0) * f;
    const arc = (a: number, b: number, r = R) => {
      ctx.beginPath();
      ctx.arc(cx, cy, r, a, b);
      ctx.stroke();
    };

    // backdrop for legibility over any scene
    const bg = ctx.createRadialGradient(cx, cy, R * 0.2, cx, cy, R * 1.15);
    bg.addColorStop(0, 'rgba(6,8,12,0.74)');
    bg.addColorStop(0.8, 'rgba(6,8,12,0.52)');
    bg.addColorStop(1, 'rgba(6,8,12,0)');
    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.15, 0, Math.PI * 2);
    ctx.fill();

    // band + redline
    ctx.lineCap = 'butt';
    ctx.lineWidth = S * 0.034;
    ctx.strokeStyle = 'rgba(255,255,255,0.13)';
    arc(A0, A1);
    const fr = (red * 0.955) / maxRpm;
    const aRed = A0 + (A1 - A0) * fr;
    ctx.strokeStyle = 'rgba(255,61,46,0.5)';
    arc(aRed, A1);

    // live rev band
    const inRed = this.rpmSm > red * 0.955;
    let grad: CanvasGradient | string = '#ffffff';
    const cg = (ctx as CanvasRenderingContext2D & { createConicGradient?: (a: number, x: number, y: number) => CanvasGradient }).createConicGradient;
    if (cg) {
      const g = cg.call(ctx, A0, cx, cy);
      const span = (A1 - A0) / (Math.PI * 2);
      g.addColorStop(0, 'rgba(255,255,255,0.5)');
      g.addColorStop(span * 0.55, '#ffffff');
      g.addColorStop(span * 0.86, '#ffb347');
      g.addColorStop(span, '#ff3d2e');
      g.addColorStop(Math.min(1, span + 0.001), '#ff3d2e');
      grad = g;
    }
    ctx.strokeStyle = inRed ? '#ff3d2e' : grad;
    if (f > 0.002) arc(A0, ang);

    // ticks + numerals
    const nT = maxRpm / 1000;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `italic 600 ${Math.round(S * 0.05)}px "Barlow Condensed", "Avenir Next Condensed", sans-serif`;
    for (let k = 0; k <= nT; k++) {
      const a = A0 + (A1 - A0) * (k / nT);
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      ctx.strokeStyle = k * 1000 >= red * 0.955 ? '#ff5a4a' : 'rgba(255,255,255,0.75)';
      ctx.lineWidth = S * 0.008;
      ctx.beginPath();
      ctx.moveTo(cx + ca * (R - S * 0.03), cy + sa * (R - S * 0.03));
      ctx.lineTo(cx + ca * (R - S * 0.058), cy + sa * (R - S * 0.058));
      ctx.stroke();
      ctx.fillStyle = k * 1000 >= red * 0.955 ? '#ff7a6a' : 'rgba(244,246,249,0.7)';
      ctx.fillText(String(k), cx + ca * (R - S * 0.1), cy + sa * (R - S * 0.1));
    }

    // needle marker
    ctx.save();
    ctx.shadowColor = inRed ? 'rgba(255,61,46,0.95)' : 'rgba(255,170,80,0.9)';
    ctx.shadowBlur = S * 0.035;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = S * 0.013;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(ang) * (R - S * 0.034), cy + Math.sin(ang) * (R - S * 0.034));
    ctx.lineTo(cx + Math.cos(ang) * (R + S * 0.03), cy + Math.sin(ang) * (R + S * 0.03));
    ctx.stroke();
    ctx.restore();

    // shift lights across the top
    const N = 12;
    const rr = this.rpmSm / red;
    const sf = clamp((rr - 0.7) / (0.965 - 0.7), 0, 1);
    const flash = rr > 0.975 && !p.hold && Math.floor(this.clock * 14) % 2 === 0;
    for (let i = 0; i < N; i++) {
      const a = (232 + (76 * i) / (N - 1)) * DEG;
      const x = cx + Math.cos(a) * (R * 0.6);
      const y = cy + Math.sin(a) * (R * 0.6);
      const on = flash || i < Math.round(sf * N);
      const col = flash ? '#9fe8ff' : i < 5 ? '#44e08b' : i < 9 ? '#ffc83d' : '#ff4d3d';
      ctx.beginPath();
      ctx.arc(x, y, S * 0.0105, 0, Math.PI * 2);
      if (on) {
        ctx.save();
        ctx.shadowColor = col;
        ctx.shadowBlur = S * 0.03;
        ctx.fillStyle = col;
        ctx.fill();
        ctx.restore();
      } else {
        ctx.fillStyle = 'rgba(255,255,255,0.14)';
        ctx.fill();
      }
    }

    // ---- nitro tank
    const bw = S * 0.56;
    const bh = S * 0.036;
    const bx = cx - bw / 2;
    const by = cy + S * 0.325;
    const sk = bh * 0.55;
    const para = (x0: number, w: number) => {
      ctx.beginPath();
      ctx.moveTo(x0 + sk, by);
      ctx.lineTo(x0 + w + sk, by);
      ctx.lineTo(x0 + w, by + bh);
      ctx.lineTo(x0, by + bh);
      ctx.closePath();
    };
    para(bx, bw);
    ctx.fillStyle = 'rgba(255,255,255,0.13)';
    ctx.fill();
    const on = p.nitroOn;
    // while burning the bar shows the burn running down, otherwise the gauge filling toward the next bottle
    const lvl = on ? clamp(p.nitroTime / p.spec.nitroDuration, 0, 1) : clamp(this.nitroSm, 0, 1);
    const ready = p.nitroBottles > 0;
    if (lvl > 0.004) {
      ctx.save();
      para(bx, bw * lvl);
      ctx.clip();
      const g = ctx.createLinearGradient(bx, 0, bx + bw, 0);
      if (on) {
        const fl = 0.75 + 0.25 * Math.sin(this.clock * 38);
        g.addColorStop(0, `rgba(255,${Math.round(90 + 60 * fl)},40,1)`);
        g.addColorStop(0.7, '#ffd27a');
        g.addColorStop(1, '#ffffff');
      } else if (ready) {
        g.addColorStop(0, '#1fa8ff');
        g.addColorStop(1, '#7df1ff');
      } else {
        g.addColorStop(0, 'rgba(60,140,180,0.9)');
        g.addColorStop(1, 'rgba(110,190,220,0.9)');
      }
      ctx.fillStyle = g;
      ctx.fillRect(bx, by, bw + sk, bh);
      if (p.drifting && !on) {
        // charging: stripes sweeping toward the tip
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        const st = bh * 1.5;
        const off = (this.clock * bh * 9) % (st * 2);
        for (let x = bx - st * 2 + off; x < bx + bw * lvl + st; x += st * 2) {
          ctx.beginPath();
          ctx.moveTo(x + sk, by);
          ctx.lineTo(x + st + sk, by);
          ctx.lineTo(x + st, by + bh);
          ctx.lineTo(x, by + bh);
          ctx.closePath();
          ctx.fill();
        }
      }
      ctx.restore();
      if (on || p.drifting) {
        // glow at the fill tip
        const tx = bx + bw * lvl;
        const gl = ctx.createRadialGradient(tx, by + bh / 2, 0, tx, by + bh / 2, bh * 2.4);
        gl.addColorStop(0, on ? 'rgba(255,200,120,0.9)' : 'rgba(125,241,255,0.85)');
        gl.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = gl;
        ctx.fillRect(tx - bh * 2.4, by - bh * 2, bh * 4.8, bh * 5);
      }
    }
    // labels
    ctx.font = `600 ${Math.round(S * 0.038)}px "Barlow Condensed", "PingFang SC", sans-serif`;
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(244,246,249,0.72)';
    ctx.fillText('N₂O', bx + sk, by - bh * 0.55);
    // stored bottles (two at most)
    {
      const pw = S * 0.05;
      const ph = S * 0.026;
      const py = by - bh * 0.55 - ph;
      let px = bx + sk + ctx.measureText('N₂O').width + S * 0.022;
      for (let i = 0; i < 2; i++) {
        const full = i < p.nitroBottles;
        ctx.beginPath();
        ctx.moveTo(px + ph * 0.5, py);
        ctx.lineTo(px + pw + ph * 0.5, py);
        ctx.lineTo(px + pw, py + ph);
        ctx.lineTo(px, py + ph);
        ctx.closePath();
        ctx.fillStyle = full ? '#7df1ff' : 'rgba(255,255,255,0.16)';
        if (full) {
          ctx.shadowColor = 'rgba(125,241,255,0.9)';
          ctx.shadowBlur = S * 0.03;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
        px += pw + S * 0.014;
      }
    }
    ctx.textAlign = 'right';
    let label = `${Math.round(p.nitro * 100)}%`;
    let lc = 'rgba(244,246,249,0.72)';
    if (on) {
      label = '喷射 BOOST';
      lc = '#ffb347';
    } else if (p.drifting) {
      label = '集气 +';
      lc = '#7df1ff';
    } else if (ready) {
      const pulse = 0.65 + 0.35 * Math.sin(this.clock * 6);
      label = this.game.input.lastDevice === 'gamepad' ? 'READY · X' : this.game.input.lastDevice === 'touch' ? 'READY' : 'READY · Ctrl';
      lc = `rgba(125,241,255,${pulse.toFixed(2)})`;
    }
    ctx.fillStyle = lc;
    ctx.fillText(label, bx + bw + sk, by - bh * 0.55);
  }
}

