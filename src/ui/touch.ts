// On-screen touch controls: analogue steering pad (left thumb), brake / drift / nitro (right thumb),
// auto-throttle, plus small pause / reset / camera buttons. Writes game.input.touch every frame.
import type { Game } from '../core/game';
import { clamp, h, toggle } from './dom';

const ICON = {
  left: '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg>',
  right: '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5l7 7-7 7"/></svg>',
};

function capture(el: Element, id: number): void {
  try {
    el.setPointerCapture(id);
  } catch {
    /* synthetic or already-released pointer */
  }
}

export class TouchControls {
  readonly root: HTMLDivElement;
  private readonly game: Game;
  private steerPad: HTMLDivElement;
  private knob: HTMLDivElement;
  private nitroBtn: HTMLDivElement;
  private steer = 0;
  private steerId = -1;
  private brake = false;
  private drift = false;
  private nitro = false;
  private active = false;

  constructor(game: Game, parent: HTMLElement, onPause: () => void, onReset: () => void, onCamera: () => void) {
    this.game = game;
    const root = h('div', 'touch');
    this.root = root;

    // steering: one wide pad, finger offset from the centre = steering amount
    this.steerPad = h('div', 'tbtn t-steer', `<span class="ar l">${ICON.left}</span><span class="ar r">${ICON.right}</span>`);
    this.knob = h('div', 'knob');
    this.steerPad.append(this.knob);
    const setSteer = (e: PointerEvent) => {
      const r = this.steerPad.getBoundingClientRect();
      const dx = (e.clientX - (r.left + r.width / 2)) / (r.width * 0.36);
      // finger right of centre = steer right = negative
      this.steer = -clamp(Math.abs(dx) < 0.08 ? 0 : dx, -1, 1);
    };
    this.steerPad.addEventListener('pointerdown', (e) => {
      this.steerId = e.pointerId;
      capture(this.steerPad, e.pointerId);
      this.steerPad.classList.add('down');
      setSteer(e);
      e.preventDefault();
    });
    this.steerPad.addEventListener('pointermove', (e) => {
      if (e.pointerId === this.steerId) setSteer(e);
    });
    const endSteer = (e: PointerEvent) => {
      if (e.pointerId !== this.steerId) return;
      this.steerId = -1;
      this.steer = 0;
      this.steerPad.classList.remove('down');
    };
    this.steerPad.addEventListener('pointerup', endSteer);
    this.steerPad.addEventListener('pointercancel', endSteer);
    this.steerPad.addEventListener('lostpointercapture', endSteer);

    const hold = (cls: string, html: string, set: (on: boolean) => void): HTMLDivElement => {
      const b = h('div', `tbtn ${cls}`, html);
      const down = (e: PointerEvent) => {
        capture(b, e.pointerId);
        b.classList.add('down');
        set(true);
        e.preventDefault();
      };
      const up = () => {
        b.classList.remove('down');
        set(false);
      };
      b.addEventListener('pointerdown', down);
      b.addEventListener('pointerup', up);
      b.addEventListener('pointercancel', up);
      b.addEventListener('lostpointercapture', up);
      return b;
    };
    const tap = (cls: string, html: string, fn: () => void): HTMLDivElement => {
      const b = h('div', `tbtn ${cls}`, html);
      b.addEventListener('pointerdown', (e) => {
        b.classList.add('down');
        e.preventDefault();
      });
      b.addEventListener('pointerup', () => {
        b.classList.remove('down');
        fn();
      });
      b.addEventListener('pointercancel', () => b.classList.remove('down'));
      return b;
    };

    const brake = hold('t-brake', '<b>刹车</b>', (on) => (this.brake = on));
    const drift = hold('t-drift', '<b>漂移</b>', (on) => (this.drift = on));
    this.nitroBtn = hold('t-nitro', '<span class="fill"></span><b>N₂O</b>', (on) => (this.nitro = on));
    const pause = tap('t-pause', '<b>II</b>', onPause);
    const reset = tap('t-reset', '<b>复位</b>', onReset);
    const cam = tap('t-cam', '<b>视角</b>', onCamera);
    root.append(this.steerPad, brake, drift, this.nitroBtn, pause, reset, cam);
    parent.append(root);
  }

  setActive(on: boolean): void {
    this.active = on;
    toggle(this.root, 'on', on);
    this.root.style.pointerEvents = on ? '' : 'none';
    for (const c of Array.from(this.root.children)) (c as HTMLElement).style.pointerEvents = on ? 'auto' : 'none';
    const t = this.game.input.touch;
    t.active = on;
    if (!on) {
      t.steer = 0;
      t.throttle = 0;
      t.brake = 0;
      t.nitro = false;
      t.handbrake = 0;
      this.steer = 0;
      this.brake = this.drift = this.nitro = false;
    }
  }

  update(): void {
    if (!this.active) return;
    const race = this.game.race;
    const t = this.game.input.touch;
    const racing = !!race && (race.state === 'racing' || race.state === 'countdown' || race.state === 'grid');
    t.active = true;
    t.steer = this.steer;
    t.brake = this.brake ? 1 : 0;
    // auto-throttle: flat out unless braking (blips on the grid are harmless: the car is held)
    t.throttle = racing && !this.brake ? 1 : 0;
    t.handbrake = this.drift ? 1 : 0;
    t.nitro = this.nitro;
    this.knob.style.transform = `translate3d(${(-this.steer * 34).toFixed(1)}%, 0, 0)`;
    if (race) {
      const p = race.player;
      this.nitroBtn.style.setProperty('--p', (p.nitroOn ? p.nitroTime / p.spec.nitroDuration : p.nitro).toFixed(3));
      toggle(this.nitroBtn, 'ready', p.nitroBottles > 0 && !p.nitroOn);
      const label = this.nitroBtn.querySelector('b');
      const text = p.nitroBottles > 0 ? `N₂O ×${p.nitroBottles}` : 'N₂O';
      if (label && label.textContent !== text) label.textContent = text;
    }
  }
}
