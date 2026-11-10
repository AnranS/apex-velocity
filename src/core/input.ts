// Player input: keyboard, gamepad and touch merged into a CarInput plus edge-triggered UI actions.
import { CarInput, makeInput } from '../sim/car';
import { clamp, moveToward } from '../sim/math';

export type Action = 'camera' | 'reset' | 'pause' | 'confirm' | 'back' | 'up' | 'down' | 'left' | 'right' | 'shiftUp' | 'shiftDown' | 'photo';

const KEY_ACTIONS: Record<string, Action> = {
  KeyC: 'camera',
  KeyR: 'reset',
  Escape: 'pause',
  KeyP: 'pause',
  Enter: 'confirm',
  NumpadEnter: 'confirm',
  Backspace: 'back',
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
  KeyE: 'shiftUp',
  KeyQ: 'shiftDown',
};

export class Input {
  readonly car: CarInput = makeInput();
  private keys = new Set<string>();
  private actions: Action[] = [];
  private steerKb = 0;
  private nitroLatch = false;
  lookBack = false;
  /** touch state written by the on-screen controls */
  touch = { steer: 0, throttle: 0, brake: 0, nitro: false, handbrake: 0, active: false };
  gamepadName = '';
  private padPrev: boolean[] = [];
  lastDevice: 'keyboard' | 'gamepad' | 'touch' = 'keyboard';

  constructor() {
    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      if (e.target instanceof HTMLInputElement) return;
      this.keys.add(e.code);
      this.lastDevice = 'keyboard';
      const a = KEY_ACTIONS[e.code];
      if (a) this.actions.push(a);
      // a quick tap can start and end between two frames: latch it until the next poll
      if (e.code === 'ControlLeft' || e.code === 'ControlRight' || e.code === 'Space') this.nitroLatch = true;
      // keep the page from scrolling / reacting to Ctrl combos while driving
      if (e.code.startsWith('Arrow') || e.code === 'Space' || (e.ctrlKey && !e.metaKey && /^(Key[WASDR]|Arrow)/.test(e.code))) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  private down(...codes: string[]): boolean {
    for (const c of codes) if (this.keys.has(c)) return true;
    return false;
  }

  /** Drain edge-triggered actions since the last call. */
  takeActions(): Action[] {
    const a = this.actions;
    this.actions = [];
    return a;
  }

  /** Refresh `car` from the devices. Call once per frame. */
  poll(dt: number): CarInput {
    const c = this.car;
    // keyboard
    let thr = this.down('KeyW', 'ArrowUp') ? 1 : 0;
    let brk = this.down('KeyS', 'ArrowDown') ? 1 : 0;
    const kbSteer = (this.down('KeyA', 'ArrowLeft') ? 1 : 0) - (this.down('KeyD', 'ArrowRight') ? 1 : 0);
    // the simulation rate-limits steering itself; pass the raw digital target through
    this.steerKb = moveToward(this.steerKb, kbSteer, dt * 30);
    let steer = this.steerKb;
    // Shift = drift (handbrake in sim handling), Ctrl = nitro (Space as a spare: Ctrl + arrows is a system
    // shortcut on macOS)
    let hb = this.down('ShiftLeft', 'ShiftRight') ? 1 : 0;
    let nitro = this.down('ControlLeft', 'ControlRight', 'Space') || this.nitroLatch;
    this.nitroLatch = false;
    this.lookBack = this.down('KeyB');
    c.shiftUp = false;
    c.shiftDown = false;

    // gamepad (first connected)
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const p of pads) {
      if (!p || !p.connected) continue;
      this.gamepadName = p.id;
      const dz = (v: number, d = 0.12) => (Math.abs(v) < d ? 0 : (v - Math.sign(v) * d) / (1 - d));
      const sx = dz(p.axes[0] ?? 0);
      const rt = p.buttons[7]?.value ?? 0;
      const lt = p.buttons[6]?.value ?? 0;
      if (Math.abs(sx) > 0.02 || rt > 0.05 || lt > 0.05) this.lastDevice = 'gamepad';
      if (this.lastDevice === 'gamepad') {
        // stick right = steer right = negative; gentle curve for precision around centre
        steer = -Math.sign(sx) * Math.pow(Math.abs(sx), 1.6);
        thr = Math.max(thr, rt);
        brk = Math.max(brk, lt);
      }
      const b = (i: number) => !!p.buttons[i]?.pressed;
      hb = Math.max(hb, b(0) ? 1 : 0);
      nitro = nitro || b(2) || b(5);
      this.lookBack = this.lookBack || b(4);
      const edges: [number, Action][] = [
        [3, 'camera'],
        [8, 'reset'],
        [9, 'pause'],
        [0, 'confirm'],
        [1, 'back'],
        [12, 'up'],
        [13, 'down'],
        [14, 'left'],
        [15, 'right'],
      ];
      for (const [i, a] of edges) {
        const now = b(i);
        if (now && !this.padPrev[i]) {
          this.actions.push(a);
          this.lastDevice = 'gamepad';
        }
        this.padPrev[i] = now;
      }
      break;
    }

    // touch
    const t = this.touch;
    if (t.active) {
      this.lastDevice = 'touch';
      steer = t.steer;
      thr = Math.max(thr, t.throttle);
      brk = Math.max(brk, t.brake);
      nitro = nitro || t.nitro;
      hb = Math.max(hb, t.handbrake);
    }

    c.throttle = clamp(thr, 0, 1);
    c.brake = clamp(brk, 0, 1);
    c.steer = clamp(steer, -1, 1);
    c.handbrake = hb;
    c.nitro = nitro;
    // no accelerator key to re-press on touch (auto throttle): small boosts fire by themselves
    c.autoBoost = t.active;
    return c;
  }

  rumble(strong: number, weak: number, ms: number): void {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const p of pads) {
      const act = (p as unknown as { vibrationActuator?: { playEffect: (t: string, o: object) => Promise<unknown> } } | null)?.vibrationActuator;
      if (act) act.playEffect('dual-rumble', { duration: ms, strongMagnitude: clamp(strong, 0, 1), weakMagnitude: clamp(weak, 0, 1) }).catch(() => {});
    }
  }
}
