// Menu framework: a Page is a focusable list of items driven by keyboard / gamepad actions,
// mouse hover + click, and touch taps.
import { audio } from '../audio';
import { clamp, h } from './dom';

export type NavAction = 'up' | 'down' | 'left' | 'right' | 'confirm' | 'back';

export interface Item {
  el: HTMLElement;
  confirm?: () => void;
  adjust?: (dir: number) => void;
  onFocus?: () => void;
  refresh?: () => void;
  disabled?: boolean;
}

export class Page {
  readonly el: HTMLDivElement;
  readonly list: HTMLDivElement;
  readonly items: Item[] = [];
  index = 0;
  /** left/right move the focus instead of adjusting (button rows) */
  horizontal = false;
  onBack: (() => void) | null = null;
  onOpen: (() => void) | null = null;
  onClose: (() => void) | null = null;
  /** key hints shown in the footer while this page is on top */
  hints: [string[], string][] = [
    [['↑', '↓'], '选择'],
    [['Enter'], '确认'],
    [['Esc'], '返回'],
  ];

  constructor(cls: string, titleCn = '', titleEn = '') {
    this.el = h('div', cls);
    if (titleCn) {
      const head = h('div', 'page-head');
      head.append(h('div', 't', titleCn), h('div', 's micro', titleEn));
      this.el.append(head);
    }
    this.list = h('div', 'items');
    this.el.append(this.list);
  }

  add(item: Item, parent: HTMLElement = this.list): Item {
    const i = this.items.length;
    item.el.style.setProperty('--i', String(i));
    item.el.addEventListener('pointerenter', (e) => {
      if ((e as PointerEvent).pointerType === 'mouse') this.focus(i, false);
    });
    item.el.addEventListener('click', () => {
      if (item.disabled) return;
      this.focus(i, false);
      if (item.confirm) {
        audio.play('ui_select');
        item.confirm();
      }
    });
    this.items.push(item);
    if (item.el.parentElement !== parent) parent.append(item.el);
    return item;
  }

  focus(i: number, sound = true): void {
    const n = this.items.length;
    if (n === 0) return;
    i = ((i % n) + n) % n;
    const changed = i !== this.index;
    this.index = i;
    for (let k = 0; k < n; k++) this.items[k].el.classList.toggle('focus', k === i);
    if (changed) {
      if (sound) audio.play('ui_move');
      this.items[i].onFocus?.();
      this.items[i].el.scrollIntoView?.({ block: 'nearest' });
    }
  }

  private move(d: number): void {
    const n = this.items.length;
    let i = this.index;
    for (let k = 0; k < n; k++) {
      i = (((i + d) % n) + n) % n;
      if (!this.items[i].disabled) break;
    }
    this.focus(i);
  }

  act(a: NavAction): void {
    const it = this.items[this.index];
    switch (a) {
      case 'up':
        this.move(-1);
        break;
      case 'down':
        this.move(1);
        break;
      case 'left':
      case 'right': {
        const d = a === 'left' ? -1 : 1;
        if (this.horizontal) this.move(d);
        else if (it?.adjust && !it.disabled) {
          audio.play('ui_move');
          it.adjust(d);
          it.refresh?.();
        }
        break;
      }
      case 'confirm':
        if (it && !it.disabled) {
          if (it.confirm) {
            audio.play('ui_select');
            it.confirm();
          } else if (it.adjust) {
            audio.play('ui_move');
            it.adjust(1);
            it.refresh?.();
          }
        }
        break;
      case 'back':
        if (this.onBack) {
          audio.play('ui_back');
          this.onBack();
        }
        break;
    }
  }

  refresh(): void {
    for (const it of this.items) it.refresh?.();
    this.focus(clamp(this.index, 0, Math.max(0, this.items.length - 1)), false);
    for (let k = 0; k < this.items.length; k++) this.items[k].el.classList.toggle('focus', k === this.index);
  }
}

// ------------------------------------------------------------------ item builders

export function bigItem(idx: number, cn: string, en: string, confirm: () => void): Item {
  const el = h('div', 'item big');
  el.append(h('span', 'idx', String(idx).padStart(2, '0')), h('span', 'l', cn), h('span', 'e micro', en));
  return { el, confirm };
}

export function ctaItem(cn: string, en: string, confirm: () => void): Item {
  const el = h('div', 'item cta');
  el.append(h('span', 'l', cn), h('span', 'micro', en));
  return { el, confirm };
}

export function ghostItem(cn: string, confirm: () => void): Item {
  const el = h('div', 'item ghost');
  el.append(h('span', 'l', cn));
  return { el, confirm };
}

/** Label + ◀ value ▶ row. `get` returns [main, sub?]. */
export function optionRow(cn: string, en: string, get: () => [string, string?], adjust: (dir: number) => void): Item {
  const el = h('div', 'item row');
  const lab = h('div', 'lab');
  lab.append(h('span', 'c', cn), h('span', 'micro', en));
  const ctl = h('div', 'ctl');
  const l = h('span', 'arrow', '‹');
  const r = h('span', 'arrow', '›');
  const val = h('div', 'val');
  ctl.append(l, val, r);
  el.append(lab, ctl);
  const refresh = () => {
    const [m, s] = get();
    val.innerHTML = `${m}${s ? `<small>${s}</small>` : ''}`;
  };
  const item: Item = { el, adjust, refresh };
  const click = (d: number) => (e: Event) => {
    e.stopPropagation();
    audio.play('ui_move');
    adjust(d);
    refresh();
  };
  l.addEventListener('click', click(-1));
  r.addEventListener('click', click(1));
  refresh();
  return item;
}

/** Label + slider (0..1) row. */
export function sliderRow(cn: string, en: string, get: () => number, set: (v: number) => void): Item {
  const el = h('div', 'item row');
  const lab = h('div', 'lab');
  lab.append(h('span', 'c', cn), h('span', 'micro', en));
  const ctl = h('div', 'ctl');
  const l = h('span', 'arrow', '‹');
  const r = h('span', 'arrow', '›');
  const sl = h('div', 'slider');
  const fill = h('i');
  sl.append(fill);
  const pct = h('span', 'pct', '');
  ctl.append(l, sl, r, pct);
  el.append(lab, ctl);
  const refresh = () => {
    const v = clamp(get(), 0, 1);
    fill.style.transform = `scaleX(${v.toFixed(3)})`;
    pct.textContent = String(Math.round(v * 100));
  };
  const adjust = (d: number) => set(clamp(Math.round((get() + d * 0.05) * 20) / 20, 0, 1));
  const item: Item = { el, adjust, refresh };
  const click = (d: number) => (e: Event) => {
    e.stopPropagation();
    audio.play('ui_move');
    adjust(d);
    refresh();
  };
  l.addEventListener('click', click(-1));
  r.addEventListener('click', click(1));
  // drag / tap directly on the bar
  const drag = (e: PointerEvent) => {
    const rect = sl.getBoundingClientRect();
    set(clamp(Math.round(((e.clientX - rect.left) / rect.width) * 20) / 20, 0, 1));
    refresh();
  };
  sl.addEventListener('pointerdown', (e) => {
    try {
      sl.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    drag(e);
    e.stopPropagation();
  });
  sl.addEventListener('pointermove', (e) => {
    if (sl.hasPointerCapture(e.pointerId)) drag(e);
  });
  sl.addEventListener('click', (e) => e.stopPropagation());
  sl.style.cursor = 'pointer';
  sl.style.touchAction = 'none';
  refresh();
  return item;
}
