// Tiny DOM + formatting helpers for the UI layer.

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, cls = '', html = ''): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

/** Set textContent only when it changed (avoids needless style/layout work). */
export function setText(el: HTMLElement, text: string): void {
  if (el.textContent !== text) el.textContent = text;
}

export function toggle(el: HTMLElement, cls: string, on: boolean): void {
  if (el.classList.contains(cls) !== on) el.classList.toggle(cls, on);
}

/** 83.456 -> "1:23.456" */
export function fmtTime(t: number, ms = 3): string {
  if (!Number.isFinite(t) || t <= 0) return ms === 3 ? '-:--.---' : '-:--.--';
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  const whole = Math.floor(s);
  const frac = Math.floor((s - whole) * (ms === 3 ? 1000 : 100));
  return `${m}:${whole < 10 ? '0' : ''}${whole}.${String(frac).padStart(ms, '0')}`;
}

export function fmtDelta(d: number): string {
  const a = Math.abs(d);
  return `${d < 0 ? '−' : '+'}${a.toFixed(a < 10 ? 3 : 2)}`;
}

export function clamp(x: number, a: number, b: number): number {
  return x < a ? a : x > b ? b : x;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export const isTouchDevice = (): boolean =>
  new URLSearchParams(location.search).get('touch') === '1' ||
  ((navigator.maxTouchPoints > 0 || 'ontouchstart' in window) && window.matchMedia('(pointer: coarse)').matches);
