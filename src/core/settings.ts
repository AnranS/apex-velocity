// Persistent player settings + records (localStorage).
import type { QualityName } from '../render/engine';
import type { WeatherId } from '../render/sky';
import type { CamMode } from '../render/camera';

export interface Settings {
  quality: QualityName | 'auto';
  handling: 'arcade' | 'sim';
  autoGear: boolean;
  motionBlur: boolean;
  master: number;
  music: number;
  sfx: number;
  engine: number;
  camera: CamMode;
  units: 'kmh' | 'mph';
  car: string;
  paint: string;
  weather: WeatherId;
  track: string;
  laps: number;
  aiCount: number;
  difficulty: number;
  /** best lap per "<track>|<car>" in seconds */
  bestLaps: Record<string, number>;
  /** races finished / won, for the profile card */
  races: number;
  wins: number;
}

const KEY = 'apex-velocity-settings-v1';

const DEFAULTS: Settings = {
  quality: 'auto',
  handling: 'arcade',
  autoGear: true,
  motionBlur: true,
  master: 0.8,
  music: 0.55,
  sfx: 0.8,
  engine: 0.8,
  camera: 'chase',
  units: 'kmh',
  car: 'aurion',
  paint: '#b3121d',
  weather: 'sunset',
  track: 'crimson_bay',
  laps: 3,
  aiCount: 7,
  difficulty: 1,
  bestLaps: {},
  races: 0,
  wins: 0,
};

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULTS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    /* storage unavailable */
  }
  return { ...DEFAULTS, bestLaps: {} };
}

export const settings: Settings = load();

export function saveSettings(): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    /* ignore */
  }
}

export function resolveQuality(q: Settings['quality']): QualityName {
  if (q !== 'auto') return q;
  const mobile = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && Math.min(screen.width, screen.height) < 900);
  if (mobile) return 'low';
  const cores = navigator.hardwareConcurrency || 4;
  const mem = (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? 8;
  if (cores >= 8 && mem >= 8) return 'high';
  return 'medium';
}

/** Record a lap; returns true if it is a new personal best for this track + car. */
export function recordLap(track: string, car: string, time: number): boolean {
  const k = `${track}|${car}`;
  const prev = settings.bestLaps[k];
  if (!prev || time < prev) {
    settings.bestLaps[k] = time;
    saveSettings();
    return true;
  }
  return false;
}
