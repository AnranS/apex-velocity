// Boot: create the game + UI layer, then either the normal title/menu flow or a direct race (automation).
//
// URL params: race=1 (or bot=1) start a race immediately · w/car/paint/laps/ai/diff override the event ·
//             bot=1 autopilot · q=low|medium|high|ultra quality · menu=1 skip the title · hud=0 hide all UI ·
//             fps=1 show the frame-rate counter · touch=1 force touch controls
import './ui/styles.css';
import { audio } from './audio';
import { Game } from './core/game';
import { resolveQuality, settings } from './core/settings';
import type { QualityName } from './render/engine';
import { WEATHER, type WeatherId } from './render/sky';
import { CAR_SPECS } from './sim/car_specs';
import { UI } from './ui/ui';

const params = new URLSearchParams(location.search);
const canvas = document.getElementById('gl') as HTMLCanvasElement;
const uiRoot = document.getElementById('ui') as HTMLDivElement;

/** Display font: Barlow Condensed (c) The Barlow Project Authors, SIL Open Font License 1.1 — see public/assets/fonts/OFL-BarlowCondensed.txt */
function installFonts(): void {
  const base = `${import.meta.env.BASE_URL}assets/fonts/`;
  const face = (file: string, weight: number, style: string) =>
    `@font-face{font-family:'Barlow Condensed';src:url('${base}${file}') format('truetype');font-weight:${weight};font-style:${style};font-display:swap;}`;
  const st = document.createElement('style');
  st.textContent =
    face('BarlowCondensed-Medium.ttf', 500, 'normal') +
    face('BarlowCondensed-SemiBold.ttf', 600, 'normal') +
    face('BarlowCondensed-SemiBoldItalic.ttf', 600, 'italic') +
    face('BarlowCondensed-ExtraBoldItalic.ttf', 800, 'italic');
  document.head.append(st);
  // warm the faces used by canvas text so the first gauge frame is not drawn with a fallback
  if (document.fonts?.load) {
    void document.fonts.load("italic 800 20px 'Barlow Condensed'");
    void document.fonts.load("italic 600 20px 'Barlow Condensed'");
    void document.fonts.load("600 20px 'Barlow Condensed'");
  }
}

async function boot(): Promise<void> {
  installFonts();
  const q = (params.get('q') as QualityName | null) ?? resolveQuality(settings.quality);
  const game = new Game(canvas, q);
  (window as unknown as { __apex: unknown }).__apex = { game, settings, audio };

  const direct = params.get('race') === '1' || params.get('bot') === '1';
  const ui = new UI(game, uiRoot, {
    hud: params.get('hud') !== '0',
    skipTitle: params.get('menu') === '1',
    fps: params.get('fps') === '1',
    autopilot: params.get('bot') === '1',
  });

  // event overrides from the URL (also used by the menu when present)
  const w = params.get('w') as WeatherId | null;
  if (w && WEATHER[w]) settings.weather = w;
  const car = params.get('car');
  if (car && CAR_SPECS[car]) settings.car = car;
  const paint = params.get('paint');
  if (paint) settings.paint = paint.startsWith('#') ? paint : `#${paint}`;
  const num = (k: string, lo: number, hi: number): number | null => {
    const v = params.get(k);
    if (v === null || v === '' || Number.isNaN(Number(v))) return null;
    return Math.min(hi, Math.max(lo, Math.round(Number(v))));
  };
  const laps = num('laps', 1, 99);
  const ai = num('ai', 0, 7);
  const diff = num('diff', 0, 2);
  if (laps !== null) settings.laps = laps;
  if (ai !== null) settings.aiCount = ai;
  if (diff !== null) settings.difficulty = diff;

  game.start();
  if (direct) await ui.startRace();
  else await game.enterMenu();
}

boot().catch((e) => {
  console.error(e);
  uiRoot.classList.remove('ui-off');
  uiRoot.innerHTML = `<pre style="position:absolute;inset:0;margin:0;padding:24px;color:#ff9a8a;background:#05070b;white-space:pre-wrap;font:14px/1.5 ui-monospace,monospace;pointer-events:auto;overflow:auto">启动失败 / failed to start\n\n${String(e && e.stack ? e.stack : e)}</pre>`;
});
