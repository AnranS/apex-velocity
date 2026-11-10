// UI controller: loading splash, title, menus over the live showroom, race HUD, pause, results,
// touch controls. Owns input-action routing (the only consumer of game.input.takeActions()).
import { audio } from '../audio';
import type { Game, GameMode, RaceOptions } from '../core/game';
import { resolveQuality, saveSettings, settings } from '../core/settings';
import type { QualityName } from '../render/engine';
import { WEATHER, WEATHER_IDS, WeatherId } from '../render/sky';
import { CAR_IDS, CAR_SPECS } from '../sim/car_specs';
import type { RaceEvent } from '../sim/race';
import { TRACKS } from '../sim/tracks';
import { clamp, fmtTime, h, isTouchDevice, setText, toggle } from './dom';
import { Hud } from './hud';
import { bigItem, ctaItem, ghostItem, Item, NavAction, optionRow, Page, sliderRow } from './menu';
import { TouchControls } from './touch';

export interface UIOptions {
  /** false = render no UI at all (clean screenshots) */
  hud: boolean;
  /** go straight to the main menu instead of the title */
  skipTitle: boolean;
  fps: boolean;
  autopilot: boolean;
}

type Ctx = 'menu' | 'sheet';
type State = 'boot' | 'title' | 'menu' | 'race';

const TIPS = [
  'Shift + 方向键 起漂；松开 Shift 后反打方向"拉车头"，车身回正漂移才结束',
  '漂移集气：集满一格得到一瓶氮气，最多存两瓶，按 Ctrl 喷射',
  '车头回正的瞬间松开再按 ↑，喷出一脚"小喷"',
  '小喷之后立刻再点一下漂移并补 ↑，就是"双喷"',
  '氮气喷射时接小喷可以叠加提速（CWW）',
  '绿灯亮起的瞬间点一下 Shift，触发"超级起步"',
  'R 复位到赛道 · C 切换视角 · B 回头看',
];

const EXTRA_PAINTS = ['#1f6feb', '#0f8a5f', '#6d28d9', '#c0c4cc', '#e84a8a', '#ff6a13', '#7fd4ff'];
const QUALITY_ORDER: (QualityName | 'auto')[] = ['auto', 'low', 'medium', 'high', 'ultra'];
const QUALITY_LABEL: Record<string, string> = { auto: '自动', low: '流畅', medium: '均衡', high: '高画质', ultra: '极致' };
const DIFF_LABEL: [string, string][] = [
  ['简单', 'EASY'],
  ['普通', 'NORMAL'],
  ['困难', 'HARD'],
];
const WX_STRIP: Record<WeatherId, string> = {
  sunset: 'linear-gradient(90deg,#ff7a2a,#ffb35c,#7a5ad8)',
  day: 'linear-gradient(90deg,#3fa9ff,#bfe6ff)',
  storm: 'linear-gradient(90deg,#5d6876,#aeb7c2)',
  night: 'linear-gradient(90deg,#0d1a3a,#4b6cd8)',
};

function logoHtml(): string {
  return (
    '<div class="cn">极速之巅</div>' +
    '<div class="en"><span>APEX</span><span class="accent-text">VELOCITY</span></div>' +
    '<div class="tag"><i></i><span class="micro">CRIMSON BAY CIRCUIT · 赤霞湾国际赛道</span></div>'
  );
}

function ordinal(n: number): string {
  return n === 1 ? 'ST' : n === 2 ? 'ND' : n === 3 ? 'RD' : 'TH';
}

export class UI {
  private readonly game: Game;
  private readonly root: HTMLElement;
  private readonly opts: UIOptions;
  private readonly isTouch: boolean;

  private loadingEl: HTMLElement;
  private loadFill: HTMLElement;
  private loadLabel: HTMLElement;
  private loadPct: HTMLElement;
  private loadTip: HTMLElement;
  private loadTarget = 0.02;
  private loadShown = 0;
  private loadingActive = true;

  private menuEl: HTMLElement;
  private pageHost: HTMLElement;
  private hintsEl: HTMLElement;
  private cardEl: HTMLElement;
  private titleEl: HTMLElement;
  private sheetEl: HTMLElement;

  private readonly hud: Hud;
  private readonly touch: TouchControls;
  private stack: { page: Page; ctx: Ctx }[] = [];
  private state: State = 'boot';
  private seenTitle = false;
  private extra: NavAction[] = [];
  private anyKey = false;

  private resultsTimer = -1;
  private resultsRefresh: (() => void) | null = null;
  private resultsT = 0;
  private pbBefore = 0;
  private qualityAtOpen: QualityName | 'auto' = settings.quality;

  // showroom camera
  private drag: { x: number; y: number; id: number } | null = null;
  private idleT = 0;
  private distTarget = 6.6;
  private userZoom = false;
  private viewOx = 0;
  private viewTarget = 0;
  private viewKey = '';
  private stickT = 0;

  constructor(game: Game, root: HTMLElement, opts: UIOptions) {
    this.game = game;
    this.root = root;
    this.opts = opts;
    this.isTouch = isTouchDevice();
    if (!opts.hud) root.classList.add('ui-off');
    if (this.isTouch) root.classList.add('is-touch');

    // ---- layers (back to front)
    this.hud = new Hud(game, root, opts.fps);

    this.menuEl = h('div', 'menu');
    this.pageHost = h('div', 'host');
    this.hintsEl = h('div', 'hints');
    this.cardEl = h('div', 'card hidden');
    const brand = h('div', 'brand', '<div class="cn">极速之巅</div><div class="en">APEX <span class="accent-text">VELOCITY</span></div><div class="rule"></div>');
    this.menuEl.append(h('div', 'menu-scrim'), brand, this.pageHost, this.cardEl, this.hintsEl);

    this.titleEl = h('div', 'title hidden');
    this.titleEl.append(
      h('div', 'scrim'),
      h('div', 'logo', logoHtml()),
      h('div', 'press', `<i></i>${this.isTouch ? '点击屏幕开始 · TAP TO START' : '按任意键开始 · PRESS ANY KEY'}`),
      h('div', 'venue', `<span class="micro">NOW AT · ${TRACKS[settings.track].name}</span><b>${TRACKS[settings.track].nameCn}</b><span class="micro">${TRACKS[settings.track].blurb}</span>`),
    );
    this.titleEl.addEventListener('click', () => {
      if (this.state === 'title') this.anyKey = true;
    });

    this.sheetEl = h('div', 'sheet hidden');
    this.touch = new TouchControls(
      game,
      root,
      () => this.openPause(),
      () => this.game.resetPlayer(),
      () => this.game.nextCamera(),
    );

    this.loadingEl = h('div', 'loading');
    this.loadFill = h('i');
    const bar = h('div', 'bar');
    bar.append(this.loadFill);
    this.loadLabel = h('b', '', '正在启动引擎');
    const status = h('div', 'status');
    status.append(this.loadLabel, h('span', 'micro', 'LOADING'));
    this.loadPct = h('div', 'pct num', '0%');
    this.loadTip = h('div', 'tip', `提示 · ${TIPS[Math.floor(Math.random() * TIPS.length)]}`);
    this.loadingEl.append(h('div', 'logo', logoHtml()), status, this.loadPct, bar, this.loadTip);

    const rotate = h(
      'div',
      'rotate',
      '<div><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="7" y="2.5" width="10" height="19" rx="2"/><path d="M11 18.5h2"/></svg><b>请横屏游玩</b><span class="micro">ROTATE YOUR DEVICE</span></div>',
    );
    root.append(this.menuEl, this.titleEl, this.sheetEl, this.loadingEl, rotate);
    document.getElementById('boot')?.remove();

    // ---- wiring
    game.onLoading((s) => {
      if (s.active) {
        if (!this.loadingActive) {
          this.loadShown = 0;
          this.loadTip.textContent = `提示 · ${TIPS[Math.floor(Math.random() * TIPS.length)]}`;
          this.loadingEl.classList.add('mini');
        }
        this.loadingActive = true;
        this.loadTarget = Math.max(0.03, s.progress);
        setText(this.loadLabel, s.label || '加载中');
        this.loadingEl.classList.remove('out');
      } else {
        this.loadTarget = 1;
        this.loadingActive = false;
        this.loadingEl.classList.add('out');
      }
    });
    game.onMode((m) => this.onMode(m));
    game.onRaceEvent((e) => this.onRaceEvent(e));
    game.onFrame((dt) => this.frame(dt));

    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement) return;
      if (this.state === 'title') {
        if (!e.repeat && !e.metaKey && !e.ctrlKey && !e.altKey) this.anyKey = true;
        return;
      }
      if (!this.top()) return;
      // extra menu keys: WASD + Space, plus key-repeat for adjust rows
      const map: Record<string, NavAction> = { KeyW: 'up', KeyS: 'down', KeyA: 'left', KeyD: 'right', Space: 'confirm' };
      const rep: Record<string, NavAction> = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down' };
      const a = map[e.code];
      if (a && (!e.repeat || a === 'left' || a === 'right')) this.extra.push(a);
      else if (e.repeat && rep[e.code]) this.extra.push(rep[e.code]);
    });
    document.addEventListener('visibilitychange', () => {
      const r = this.game.race;
      if (document.hidden && this.game.mode === 'race' && r && r.state === 'racing' && !this.top()) this.openPause();
    });
    this.bindOrbit();
  }

  // ------------------------------------------------------------------ page stack
  private top(): Page | null {
    return this.stack.length ? this.stack[this.stack.length - 1].page : null;
  }

  private show(): void {
    const t = this.stack[this.stack.length - 1];
    const menuOpen = !!t && t.ctx === 'menu';
    const sheetOpen = !!t && t.ctx === 'sheet';
    toggle(this.menuEl, 'open', menuOpen);
    toggle(this.sheetEl, 'hidden', !sheetOpen);
    if (menuOpen) this.pageHost.replaceChildren(t.page.el);
    else this.pageHost.replaceChildren();
    if (sheetOpen) this.sheetEl.replaceChildren(t.page.el);
    else this.sheetEl.replaceChildren();
    const pad = this.game.input.lastDevice === 'gamepad';
    const padKey: Record<string, string> = { Enter: 'A', Esc: 'B', '↑': '▲', '↓': '▼', '←': '◀', '→': '▶' };
    this.hintsEl.innerHTML =
      t && menuOpen
        ? t.page.hints.map(([keys, label]) => `<span>${keys.map((k) => `<kbd>${pad ? (padKey[k] ?? k) : k}</kbd>`).join('')}${label}</span>`).join('')
        : '';
    if (!t || t.page !== this.mainPage) this.cardEl.classList.add('hidden');
  }

  private push(page: Page, ctx: Ctx): void {
    this.stack.push({ page, ctx });
    // restart entry animations
    page.el.style.animation = 'none';
    void page.el.offsetWidth;
    page.el.style.animation = '';
    this.show();
    page.onOpen?.();
    page.refresh();
  }

  private pop(): void {
    const t = this.stack.pop();
    t?.page.onClose?.();
    this.show();
    const n = this.top();
    if (n) {
      n.onOpen?.();
      n.refresh();
    }
  }

  private closeAll(): void {
    while (this.stack.length) this.stack.pop()?.page.onClose?.();
    this.show();
  }

  // ------------------------------------------------------------------ flow
  private onMode(m: GameMode): void {
    if (m === 'menu') {
      this.hud.setVisible(false);
      this.touch.setActive(false);
      this.closeAll();
      this.resultsTimer = -1;
      this.resultsRefresh = null;
      if (!this.seenTitle && !this.opts.skipTitle) this.showTitle();
      else this.openMain();
    } else if (m === 'race') {
      this.state = 'race';
      this.closeAll();
      this.titleEl.classList.add('hidden');
      this.resultsTimer = -1;
      this.resultsRefresh = null;
      const key = `${settings.track}|${this.game.options?.car ?? settings.car}`;
      this.pbBefore = settings.bestLaps[key] ?? 0;
      this.hud.reset();
      this.hud.setVisible(true);
      this.touch.setActive(this.isTouch);
      this.clearViewOffset();
    }
  }

  private showTitle(): void {
    this.state = 'title';
    this.seenTitle = true;
    this.anyKey = false;
    this.titleEl.classList.remove('hidden');
    this.viewTarget = 0.07;
    this.distTarget = 7.0;
    this.userZoom = false;
  }

  private openMain(): void {
    this.state = 'menu';
    this.seenTitle = true;
    this.titleEl.classList.add('hidden');
    this.closeAll();
    this.push(this.mainPage, 'menu');
  }

  /** Start a race from the current settings (optionally overridden). */
  async startRace(over: Partial<RaceOptions> = {}): Promise<void> {
    const opts: RaceOptions = {
      trackId: settings.track,
      weather: settings.weather,
      car: settings.car,
      paint: settings.paint,
      laps: settings.laps,
      aiCount: settings.aiCount,
      difficulty: settings.difficulty,
      mode: 'race',
      autopilot: this.opts.autopilot,
      ...over,
    };
    saveSettings();
    this.closeAll();
    this.titleEl.classList.add('hidden');
    this.state = 'race';
    await this.game.startRace(opts);
  }

  private onRaceEvent(e: RaceEvent): void {
    this.hud.onEvent(e);
    const race = this.game.race;
    if (e.type === 'finish' && race && e.car === race.playerIdx) this.resultsTimer = 2.6;
  }

  // ------------------------------------------------------------------ pages
  private _main: Page | null = null;
  private get mainPage(): Page {
    if (this._main) return this._main;
    const p = new Page('page');
    p.add(bigItem(1, '开始比赛', 'START RACE', () => void this.startRace()));
    p.add(bigItem(2, '车库', 'GARAGE', () => this.push(this.garagePage(), 'menu')));
    p.add(bigItem(3, '赛事设置', 'EVENT SETUP', () => this.push(this.eventPage(), 'menu')));
    p.add(bigItem(4, '设置', 'SETTINGS', () => this.push(this.settingsPage('menu'), 'menu')));
    p.add(bigItem(5, '操作说明', 'CONTROLS', () => this.push(this.controlsPage(), 'menu')));
    p.onBack = () => {
      this.closeAll();
      this.showTitle();
    };
    p.onOpen = () => {
      this.viewTarget = 0.14;
      this.distTarget = 7.3;
      this.userZoom = false;
      this.refreshCard();
      this.cardEl.classList.remove('hidden');
    };
    this._main = p;
    return p;
  }

  private refreshCard(): void {
    const spec = CAR_SPECS[settings.car] ?? CAR_SPECS.aurion;
    const tr = TRACKS[settings.track];
    const best = settings.bestLaps[`${settings.track}|${settings.car}`];
    this.cardEl.innerHTML =
      `<span class="micro">CURRENT SETUP 当前配置</span>` +
      `<div class="name">${spec.name}</div><div class="cn">${spec.nameCn} · ${spec.tagline}</div>` +
      `<div class="grid">` +
      `<div><span class="micro">TRACK 赛道</span><b>${tr.nameCn.replace('国际赛道', '')}</b></div>` +
      `<div><span class="micro">WEATHER 天气</span><b>${WEATHER[settings.weather].nameCn}</b></div>` +
      `<div><span class="micro">RIVALS 对手</span><b>${settings.aiCount} · ${DIFF_LABEL[settings.difficulty]?.[0] ?? ''}</b></div>` +
      `<div><span class="micro">LAPS 圈数</span><b class="num">${settings.laps}</b></div>` +
      `<div><span class="micro">BEST LAP 最快圈</span><b class="num">${best ? fmtTime(best) : '-:--.---'}</b></div>` +
      `<div><span class="micro">WINS 胜场</span><b class="num">${settings.wins}/${settings.races}</b></div>` +
      `</div>`;
  }

  private garagePage(): Page {
    const p = new Page('page', '车库', 'GARAGE · 选择座驾与涂装');
    p.hints = [
      [['←', '→'], '切换'],
      [['↑', '↓'], '选择'],
      [['Esc'], '返回'],
    ];
    // car tabs
    const carsEl = h('div', 'cars');
    const tabs: HTMLElement[] = [];
    const spec = h('div', 'spec');
    const paintItemEl = h('div', 'item paintRow');
    const swWrap = h('div', 'paints');
    paintItemEl.append(h('div', 'lab', '<span class="c">涂装</span><span class="micro">PAINT</span>'), swWrap);
    let paints: string[] = [];

    const renderSpec = () => {
      const s = CAR_SPECS[settings.car];
      const st = s.stats;
      const row = (k: string, v: number) => `<div class="stat"><span class="k">${k}</span><span class="track"><i style="transform:scaleX(${(v / 10).toFixed(3)})"></i></span><span class="v">${v.toFixed(1)}</span></div>`;
      spec.innerHTML =
        `<div class="tag">${s.tagline}</div>` +
        `<div class="eng"><span>${s.engineLabel}</span><span><span class="num">${s.powerHp}</span> HP</span><span>${s.drive}</span><span><span class="num">${s.mass}</span> KG</span></div>` +
        `<div class="stats">${row('极速', st.speed)}${row('加速', st.accel)}${row('操控', st.handling)}${row('制动', st.braking)}</div>`;
    };
    const renderPaints = () => {
      const s = CAR_SPECS[settings.car];
      paints = [...s.paints];
      const rgb = (c: string) => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
      const far = (c: string) => paints.every((q) => Math.hypot(...rgb(c).map((v, i) => v - rgb(q)[i])) > 70);
      for (const c of EXTRA_PAINTS) if (far(c)) paints.push(c);
      if (!paints.includes(settings.paint)) paints.unshift(settings.paint);
      swWrap.replaceChildren();
      paints.forEach((c) => {
        const sw = h('div', 'sw');
        sw.style.background = c;
        toggle(sw, 'sel', c === settings.paint);
        sw.addEventListener('click', (e) => {
          e.stopPropagation();
          audio.play('ui_move');
          setPaint(c);
        });
        swWrap.append(sw);
      });
    };
    const setPaint = (c: string) => {
      settings.paint = c;
      this.game.setShowroomPaint(c);
      Array.from(swWrap.children).forEach((el, i) => toggle(el as HTMLElement, 'sel', paints[i] === c));
    };
    const setCar = (id: string) => {
      if (id === settings.car) return;
      settings.car = id;
      settings.paint = CAR_SPECS[id].paints[0];
      tabs.forEach((t, i) => toggle(t, 'sel', CAR_IDS[i] === id));
      renderSpec();
      renderPaints();
      void this.game.setShowroomCar(id, settings.paint);
    };
    CAR_IDS.forEach((id) => {
      const s = CAR_SPECS[id];
      const t = h('div', 'item carTab', `<div class="n">${s.name}</div><div class="c">${s.nameCn}</div>`);
      toggle(t, 'sel', id === settings.car);
      t.addEventListener('click', (e) => {
        e.stopPropagation();
        audio.play('ui_select');
        setCar(id);
      });
      tabs.push(t);
      carsEl.append(t);
    });
    const carsItem: Item = {
      el: carsEl,
      adjust: (d) => {
        const i = CAR_IDS.indexOf(settings.car);
        setCar(CAR_IDS[(i + d + CAR_IDS.length) % CAR_IDS.length]);
      },
    };
    p.add(carsItem);
    p.list.append(spec);
    p.add({
      el: paintItemEl,
      adjust: (d) => {
        const i = Math.max(0, paints.indexOf(settings.paint));
        setPaint(paints[(i + d + paints.length) % paints.length]);
      },
    });
    p.add(ghostItem('确认并返回', () => p.onBack?.()));
    p.onBack = () => {
      saveSettings();
      this.pop();
    };
    p.onOpen = () => {
      this.viewTarget = 0.16;
      this.distTarget = 6.5;
      this.userZoom = false;
    };
    renderSpec();
    renderPaints();
    return p;
  }

  private eventPage(): Page {
    const p = new Page('page', '赛事设置', 'EVENT SETUP · 赤霞湾国际赛道');
    p.hints = [
      [['←', '→'], '调整'],
      [['↑', '↓'], '选择'],
      [['Enter'], '确认'],
      [['Esc'], '返回'],
    ];
    const wxEl = h('div', 'item wxRow');
    const wxGrid = h('div', 'wx');
    wxEl.append(h('div', 'lab', '<span class="c">天气与时段</span><span class="micro">CONDITIONS</span>'), wxGrid);
    const tiles: HTMLElement[] = [];
    const setWx = (id: WeatherId) => {
      if (settings.weather === id) return;
      settings.weather = id;
      tiles.forEach((t, i) => toggle(t, 'sel', WEATHER_IDS[i] === id));
      void this.game.setMenuWeather(id);
    };
    WEATHER_IDS.forEach((id) => {
      const w = WEATHER[id];
      const t = h('div', 'wxTile', `<span class="dot" style="background:${WX_STRIP[id]}"></span><b>${w.nameCn}</b><small>${w.desc}</small>`);
      toggle(t, 'sel', id === settings.weather);
      t.addEventListener('click', (e) => {
        e.stopPropagation();
        audio.play('ui_move');
        setWx(id);
      });
      tiles.push(t);
      wxGrid.append(t);
    });
    p.add({
      el: wxEl,
      adjust: (d) => {
        const i = WEATHER_IDS.indexOf(settings.weather);
        setWx(WEATHER_IDS[(i + d + WEATHER_IDS.length) % WEATHER_IDS.length]);
      },
    });
    p.add(
      optionRow('圈数', 'LAPS', () => [`${settings.laps} 圈`, `约 ${Math.round((settings.laps * 80) / 60)} 分钟`], (d) => {
        settings.laps = clamp(settings.laps + d, 1, 10);
      }),
    );
    p.add(
      optionRow('对手', 'OPPONENTS', () => [settings.aiCount === 0 ? '单人练习' : `${settings.aiCount} 台`, settings.aiCount === 0 ? 'SOLO RUN' : `GRID OF ${settings.aiCount + 1}`], (d) => {
        settings.aiCount = clamp(settings.aiCount + d, 0, 7);
      }),
    );
    p.add(
      optionRow('难度', 'DIFFICULTY', () => [DIFF_LABEL[settings.difficulty][0], DIFF_LABEL[settings.difficulty][1]], (d) => {
        settings.difficulty = clamp(settings.difficulty + d, 0, 2);
      }),
    );
    p.add(ctaItem('开始比赛', 'START RACE', () => void this.startRace()));
    p.add(ghostItem('返回', () => p.onBack?.()));
    p.onBack = () => {
      saveSettings();
      this.pop();
    };
    p.onOpen = () => {
      this.viewTarget = 0.17;
      this.distTarget = 7.2;
      this.userZoom = false;
    };
    return p;
  }

  private settingsPage(ctx: Ctx): Page {
    const inRace = ctx === 'sheet';
    const p = new Page(inRace ? 'pausePanel wide' : 'page', '设置', 'SETTINGS');
    p.list.classList.add('scroll');
    p.hints = [
      [['←', '→'], '调整'],
      [['↑', '↓'], '选择'],
      [['Esc'], '返回'],
    ];
    this.qualityAtOpen = settings.quality;
    const apply = () => this.game.applySettings();
    p.add(
      optionRow(
        '画质',
        'QUALITY',
        () => {
          const q = settings.quality;
          const sub = q === 'auto' ? `AUTO → ${resolveQuality('auto').toUpperCase()}` : q.toUpperCase();
          return [QUALITY_LABEL[q], q !== this.qualityAtOpen ? (inRace ? '返回主菜单后重新加载生效' : '返回时重新加载') : sub];
        },
        (d) => {
          const i = QUALITY_ORDER.indexOf(settings.quality);
          settings.quality = QUALITY_ORDER[(i + d + QUALITY_ORDER.length) % QUALITY_ORDER.length];
          saveSettings();
        },
      ),
    );
    p.add(
      optionRow('操控手感', 'HANDLING', () => (settings.handling === 'arcade' ? ['街机漂移', inRace ? '下一场比赛生效' : 'ARCADE DRIFT'] : ['拟真', inRace ? '下一场比赛生效' : 'SIMULATION']), () => {
        settings.handling = settings.handling === 'arcade' ? 'sim' : 'arcade';
        apply();
      }),
    );
    p.add(
      optionRow('变速箱', 'GEARBOX', () => (settings.autoGear ? ['自动', inRace ? '下一场比赛生效' : 'AUTOMATIC'] : ['手动', 'E 升档 · Q 降档']), () => {
        settings.autoGear = !settings.autoGear;
        apply();
      }),
    );
    p.add(
      optionRow('动态模糊', 'MOTION BLUR', () => (settings.motionBlur ? ['开启', 'ON'] : ['关闭', 'OFF']), () => {
        settings.motionBlur = !settings.motionBlur;
        apply();
      }),
    );
    p.add(
      optionRow('速度单位', 'UNITS', () => (settings.units === 'kmh' ? ['KM/H', '公制'] : ['MPH', '英制']), () => {
        settings.units = settings.units === 'kmh' ? 'mph' : 'kmh';
        apply();
        this.hud.reset();
      }),
    );
    const vol = (cn: string, en: string, key: 'master' | 'music' | 'sfx' | 'engine') =>
      sliderRow(cn, en, () => settings[key], (v) => {
        settings[key] = v;
        apply();
      });
    p.add(vol('主音量', 'MASTER', 'master'));
    p.add(vol('音乐', 'MUSIC', 'music'));
    p.add(vol('音效', 'EFFECTS', 'sfx'));
    p.add(vol('引擎声', 'ENGINE', 'engine'));
    p.add(ghostItem('返回', () => p.onBack?.()));
    p.onBack = () => {
      saveSettings();
      if (!inRace && settings.quality !== this.qualityAtOpen) {
        location.reload();
        return;
      }
      this.pop();
    };
    return p;
  }

  private controlsPage(): Page {
    const p = new Page('page', '操作说明', 'CONTROLS');
    p.hints = [[['Esc'], '返回']];
    const k = (...keys: string[]) => keys.map((x) => `<kbd>${x}</kbd>`).join('');
    const r = (keys: string, d: string) => `<div class="r"><div class="k">${keys}</div><div class="d">${d}</div></div>`;
    const help = h(
      'div',
      'help',
      `<h4 class="micro">KEYBOARD 键盘</h4>` +
        r(k('↑', '↓', '←', '→'), '加速 · 刹车 / 倒车 · 转向（也可用 W A S D）') +
        r(k('Shift'), '漂移（配合方向键）') +
        r(k('Ctrl'), '使用氮气（Mac 上 Ctrl + 方向键是系统快捷键，可改用 空格）') +
        r(k('R'), '复位') +
        r(k('C') + k('B'), '切换视角 · 回头看') +
        r(k('Esc') + ' / ' + k('P'), '暂停') +
        `<h4 class="micro">GAMEPAD 手柄</h4>` +
        r(k('RT') + k('LT') + k('L 摇杆'), '油门 · 刹车 · 转向') +
        r(k('A') + k('X') + k('Y') + k('START'), '漂移 · 氮气 · 视角 · 暂停') +
        `<h4 class="micro">HOW TO DRIFT 漂移要领</h4>` +
        `<p><b>漂移</b>：按住方向键再按 Shift 起漂，按住 Shift 角度越甩越大；松开 Shift 后车仍在滑行，反打方向键"拉车头"让车身回正，漂移才算结束。<br><b>小喷 / 双喷</b>：车身回正的瞬间松开再按一下 ↑ 出"小喷"；小喷后立刻再点一下 Shift + 方向并补 ↑，就是"双喷"。<br><b>氮气</b>：漂移时集气，集满得到一瓶（最多两瓶），按 Ctrl 喷射约 3 秒；喷射中接小喷可以叠加提速。<br><b>起步 / 落地</b>：绿灯亮起瞬间点一下 Shift 是"超级起步"；腾空落地的瞬间按 ↑ 是"落地喷"。</p>`,
    );
    p.list.append(help);
    p.add(ghostItem('返回', () => p.onBack?.()));
    p.onBack = () => this.pop();
    return p;
  }

  // ------------------------------------------------------------------ pause / results
  private openPause(): void {
    const race = this.game.race;
    if (this.game.mode !== 'race' || !race || this.top() || race.state === 'finished') return;
    this.game.setPaused(true);
    this.touch.setActive(false);
    const p = new Page('pausePanel', '暂停', 'PAUSED');
    const resume = () => {
      this.closeAll();
      this.game.setPaused(false);
      this.touch.setActive(this.isTouch);
    };
    p.add(bigItem(1, '继续比赛', 'RESUME', resume));
    p.add(
      bigItem(2, '重新开始', 'RESTART', () => {
        this.closeAll();
        void this.game.restartRace();
      }),
    );
    p.add(bigItem(3, '设置', 'SETTINGS', () => this.push(this.settingsPage('sheet'), 'sheet')));
    p.add(
      bigItem(4, '返回主菜单', 'QUIT TO MENU', () => {
        this.closeAll();
        void this.game.quitToMenu();
      }),
    );
    p.onBack = resume;
    this.push(p, 'sheet');
  }

  private openResults(): void {
    const race = this.game.race;
    if (!race || this.game.mode !== 'race') return;
    this.closeAll();
    this.hud.setVisible(false);
    this.touch.setActive(false);
    const p = new Page('results');
    p.horizontal = true;
    p.list.className = 'acts';
    const inf = race.playerInfo;
    const pos = inf.position;
    const solo = race.cars.length < 2;
    const title = solo ? '练习结束' : pos === 1 ? '冠军' : pos === 2 ? '亚军' : pos === 3 ? '季军' : '比赛结束';
    const newPb = inf.bestLap > 0 && (this.pbBefore === 0 || inf.bestLap < this.pbBefore - 1e-4);
    const top = h('div', 'top');
    top.innerHTML =
      (solo ? '' : `<div class="place"><span class="${pos <= 3 ? 'accent-text' : ''}">${pos}</span><sup>${ordinal(pos)}</sup></div>`) +
      `<div class="hd"><div class="t">${title}</div><div class="micro">RACE COMPLETE · ${TRACKS[settings.track].name}</div></div>` +
      `<div class="pb ${newPb ? 'new' : ''}"><span class="micro">BEST LAP 最快圈</span><span class="num">${fmtTime(inf.bestLap)}</span>` +
      (newPb ? `<span class="badge">个人最佳 NEW PB</span>` : this.pbBefore ? `<span class="micro">PB ${fmtTime(this.pbBefore)}</span>` : '') +
      `</div>`;
    const table = h('div', 'rtable');
    const render = () => {
      let html = `<div class="rrow head"><span>名次</span><span>车手</span><span>车型</span><span style="text-align:right">用时</span><span style="text-align:right">最快圈</span></div>`;
      const leaderT = race.info[race.order[0]].finishTime;
      race.order.forEach((ci, k) => {
        const x = race.info[ci];
        const spec = race.cars[ci].spec;
        let t: string;
        if (!x.finished) t = '<span class="t dnf">比赛中…</span>';
        else if (k === 0 || !leaderT) t = `<span class="t">${fmtTime(x.finishTime)}</span>`;
        else t = `<span class="t">+${(x.finishTime - leaderT).toFixed(3)}</span>`;
        html +=
          `<div class="rrow ${x.isPlayer ? 'me' : ''}" style="--i:${k}"><span class="p">${k + 1}</span>` +
          `<span class="n">${x.isPlayer ? '你 YOU' : x.name}</span><span class="c">${spec.name}</span>${t}<span class="b">${fmtTime(x.bestLap)}</span></div>`;
      });
      table.innerHTML = html;
    };
    render();
    this.resultsRefresh = () => {
      // rows keep their entry animation only on first paint
      render();
      table.querySelectorAll<HTMLElement>('.rrow').forEach((r) => (r.style.animation = 'none'));
    };
    p.el.insertBefore(top, p.list);
    p.el.insertBefore(table, p.list);
    p.add(
      ctaItem('再来一局', 'RACE AGAIN', () => {
        this.closeAll();
        void this.game.restartRace();
      }),
    );
    p.add(
      ctaItem('返回菜单', 'BACK TO MENU', () => {
        this.closeAll();
        void this.game.quitToMenu();
      }),
    );
    p.items[1].el.classList.replace('cta', 'ghostBtn');
    this.push(p, 'sheet');
  }

  // ------------------------------------------------------------------ showroom camera
  private bindOrbit(): void {
    const canvas = this.game.engine.canvas;
    const mc = this.game.menuCam;
    canvas.addEventListener('pointerdown', (e) => {
      if (this.game.mode !== 'menu' || this.state !== 'menu') return;
      this.drag = { x: e.clientX, y: e.clientY, id: e.pointerId };
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      mc.auto = false;
    });
    canvas.addEventListener('pointermove', (e) => {
      const d = this.drag;
      if (!d || d.id !== e.pointerId) return;
      mc.yaw -= (e.clientX - d.x) * 0.006;
      mc.pitch = clamp(mc.pitch + (e.clientY - d.y) * 0.004, 0.03, 0.62);
      d.x = e.clientX;
      d.y = e.clientY;
    });
    const end = (e: PointerEvent) => {
      if (this.drag && this.drag.id === e.pointerId) {
        this.drag = null;
        this.idleT = 0;
      }
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener(
      'wheel',
      (e) => {
        if (this.game.mode !== 'menu' || this.state !== 'menu') return;
        e.preventDefault();
        this.userZoom = true;
        mc.dist = clamp(mc.dist * Math.exp(e.deltaY * 0.0012), 4.2, 10.5);
        mc.auto = false;
        this.idleT = 0;
      },
      { passive: false },
    );
  }

  /** Shift the showroom framing sideways so the car sits beside the menu column. */
  private updateViewOffset(dt: number): void {
    const eng = this.game.engine;
    const cam = eng.camera;
    this.viewOx += (this.viewTarget - this.viewOx) * (1 - Math.exp(-dt * 4));
    const w = eng.width;
    const hgt = eng.height;
    const key = `${w}|${hgt}|${this.viewOx.toFixed(3)}`;
    if (key === this.viewKey) return;
    this.viewKey = key;
    // narrow / portrait-ish screens: keep the car centred
    const ox = w / hgt < 1.45 ? this.viewOx * 0.35 : this.viewOx;
    cam.setViewOffset(w, hgt, -w * ox, hgt * 0.045, w, hgt);
    eng.cameraChanged();
  }

  private clearViewOffset(): void {
    const eng = this.game.engine;
    if (eng.camera.view && eng.camera.view.enabled) {
      eng.camera.clearViewOffset();
      eng.cameraChanged();
    }
    this.viewKey = '';
    this.viewOx = 0;
  }

  // ------------------------------------------------------------------ per frame
  private frame(dt: number): void {
    const game = this.game;
    // ---- loading bar easing
    if (this.loadShown < this.loadTarget) {
      this.loadShown = Math.min(this.loadTarget, this.loadShown + dt * (this.loadingActive ? 0.9 : 3));
      this.loadFill.style.transform = `scaleX(${this.loadShown.toFixed(3)})`;
      setText(this.loadPct, `${Math.round(this.loadShown * 100)}%`);
    }

    // ---- input actions (single consumer)
    const acts = game.input.takeActions();
    this.pollStick(dt);
    const nav = this.extra;
    this.extra = [];
    if (!this.loadingActive) {
      if (this.state === 'title') {
        if (this.anyKey || acts.length) {
          this.anyKey = false;
          audio.play('ui_select');
          this.openMain();
        }
      } else {
        const page = this.top();
        if (page) {
          for (const a of acts) {
            if (a === 'pause') page.act('back');
            else if (a === 'up' || a === 'down' || a === 'left' || a === 'right' || a === 'confirm' || a === 'back') page.act(a);
            if (this.top() !== page) break;
          }
          if (this.top() === page) for (const a of nav) page.act(a);
        } else if (game.mode === 'race') {
          for (const a of acts) {
            if (a === 'camera') game.nextCamera();
            else if (a === 'reset') game.resetPlayer();
            else if (a === 'pause') this.openPause();
          }
        }
      }
    }

    // ---- showroom camera
    if (game.mode === 'menu') {
      const mc = game.menuCam;
      if (!this.drag && !mc.auto) {
        this.idleT += dt;
        if (this.idleT > 4) mc.auto = true;
      }
      if (!this.userZoom) mc.dist += (this.distTarget - mc.dist) * (1 - Math.exp(-dt * 2.5));
      this.updateViewOffset(dt);
    }

    // ---- results
    if (this.resultsTimer > 0) {
      this.resultsTimer -= dt;
      if (this.resultsTimer <= 0) this.openResults();
    }
    if (this.resultsRefresh && this.top()?.el.classList.contains('results')) {
      this.resultsT -= dt;
      if (this.resultsT <= 0) {
        this.resultsT = 0.6;
        this.resultsRefresh();
      }
    }

    this.hud.update(dt);
    this.touch.update();
  }

  /** Left-stick menu navigation with auto-repeat. */
  private pollStick(dt: number): void {
    if (!this.top()) return;
    this.stickT -= dt;
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const p of pads) {
      if (!p || !p.connected) continue;
      const x = p.axes[0] ?? 0;
      const y = p.axes[1] ?? 0;
      const ax = Math.abs(x) > 0.6 ? (x > 0 ? 'right' : 'left') : null;
      const ay = Math.abs(y) > 0.6 ? (y > 0 ? 'down' : 'up') : null;
      const a = (ay ?? ax) as NavAction | null;
      if (!a) {
        this.stickT = 0;
      } else if (this.stickT <= 0) {
        this.extra.push(a);
        this.stickT = 0.22;
      }
      break;
    }
  }
}
