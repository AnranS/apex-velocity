// UI flow driver:  npx tsx tools/ui/flow.ts <scenario> [width] [height] [outDir]
// scenarios: menus | race | results | touch | perf
import { mkdirSync } from 'node:fs';
import { chromium, type Page } from 'playwright';

const scenario = process.argv[2] ?? 'menus';
const width = Number(process.argv[3] ?? 1920);
const height = Number(process.argv[4] ?? 1080);
const out = process.argv[5] ?? 'tools/ui/_work';
const base = process.env.APEX_URL ?? 'http://localhost:5199/';
mkdirSync(out, { recursive: true });
const tag = `${width}x${height}`;

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl', '--autoplay-policy=no-user-gesture-required'],
});
const touch = scenario === 'touch';
const ctx = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: touch ? 2 : 1, hasTouch: touch, isMobile: touch });
const page = await ctx.newPage();
const logs: string[] = [];
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text().slice(0, 300)}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${String(e).slice(0, 500)}`));

const shot = async (name: string) => {
  const f = `${out}/${scenario}_${tag}_${name}.png`;
  await page.screenshot({ path: f });
  console.log('shot', f);
};
const wait = (ms: number) => page.waitForTimeout(ms);
const key = async (k: string, n = 1, gap = 140) => {
  for (let i = 0; i < n; i++) {
    await page.keyboard.press(k);
    await wait(gap);
  }
};
const info = (p: Page) =>
  p.evaluate(() => {
    const g = (window as unknown as { __apex?: { game: any } }).__apex?.game;
    if (!g) return null;
    const r = g.race;
    return { mode: g.mode, fps: +g.fps.toFixed(1), state: r?.state, t: r ? +r.time.toFixed(1) : 0, pos: r?.playerInfo.position, laps: r?.playerInfo.lapsDone, paused: g.paused, loading: g.loading.active };
  });
const waitMode = async (mode: string, timeout = 60000) => {
  await page.waitForFunction((m) => { const g = (window as any).__apex?.game; return g && g.mode === m && !g.loading.active; }, mode, { timeout });
};

if (scenario === 'menus') {
  await page.goto(base, { waitUntil: 'load' });
  await wait(700);
  await shot('00_loading');
  await waitMode('menu');
  await wait(1800);
  await shot('01_title');
  await key('Enter');
  await wait(1300);
  await shot('02_main');
  await key('ArrowDown');
  await key('Enter');
  await wait(1400);
  await shot('03_garage');
  await key('ArrowRight');
  await wait(1800);
  await key('ArrowDown');
  await key('ArrowRight', 3);
  await wait(900);
  await shot('04_garage_car2_paint');
  await key('ArrowUp');
  await key('ArrowLeft');
  await wait(1200);
  await key('Escape');
  await wait(500);
  await key('ArrowDown');
  await key('Enter');
  await wait(1200);
  await shot('05_event');
  await key('ArrowRight', 3, 900);
  await wait(1500);
  await shot('06_event_night');
  await key('ArrowLeft', 3, 700);
  await key('ArrowDown');
  await key('ArrowRight', 2);
  await key('Escape');
  await wait(500);
  await key('ArrowDown');
  await key('Enter');
  await wait(900);
  await shot('07_settings');
  await key('ArrowDown', 5);
  await key('ArrowLeft', 3);
  await wait(300);
  await shot('08_settings_slider');
  await key('ArrowRight', 3);
  await key('Escape');
  await wait(500);
  await key('ArrowDown');
  await key('Enter');
  await wait(900);
  await shot('09_controls');
  await key('Escape');
  await wait(600);
  // mouse: hover an item and drag-orbit the showroom
  await page.mouse.move(width * 0.7, height * 0.5);
  await page.mouse.down();
  await page.mouse.move(width * 0.55, height * 0.42, { steps: 8 });
  await page.mouse.up();
  await wait(500);
  await shot('10_main_orbit');
  console.log('info', JSON.stringify(await info(page)));
} else if (scenario === 'race') {
  await page.goto(`${base}?menu=1`, { waitUntil: 'load' });
  await waitMode('menu');
  await wait(800);
  await key('Enter'); // START RACE
  await waitMode('race');
  await wait(2600);
  await shot('01_countdown');
  await wait(2600);
  await shot('02_go');
  // drive: throttle, then a drift with nitro
  await page.keyboard.down('KeyW');
  await wait(5200);
  await page.keyboard.down('Space');
  await page.keyboard.down('KeyA');
  await wait(1100);
  await shot('03_drift');
  await page.keyboard.up('KeyA');
  await page.keyboard.up('Space');
  await wait(350);
  await shot('04_miniboost');
  await page.evaluate(() => { (window as any).__apex.game.race.player.nitro = 1; });
  await page.keyboard.down('KeyW');
  await page.keyboard.down('KeyF');
  await wait(450);
  await page.keyboard.up('KeyF');
  await wait(250);
  await shot('05_nitro');
  await page.keyboard.up('KeyW');
  await key('Escape');
  await wait(600);
  await shot('06_pause');
  await key('ArrowDown', 2);
  await key('Enter');
  await wait(600);
  await shot('07_pause_settings');
  await key('Escape');
  await wait(300);
  await key('Escape');
  await wait(500);
  console.log('info', JSON.stringify(await info(page)));
  // quit to menu through the pause menu
  await key('Escape');
  await wait(400);
  await key('ArrowUp');
  await key('Enter');
  await waitMode('menu');
  await wait(1200);
  await shot('08_back_in_menu');
  console.log('info', JSON.stringify(await info(page)));
} else if (scenario === 'results') {
  await page.goto(`${base}?race=1&bot=1&laps=1&ai=7`, { waitUntil: 'load' });
  await waitMode('race');
  await wait(9000);
  await shot('01_racing');
  await page.waitForFunction(() => (window as any).__apex.game.race?.state === 'finished', null, { timeout: 180000 });
  await wait(4200);
  await shot('02_results');
  await wait(9000);
  await shot('03_results_later');
  await key('ArrowRight');
  await wait(300);
  await shot('04_results_focus2');
  await key('Enter');
  await waitMode('menu');
  await wait(1500);
  await shot('05_menu_after');
  console.log('info', JSON.stringify(await info(page)));
} else if (scenario === 'touch') {
  await page.goto(`${base}?touch=1`, { waitUntil: 'load' });
  await waitMode('menu');
  await wait(1500);
  await shot('01_title');
  await page.touchscreen.tap(width * 0.5, height * 0.5);
  await wait(1200);
  await shot('02_main');
  // tap 车库
  const items = page.locator('.page .item.big');
  await items.nth(1).tap();
  await wait(1500);
  await shot('03_garage');
  await page.locator('.item.ghost').last().tap();
  await wait(600);
  await items.nth(2).tap();
  await wait(1000);
  await shot('04_event');
  await page.locator('.item.cta').tap();
  await waitMode('race');
  await wait(7000);
  await shot('05_race');
  // hold drift + steer
  const box = await page.locator('.t-steer').boundingBox();
  if (box) {
    await page.evaluate(([x, y]) => {
      const el = document.querySelector('.t-steer')!;
      el.dispatchEvent(new PointerEvent('pointerdown', { clientX: x, clientY: y, pointerId: 11, bubbles: true, pointerType: 'touch' }));
    }, [box.x + box.width * 0.2, box.y + box.height / 2]);
  }
  await wait(900);
  await shot('06_race_steer');
  await page.locator('.t-pause').tap();
  await wait(600);
  await shot('07_pause');
  console.log('info', JSON.stringify(await info(page)));
} else if (scenario === 'perf') {
  for (const q of ['race=1&bot=1&hud=0', 'race=1&bot=1&fps=1']) {
    await page.goto(`${base}?${q}`, { waitUntil: 'load' });
    await waitMode('race');
    await wait(16000);
    const fpsSamples: number[] = [];
    for (let i = 0; i < 8; i++) {
      await wait(500);
      fpsSamples.push((await info(page))!.fps);
    }
    console.log(q, 'fps', fpsSamples.join(' '));
  }
  await shot('fps_hud');
}
const uniq = [...new Set(logs)].filter((l) => !l.includes('.vite/deps'));
console.log(uniq.length ? `--- console (${uniq.length})\n${uniq.slice(0, 20).join('\n')}` : '--- console clean');
await browser.close();
