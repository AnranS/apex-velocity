// Scripted keyboard driving with screenshots:
//   npx tsx tools/drive.ts "<query>" <out-prefix> "<script>" [width] [height]
// script: semicolon-separated steps, each "<seconds>:<keys>[:shot]" where keys is a +-joined list of
// KeyboardEvent codes held during that step ("-" = none), e.g. "5:KeyW;1.5:KeyW+Space+KeyA:shot;2:KeyW"
// The clock starts at GO (race.state === 'racing').
import { chromium } from 'playwright';

const query = process.argv[2] ?? 'race=1';
const prefix = process.argv[3] ?? '/tmp/apex_drive';
const script = process.argv[4] ?? '4:KeyW';
const width = Number(process.argv[5] ?? 1280);
const height = Number(process.argv[6] ?? 720);
const base = process.env.APEX_URL ?? 'http://localhost:5199/';

const browser = await chromium.launch({
  channel: 'chrome',
  headless: process.env.HEADED !== '1',
  // REAL_AUDIO=1: keep the real output device but start the game with master volume 0 (silent, no fake-device artifacts)
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required', ...(process.env.REAL_AUDIO === '1' ? [] : ['--mute-audio'])],
});
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
if (process.env.REAL_AUDIO === '1') await page.addInitScript(`localStorage.setItem('apex-velocity-settings-v1', JSON.stringify({ master: 0 }))`);
const logs: string[] = [];
page.on('console', (m) => {
  if (m.type() === 'error') logs.push(`[error] ${m.text().slice(0, 300)}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${String(e).slice(0, 400)}`));
await page.goto(base + `?${query}`, { waitUntil: 'load' });
// first user gesture (unlocks audio) before the race, like the title screen does in the normal flow
if (process.env.APEX_PREGESTURE !== '0') {
  await page.waitForTimeout(1500);
  await page.keyboard.press('KeyZ');
}
await page.waitForFunction(() => (window as any).__apex?.game?.race?.state === 'racing', undefined, { timeout: 60000 });
await page.evaluate(`(() => {
  window.__hitch = { max: 0, sum: 0 };
  let last = performance.now();
  function tick(now) {
    const d = now - last;
    last = now;
    if (d > window.__hitch.max) window.__hitch.max = d;
    if (d > 50) window.__hitch.sum += d;
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})()`);
const tele = () =>
  page.evaluate(() => {
    const g = (window as any).__apex.game;
    const p = g.race.player;
    const h = (window as any).__hitch;
    const hitch = Math.round(h.max);
    const lost = Math.round(h.sum);
    h.max = 0;
    h.sum = 0;
    const pr = g.prof;
    const prof = pr ? Object.fromEntries(Object.entries(pr).map(([k, v]) => [k, Math.round(v as number)]).filter(([, v]) => (v as number) >= 12)) : {};
    if (pr) for (const k of Object.keys(pr)) pr[k] = 0;
    const ri = g.engine.renderer.info;
    return {
      hitch, lost, prof, gl: [ri.programs.length, ri.memory.geometries, ri.memory.textures],
      kmh: Math.round(p.speed * 3.6), gear: p.gear, beta: +((p.beta * 180) / Math.PI).toFixed(0), drift: p.drifting, nos: +p.nitro.toFixed(2), bottles: p.nitroBottles, kind: p.boostKind, nitroOn: p.nitroOn,
      mb: +p.miniBoost.toFixed(2), s: Math.round(p.s), d: +p.d.toFixed(1), pos: g.race.playerInfo.position, fps: Math.round(g.fps),
    };
  });
let held: string[] = [];
let n = 0;
for (const step of script.split(';')) {
  const [secs, keys, flag] = step.split(':');
  const want = keys && keys !== '-' ? keys.split('+') : [];
  for (const k of held) if (!want.includes(k)) await page.keyboard.up(k);
  for (const k of want) if (!held.includes(k)) await page.keyboard.down(k);
  held = want;
  // condition steps: "s>320" (lap distance reached) or "v<118" / "v>200" (speed in km/h)
  const cond = /^([sv])([<>])([\d.]+)$/.exec(secs);
  if (cond) {
    const [, what, op, val] = cond;
    const target = Number(val);
    const t0 = Date.now();
    for (;;) {
      const cur = await page.evaluate((w) => {
        const p = (window as any).__apex.game.race.player;
        return w === 's' ? (p.s > 3000 ? p.s - (window as any).__apex.game.track.length : p.s) : p.speed * 3.6;
      }, what);
      if ((op === '>' && cur >= target) || (op === '<' && cur <= target) || Date.now() - t0 > 40000) break;
      await page.waitForTimeout(25);
    }
    console.log(`step ${step}`, JSON.stringify(await tele()));
    continue;
  }
  const dur = Number(secs) * 1000;
  if (flag === 'shot') {
    await page.waitForTimeout(dur * 0.6);
    const file = `${prefix}_${n++}.png`;
    await page.screenshot({ path: file });
    console.log(`shot ${file}`, JSON.stringify(await tele()));
    await page.waitForTimeout(dur * 0.4);
  } else {
    await page.waitForTimeout(dur);
    console.log(`step ${step}`, JSON.stringify(await tele()));
  }
}
for (const k of held) await page.keyboard.up(k);
console.log(logs.length ? [...new Set(logs)].slice(0, 10).join('\n') : 'console clean');
await browser.close();
