// Browser smoke test + screenshots:
//   npx tsx tools/shot.ts "<url query>" <out-prefix> [seconds,seconds,...] [width] [height] [headed]
// Opens the dev server page, logs console errors, and saves a screenshot at each timestamp.
import { chromium } from 'playwright';

const query = process.argv[2] ?? '';
const prefix = process.argv[3] ?? '/tmp/apex_shot';
const times = (process.argv[4] ?? '6').split(',').map(Number);
const width = Number(process.argv[5] ?? 1280);
const height = Number(process.argv[6] ?? 720);
const headed = process.argv[7] === 'headed';
const base = process.env.APEX_URL ?? 'http://localhost:5199/';

const browser = await chromium.launch({
  channel: 'chrome',
  headless: !headed,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-webgl', '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
const logs: string[] = [];
page.on('console', (m) => {
  const t = m.type();
  if (t === 'error' || t === 'warning') logs.push(`[${t}] ${m.text().slice(0, 400)}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${String(e).slice(0, 600)}`));
page.on('requestfailed', (r) => logs.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText ?? ''}`));
page.on('response', (r) => {
  if (r.status() >= 400) logs.push(`[http ${r.status()}] ${r.url()}`);
});
await page.goto(base + (query ? `?${query}` : ''), { waitUntil: 'load' });
let prev = 0;
for (const t of times) {
  await page.waitForTimeout(Math.max(0, (t - prev) * 1000));
  prev = t;
  const file = `${prefix}_${String(t).replace('.', '_')}.png`;
  await page.screenshot({ path: file });
  const info = await page.evaluate(() => {
    const g = (window as unknown as { __apex?: { game: any } }).__apex?.game;
    if (!g) return null;
    const r = g.race;
    const p = r?.player;
    const ri = g.engine.renderer.info;
    return {
      fps: +g.fps.toFixed(1), gpu: g.engine.gpu, q: g.engine.q.name, dpr: +g.engine.dpr.toFixed(2),
      calls: ri.render.calls, tris: ri.render.triangles, programs: ri.programs?.length, trees: g.scenery?.treeCount,
      state: r?.state, t: r ? +r.time.toFixed(1) : 0, kmh: p ? Math.round(p.speed * 3.6) : 0, pos: r?.playerInfo.position, s: p ? Math.round(p.s) : 0,
    };
  });
  console.log(`shot ${file}`, JSON.stringify(info));
}
const uniq = [...new Set(logs)];
console.log(uniq.length ? `--- console (${uniq.length})\n${uniq.slice(0, 25).join('\n')}` : '--- console clean');
await browser.close();
