// Screenshot tour around the lap: teleports the (autopilot) player to each distance and captures a frame.
//   npx tsx tools/tour.ts "<query>" <out-prefix> "<s1,s2,...>" [width] [height] [settle seconds]
import { chromium } from 'playwright';

const query = process.argv[2] ?? 'race=1&bot=1&hud=0';
const prefix = process.argv[3] ?? '/tmp/apex_tour';
const stops = (process.argv[4] ?? '0,600,1200,1800,2400,3000').split(',').map(Number);
const width = Number(process.argv[5] ?? 1280);
const height = Number(process.argv[6] ?? 720);
const settle = Number(process.argv[7] ?? 1.6);
const base = process.env.APEX_URL ?? 'http://localhost:5199/';

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
const logs: string[] = [];
page.on('console', (m) => {
  if (m.type() === 'error') logs.push(`[error] ${m.text().slice(0, 300)}`);
});
page.on('pageerror', (e) => logs.push(`[pageerror] ${String(e).slice(0, 400)}`));
await page.goto(base + `?${query}`, { waitUntil: 'load' });
await page.waitForFunction(`window.__apex && window.__apex.game && window.__apex.game.race && window.__apex.game.race.state === 'racing'`, undefined, { timeout: 90000 });
for (const s of stops) {
  await page.evaluate(`(() => {
    const g = window.__apex.game;
    const r = g.race;
    const c = r.player;
    c.placeOnTrack(g.track, ${s}, g.track.line[g.track.indexAt(${s})] * 0.6);
    c.hold = false;
    // roll in at the reference pace so the camera sits naturally
    const f = Math.hypot(c.m02, c.m22) || 1;
    const v = 38;
    c.vx = (c.m02 / f) * v; c.vz = (c.m22 / f) * v;
    for (const w of c.wheels) w.omega = v / w.radius;
    c.gear = 3;
    r.info[r.playerIdx].lastS = c.s;
    g.cam.snap();
  })()`);
  await page.waitForTimeout(settle * 1000);
  const file = `${prefix}_${s}.png`;
  await page.screenshot({ path: file });
  const info = await page.evaluate(`(() => { const g = window.__apex.game; const i = g.engine.renderer.info; return { fps: Math.round(g.fps), calls: i.render.calls, tris: i.render.triangles, kmh: Math.round(g.race.player.speed * 3.6) }; })()`);
  console.log(`shot ${file}`, JSON.stringify(info));
}
console.log(logs.length ? [...new Set(logs)].slice(0, 8).join('\n') : 'console clean');
await browser.close();
