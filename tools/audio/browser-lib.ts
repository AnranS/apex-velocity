/**
 * Shared Playwright plumbing for the browser-side audio tools.
 */
import { chromium, type Browser, type Page } from 'playwright';
import { build as viteBuild } from 'vite';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { WORK } from './lib';

export const ROOT = new URL('../../', import.meta.url).pathname;
export const ENTRY = `${ROOT}tools/audio/browser-entry.ts`;
// --disable-audio-output: Chromium renders into a fake (timer-driven) sink, so the tests neither make a sound nor
// depend on the state of the machine's audio device.
const ARGS = ['--mute-audio', '--disable-audio-output', '--autoplay-policy=no-user-gesture-required'];

export async function launch(): Promise<Browser> {
  try {
    return await chromium.launch({ channel: 'chrome', headless: true, args: ARGS });
  } catch {
    /* fall through to a cached Chromium */
  }
  const cache = `${process.env.HOME}/Library/Caches/ms-playwright`;
  const dirs = readdirSync(cache)
    .filter((d) => /^chromium-\d+$/.test(d))
    .sort()
    .reverse();
  for (const d of dirs) {
    for (const sub of readdirSync(`${cache}/${d}`)) {
      for (const app of ['Chromium.app/Contents/MacOS/Chromium', 'Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing']) {
        const p = `${cache}/${d}/${sub}/${app}`;
        if (existsSync(p)) return chromium.launch({ executablePath: p, headless: true, args: ARGS });
      }
    }
  }
  throw new Error('no Chromium available');
}

export async function openPage(browser: Browser, js: string): Promise<Page> {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('   [pageerror]', e.message));
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning') console.log(`   [console.${m.type()}]`, m.text());
  });
  await page.route('https://audio.test/**', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><html><body>audio test</body></html>' }));
  await page.goto('https://audio.test/');
  await page.addScriptTag({ content: js });
  return page;
}

export function decode(b64: string): [Float32Array, Float32Array] {
  const buf = Buffer.from(b64, 'base64');
  const n = buf.length / 4;
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    L[i] = buf.readInt16LE(i * 4) / 32767;
    R[i] = buf.readInt16LE(i * 4 + 2) / 32767;
  }
  return [L, R];
}

/** Bundle the in-page test entry into an IIFE that exposes `AT` (same Vite/Rolldown pipeline as the game build). */
export async function bundle(minify: boolean): Promise<string> {
  const outDir = `${WORK}bundle-${minify ? 'min' : 'dev'}`;
  await viteBuild({
    configFile: false,
    root: ROOT,
    logLevel: 'silent',
    build: { outDir, emptyOutDir: true, target: 'es2022', minify, lib: { entry: ENTRY, formats: ['iife'], name: 'AT', fileName: () => 'at.js' } },
  });
  return readFileSync(`${outDir}/at.js`, 'utf8');
}
