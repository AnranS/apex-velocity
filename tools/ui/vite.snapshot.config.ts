// Dev server for UI test runs: same working tree, but no HMR, so a page under test is never
// reloaded when another file is saved mid-run.  npx vite --config tools/ui/vite.snapshot.config.ts
import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('../..', import.meta.url)),
  base: './',
  cacheDir: 'node_modules/.vite-ui',
  // hmr off = pages are never auto-reloaded; the watcher stays on so each new page load gets fresh modules
  server: { port: 5183, strictPort: true, host: 'localhost', hmr: false },
});
