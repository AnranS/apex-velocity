import { defineConfig } from 'vite';

// Static H5 build: `npm run build` -> dist/ (relative paths, host anywhere).
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 4000,
    sourcemap: false,
  },
  server: { port: 5199, host: true, strictPort: false },
  preview: { port: 4173, host: true },
});
