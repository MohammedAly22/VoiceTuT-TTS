import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev`   -> UI on :5173, API calls proxied to the Python server on :7860
// `npm run build` -> static build written into the Python package (voicetut_tts/webui)
const API = process.env.VOICETUT_API ?? "http://localhost:7860";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../voicetut_tts/webui",
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
  server: {
    port: 5173,
    proxy: {
      "/gradio": { target: API, changeOrigin: false, ws: true },
      "/voices": { target: API, changeOrigin: false },
    },
  },
});
