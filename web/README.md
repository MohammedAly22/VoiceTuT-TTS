# VoiceTut-TTS — web UI

React + Vite + framer-motion front end for VoiceTut-TTS. It talks to the Python server
(`voicetut_tts/server.py`) through its Gradio API (`/gradio`), so it works the same locally
and on a HuggingFace ZeroGPU Space.

```
web/                      sources (this folder)
voicetut_tts/webui/       production build, shipped inside the Python package
```

## Develop

```bash
# 1) API server — no GPU needed in mock mode (synthetic audio)
VOICETUT_MOCK=1 python app.py          # http://localhost:7860

# 2) UI with hot reload, proxied to the server above
cd web
npm install
npm run dev                            # http://localhost:5173
```

## Build

```bash
cd web
npm run build                          # writes ../voicetut_tts/webui
```

Commit the updated `voicetut_tts/webui` with your source changes. The Space and
`pip install voicetut-tts[web]` serve that build directly.
