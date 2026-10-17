"""
VoiceTut-TTS — web app (React UI + Gradio API). Also the HuggingFace Space entry point.

Run:
    pip install "voicetut-tts[web]"
    python app.py                                   # default HF checkpoint, port 7860
    VOICETUT_CKPT=exp/omnivoice_egy/checkpoint-8000 python app.py
    VOICETUT_MOCK=1 python app.py                   # UI preview without a model / GPU

The UI is a prebuilt React app shipped inside the package (voicetut_tts/webui);
its sources live in web/ (see web/README.md).
"""

import logging

# keep this import first: on ZeroGPU Spaces `spaces` must load before torch
from voicetut_tts.server import launch

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    launch()
