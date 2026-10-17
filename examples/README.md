# 📓 VoiceTut-TTS — Example Notebooks

Colab-ready notebooks. Each installs the **OmniVoice backbone first**, then `voicetut-tts`.
Set runtime to **GPU** (T4 is enough).

| Notebook | What it covers | Open |
|---|---|---|
| [01_quickstart.ipynb](01_quickstart.ipynb) | Install, load the model, and synthesize with a **built-in voice**. Pure Egyptian, code-switching, and **streaming** for long text. Start here. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/01_quickstart.ipynb) |
| [02_voice_cloning.ipynb](02_voice_cloning.ipynb) | **Zero-shot voice cloning** from a short reference (upload/record), plus the **text-normalization** pipeline (numbers, dates, times, phones, emails) and **fixing a word's pronunciation with diacritics**. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/02_voice_cloning.ipynb) |
| [03_web_ui.ipynb](03_web_ui.ipynb) | Launch the **VoiceTut web UI** from Colab — voice gallery, diacritics toolbar, cloning (upload or mic), streaming, light/dark mode. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/03_web_ui.ipynb) |
| [04_evaluation.ipynb](04_evaluation.ipynb) | **Evaluate the model**: RTF, time-to-first-audio, peak VRAM, WER (ASR round-trip), speaker similarity, and UTMOS naturalness. Prints a summary table to paste into the README / model card. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/04_evaluation.ipynb) |

## Tips

- **First run is slow** — Colab downloads the model + (for eval) the ASR/MOS models. Subsequent cells are fast.
- **Pick a voice:** `tts.list_speakers()` lists the 19 built-in voices (name, gender, tags).
- **Language:** pass `language="arz"` (Egyptian Arabic, default) or `language="en"` (English). Other languages are not supported.
- **Quality vs. speed:** raise `num_step` (e.g. 48) for quality, lower it (e.g. 16) for speed.

See the [main README](../README.md) for full installation and API docs.
