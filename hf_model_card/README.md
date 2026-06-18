---
license: apache-2.0
language:
- ar
- arz
- en
library_name: omnivoice
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- egyptian-arabic
- arabic
- code-switching
- voice-cloning
- omnivoice
base_model:
- k2-fsa/OmniVoice
---

<div align="center">

<img src="https://raw.githubusercontent.com/MohammedAly22/VoiceTuT-TTS/main/assets/VoiceTut-TTS-Banner.png" alt="VoiceTut-TTS" width="100%" />

# 𓋹 VoiceTut-TTS

**The best open-source text-to-speech model for Egyptian Arabic & code-switching**

[![🤗 Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/mohammedaly22/VoiceTut-TTS)
[![🤗 Space](https://img.shields.io/badge/🤗%20Space-Demo-blue)](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS)
[![PyPI](https://img.shields.io/pypi/v/voicetut-tts?color=blue&label=PyPI)](https://pypi.org/project/voicetut-tts/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/MohammedAly22/VoiceTuT-TTS)
[![Base](https://img.shields.io/badge/Base-OmniVoice-purple)](https://github.com/k2-fsa/OmniVoice)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/MohammedAly22/VoiceTuT-TTS/blob/main/LICENSE)

</div>

VoiceTut-TTS is an Egyptian-Arabic text-to-speech model fine-tuned from [OmniVoice](https://github.com/k2-fsa/OmniVoice) on ~380 hours of Egyptian podcast speech. It produces natural Egyptian speech with seamless **Arabic ↔ English code-switching**, ships **15 built-in studio voices**, supports **zero-shot voice cloning**, and includes a robust Egyptian-Arabic **text normalization** pipeline plus **true streaming** for long text.

> **Why "VoiceTut"?** *Tut* — after the boy-king **Tutankhamun** (توت عنخ آمون) — anchors the model in Egyptian identity, just as our companion ASR model **[QwenCleo-ASR](https://github.com/MohammedAly22/qwencleo-asr)** is named after **Cleopatra**. Together they form an Egyptian speech stack: **Cleo listens, Tut speaks.** 🎙️🗣️


## ✨ Features

- 🎯 **Egyptian-first** — fine-tuned specifically on Egyptian Arabic, not generic MSA.
- 🔀 **Code-switching** — handles real Arabic + English mixed speech (`عندي meeting بكرة`).
- 🗣️ **15 built-in voices** — male & female studio speakers, each with style tags.
- 🧬 **Zero-shot cloning** — clone any voice from a few seconds of reference audio.
- 🔢 **Robust normalization** — numbers, dates, times, currencies, phones, emails, URLs, abbreviations + diacritics & name dictionaries.
- ⚡ **True streaming** — long text is split into sentences and yielded as audio chunks.

## 📦 Installation

```bash
# PyTorch matching your CUDA (see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# OmniVoice backbone (not on PyPI — install from GitHub)
pip install git+https://github.com/k2-fsa/OmniVoice.git
pip install voicetut-tts
```

## 🚀 Usage

```python
from voicetut_tts import VoiceTutTTS

tts = VoiceTutTTS.from_pretrained("mohammedaly22/VoiceTut-TTS")

# 1) Built-in speaker
tts.synthesize("ازيك عامل ايه النهاردة؟", speaker="Mohamed", output="out.wav")

# 2) Zero-shot voice cloning
tts.synthesize("النهارده الجو حلو اوي",
               ref_audio="my_voice.wav", ref_text="ده الصوت بتاعي", output="clone.wav")

# 3) Code-switching + generation params
tts.synthesize("عندي meeting الساعة 3:30 ومعايا ال presentation",
               speaker="Asmaa", num_step=48, guidance_scale=2.5, speed=1.05, output="cs.wav")
```

Streaming long text:

```python
for sr, chunk in tts.stream(long_paragraph, speaker="Sayed"):
    play(chunk)                    # plays each sentence as it's generated
tts.synthesize_long(long_paragraph, "long.wav", speaker="Sayed")
```

## 🗣️ Built-in Voices

| | Male | Female |
|---|---|---|
| Names | Abdelrahman, Abdullah, Kamal, Hossam, Mohamed, Omar, Sayed, Zaki, Aly | Asmaa, Esraa, Hanan, Sarah, Yasmin, Omnia |

Each voice ships with a reference clip + Arabic style tags (e.g. `شبابي`, `حيوي`, `هادي`). Browse and listen in the [Space](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS).


## 🏗️ Training

- **Base model:** [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (Qwen3-0.6B text backbone + Higgs audio tokenizer)
- **Data:** ~380 h Egyptian-Arabic YouTube podcasts (`language_id = arz`)
- **Steps:** 20,000 · **LR:** 3e-5 · **bf16**

## 📜 License & Citation

Apache-2.0.

```bibtex
@software{voicetut_tts_2026,
  author  = {Mohammed Aly},
  title   = {VoiceTut-TTS: Egyptian Arabic & Code-Switching Text-to-Speech},
  year    = {2026},
  url     = {https://github.com/MohammedAly22/VoiceTuT-TTS},
  note    = {Fine-tuned from OmniVoice}
}
```
