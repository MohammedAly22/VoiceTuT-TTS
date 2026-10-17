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
- diacritics
base_model:
- k2-fsa/OmniVoice
---

<div align="center">

<img src="https://raw.githubusercontent.com/MohammedAly22/VoiceTuT-TTS/main/assets/VoiceTut-TTS-Banner.png" alt="VoiceTut-TTS" width="100%" />

# 𓋹 VoiceTut-TTS

**An Open-Source Text-to-Speech Model for Egyptian Arabic & Code-Switching**

[![🤗 Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/mohammedaly22/VoiceTut-TTS)
[![🤗 Space](https://img.shields.io/badge/🤗%20Space-Demo-blue)](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS)
[![🎧 Samples](https://img.shields.io/badge/🎧%20Audio-Demos-ff9800)](https://mohammedaly22.github.io/VoiceTuT-TTS/)
[![PyPI](https://img.shields.io/pypi/v/voicetut-tts?color=blue&label=PyPI)](https://pypi.org/project/voicetut-tts/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/MohammedAly22/VoiceTuT-TTS)
[![Base](https://img.shields.io/badge/Base-OmniVoice-purple)](https://github.com/k2-fsa/OmniVoice)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/MohammedAly22/VoiceTuT-TTS/blob/main/LICENSE)

VoiceTut-TTS is an Egyptian-Arabic text-to-speech model fine-tuned from [OmniVoice](https://github.com/k2-fsa/OmniVoice). It produces natural Egyptian speech with seamless **Arabic ↔ English code-switching**, ships **19 built-in voices**, supports **zero-shot voice cloning**, and includes a robust Egyptian-Arabic **text normalization** pipeline plus **true streaming** for long text.

</div>

> **Why "VoiceTut"?** *Tut* — after the boy-king **Tutankhamun** (توت عنخ آمون) — anchors the model in Egyptian identity, just as our companion ASR model **[QwenCleo-ASR](https://github.com/MohammedAly22/qwencleo-asr)** is named after **Cleopatra**. Together they form an Egyptian speech stack: **Cleo listens, Tut speaks.** 🎙️🗣️

## 🆕 What's new in v2

**VoiceTut-TTS-v2** is a more capable model that **supports Arabic diacritics (التشكيل)**. Plain text works exactly as before, and if a word is pronounced wrong you can fix it by adding diacritics to that word — the model follows them.

```python
tts.synthesize("انا مْخَطَّط للموضوع ده", speaker="Mohammed", output="fixed.wav")
```

The previous weights remain available on the [`v1` branch](https://huggingface.co/mohammedaly22/VoiceTut-TTS/tree/v1).

## 🔗 Links

- 🎧 **Audio demo (VoiceTut vs. base OmniVoice):** https://mohammedaly22.github.io/VoiceTuT-TTS/
- 🚀 **Interactive Space:** https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS
- 💻 **GitHub (code, notebooks):** https://github.com/MohammedAly22/VoiceTuT-TTS
- 📦 **PyPI:** https://pypi.org/project/voicetut-tts/

## ✨ Features

- 🎯 **Egyptian-first** — fine-tuned specifically on Egyptian Arabic, not generic MSA.
- 🔀 **Code-switching** — handles real Arabic + English mixed speech (`عندي meeting بكرة`).
- 🗣️ **19 built-in voices** — male & female speakers.
- 🧬 **Zero-shot cloning** — clone any voice from a few seconds of reference audio.
- ✍️ **Diacritics support (v2)** — add التشكيل to any word to control its pronunciation.
- 🔢 **Robust normalization** — numbers, dates, times, currencies, phones, emails, URLs, abbreviations + an English→Arabic name map.
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
tts.synthesize("ازيك عامل ايه النهاردة؟", speaker="Mohammed", output="out.wav")

# 2) Zero-shot voice cloning
tts.synthesize("النهارده الجو حلو اوي",
               ref_audio="my_voice.wav", ref_text="ده الصوت بتاعي", output="clone.wav")

# 3) Code-switching + generation params
tts.synthesize("عندي meeting الساعة 3:30 ومعايا ال presentation",
               speaker="Yasmin", num_step=48, guidance_scale=2.5, speed=1.05, output="cs.wav")

# 4) Fix a word's pronunciation with diacritics
tts.synthesize("دي عِشْرة عمر يا صاحبي", speaker="Ahmed", output="fixed.wav")
```

Streaming long text:

```python
for sr, chunk in tts.stream(long_paragraph, speaker="Ahmed"):
    play(chunk)                    # plays each sentence as it's generated
tts.synthesize_long(long_paragraph, "long.wav", speaker="Ahmed")
```

## 🗣️ Built-in Voices

| | Male | Female |
|---|---|---|
| Names | Ahmed, Ali, Hassan, Karim, Mahmoud, Mohammed, Mostafa, Ramy, Shaker, Tareq, Yehia, Youssef | Hadeer, Hanan, Mariam, Rana, Sarah, Yasmin, Yara |

Each voice ships with a reference clip. Browse and listen in the [Space](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS).


## 📊 Performance

> Measured on a single **NVIDIA T4 (Colab)**, `float16`, `num_step=32`. Reproduce with [`examples/04_evaluation.ipynb`](https://github.com/MohammedAly22/VoiceTuT-TTS/blob/main/examples/04_evaluation.ipynb).

| Metric | Value |
|---|---|
| Real-time factor (RTF, mean) | **1.13×** |
| RTF (best) | **0.49×** |
| Time-to-first-audio (streaming) | **1.68 s** |
| Peak VRAM (fp16) | **2.93 GB** |
| WER — Egyptian Arabic | **0.40** |
| WER — English | **0.07** |
| Speaker similarity (cloning, cosine) | **0.83** |
| Naturalness (UTMOS, 1–5) | **3.47** |
| Sampling rate | 24 kHz |

> On A100 / H100 expect markedly lower RTF and TTFA.

## 🏗️ Model

- **Base model:** [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (Qwen3-0.6B text backbone + Higgs audio tokenizer)
- **Language id:** `arz` (Egyptian Arabic), with English code-switching (`en`)

## ⚠️ Responsible Use

Voice cloning is provided to enable **beneficial** use cases — voice assistants, accessibility,
educational and creative content. **Do not** use it to impersonate real people, produce deceptive
or misleading audio, or harm, harass, or defraud anyone. Always obtain consent before cloning a
real person's voice, and disclose synthetic audio where appropriate.

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
