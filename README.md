<div align="center">

<img src="assets/VoiceTut-TTS-Banner.png" alt="VoiceTut-TTS" width="100%" />

# 𓋹 VoiceTut-TTS

**An Open-Source Text-to-Speech Model for Egyptian Arabic & Code-Switching**

[![🤗 Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/mohammedaly22/VoiceTut-TTS)
[![🤗 Space](https://img.shields.io/badge/🤗%20Space-Demo-blue)](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS)
[![🎧 Samples](https://img.shields.io/badge/🎧%20Audio-Demos-ff9800)](https://mohammedaly22.github.io/VoiceTuT-TTS/)
[![PyPI](https://img.shields.io/pypi/v/voicetut-tts?color=blue&label=PyPI)](https://pypi.org/project/voicetut-tts/)
[![Base](https://img.shields.io/badge/Base-OmniVoice-purple)](https://github.com/k2-fsa/OmniVoice)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/01_quickstart.ipynb)

</div>

> 🎧 **Listen:** hear VoiceTut-TTS vs. the base OmniVoice on Egyptian Arabic & code-switching → **[Audio demos](https://mohammedaly22.github.io/VoiceTuT-TTS/)** · 🚀 **Try it live:** **[HuggingFace Space](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS)**

VoiceTut-TTS is an Egyptian-Arabic text-to-speech system fine-tuned from [OmniVoice](https://github.com/k2-fsa/OmniVoice). It produces natural Egyptian speech with seamless **Arabic ↔ English code-switching**, ships **19 built-in voices**, supports **zero-shot voice cloning**, and includes a robust Egyptian-Arabic **text normalization** pipeline plus **true streaming** for long text.

> **Why "VoiceTut"?** *Tut* — after the boy-king **Tutankhamun** (توت عنخ آمون) — anchors the model in Egyptian identity, just as our companion ASR model **[QwenCleo-ASR](https://github.com/MohammedAly22/qwencleo-asr)** is named after **Cleopatra**. Together they form an Egyptian speech stack: **Cleo listens, Tut speaks.** 🎙️🗣️

## ✨ Features

- 🎯 **Egyptian-first** — fine-tuned specifically on Egyptian Arabic, not generic MSA.
- 🔀 **Code-switching** — handles real Arabic + English mixed speech (`عندي meeting بكرة`).
- 🗣️ **19 built-in voices** — male & female speakers.
- 🧬 **Zero-shot cloning** — clone any voice from a few seconds of reference audio.
- 🔢 **Robust normalization** — numbers, dates, times, currencies, phones, emails, URLs, abbreviations + an English→Arabic name map. Your diacritics are always kept.
- ⚡ **True streaming** — long text is split into sentences and yielded as audio chunks for low time-to-first-audio.
- 📦 **pip-installable** — `pip install voicetut-tts` (+ OmniVoice from GitHub), or clone and run locally.

## 📊 Performance

> Measured on a single **NVIDIA T4 (Colab)**, `float16`, `num_step=32`. Reproduce with [`examples/04_evaluation.ipynb`](examples/04_evaluation.ipynb).

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

> Measured on a T4; expect markedly lower RTF / TTFA on an A100 / H100. RTF and TTFA scale with `num_step` — drop to `num_step=16` for faster, slightly lower-quality output.

## 🗣️ Sample Outputs

🎧 **[Listen to all built-in voices & VoiceTut vs. base OmniVoice comparisons →](https://mohammedaly22.github.io/VoiceTuT-TTS/)** — pure Egyptian, code-switching, English, normalization, per-speaker, customer-service, and zero-shot cloning samples. Or try them live in the **[HuggingFace Space](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS)**.

## 📦 Installation

**Option A — PyPI**
```bash
# 1. PyTorch matching your CUDA (see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2. The OmniVoice backbone (not on PyPI, so install it from GitHub)
pip install git+https://github.com/k2-fsa/OmniVoice.git
# 3. VoiceTut-TTS
pip install voicetut-tts
# optional: web UI deps
pip install "voicetut-tts[web]"
```

**Option B — from source**
```bash
git clone https://github.com/MohammedAly22/VoiceTuT-TTS.git
cd VoiceTuT-TTS
conda create -n voicetut python=3.10 -y && conda activate voicetut
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/k2-fsa/OmniVoice.git    # backbone
pip install -e ".[web,dev]"
```

## 🚀 Usage

### Python API

```python
from voicetut_tts import VoiceTutTTS

tts = VoiceTutTTS.from_pretrained("mohammedaly22/VoiceTut-TTS")

# 1) Built-in speaker
tts.synthesize("ازيك عامل ايه النهاردة؟", speaker="Mohammed", output="out.wav")

# 2) Zero-shot voice cloning
tts.synthesize("النهارده الجو حلو اوي",
               ref_audio="my_voice.wav", ref_text="ده الصوت بتاعي",
               output="clone.wav")

# 3) Generation parameters
tts.synthesize("عندي meeting الساعة 3:30",
               speaker="Yasmin", num_step=48, guidance_scale=2.5, speed=1.1,
               output="cs.wav")
```

### True streaming (long text)

```python
import sounddevice as sd
for sr, chunk in tts.stream(long_paragraph, speaker="Ahmed"):
    sd.play(chunk, sr); sd.wait()          # play each sentence as it's ready

# or write a single concatenated file
tts.synthesize_long(long_paragraph, "long.wav", speaker="Ahmed")
```

### Text normalization

```python
from voicetut_tts import ArabicNormalizer

norm = ArabicNormalizer()
norm("عندي 250 جنيه والميعاد 3:30 يوم 14/3/2024")
# -> "عندي ميتين وخمسين جنيه والميعاد تلاتة و نص يوم أربعتاشر مارس ألفين وأربعة وعشرين"
```

**What gets normalized**

| Input | Output (spoken) |
|---|---|
| `3:30` / `7:45` / `9:50` | تلاتة و نص / تمانية الا ربع / عشرة الا عشرة *(colloquial Egyptian clock)* |
| `01147450629` | زيرو حْدَاشَر، سَبْعَه وأَرْبِعين، خَمْسَه وأَرْبِعين، زيرو سِتَّه، تِسْعَه وعِشْرين *(Egyptian prefix + 2-digit groups, paused)* |
| `Ahmed` / `Mohamed` / `Mona` | أحمد / محمد / منى *(English→Arabic name map)* |
| `250 جنيه` / `75$` / `25%` | ميتين وخمسين جنيه / خمسة وسبعين دولار / خمسة وعشرين في المية |
| `14/3/2024` | أربعتاشر مارس ألفين وأربعة وعشرين |
| `a.b@gmail.com` | a dot b at gmail dot com |

Diacritics you write are passed to the model untouched.

**English names**

```python
# English-name -> Arabic for correct pronunciation
tts.add_names({"Ziad": "زياد", "Kareem": "كريم"})
```

The name table ships with the package: [`data/names_en_ar.csv`](voicetut_tts/data/names_en_ar.csv) — `english,arabic`.

### CLI

```bash
voicetut --list-speakers
voicetut --text "ازيك عامل ايه؟" --speaker Mohammed --output out.wav
voicetut --text "نص طويل..." --speaker Ahmed --stream --output long.wav
```

## 🌐 Serving

A modern Arabic-first (RTL) **web app** built with React: light/dark mode, an animated live waveform, a diacritics toolbar, a searchable voice gallery with previews, voice cloning by upload or microphone, one-shot or streaming generation, and live latency metrics. It is the same UI that runs on the [HuggingFace Space](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS).

```bash
pip install "voicetut-tts[web]"
python app.py                                   # default HF checkpoint -> http://localhost:7860
VOICETUT_CKPT=path/to/checkpoint python app.py  # local checkpoint
VOICETUT_MOCK=1 python app.py                   # preview the UI without a model / GPU
```

Behind the UI, the server exposes a small Gradio API (`info`, `normalize`, `synthesize`, `stream`) that you can call with `gradio_client` on the same URL. UI sources live in [`web/`](web/README.md).

## 📓 Examples (Colab)

| Notebook | Description | |
|---|---|---|
| [01_quickstart.ipynb](examples/01_quickstart.ipynb) | Install, load, synthesize with a built-in voice | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/01_quickstart.ipynb) |
| [02_voice_cloning.ipynb](examples/02_voice_cloning.ipynb) | Zero-shot cloning, normalization & diacritics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/02_voice_cloning.ipynb) |
| [03_web_ui.ipynb](examples/03_web_ui.ipynb) | Launch the web UI from Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/03_web_ui.ipynb) |
| [04_evaluation.ipynb](examples/04_evaluation.ipynb) | Measure RTF, TTFA, VRAM, WER, similarity, UTMOS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/04_evaluation.ipynb) |

## 🔗 Links

- 🤗 Model: https://huggingface.co/mohammedaly22/VoiceTut-TTS
- 📦 PyPI: https://pypi.org/project/voicetut-tts/
- 🧠 Base model: https://github.com/k2-fsa/OmniVoice
- 🎧 Companion ASR: https://github.com/MohammedAly22/qwencleo-asr

## ⚠️ Responsible Use

Voice cloning is provided for **beneficial** use cases — voice assistants, accessibility,
educational and creative content. **Do not** use it to impersonate real people, produce deceptive
audio, or harm, harass, or defraud anyone. Obtain consent before cloning a real person's voice and
disclose synthetic audio where appropriate.

## 📜 License & Citation

Released under the **Apache-2.0** license.

```bibtex
@software{voicetut_tts_2026,
  author  = {Mohammed Aly},
  title   = {VoiceTut-TTS: Egyptian Arabic & Code-Switching Text-to-Speech},
  year    = {2026},
  url     = {https://github.com/MohammedAly22/VoiceTuT-TTS},
  note    = {Fine-tuned from OmniVoice}
}
```
