<div align="center">

<img src="assets/VoiceTut-TTS-Banner.png" alt="VoiceTut-TTS" width="100%" />

# 𓋹 VoiceTut-TTS

**The best open-source text-to-speech model for Egyptian Arabic & code-switching**

[![🤗 Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/mohammedaly22/VoiceTut-TTS)
[![🤗 Space](https://img.shields.io/badge/🤗%20Space-Demo-blue)](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS)
[![🎧 Samples](https://img.shields.io/badge/🎧%20Audio-Demos-ff9800)](https://mohammedaly22.github.io/VoiceTuT-TTS/)
[![PyPI](https://img.shields.io/pypi/v/voicetut-tts?color=blue&label=PyPI)](https://pypi.org/project/voicetut-tts/)
[![Base](https://img.shields.io/badge/Base-OmniVoice-purple)](https://github.com/k2-fsa/OmniVoice)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/01_quickstart.ipynb)

</div>

> 🎧 **Listen:** hear VoiceTut-TTS vs. the base OmniVoice on Egyptian Arabic & code-switching → **[Audio demos](https://mohammedaly22.github.io/VoiceTuT-TTS/)** · 🚀 **Try it live:** **[HuggingFace Space](https://huggingface.co/spaces/mohammedaly22/VoiceTut-TTS)**

VoiceTut-TTS is an Egyptian-Arabic text-to-speech system fine-tuned from [OmniVoice](https://github.com/k2-fsa/OmniVoice) on ~380 hours of Egyptian podcast speech. It produces natural Egyptian speech with seamless **Arabic ↔ English code-switching**, ships **14 built-in studio voices**, supports **zero-shot voice cloning**, and includes a robust Egyptian-Arabic **text normalization** pipeline plus **true streaming** for long text.

> **Why "VoiceTut"?** *Tut* — after the boy-king **Tutankhamun** (توت عنخ آمون) — anchors the model in Egyptian identity, just as our companion ASR model **[QwenCleo-ASR](https://github.com/MohammedAly22/qwencleo-asr)** is named after **Cleopatra**. Together they form an Egyptian speech stack: **Cleo listens, Tut speaks.** 🎙️🗣️

## ✨ Features

- 🎯 **Egyptian-first** — fine-tuned specifically on Egyptian Arabic, not generic MSA.
- 🔀 **Code-switching** — handles real Arabic + English mixed speech (`عندي meeting بكرة`).
- 🗣️ **14 built-in voices** — male & female studio speakers, each with style tags.
- 🧬 **Zero-shot cloning** — clone any voice from a few seconds of reference audio.
- 🔢 **Robust normalization** — numbers, dates, times, currencies, phones, emails, URLs, abbreviations + a diacritics override table and a custom lexicon.
- ⚡ **True streaming** — long text is split into sentences and yielded as audio chunks for low time-to-first-audio.
- 📦 **pip-installable** — `pip install voicetut-tts` (+ OmniVoice from GitHub), or clone and run locally.

## 📊 Performance

> Measured on a single **NVIDIA H100 80GB**, `float16`, `num_step=32`. Numbers are indicative; see [`examples/`](examples/) to reproduce.

| Metric | Value |
|---|---|
| Real-time factor (RTF) | **~0.10** (≈10× faster than real-time) |
| Time-to-first-audio (streaming, 1st sentence) | **~0.4–0.7 s** |
| Peak VRAM (inference, fp16) | **~6.5 GB** |
| Sampling rate | 24 kHz |
| Speaker similarity (cloning, cosine) | **0.78** |
| Naturalness (internal MOS, 1–5) | **4.1** |

> TTFA and RTF scale with `num_step`; drop to `num_step=16` for faster, slightly lower-quality output.

## 🗣️ Sample Outputs

| Type | Text | Speaker |
|---|---|---|
| Pure Egyptian | `ازيك عامل ايه النهاردة؟ يا رب تكون كويس` | Mohamed |
| Code-switching | `عندي meeting الساعة 3:30 ومعايا ال presentation` | Asmaa |
| Long / streaming | *(multi-sentence paragraph, streamed)* | Sayed |

🎧 Listen to all built-in voices in the [web demo](#-serving) or the [Colab notebook](examples/01_quickstart.ipynb).

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
tts.synthesize("ازيك عامل ايه النهاردة؟", speaker="Mohamed", output="out.wav")

# 2) Zero-shot voice cloning
tts.synthesize("النهارده الجو حلو اوي",
               ref_audio="my_voice.wav", ref_text="ده الصوت بتاعي",
               output="clone.wav")

# 3) Generation parameters
tts.synthesize("عندي meeting الساعة 3:30",
               speaker="Asmaa", num_step=48, guidance_scale=2.5, speed=1.1,
               output="cs.wav")
```

### True streaming (long text)

```python
import sounddevice as sd
for sr, chunk in tts.stream(long_paragraph, speaker="Sayed"):
    sd.play(chunk, sr); sd.wait()          # play each sentence as it's ready

# or write a single concatenated file
tts.synthesize_long(long_paragraph, "long.wav", speaker="Sayed")
```

### Text normalization & custom lexicon

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
| `01147450629` | زيرو حداشر سبعة وأربعين خمسة وأربعين ستة تسعة وعشرين *(Egyptian prefix + 2-digit groups)* |
| `Ahmed` / `Mohamed` / `Mona` | أحمد / محمد / منى *(English→Arabic name map)* |
| `250 جنيه` / `75$` / `25%` | ميتين وخمسين جنيه / خمسة وسبعين دولار / خمسة وعشرين في المية |
| `14/3/2024` | أربعتاشر مارس ألفين وأربعة وعشرين |
| `a.b@gmail.com` | a dot b at gmail dot com |

**Override tables & runtime dictionaries**

```python
# diacritized-form overrides (win over the CSV table)
tts.add_lexicon({"تيوت": "تُوت", "نايل": "نَايِل"})

# English-name -> Arabic for correct pronunciation
tts.add_names({"Ziad": "زياد", "Kareem": "كريم"})
```

Editable data tables ship with the package:
- [`data/diacritics.csv`](voicetut_tts/data/diacritics.csv) — `word,diacritized` (Arabic word → diacritized form)
- [`data/names_en_ar.csv`](voicetut_tts/data/names_en_ar.csv) — `english,arabic` (name transliteration)

### CLI

```bash
voicetut --list-speakers
voicetut --text "ازيك عامل ايه؟" --speaker Mohamed --output out.wav
voicetut --text "نص طويل..." --speaker Sayed --stream --output long.wav
```

## 🌐 Serving

A custom-styled **Gradio** web UI (black/white + blue theme, speaker dropdown with gender + style tags, reference preview, voice cloning with mic recording, AR/EN language switch, generation params, examples):

```bash
pip install "voicetut-tts[web]"
python app.py                                   # default HF checkpoint, port 7860
OMNICLEO_CKPT=exp/omnivoice_egy/checkpoint-8000 python app.py   # local checkpoint
OMNICLEO_SHARE=1 python app.py                  # public share link
```

## 📓 Examples (Colab)

| Notebook | Description | |
|---|---|---|
| [01_quickstart.ipynb](examples/01_quickstart.ipynb) | Install, load, synthesize with a built-in voice | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/01_quickstart.ipynb) |
| [02_voice_cloning.ipynb](examples/02_voice_cloning.ipynb) | Zero-shot cloning + normalization & lexicon | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/02_voice_cloning.ipynb) |
| [03_web_ui.ipynb](examples/03_web_ui.ipynb) | Launch the Gradio web UI from Colab (public link) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/VoiceTuT-TTS/blob/main/examples/03_web_ui.ipynb) |

## 🔗 Links

- 🤗 Model: https://huggingface.co/mohammedaly22/VoiceTut-TTS
- 📦 PyPI: https://pypi.org/project/voicetut-tts/
- 🧠 Base model: https://github.com/k2-fsa/OmniVoice
- 🎧 Companion ASR: https://github.com/MohammedAly22/qwencleo-asr

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
