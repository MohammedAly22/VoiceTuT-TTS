# 📤 Publishing VoiceTut-TTS — command sheet

Everything is built and committed. Run these with **your** credentials.
Order suggested: GitHub → HF model (weights) → HF Space → PyPI → GitHub Pages.

Names used everywhere:
- GitHub: `MohammedAly22/VoiceTuT-TTS`
- HF model: `MohammedAly22/VoiceTut-TTS`
- HF Space: `MohammedAly22/VoiceTut-TTS`
- PyPI: `voicetut-tts`

---

## 1) Push code to GitHub

From the repo root (`VoiceTut-TTS/`), already a git repo with one commit on `main`:

```bash
git remote add origin https://github.com/MohammedAly22/VoiceTuT-TTS.git
git push -u origin main
```
If it rejects (remote not empty), pull/rebase first:
```bash
git pull --rebase origin main && git push -u origin main
```

---

## 2) Upload the model + reference speakers to HuggingFace

The fine-tuned **checkpoint-20000** lives on the GPU pod, not this machine.
Run this **on the pod** (where `hf` is logged in as you):

```bash
# create the model repo (once)
hf repo create MohammedAly22/VoiceTut-TTS --repo-type model -y

# upload the checkpoint weights + config + tokenizer
hf upload MohammedAly22/VoiceTut-TTS \
    /home/workspace/m.aly/OmniVoice-FT/OmniVoice/exp/omnivoice_egy/checkpoint-20000 . \
    --repo-type model

# upload the built-in speakers (so from_pretrained can fetch them)
hf upload MohammedAly22/VoiceTut-TTS \
    reference_speakers reference_speakers --repo-type model
#   ^ run from the VoiceTut-TTS repo dir, or give the absolute path to reference_speakers/

# upload the model card (this overwrites the auto README)
hf upload MohammedAly22/VoiceTut-TTS hf_model_card/README.md README.md --repo-type model
```

> The model card is at `hf_model_card/README.md`. Its banner points to the raw GitHub
> image, so push to GitHub (step 1) first or the banner 404s until then.

Verify it loads:
```bash
python -c "from voicetut_tts import VoiceTutTTS; VoiceTutTTS.from_pretrained('MohammedAly22/VoiceTut-TTS')"
```

---

## 3) Create the HuggingFace Space (interactive demo)

A Space needs a **GPU** to run this TTS model (set hardware to a GPU tier in Space settings).

```bash
# create the Space (gradio sdk)
hf repo create MohammedAly22/VoiceTut-TTS --repo-type space --space_sdk gradio -y

# upload the Space files
hf upload MohammedAly22/VoiceTut-TTS hf_space/README.md README.md --repo-type space
hf upload MohammedAly22/VoiceTut-TTS hf_space/requirements.txt requirements.txt --repo-type space
hf upload MohammedAly22/VoiceTut-TTS app.py app.py --repo-type space
```

Then in the Space UI → **Settings → Hardware → pick a GPU** (e.g. T4/L4). The Space's
`app.py` defaults to `MohammedAly22/VoiceTut-TTS`, so it loads the model you pushed in step 2.

---

## 4) Publish the package to PyPI

The distributions are already built in `dist/` (validated with `twine check`).

```bash
# (recommended) test on TestPyPI first
python -m twine upload --repo testpypi dist/*
# real upload (needs your PyPI token; user = __token__)
python -m twine upload dist/*
```

If you change anything, rebuild first:
```bash
rm -rf dist build *.egg-info && python -m build && python -m twine check dist/*
```

---

## 5) Enable the audio-demo site (GitHub Pages)

After step 1, on github.com → repo **Settings → Pages**:
- **Source:** Deploy from a branch
- **Branch:** `main`  · **Folder:** `/docs`

Live at: `https://mohammedaly22.github.io/VoiceTuT-TTS/`

**Add the audio samples:** generate the WAVs and drop them into `docs/audio/` with the
exact names listed in `docs/audio/README.md` (e.g. `eg1_voicetut.wav`, `eg1_base.wav`, ...),
then commit & push. The page already has the players wired to those filenames.

```bash
# generate VoiceTut samples (on the pod)
python -c "
from voicetut_tts import VoiceTutTTS
t = VoiceTutTTS.from_pretrained('MohammedAly22/VoiceTut-TTS')
t.synthesize('ازيك عامل ايه النهاردة؟ يا رب تكون كويس وكله تمام.', speaker='Mohamed', output='docs/audio/eg1_voicetut.wav')
# ... repeat for eg2/cs1/cs2/clone1 per docs/audio/README.md
"
git add docs/audio/*.wav && git commit -m "Add audio demo samples" && git push
```

---

## Checklist

- [ ] `git push` → GitHub repo populated
- [ ] HF model repo: weights + reference_speakers + model card
- [ ] `from_pretrained` loads from HF
- [ ] HF Space created + GPU hardware enabled
- [ ] `twine upload` → PyPI shows voicetut-tts 0.1.0
- [ ] GitHub Pages enabled (/docs) → demo site live
- [ ] audio samples added to docs/audio/
