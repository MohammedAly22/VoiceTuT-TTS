"""
VoiceTut-TTS — Gradio web app (custom-styled, black/white + blue).

Two tabs (built-in speakers / voice cloning), a language switch (Egyptian Arabic ⇄
English) that sets the synthesis language, generation parameters, and examples.

Run:
    pip install "voicetut-tts[web]"
    python app.py
    OMNICLEO_CKPT=exp/omnivoice_egy/checkpoint-8000 python app.py
    OMNICLEO_SHARE=1 python app.py        # public link
"""

import os

import gradio as gr

from voicetut_tts import VoiceTutTTS, GenerationParams
from voicetut_tts.engine import DEFAULT_REPO

CKPT = os.environ.get("OMNICLEO_CKPT", DEFAULT_REPO)

print(f"Loading VoiceTut-TTS from {CKPT} ...")
TTS = VoiceTutTTS.from_pretrained(CKPT)
SPEAKERS = TTS.list_speakers()

# Copy each speaker's reference WAV into a local dir under the app's CWD.
# When speakers come from an HF snapshot they live in ~/.cache/huggingface, which Gradio
# refuses to serve. Mirroring them under ./reference_audio (always inside CWD) avoids the
# allowed_paths / InvalidPathError problem on HF Spaces and locally alike.
import shutil
_REF_DIR = os.path.join(os.getcwd(), "reference_audio")
os.makedirs(_REF_DIR, exist_ok=True)
REF_AUDIO = {}   # speaker_name -> local servable wav path
for s in SPEAKERS:
    try:
        dst = os.path.join(_REF_DIR, os.path.basename(s.audio_path))
        if os.path.abspath(s.audio_path) != os.path.abspath(dst):
            shutil.copyfile(s.audio_path, dst)
        REF_AUDIO[s.speaker_name] = dst
    except Exception as e:
        print(f"  (warn) couldn't stage reference for {s.speaker_name}: {e}")
        REF_AUDIO[s.speaker_name] = s.audio_path

SPK_BY_NAME = {s.speaker_name: s for s in SPEAKERS}
DEFAULT_SPK = SPEAKERS[0].speaker_name if SPEAKERS else None

# "Name  ·  ♀/♂  ·  tags" as the label, speaker_name as the value
SPEAKER_CHOICES = [
    (f"{s.speaker_name}   ·   {'♀ أنثى' if s.gender == 'female' else '♂ ذكر'}"
     f"{('   ·   ' + ' · '.join(s.tags)) if s.tags else ''}", s.speaker_name)
    for s in SPEAKERS
]

EXAMPLES = [
    ["Mohamed", "بصراحة ال feedback اللي جالي من ال manager كان كويس اوي، بس في شوية comments محتاجين نخلصها."],
    ["Sayed", "و القصة دي اتكتبت في كتاب اسمه رحلات ماركو بولو 'The Travels of Marco Polo' ، و الكتاب ده بيوري الناس ان العالم أكبر بكتير من اللي هما فاكرينه."],
    ["Asmaa", "اتفقنا نعمل ال meeting بكرة الصبح، فياريت كل واحد يجهز ال presentation بتاعته."],
    ["Yasmin", "النهارده الجو حلو اوي، يلا نطلع نتمشى وناخد بريك من ال laptop."],
    ["Abdelrahman", "يا صباح الفل يا سيدي، اخبارك ايه؟ تعالى نشرب قهوة ونتكلم في الشغل."],
]

# blue waveform for all audio players (Gradio renders orange by default)
WAVE = gr.WaveformOptions(waveform_color="#3a6bd6", waveform_progress_color="#2f6bff")

# ---------------------------------------------------------------- theme + css
THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate", radius_size="lg").set(
    body_background_fill="#000000", body_background_fill_dark="#000000",
    block_background_fill="#0f0f12", block_background_fill_dark="#0f0f12",
    block_border_color="#222228", block_border_color_dark="#222228",
    block_label_background_fill="#0f0f12", block_label_background_fill_dark="#0f0f12",
    input_background_fill="#16161b", input_background_fill_dark="#16161b",
    border_color_primary="#222228", border_color_primary_dark="#222228",
    button_primary_background_fill="#2f6bff", button_primary_background_fill_hover="#4f86ff",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#16161b", button_secondary_background_fill_hover="#1d1d23",
    color_accent_soft="rgba(47,107,255,.14)",
)

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; width: 100% !important; margin: 0 auto !important; direction: rtl!important;
  font-family: 'Cairo','Inter',system-ui,sans-serif !important; }
footer { display: none !important; }

/* hero */
#vt-hero { display:flex; align-items:center; gap:16px; padding:22px 26px; margin:8px 0 18px;
  border:1px solid #222228; border-radius:20px;
  background:linear-gradient(120deg,#0d0d10,#121218);
  box-shadow:0 16px 50px rgba(0,0,0,.6); }
#vt-hero .logo { width:52px; height:52px; border-radius:14px; display:grid; place-items:center;
  font-size:26px; background:linear-gradient(135deg,#2f6bff,#4f86ff); color:#fff;
  box-shadow:0 8px 26px rgba(47,107,255,.45); animation:vtfloat 5s ease-in-out infinite; }
@keyframes vtfloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
#vt-hero h1 { font-size:26px; font-weight:800; margin:0; color:#f5f6f8; letter-spacing:-.5px; }
#vt-hero .accent { background:linear-gradient(90deg,#4f86ff,#8ab0ff);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
#vt-hero p { margin:3px 0 0; color:#8a8d96; font-size:14px; }
#vt-hero .badges { margin-inline-start:auto; display:flex; gap:8px; }
#vt-hero .badge { font-size:12px; font-weight:700; color:#9fb6ff; padding:5px 12px;
  border:1px solid rgba(47,107,255,.3); border-radius:20px; background:rgba(47,107,255,.1); }

/* generate button */
#vt-generate { font-weight:800 !important; font-size:16px !important; padding:14px !important;
  box-shadow:0 8px 26px rgba(47,107,255,.35) !important; transition:all .2s ease !important; }
#vt-generate:hover { transform:translateY(-2px) !important; box-shadow:0 12px 34px rgba(47,107,255,.45) !important; }

/* tabs */
.tab-nav button { font-weight:700 !important; font-size:15px !important; }
.tab-nav button.selected { color:#4f86ff !important; border-bottom:2px solid #2f6bff !important; }

/* single column: full container width, each child a flat card, evenly spaced */
.vt-col { width:100% !important; max-width:100% !important; margin:0 auto !important;
  display:flex !important; flex-direction:column !important; gap:14px !important; }
.vt-col > * { width:100% !important; }

/* streaming metrics cards */
.vt-metrics { display:flex; flex-wrap:wrap; gap:12px; margin-top:6px; }
.vt-metric { flex:1 1 130px; display:flex; flex-direction:column; align-items:center; gap:4px;
  padding:14px 10px; border:1px solid #222228; border-radius:14px;
  background:linear-gradient(160deg,#101218,#0c0c10); box-shadow:0 8px 24px rgba(0,0,0,.35); }
.vt-metric .ic { font-size:20px; }
.vt-metric .val { font-size:20px; font-weight:800; color:#4f86ff; letter-spacing:-.5px; }
.vt-metric .lbl { font-size:11px; color:#8a8d96; font-weight:600; text-align:center; }

/* streaming toggle */
#vt-stream { background:#101218 !important; border:1px solid #222228 !important;
  border-radius:12px !important; padding:10px 14px !important; }

/* inputs */
textarea, input { font-size:15px !important; }
textarea:focus, input:focus { border-color:#2f6bff !important;
  box-shadow:0 0 0 2px rgba(47,107,255,.18) !important; }
input[type=range]::-webkit-slider-thumb { background:#2f6bff !important; }

/* audio: remove the stray white border/outline, keep a clean blue-tinted card */
.vt-audio, .vt-audio * { outline:none !important; }
.vt-audio { border:1px solid #222228 !important; border-radius:14px !important; }

/* language switch -> segmented blue pill */
#vt-lang fieldset { border:none !important; display:flex !important; gap:6px !important;
  background:#0f0f12 !important; border:1px solid #222228 !important; padding:6px !important;
  border-radius:30px !important; width:fit-content !important; }
#vt-lang fieldset > div { display:flex !important; gap:6px !important; }
#vt-lang label { border-radius:24px !important; transition:all .2s ease !important;
  font-weight:700 !important; padding:8px 18px !important; border:none !important; cursor:pointer; }
#vt-lang label:has(input:checked) { background:#2f6bff !important; color:#fff !important;
  box-shadow:0 4px 14px rgba(47,107,255,.4) !important; }

/* generate button spacing */
#vt-generate { margin:6px 0 !important; }

/* examples */
.gr-samples-table tr:hover { background:rgba(47,107,255,.12) !important; }
"""


# ---------------------------------------------------------------- callbacks
def _params(num_step, guidance, speed):
    return GenerationParams(num_step=int(num_step), guidance_scale=float(guidance), speed=float(speed))


def _lang_code(language):
    return "en" if "English" in (language or "") else "arz"


def gen_builtin(speaker_name, text, language, num_step, guidance, speed, normalize):
    if not text or not text.strip():
        raise gr.Error("اكتب النص الأول من فضلك")
    if not speaker_name:
        raise gr.Error("اختار صوت")
    wav = TTS.synthesize(text.strip(), speaker=speaker_name, language=_lang_code(language),
                         normalize=normalize, params=_params(num_step, guidance, speed))
    return (TTS.sampling_rate, wav)


def gen_clone(ref_audio, ref_text, text, language, num_step, guidance, speed, normalize):
    if not text or not text.strip():
        raise gr.Error("اكتب النص الأول")
    if not ref_audio:
        raise gr.Error("ارفع ملف صوتي الأول")
    wav = TTS.synthesize(text.strip(), ref_audio=ref_audio, ref_text=(ref_text or None),
                         language=_lang_code(language), normalize=normalize,
                         params=_params(num_step, guidance, speed))
    return (TTS.sampling_rate, wav)


def _metrics_html(chunks, ttfa, total, audio_secs, vram_gb):
    """Render the streaming metrics as styled cards."""
    rtf = (total / audio_secs) if audio_secs else 0.0
    vram = f"{vram_gb:.2f} GB" if vram_gb is not None else "—"
    cards = [
        ("🔊", "مقاطع / Chunks", str(chunks)),
        ("⚡", "زمن أول صوت / TTFA", f"{ttfa:.2f}s"),
        ("⏱️", "الزمن الكلي / Latency", f"{total:.2f}s"),
        ("🎚️", "RTF", f"{rtf:.2f}×"),
        ("💾", "ذاكرة الكارت / VRAM", vram),
    ]
    body = "".join(
        f'<div class="vt-metric"><span class="ic">{ic}</span>'
        f'<span class="val">{val}</span><span class="lbl">{lbl}</span></div>'
        for ic, lbl, val in cards
    )
    return f'<div class="vt-metrics">{body}</div>'


def _peak_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return None


def stream_tts(text, speaker_name, ref_audio, ref_text, language,
               num_step, guidance, speed, normalize, *, use_clone):
    """Stream long text sentence-by-sentence: yields (audio_chunk, metrics_html)."""
    import time
    if not text or not text.strip():
        raise gr.Error("اكتب النص الأول")
    voice = dict(ref_audio=ref_audio, ref_text=(ref_text or None)) if use_clone \
        else dict(speaker=speaker_name)
    if use_clone and not ref_audio:
        raise gr.Error("ارفع ملف صوتي الأول")
    if not use_clone and not speaker_name:
        raise gr.Error("اختار صوت")

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t0 = time.time()
    ttfa = 0.0
    n = 0
    audio_secs = 0.0
    for sr, chunk in TTS.stream(text.strip(), language=_lang_code(language),
                                normalize=normalize, params=_params(num_step, guidance, speed),
                                **voice):
        n += 1
        if n == 1:
            ttfa = time.time() - t0
        audio_secs += len(chunk) / sr
        total = time.time() - t0
        metrics = _metrics_html(n, ttfa, total, audio_secs, _peak_vram_gb())
        yield (sr, chunk), metrics


def preview_reference(speaker_name):
    """Show the selected built-in speaker's reference audio + text + a tags chip row."""
    if not speaker_name or speaker_name not in SPK_BY_NAME:
        return None, "", ""
    spk = SPK_BY_NAME[speaker_name]
    chips = "".join(
        f'<span class="vt-chip">{tg}</span>' for tg in spk.tags
    )
    gender = "أنثى ♀" if spk.gender == "female" else "ذكر ♂"
    header = (f'<div class="vt-spk-meta"><span class="vt-gender">{gender}</span>'
              f'<div class="vt-chips">{chips}</div></div>')
    # use the locally-staged copy so Gradio can serve it (HF snapshot paths are blocked)
    return REF_AUDIO.get(speaker_name, spk.audio_path), spk.reference_text, header


# ---------------------------------------------------------------- reusable blocks
def advanced_block():
    with gr.Accordion("⚙️ إعدادات متقدمة", open=False):
        with gr.Row():
            ns = gr.Slider(8, 64, value=32, step=1, label="خطوات التوليد / Steps")
            gs = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="قوة التوجيه / Guidance")
        with gr.Row():
            sp = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="السرعة / Speed")
            nm = gr.Checkbox(value=True, label="تطبيع النص / Normalize")
    return ns, gs, sp, nm


# extra CSS for the speaker chips (injected via the HTML preview)
CHIP_CSS = """
<style>
.vt-spk-meta { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:2px 0 4px; }
.vt-gender { font-size:12px; font-weight:700; color:#cfd3da; padding:3px 12px;
  border:1px solid #2a2a31; border-radius:20px; }
.vt-chips { display:flex; gap:6px; flex-wrap:wrap; }
.vt-chip { font-size:12px; font-weight:700; color:#9fb6ff; padding:3px 12px;
  border:1px solid rgba(47,107,255,.3); border-radius:20px; background:rgba(47,107,255,.12); }
</style>
"""


# set RTL leading order on load
RTL_JS = """
() => {
  const root = document.querySelector('gradio-app') || document.body;
  root.setAttribute('dir', 'rtl');
  document.documentElement.setAttribute('dir', 'rtl');
  document.documentElement.setAttribute('lang', 'ar');
}
"""

# ---------------------------------------------------------------- build app
_GR_MAJOR = int(gr.__version__.split(".")[0])
_blocks_kwargs = {"title": "VoiceTut-TTS"}
if _GR_MAJOR < 6:
    _blocks_kwargs.update(css=CUSTOM_CSS, theme=THEME, js=RTL_JS)

with gr.Blocks(**_blocks_kwargs) as demo:
    gr.HTML(CHIP_CSS +
        '<div id="vt-hero">'
        '<div class="logo">𓋹</div>'
        '<div><h1>VoiceTut<span class="accent">-TTS</span></h1>'
        '<p>تحويل النص إلى كلام — مصري وإنجليزي · Egyptian Arabic & code-switching TTS</p></div>'
        '<div class="badges"><span class="badge">15 صوت</span>'
        '<span class="badge">Zero-shot</span><span class="badge">Streaming</span></div>'
        '</div>'
    )

    with gr.Tabs():
        # ============================================= built-in (single column)
        with gr.Tab("🎙️ الأصوات الجاهزة"):
            with gr.Column(elem_classes="vt-col"):
                speaker = gr.Dropdown(
                    choices=SPEAKER_CHOICES, value=DEFAULT_SPK,
                    label="🗣️ الصوت", interactive=True, allow_custom_value=False,
                    filterable=True,
                )
                spk_meta = gr.HTML()
                ref_prev = gr.Audio(label="الصوت المرجعي", interactive=False,
                                    elem_classes="vt-audio", waveform_options=WAVE)
                ref_txt = gr.Textbox(label="النص المرجعي",
                                     interactive=False, lines=2)

                text_b = gr.Textbox(label="📝 النص اللي عايز تنطقه", lines=5,
                                    placeholder="اكتب النص هنا... مثال: ازيك عامل ايه النهاردة؟")

                language = gr.Radio(["العربية (Egyptian)", "English"], value="العربية (Egyptian)",
                                    label="🌐 اللغة / Language", elem_id="vt-lang")

                ns_b, gs_b, sp_b, nm_b = advanced_block()
                stream_b = gr.Checkbox(value=False, elem_id="vt-stream",
                                       label="🌊 بث مباشر للنصوص الطويلة / Stream long text (audio chunks)")
                gen_b = gr.Button("🔊 توليد الصوت", variant="primary", elem_id="vt-generate")
                out_b = gr.Audio(label="الناتج", type="numpy", elem_classes="vt-audio",
                                 waveform_options=WAVE, autoplay=True)
                metrics_b = gr.HTML()

                gr.Examples(EXAMPLES, inputs=[speaker, text_b], label="أمثلة",
                            run_on_click=False, cache_examples=False)

            speaker.change(preview_reference, speaker, [ref_prev, ref_txt, spk_meta])

            def run_b(stream_on, speaker_name, text, language, ns, gs, sp, nm):
                if stream_on:
                    for chunk, met in stream_tts(text, speaker_name, None, None, language,
                                                 ns, gs, sp, nm, use_clone=False):
                        yield chunk, met
                else:
                    wav = gen_builtin(speaker_name, text, language, ns, gs, sp, nm)
                    yield wav, ""

            gen_b.click(run_b, [stream_b, speaker, text_b, language, ns_b, gs_b, sp_b, nm_b],
                        [out_b, metrics_b])

        # ============================================= clone (single column)
        with gr.Tab("استنساخ صوت"):
            with gr.Column(elem_classes="vt-col"):
                ref_audio_c = gr.Audio(label="🎤 الصوت المرجعي audio",
                                       sources=["upload", "microphone"], type="filepath",
                                       elem_classes="vt-audio", waveform_options=WAVE)
                ref_text_c = gr.Textbox(label="النص المرجعي (اختياري)", lines=2,
                                        placeholder="اتركه فاضي للكتابة التلقائية / leave empty to auto-transcribe")

                text_c = gr.Textbox(label="📝 النص اللي عايز تنطقه", lines=5,
                                    placeholder="اكتب النص هنا...")

                language_c = gr.Radio(["العربية (Egyptian)", "English"], value="العربية (Egyptian)",
                                      label="🌐 اللغة / Language", elem_id="vt-lang")

                ns_c, gs_c, sp_c, nm_c = advanced_block()
                stream_c = gr.Checkbox(value=False, elem_id="vt-stream",
                                       label="🌊 بث مباشر للنصوص الطويلة / Stream long text (audio chunks)")
                gen_c = gr.Button("🔊 توليد الصوت", variant="primary", elem_id="vt-generate")
                out_c = gr.Audio(label="الناتج", type="numpy", elem_classes="vt-audio",
                                 waveform_options=WAVE, autoplay=True)
                metrics_c = gr.HTML()

            def run_c(stream_on, ref_audio, ref_text, text, language, ns, gs, sp, nm):
                if stream_on:
                    for chunk, met in stream_tts(text, None, ref_audio, ref_text, language,
                                                 ns, gs, sp, nm, use_clone=True):
                        yield chunk, met
                else:
                    wav = gen_clone(ref_audio, ref_text, text, language, ns, gs, sp, nm)
                    yield wav, ""

            gen_c.click(run_c, [stream_c, ref_audio_c, ref_text_c, text_c, language_c,
                                ns_c, gs_c, sp_c, nm_c], [out_c, metrics_c])

    # initial preview for the default speaker
    demo.load(preview_reference, speaker, [ref_prev, ref_txt, spk_meta])


if __name__ == "__main__":
    # Allow serving the staged reference dir, the registry dir, and the HF cache
    # (covers both local checkpoints and HF-snapshot speakers).
    allowed = [_REF_DIR]
    if TTS.registry:
        allowed.append(TTS.registry.base_dir)
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    if os.path.isdir(hf_cache):
        allowed.append(hf_cache)
    launch_kwargs = dict(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("OMNICLEO_PORT", "7860")),
        share=os.environ.get("OMNICLEO_SHARE", "0") == "1",
        allowed_paths=allowed,
    )
    if _GR_MAJOR >= 6:                    # Gradio 6: css/theme/js belong on launch()
        launch_kwargs.update(css=CUSTOM_CSS, theme=THEME, js=RTL_JS)
    demo.queue().launch(**launch_kwargs)
