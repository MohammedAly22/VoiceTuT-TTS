#!/usr/bin/env python3
"""
VoiceTut-TTS web server — a React web UI on top of a small Gradio API.

    /            the web UI (static build in ``voicetut_tts/webui``, sources in ``web/``)
    /gradio      Gradio API used by the UI (queueing + HuggingFace ZeroGPU support)
    /voices/<id> reference clip of a built-in speaker

Endpoints (``/gradio``, callable with ``@gradio/client`` or ``gradio_client``):

    info        -> {version, device, speakers, examples, ...}
    normalize   (text) -> the text exactly as the model will read it
    synthesize  (text, language, voice_mode, speaker, ref_audio, ref_text,
                 num_step, guidance, speed, normalize) -> (wav file, metrics)
    stream      same inputs, yields one (wav chunk, metrics) per sentence

Run:
    python app.py                       # default HF checkpoint on port 7860
    VOICETUT_CKPT=path/to/ckpt python app.py
    VOICETUT_MOCK=1 python app.py       # no model / GPU: synthetic audio, for UI work
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from typing import Optional

# HuggingFace ZeroGPU allocates a GPU per call inside @spaces.GPU functions. `spaces`
# must be imported before torch, and the decorated functions must exist at startup.
try:
    import spaces

    gpu = spaces.GPU
except ImportError:                       # local / dedicated GPU: no-op decorator
    def gpu(*dargs, **dkwargs):
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return lambda fn: fn

import numpy as np

from . import __version__
from .engine import DEFAULT_REPO, GenerationParams, resolve_language, split_sentences
from .normalization import ArabicNormalizer
from .speakers import SpeakerRegistry

log = logging.getLogger("voicetut.server")

WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")
MAX_CHARS = 3000

EXAMPLES = [
    # everyday Egyptian
    "النهارده الجو حلو اوي، يلا نطلع نتمشى شوية ونشرب حاجة على الكورنيش.",
    "انا لسه صاحي حالاً، استنى عليا عشر دقايق وهنزلك على طول.",
    "ماما عاملة محشي النهارده، فمتتأخرش عشان هنتغدى بدري.",
    "الزحمة في الشارع كانت فظيعة، قعدت ساعة في الطريق عشان أوصل.",
    "بقولك ايه، ما تيجي نروح السينما بالليل ونتعشى برة؟",
    "الحمد لله الامتحان كان سهل، وإن شاء الله النتيجة تطلع كويسة.",
    "خلي بالك من نفسك، والجو برد بالليل فخد معاك جاكت.",
    "احنا مسافرين اسكندرية الخميس الجاي، تحب تيجي معانا؟",
    "الواد ده جدع اوي، وقف جنبي في كل حاجة من غير ما أطلب.",
    "ان انت تبقى مش stressed، ومركز في حياتك وفي بيتك وعايش عيشة كويسة، ده اهم حاجة.",
    # code-switching
    "بصراحة ال feedback اللي جالي من ال manager كان كويس اوي، بس في شوية comments محتاجين نخلصها.",
    "اتفقنا نعمل ال meeting بكرة الصبح، فياريت كل واحد يجهز ال presentation بتاعته.",
    "النهارده الجو حلو اوي، يلا نطلع نتمشى وناخد بريك من ال laptop.",
    "ال deadline بتاع ال project اتأجل لأسبوع كمان، فعندنا وقت نعمل review كويس.",
    "لو عندك أي question بخصوص ال offer، ابعتلي email وأنا هرد عليك النهارده.",
    "ال app بيعمل crash كل ما أفتح ال settings، محتاجين نعمل update بسرعة.",
    "انا بشتغل remote من البيت، بس بروح ال office يومين في الأسبوع.",
    "ال team كله عمل overtime امبارح عشان نلحق ال release قبل ال weekend.",
    # numbers, dates, contacts
    "عندي ميعاد مع الدكتور الساعة 3:30 وبعدها هكلمك على 01147450629.",
    "الطلب هيوصلك يوم 15/11/2026، والحساب كله 1250 جنيه شامل الشحن.",
    "في خصم 30% على كل المنتجات لحد آخر الشهر، الحق قبل ما يخلص.",
    "القطر بيقوم الساعة 7:45 الصبح، فلازم نكون في المحطة 7 ونص.",
    "ابعتلي الملف على ahmed.ali@gmail.com وأنا هراجعه وأرد عليك.",
    # diacritics: fix a word in place
    "الموضوع محتاج دراسة وتفكير وتخطيط، وكل حاجة ماشية احسن مما انت مْخَطَّط ومُتَخَيِّل.",
    "دي عِشْرة عمر يا صاحبي، مش يوم ولا اتنين.",
    "العصافير بتاكل الحَبّ من الأرض، والقطة نايمة في الشمس.",
    "هو كان مُدَرِّس في المَدْرَسة دي من عشرين سنة.",
    "انا عايز آكُل حاجة خفيفة قبل ما أنام.",
    "الوَرَق ده مهم جداً، حطه في الدُرْج وماتنساهوش.",
    "المسلسل ده مِتْخَيِّل وكله تأليف، مفيش حاجة منه حقيقية.",
]

# one dir for every generated wav (also whitelisted for Gradio file serving)
_OUT_DIR = os.path.join(tempfile.gettempdir(), "voicetut_out")
os.makedirs(_OUT_DIR, exist_ok=True)

_STATE: dict = {"tts": None, "device": "—"}


# --------------------------------------------------------------------------- mock engine
class MockTTS:
    """Model-free stand-in (``VOICETUT_MOCK=1``) so the web UI can be built and previewed
    without torch / OmniVoice / a GPU. Produces speech-like synthetic audio whose length
    follows the text, so waveforms, streaming and metrics all behave realistically."""

    sampling_rate = 24000

    def __init__(self, registry: Optional[SpeakerRegistry] = None):
        self.registry = registry
        self.normalizer = ArabicNormalizer()

    def list_speakers(self):
        return self.registry.all() if self.registry else []

    def _voice_pitch(self, speaker, ref_audio) -> float:
        if speaker and self.registry:
            spk = self.registry.get(speaker)
            base = 205.0 if spk.gender == "female" else 118.0
            return base + (sum(map(ord, spk.speaker_name)) % 9) * 4.0
        return 160.0 if ref_audio else 140.0

    def _babble(self, text: str, f0: float, speed: float) -> np.ndarray:
        sr = self.sampling_rate
        dur = float(np.clip(0.062 * len(text) / max(speed, 0.3), 0.8, 40.0))
        n = int(sr * dur)
        t = np.arange(n) / sr
        rng = np.random.default_rng(len(text) * 7919 + int(f0))
        # syllable envelope (~4.5 Hz) with word gaps
        syl = 0.5 * (1 + np.sin(2 * np.pi * 4.6 * t + rng.uniform(0, 6) + 1.7 * np.sin(2 * np.pi * 0.9 * t)))
        words = (np.sin(2 * np.pi * 1.3 * t + rng.uniform(0, 6)) > -0.75).astype(np.float32)
        env = (syl ** 1.6) * words
        env = np.convolve(env, np.ones(240) / 240, mode="same")
        # intonation: slow drift + vibrato
        pitch = f0 * (1 + 0.12 * np.sin(2 * np.pi * 0.35 * t) + 0.015 * np.sin(2 * np.pi * 5.5 * t))
        phase = 2 * np.pi * np.cumsum(pitch) / sr
        voiced = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 7))
        breath = 0.04 * rng.standard_normal(n)
        wav = (voiced + breath) * env
        fade = int(0.03 * sr)
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
        return (0.45 * wav / (np.abs(wav).max() + 1e-6)).astype(np.float32)

    def synthesize(self, text, *, speaker=None, ref_audio=None, ref_text=None,
                   language=None, normalize=True, params=None, **_):
        params = params or GenerationParams()
        time.sleep(0.4 + 0.004 * len(text))
        return self._babble(text, self._voice_pitch(speaker, ref_audio), params.speed)

    def stream(self, text, *, speaker=None, ref_audio=None, ref_text=None,
               language=None, normalize=True, params=None, **_):
        params = params or GenerationParams()
        f0 = self._voice_pitch(speaker, ref_audio)
        for sent in split_sentences(text):
            time.sleep(0.25 + 0.003 * len(sent))
            yield self.sampling_rate, self._babble(sent, f0, params.speed)


# --------------------------------------------------------------------------- GPU work
# Module-level so ZeroGPU sees them at startup. Arguments/results are plain picklable data.
def _peak_vram_gb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except Exception:
        pass
    return None


def _reset_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


@gpu(duration=120)
def _gpu_synthesize(text: str, language: str, voice: dict, params: GenerationParams,
                    normalize: bool):
    _reset_vram()
    t0 = time.perf_counter()
    wav = _STATE["tts"].synthesize(text, language=language, normalize=normalize,
                                   params=params, **voice)
    return np.asarray(wav, dtype=np.float32), time.perf_counter() - t0, _peak_vram_gb()


@gpu(duration=180)
def _gpu_stream(text: str, language: str, voice: dict, params: GenerationParams,
                normalize: bool):
    _reset_vram()
    for _sr, chunk in _STATE["tts"].stream(text, language=language, normalize=normalize,
                                           params=params, **voice):
        yield np.asarray(chunk, dtype=np.float32), _peak_vram_gb()


# --------------------------------------------------------------------------- helpers
def _gr():
    import gradio as gr
    return gr


def _write_wav(wav: np.ndarray, sr: int) -> str:
    import soundfile as sf
    path = os.path.join(_OUT_DIR, f"voicetut-{uuid.uuid4().hex[:12]}.wav")
    sf.write(path, wav, sr, subtype="PCM_16")
    return path


def _check_text(text: Optional[str]) -> str:
    text = (text or "").strip()
    if not text:
        raise _gr().Error("اكتب النص الأول")
    if len(text) > MAX_CHARS:
        raise _gr().Error(f"النص أطول من {MAX_CHARS} حرف")
    return text


def _voice(voice_mode, speaker, ref_audio, ref_text) -> dict:
    if voice_mode == "clone":
        if not ref_audio:
            raise _gr().Error("ارفع أو سجّل صوت مرجعي الأول")
        return {"ref_audio": ref_audio, "ref_text": (ref_text or "").strip() or None}
    if not speaker:
        raise _gr().Error("اختار صوت")
    return {"speaker": speaker}


def _params(num_step, guidance, speed) -> GenerationParams:
    return GenerationParams(
        num_step=int(np.clip(num_step or 32, 4, 64)),
        guidance_scale=float(np.clip(guidance or 2.0, 0.0, 6.0)),
        speed=float(np.clip(speed or 1.0, 0.5, 2.0)),
    )


def _language(language) -> str:
    try:
        return resolve_language(language or "arz")
    except ValueError as e:
        raise _gr().Error(str(e))


# --------------------------------------------------------------------------- endpoints
def api_info() -> dict:
    tts = _STATE["tts"]
    speakers = []
    for s in tts.list_speakers():
        d = s.to_public()
        d["audio_url"] = f"voices/{s.speaker_id}"
        speakers.append(d)
    return {
        "name": "VoiceTut-TTS",
        "version": __version__,
        "model_version": "v2",
        "device": _STATE["device"],
        "mock": isinstance(tts, MockTTS),
        "sampling_rate": tts.sampling_rate,
        "max_chars": MAX_CHARS,
        "speakers": speakers,
        "examples": EXAMPLES,
    }


def api_normalize(text: str) -> str:
    return _STATE["tts"].normalizer.normalize(text or "")


def api_synthesize(text, language, voice_mode, speaker, ref_audio, ref_text,
                   num_step, guidance, speed, normalize):
    text = _check_text(text)
    voice = _voice(voice_mode, speaker, ref_audio, ref_text)
    t0 = time.perf_counter()
    wav, gen_secs, vram = _gpu_synthesize(text, _language(language), voice,
                                          _params(num_step, guidance, speed), bool(normalize))
    sr = _STATE["tts"].sampling_rate
    total = time.perf_counter() - t0
    audio = len(wav) / sr
    metrics = {"total": round(total, 3), "generation": round(gen_secs, 3),
               "audio": round(audio, 3), "rtf": round(total / audio, 3) if audio else None,
               "ttfa": round(total, 3), "chunks": 1, "vram_gb": vram}
    return _write_wav(wav, sr), metrics


def api_stream(text, language, voice_mode, speaker, ref_audio, ref_text,
               num_step, guidance, speed, normalize):
    text = _check_text(text)
    voice = _voice(voice_mode, speaker, ref_audio, ref_text)
    sr = _STATE["tts"].sampling_rate
    t0 = time.perf_counter()
    ttfa, audio, n = None, 0.0, 0
    for chunk, vram in _gpu_stream(text, _language(language), voice,
                                   _params(num_step, guidance, speed), bool(normalize)):
        n += 1
        now = time.perf_counter() - t0
        ttfa = now if ttfa is None else ttfa
        audio += len(chunk) / sr
        yield _write_wav(chunk, sr), {
            "total": round(now, 3), "audio": round(audio, 3),
            "rtf": round(now / audio, 3) if audio else None,
            "ttfa": round(ttfa, 3), "chunks": n, "vram_gb": vram,
        }


def build_api():
    """Gradio Blocks exposing the endpoints (no visible UI; the React app is the UI)."""
    gr = _gr()
    with gr.Blocks(title="VoiceTut-TTS API", analytics_enabled=False) as demo:
        gr.Markdown("### VoiceTut-TTS API\nThe web UI is served at the site root.")
        with gr.Row(visible=False):
            text = gr.Textbox()
            language = gr.Textbox()
            voice_mode = gr.Textbox()
            speaker = gr.Textbox()
            ref_audio = gr.File(type="filepath")
            ref_text = gr.Textbox()
            num_step = gr.Number()
            guidance = gr.Number()
            speed = gr.Number()
            normalize = gr.Checkbox()
            out_file = gr.File()
            out_json = gr.JSON()
            out_text = gr.Textbox()
            btn = gr.Button()
        inputs = [text, language, voice_mode, speaker, ref_audio, ref_text,
                  num_step, guidance, speed, normalize]
        btn.click(api_info, None, out_json, api_name="info", concurrency_limit=None)
        btn.click(api_normalize, text, out_text, api_name="normalize", concurrency_limit=None)
        btn.click(api_synthesize, inputs, [out_file, out_json], api_name="synthesize")
        btn.click(api_stream, inputs, [out_file, out_json], api_name="stream")
    return demo


# --------------------------------------------------------------------------- app
def load_tts(model: Optional[str] = None, mock: Optional[bool] = None):
    """Load the engine (or the mock) into the server state."""
    mock = (os.environ.get("VOICETUT_MOCK", "0") == "1") if mock is None else mock
    if mock:
        ref = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            os.pardir, "reference_speakers", "references.json"))
        tts = MockTTS(SpeakerRegistry(ref) if os.path.exists(ref) else None)
        device = "mock"
    else:
        from .engine import VoiceTutTTS
        ckpt = model or os.environ.get("VOICETUT_CKPT") or os.environ.get("OMNICLEO_CKPT") or DEFAULT_REPO
        log.info(f"Loading VoiceTut-TTS from {ckpt} ...")
        tts = VoiceTutTTS.from_pretrained(ckpt)
        device = "ZeroGPU" if os.environ.get("SPACES_ZERO_GPU") else str(tts.model.device)
    _STATE.update(tts=tts, device=device)
    return tts


def create_app(model: Optional[str] = None, mock: Optional[bool] = None):
    """FastAPI app: React UI at ``/``, Gradio API at ``/gradio``, speaker clips at ``/voices``."""
    import gradio as gr
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    if _STATE["tts"] is None:
        load_tts(model, mock)

    app = FastAPI(title="VoiceTut-TTS", docs_url=None, redoc_url=None)

    @app.get("/voices/{speaker_id}", include_in_schema=False)
    def voice_clip(speaker_id: str):
        reg = getattr(_STATE["tts"], "registry", None)
        try:
            spk = reg.get(speaker_id) if reg else None
        except KeyError:
            spk = None
        if spk is None or not os.path.exists(spk.audio_path):
            raise HTTPException(404, "unknown speaker")
        return FileResponse(spk.audio_path, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"ok": True, "device": _STATE["device"]}

    app = gr.mount_gradio_app(app, build_api(), path="/gradio", allowed_paths=[_OUT_DIR])

    if os.path.isfile(os.path.join(WEBUI_DIR, "index.html")):
        app.mount("/", StaticFiles(directory=WEBUI_DIR, html=True), name="webui")
    else:
        @app.get("/", include_in_schema=False)
        def no_ui():
            return HTMLResponse(
                "<h3>VoiceTut-TTS</h3><p>Web UI not built. Run <code>cd web && npm install "
                "&& npm run build</code>. The API is live at <a href='gradio/'>/gradio</a>.</p>")
    return app


def launch(host: str = "0.0.0.0", port: Optional[int] = None, model: Optional[str] = None,
           mock: Optional[bool] = None):
    import uvicorn
    port = port or int(os.environ.get("VOICETUT_PORT") or os.environ.get("OMNICLEO_PORT") or 7860)
    app = create_app(model, mock)
    print(f"VoiceTut-TTS UI  ->  http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    launch()
