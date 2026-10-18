#!/usr/bin/env python3
"""
VoiceTut-TTS inference engine.

A thin, well-documented layer over the fine-tuned Egyptian-Arabic OmniVoice
checkpoint, adding:
  * built-in speakers + custom zero-shot voice cloning
  * Egyptian-Arabic + English only (language is fixed/validated)
  * Arabic text normalization (numbers, dates, currency, ...) that keeps user diacritics,
    so any mispronounced word can be fixed by adding harakat to it
  * TRUE streaming for long text: splits into sentences and yields audio chunks
    as soon as each is generated (low time-to-first-audio).

Quick start
-----------
    from voicetut_tts import VoiceTutTTS

    tts = VoiceTutTTS.from_pretrained("mohammedaly22/VoiceTut-TTS")

    # built-in speaker
    tts.synthesize("ازيك عامل ايه؟", speaker="Mohammed", output="out.wav")

    # custom zero-shot clone
    tts.synthesize("النهارده الجو حلو", ref_audio="me.wav", ref_text="...", output="o.wav")

    # fix a word's pronunciation by diacritizing it
    tts.synthesize("انا مْخطط للموضوع ده", speaker="Mohammed", output="fixed.wav")

    # streaming (yields (sample_rate, np.ndarray) chunks)
    for sr, chunk in tts.stream("نص طويل... جملة تانية... جملة تالتة", speaker="Ahmed"):
        play(chunk)            # play / send over the wire as it arrives
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .normalization import ArabicNormalizer, NormalizerConfig
from .speakers import Speaker, SpeakerRegistry

log = logging.getLogger("voicetut")

# Egyptian Arabic is the fine-tuned language; English is supported for code-switching.
LANG_ALIASES = {
    "ar": "arz", "arz": "arz", "egyptian": "arz", "arabic": "arz", "ar-eg": "arz",
    "en": "en", "english": "en", "eng": "en",
}
DEFAULT_LANGUAGE = "arz"

# Default HF repo (checkpoint + reference_speakers/ + references.json live here).
DEFAULT_REPO = "mohammedaly22/VoiceTut-TTS"

# sentence boundaries: Arabic + Latin punctuation
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\؟\…\n])\s+|(?<=[\.\!\?\؟])(?=\S)")


@dataclass
class GenerationParams:
    """Generation knobs exposed to users (sane Egyptian-TTS defaults)."""
    num_step: int = 32              # diffusion steps: quality <-> speed
    guidance_scale: float = 2.0     # classifier-free guidance strength
    speed: float = 1.0              # >1 faster, <1 slower
    duration: Optional[float] = None
    t_shift: float = 0.1
    denoise: bool = True
    postprocess_output: bool = True
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0

    def as_kwargs(self) -> dict:
        return {
            "num_step": int(self.num_step),
            "guidance_scale": float(self.guidance_scale),
            "speed": float(self.speed),
            "duration": self.duration,
            "t_shift": float(self.t_shift),
            "denoise": bool(self.denoise),
            "postprocess_output": bool(self.postprocess_output),
            "layer_penalty_factor": float(self.layer_penalty_factor),
            "position_temperature": float(self.position_temperature),
            "class_temperature": float(self.class_temperature),
        }


def split_sentences(text: str, max_chars: int = 220) -> List[str]:
    """Split text into speakable sentences for streaming.

    Splits on sentence punctuation, then hard-wraps any over-long sentence on the
    nearest comma/space so each chunk stays short enough for low-latency synthesis.
    """
    raw = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    chunks: List[str] = []
    for sent in raw:
        if len(sent) <= max_chars:
            chunks.append(sent)
            continue
        # wrap long sentence on commas, then spaces
        buf = ""
        for piece in re.split(r"(?<=[،,])\s+", sent):
            if len(buf) + len(piece) + 1 <= max_chars:
                buf = (buf + " " + piece).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = piece
        if buf:
            chunks.append(buf)
    return chunks or ([text.strip()] if text.strip() else [])


# --------------------------------------------------------------------------- chunk planning
# Silence (seconds) inserted where the writer asked for a pause, or where a chunk is cut.
PAUSES = {"paragraph": 0.75, "line": 0.45, "ellipsis": 0.5, "sentence": 0.2, "clause": 0.1, "word": 0.0}
_HARD_BREAK = re.compile(r"(\n[ \t]*\n\s*)|([ \t]*\n\s*)|(\s*(?:\.{2,}|…)\s*)")
_SENT_END = re.compile(r"(?<=[.!?؟])\s+")
_CLAUSE_END = re.compile(r"(?<=[،,;؛:])\s+")


def _split_long(seg: str, budget: int) -> List[Tuple[str, float]]:
    """Break one paragraph-piece into units no longer than `budget` characters:
    sentences first, then clauses, then words. Each unit carries the pause after it."""
    units: List[Tuple[str, float]] = []
    for sent in (s for s in _SENT_END.split(seg) if s.strip()):
        if len(sent) <= budget:
            units.append((sent, PAUSES["sentence"]))
            continue
        for clause in (c for c in _CLAUSE_END.split(sent) if c.strip()):
            if len(clause) <= budget:
                units.append((clause, PAUSES["clause"]))
                continue
            buf = ""
            for word in clause.split():
                if buf and len(buf) + 1 + len(word) > budget:
                    units.append((buf, PAUSES["word"]))
                    buf = word
                else:
                    buf = f"{buf} {word}".strip()
            if buf:
                units.append((buf, PAUSES["clause"]))
    return units


def _is_fragment(seg: str, min_chars: int) -> bool:
    """Too short to synthesize alone: unfinished bits ("فاا", "كل ده") and very short lines."""
    if len(seg) < 8:
        return True
    return len(seg) < min_chars and not seg.rstrip().endswith((".", "!", "?", "؟"))


def _join(a: str, b: str) -> str:
    a = a.rstrip("،, ")
    return f"{a} {b}" if a.endswith((".", "!", "?", "؟")) else f"{a.rstrip('.')}، {b}"


def plan_chunks(text: str, first_max: int = 110, max_chars: int = 220,
                min_chars: int = 25) -> List[Tuple[str, float]]:
    """Plan synthesis chunks by character budget, honouring the writer's pauses.

    * Blank lines, line breaks and ellipses ("...", "…") are pauses: they end a chunk and
      become real silence instead of being read as text.
    * Inside a paragraph, sentences are packed together up to the budget (`first_max` for
      the first chunk, so streaming starts fast, then `max_chars`), cutting at sentence,
      then clause, then word boundaries only when needed.
    * Fragments shorter than `min_chars` (e.g. "فاا...") are merged into the next chunk
      with a comma, because tiny generations sound unnatural.

    Returns ``[(chunk_text, pause_seconds_after), ...]``; the last pause is 0.
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    # 1) cut at hard breaks
    pieces: List[List] = []
    pos = 0
    for m in _HARD_BREAK.finditer(text):
        seg = text[pos:m.start()].strip()
        kind = "paragraph" if m.group(1) else "line" if m.group(2) else "ellipsis"
        if seg:
            pieces.append([seg, PAUSES[kind]])
        elif pieces:
            pieces[-1][1] = max(pieces[-1][1], PAUSES[kind])
        pos = m.end()
    if text[pos:].strip():
        pieces.append([text[pos:].strip(), 0.0])
    # 2) merge tiny fragments forward (or backward for a trailing one)
    merged: List[List] = []
    for seg, pause in pieces:
        if merged and _is_fragment(merged[-1][0], min_chars):
            seg = _join(merged.pop()[0], seg)
        merged.append([seg, pause])
    if len(merged) > 1 and _is_fragment(merged[-1][0], min_chars):
        last = merged.pop()
        merged[-1] = [_join(merged[-1][0], last[0]), last[1]]
    # 3) pack each piece into budget-sized chunks
    chunks: List[Tuple[str, float]] = []
    for seg, pause in merged:
        budget = first_max if not chunks else max_chars
        if len(seg) <= budget:
            chunks.append((seg, pause))
            continue
        buf, buf_pause = "", 0.0
        for unit, unit_pause in _split_long(seg, budget):
            budget = first_max if not chunks else max_chars
            if buf and len(buf) + 1 + len(unit) > budget:
                chunks.append((buf, buf_pause))
                buf = unit
            else:
                buf = f"{buf} {unit}".strip()
            buf_pause = unit_pause
        if buf:
            chunks.append((buf, pause))
    if chunks:
        chunks[-1] = (chunks[-1][0], 0.0)
    return chunks


def with_pause(wav: np.ndarray, sr: int, pause: float) -> np.ndarray:
    if pause <= 0:
        return wav
    return np.concatenate([np.asarray(wav, dtype=np.float32), np.zeros(int(sr * pause), dtype=np.float32)])


def resolve_language(language: Optional[str]) -> str:
    """Map any English/Egyptian alias to a supported code; reject everything else."""
    if language is None:
        return DEFAULT_LANGUAGE
    key = str(language).strip().lower()
    if key not in LANG_ALIASES:
        raise ValueError(
            f"Unsupported language '{language}'. VoiceTut-TTS supports only Egyptian "
            f"Arabic ('arz'/'ar') and English ('en')."
        )
    return LANG_ALIASES[key]


class VoiceTutTTS:
    """High-level Egyptian-Arabic TTS engine."""

    def __init__(
        self,
        model,                                  # a loaded OmniVoice model
        registry: Optional[SpeakerRegistry] = None,
        normalizer: Optional[ArabicNormalizer] = None,
        language: str = DEFAULT_LANGUAGE,
    ):
        self.model = model
        self.sampling_rate = model.sampling_rate
        self.registry = registry
        self.normalizer = normalizer or ArabicNormalizer()
        self.language = resolve_language(language)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEFAULT_REPO,
        *,
        references: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float16",
        normalizer_config: Optional[NormalizerConfig] = None,
        language: str = DEFAULT_LANGUAGE,
        **hf_kwargs,
    ) -> "VoiceTutTTS":
        """
        Load the fine-tuned checkpoint and the built-in speakers.

        Args:
          model_path: HF repo id or local checkpoint dir.
          references: path to references.json. If None, looks for
                      <model_path>/reference_speakers/references.json (works for both
                      a local dir and a downloaded HF snapshot).
          device:     "cuda" / "cuda:0" / "cpu" (auto if None).
          dtype:      "float16" (default), "bfloat16", or "float32".
        """
        import torch
        from omnivoice.models.omnivoice import OmniVoice
        from omnivoice.utils.common import get_best_device

        dev = device or get_best_device()
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}.get(dtype, torch.float16)
        log.info(f"Loading VoiceTut-TTS from '{model_path}' on {dev} ({dtype}) ...")
        model = OmniVoice.from_pretrained(model_path, device_map=dev, dtype=torch_dtype,
                                          **hf_kwargs)

        # locate references.json
        registry = None
        ref_path = references or cls._find_references(model_path)
        if ref_path and os.path.exists(ref_path):
            registry = SpeakerRegistry(ref_path)
            log.info(f"Loaded {len(registry)} built-in speakers from {ref_path}")
        else:
            log.warning("No references.json found — built-in speakers disabled "
                        "(custom voice cloning still works).")

        norm = ArabicNormalizer(normalizer_config) if normalizer_config else ArabicNormalizer()
        return cls(model, registry=registry, normalizer=norm, language=language)

    @staticmethod
    def _find_references(model_path: str) -> Optional[str]:
        # 1) next to a local checkpoint dir
        local = os.path.join(model_path, "reference_speakers", "references.json")
        if os.path.exists(local):
            return local
        # 2) in an HF snapshot of the model repo
        if not os.path.exists(model_path):          # looks like an HF repo id
            try:
                from huggingface_hub import snapshot_download
                snap = snapshot_download(model_path, allow_patterns=["reference_speakers/*"])
                cand = os.path.join(snap, "reference_speakers", "references.json")
                if os.path.exists(cand):
                    return cand
            except Exception:
                pass
        # 3) fall back to the speakers bundled with this repo (one level up from the package)
        bundled = os.path.join(os.path.dirname(__file__), os.pardir,
                               "reference_speakers", "references.json")
        bundled = os.path.normpath(bundled)
        return bundled if os.path.exists(bundled) else None

    # ------------------------------------------------------------------ speakers / names
    def list_speakers(self) -> List[Speaker]:
        return self.registry.all() if self.registry else []

    def add_names(self, mapping: Dict[str, str]) -> None:
        """Add English-name -> Arabic-form overrides (e.g. {'Ziad': 'زياد'})."""
        self.normalizer.add_names(mapping)

    def _resolve_voice(self, speaker, ref_audio, ref_text, instruct):
        modes = sum(x is not None for x in (speaker, ref_audio, instruct))
        if modes > 1:
            raise ValueError("Choose ONE of: speaker, ref_audio(+ref_text), or instruct.")
        if speaker is not None:
            if not self.registry:
                raise RuntimeError("Built-in speakers unavailable (no references.json).")
            spk = self.registry.get(speaker)
            return spk.audio_path, spk.reference_text, None
        if ref_audio is not None and not os.path.exists(ref_audio):
            raise FileNotFoundError(f"ref_audio not found: {ref_audio}")
        return ref_audio, ref_text, instruct

    # ------------------------------------------------------------------ core generate
    def _generate_one(self, text, language, ref_audio, ref_text, instruct, params,
                      normalize) -> np.ndarray:
        if normalize:
            text = self.normalizer.normalize(text, language=language)
        audios = self.model.generate(
            text=text, language=language, ref_audio=ref_audio, ref_text=ref_text,
            instruct=instruct, **params.as_kwargs(),
        )
        return audios[0]

    def synthesize(
        self,
        text: str,
        *,
        speaker: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        language: Optional[str] = None,
        normalize: bool = True,
        output: Optional[str] = None,
        params: Optional[GenerationParams] = None,
        **param_overrides,
    ) -> np.ndarray:
        """
        Synthesize `text` in one shot. Returns the waveform (np.float32, 1-D).
        If `output` is given, also writes a WAV there.

        Voice mode (pick one): speaker=... | ref_audio=...(+ref_text) | instruct=...
        Generation knobs: pass a GenerationParams, or override fields via kwargs
        (e.g. num_step=48, speed=1.1).
        """
        if not text or not text.strip():
            raise ValueError("text is empty.")
        lang = resolve_language(language or self.language)
        params = self._merge_params(params, param_overrides)
        ra, rt, ins = self._resolve_voice(speaker, ref_audio, ref_text, instruct)

        chunks = plan_chunks(text)
        if len(chunks) == 1:
            wav = self._generate_one(chunks[0][0], lang, ra, rt, ins, params, normalize)
        else:   # long text / explicit pauses: generate chunk by chunk and join with real silences
            wav = np.concatenate([
                with_pause(self._generate_one(chunk, lang, ra, rt, ins, params, normalize), self.sampling_rate, pause)
                for chunk, pause in chunks
            ])
        if output:
            self.save(wav, output)
        return wav

    # ------------------------------------------------------------------ streaming
    def stream(
        self,
        text: str,
        *,
        speaker: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        language: Optional[str] = None,
        normalize: bool = True,
        first_chars: int = 110,
        max_chars: int = 220,
        params: Optional[GenerationParams] = None,
        **param_overrides,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        TRUE streaming: yield ``(sampling_rate, waveform_chunk)`` as each chunk finishes.

        Chunks are planned by character budget (see :func:`plan_chunks`): a short first
        chunk for low time-to-first-audio, then ~`max_chars` per chunk. Line breaks and
        ellipses become real silence appended to the chunk, so playback stays smooth.
        """
        if not text or not text.strip():
            return
        lang = resolve_language(language or self.language)
        params = self._merge_params(params, param_overrides)
        ra, rt, ins = self._resolve_voice(speaker, ref_audio, ref_text, instruct)

        for chunk, pause in plan_chunks(text, first_max=first_chars, max_chars=max_chars):
            wav = self._generate_one(chunk, lang, ra, rt, ins, params, normalize)
            yield self.sampling_rate, with_pause(wav, self.sampling_rate, pause)

    def synthesize_long(self, text: str, output: str, **kwargs) -> np.ndarray:
        """Stream-generate long text and concatenate the chunks (pauses included) into one WAV."""
        pieces = [chunk.astype(np.float32) for _, chunk in self.stream(text, **kwargs)]
        wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
        self.save(wav, output)
        return wav

    # ------------------------------------------------------------------ utils
    def _merge_params(self, params, overrides) -> GenerationParams:
        base = params or GenerationParams()
        if overrides:
            base = GenerationParams(**{**base.__dict__, **overrides})
        return base

    def save(self, wav: np.ndarray, output: str) -> str:
        import soundfile as sf
        output = os.path.abspath(output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        sf.write(output, wav, self.sampling_rate)
        log.info(f"Saved -> {output}")
        return output


# --------------------------------------------------------------------------- demo / self-test
if __name__ == "__main__":
    # No model needed: exercise the text-side logic (split + language resolution).
    print("# language resolution")
    for l in ["ar", "arz", "Egyptian", "en", "English", None]:
        print(f"  {str(l):>10} -> {resolve_language(l)}")
    try:
        resolve_language("fr")
    except ValueError as e:
        print("  rejected 'fr':", str(e)[:50], "...")

    print("\n# sentence splitting for streaming")
    txt = ("ازيك عامل ايه النهاردة؟ انا تمام الحمد لله. تعالى نتكلم في موضوع مهم، "
           "وهو ان احنا محتاجين نخلص الشغل بسرعة! يلا بينا.")
    for i, s in enumerate(split_sentences(txt), 1):
        print(f"  [{i}] {s}")
