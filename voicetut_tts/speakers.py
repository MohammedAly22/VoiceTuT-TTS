#!/usr/bin/env python3
"""Built-in speaker registry for VoiceTut-TTS.

Speakers ship inside the released HuggingFace model repo under ``reference_speakers/``
together with a ``references.json``. This module resolves and loads them, and also
carries lightweight UI metadata (gender + Arabic style tags).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Arabic style tags per speaker (used by the web UI chips). Purely cosmetic.
STYLE_TAGS: Dict[str, List[str]] = {
    "Abdelrahman": ["محترف", "حماسي"],
    "Aly":         ["هادي", "فخور"],
    "Abdullah":    ["رسمي", "محترف"],
    "Kamal":       ["جاد", "واضح"],
    "Hossam":      ["دافي", "ودود"],
    "Mohamed":     ["هادي", "عميق"],
    "Omar":        ["حكواتي", "بطئ"],
    "Sayed":       ["شبابي", "معبّر"],
    "Zaki":        ["رزين", "موثوق"],
    "Asmaa":       ["ناعم", "ودود"],
    "Esraa":       ["مرح", "حيوي"],
    "Hanan":       ["دافي", "هادي"],
    "Sarah":       ["أنيق", "أمومي"],
    "Yasmin":      ["شبابي", "مشرق"],
    "Omnia":       ["ودود", "عفوي"],
    "Essam":       ["جاد", "واضح"],
    "Ahmed":       ["شبابي", "حيوي"],
}


@dataclass
class Speaker:
    speaker_id: str
    speaker_name: str
    audio_path: str
    reference_text: str
    gender: str = ""
    tags: List[str] = field(default_factory=list)

    def to_public(self) -> dict:
        """JSON-safe dict for the web UI (audio served separately)."""
        return {
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "gender": self.gender,
            "tags": self.tags,
            "reference_text": self.reference_text,
            "audio_filename": os.path.basename(self.audio_path),
        }


class SpeakerRegistry:
    """Loads built-in speakers from a references.json and indexes them by name & id."""

    def __init__(self, references_path: str):
        self.references_path = references_path
        self.base_dir = os.path.dirname(os.path.abspath(references_path))
        self._by_key: Dict[str, Speaker] = {}
        self._ordered: List[Speaker] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.references_path):
            raise FileNotFoundError(f"references.json not found: {self.references_path}")
        with open(self.references_path, encoding="utf-8") as f:
            for e in json.load(f):
                audio = e["audio_path"]
                if not os.path.isabs(audio):
                    # resolve relative to the references.json location
                    cand = os.path.join(self.base_dir, os.path.basename(audio))
                    audio = cand if os.path.exists(cand) else os.path.join(self.base_dir, audio)
                spk = Speaker(
                    speaker_id=e["speaker_id"],
                    speaker_name=e["speaker_name"],
                    audio_path=audio,
                    reference_text=e["reference_text"],
                    gender=e.get("gender", ""),
                    tags=STYLE_TAGS.get(e["speaker_name"], []),
                )
                self._ordered.append(spk)
                self._by_key[spk.speaker_name.lower()] = spk
                self._by_key[spk.speaker_id.lower()] = spk

    def get(self, name_or_id: str) -> Speaker:
        spk = self._by_key.get(name_or_id.lower())
        if spk is None:
            raise KeyError(f"Unknown speaker '{name_or_id}'. Available: {self.names()}")
        return spk

    def names(self) -> List[str]:
        return [s.speaker_name for s in self._ordered]

    def all(self) -> List[Speaker]:
        return list(self._ordered)

    def __len__(self) -> int:
        return len(self._ordered)
