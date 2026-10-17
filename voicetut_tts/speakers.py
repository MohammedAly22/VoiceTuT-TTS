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

# Optional Arabic style tags per speaker (shown in the web UI). Purely cosmetic,
# e.g. {"Ahmed": ["شبابي", "حيوي"]}. Speakers without tags just show their gender.
STYLE_TAGS: Dict[str, List[str]] = {}

# Arabic display names per speaker (web UI). Purely cosmetic.
ARABIC_NAMES: Dict[str, str] = {
    "Ahmed": "أحمد", "Ali": "علي", "Hassan": "حسن", "Karim": "كريم", "Mahmoud": "محمود",
    "Mohammed": "محمد", "Mostafa": "مصطفى", "Ramy": "رامي", "Shaker": "شاكر", "Tareq": "طارق",
    "Yehia": "يحيى", "Youssef": "يوسف", "Hadeer": "هدير", "Hanan": "حنان", "Mariam": "مريم",
    "Rana": "رنا", "Sarah": "سارة", "Yasmin": "ياسمين", "Yara": "يارا",
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
            "arabic_name": ARABIC_NAMES.get(self.speaker_name, self.speaker_name),
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
