#!/usr/bin/env python3
"""Command-line interface for VoiceTut-TTS.

    voicetut --text "ازيك عامل ايه؟" --speaker Mohammed --output out.wav
    voicetut --text "..." --ref_audio me.wav --ref_text "..." --output o.wav
    voicetut --list-speakers
"""

import argparse

from .engine import VoiceTutTTS, GenerationParams, DEFAULT_REPO


def main():
    p = argparse.ArgumentParser(description="VoiceTut-TTS — Egyptian Arabic TTS")
    p.add_argument("--model", default=DEFAULT_REPO, help="HF repo id or local checkpoint dir.")
    p.add_argument("--text", help="Text to synthesize.")
    p.add_argument("--output", default="output.wav", help="Output WAV path.")
    p.add_argument("--list-speakers", action="store_true")
    # voice modes
    p.add_argument("--speaker", default=None)
    p.add_argument("--ref_audio", default=None)
    p.add_argument("--ref_text", default=None)
    p.add_argument("--instruct", default=None)
    # generation
    p.add_argument("--language", default="arz", help="'arz' (Egyptian) or 'en'.")
    p.add_argument("--num_step", type=int, default=32)
    p.add_argument("--guidance_scale", type=float, default=2.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-normalize", action="store_true", help="Disable text normalization.")
    p.add_argument("--stream", action="store_true", help="Stream long text sentence by sentence.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    tts = VoiceTutTTS.from_pretrained(args.model, device=args.device, language=args.language)

    if args.list_speakers:
        for s in tts.list_speakers():
            print(f"{s.speaker_id:16} {s.speaker_name:14} {s.gender:7} {', '.join(s.tags)}")
        return

    if not args.text:
        p.error("--text is required (or use --list-speakers).")

    params = GenerationParams(num_step=args.num_step, guidance_scale=args.guidance_scale,
                              speed=args.speed)
    kw = dict(speaker=args.speaker, ref_audio=args.ref_audio, ref_text=args.ref_text,
              instruct=args.instruct, language=args.language,
              normalize=not args.no_normalize, params=params)

    if args.stream:
        tts.synthesize_long(args.text, args.output, **kw)
    else:
        tts.synthesize(args.text, output=args.output, **kw)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
