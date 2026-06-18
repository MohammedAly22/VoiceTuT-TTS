#!/usr/bin/env python3
"""
Generate the audio-demo sample set for VoiceTut-TTS.

Runs on the GPU pod. For each example it synthesizes with BOTH:
  • VoiceTut-TTS  (the fine-tuned checkpoint)
  • base OmniVoice (k2-fsa/OmniVoice)
so the demo page can A/B them. Writes WAVs into docs/audio/ and a manifest.json
that the demo page renders from.

Usage
-----
    cd VoiceTut-TTS
    # point at your fine-tuned checkpoint (local dir or HF repo)
    VOICETUT_CKPT=../OmniVoice/exp/omnivoice_egy/checkpoint-20000 \
        python generate_samples.py

    # options:
    #   --skip-base        only generate VoiceTut samples (faster)
    #   --only SECTIONS    comma list: egyptian,codeswitch,english,normalization,
    #                      speakers,customer_service,cloning   (default: all)
    #   --out docs/audio   output dir
"""

from __future__ import annotations

import argparse
import json
import os
import time

import soundfile as sf

# ------------------------------------------------------------------ config
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "docs", "audio")
VOICETUT_CKPT = os.environ.get("VOICETUT_CKPT", "mohammedaly22/VoiceTut-TTS")
BASE_REPO = "k2-fsa/OmniVoice"

# zero-shot reference (shipped in the repo)
CLONE_REF_AUDIO = os.path.join(DEFAULT_OUT, "zero_shot_reference.mp3")
CLONE_REF_TEXT = ("و كذا بتاع ف دخلت صورت و انا مش مقتنع بس خلاص تمام و رغم ان "
                  "بالأرقام هو انجح في الايرادات من درس خصوصي")

# a default male / female built-in speaker for mixed examples
MALE_SPK = "Abdullah"
FEMALE_SPK = "Yasmin"


# ==================================================================
# Example bank — (key, text, speaker, lang, note)
# lang: "arz" (Egyptian) or "en" (English). note: shown on the demo card.
# ==================================================================
def build_examples():
    EX = {}

    # ---------------- Pure Egyptian Arabic: short / medium / long ----------------
    EX["egyptian"] = [
        ("eg_short_1", "أهلا و سهلا بيكم يا جماعة", "Abdullah", "arz", "قصير"),
        ("eg_short_2", "ما تيجي نخرج نتمشى شويه بعد الشغل.", "Yasmin", "arz", "قصير"),
        ("eg_med_1", "النهاردَة الجو حلو اوي، فقلت اخرج اتمشى على الكورنيش واشرب حاجة ساقعة.",
         "Omnia", "arz", "متوسط"),
        ("eg_med_2", "انا مكنتش عارف اقرر اروح المشوار ده ولا اقعد في البيت، بس في الاخر قررت اروح عادي.",
         "Abdelrahman", "arz", "متوسط"),
        ("eg_long_1",
         "بصراحة اليوم كان طويل جدا ومليان مشاوير، صحيت بدري ورُحت الشغل، وبعد كده عديت على "
         "السوبر ماركت جبت شويةْ حاجات، ولما رجعت البيت لقيت ضيوف مستنينِّي، فقعدت معاهم لحد بالليل "
         "وانا تعبان بس مبسوط.", "Sayed", "arz", "طويل"),
    ]

    # ---------------- Code-switching (Arabic + English) ----------------
    EX["codeswitch"] = [
        ("cs_short_1", "عندي presentation مهمة بُكْرَه الصبح.", "Sayed", "arz", "قصير"),
        ("cs_short_2", "ابعتلي ال file على ال email.", "Esraa", "arz", "قصير"),
        ("cs_med_1",
         "بصراحة ال feedback اللي جالي من ال manager كان كويس اوي، بس في شويةْ comments "
         "محتاجين نخلصها قبل ال deadline.", "Hanan", "arz", "متوسط"),
        ("cs_med_2",
         "النهاردَة هنعمل review لل code وبعدها هنبدأ ال deployment على ال server الجديد.",
         "Abdullah", "arz", "متوسط"),
        ("cs_long_1",
         "القصة دي اتكتبت في كتاب اسمه رحلات ماركو بولو 'The Travels of Marco Polo'، و الكتاب ده بيوري الناس ان العالم "
         "أكبر بكتير من اللي هما فاكرينُه، وان في حضارات وثقافات كتير حوالينا، فلازم نكون "
         "open-minded ونتعلم من بعض على طول.", "Sarah", "arz", "طويل"),
    ]

    # ---------------- Pure English ----------------
    EX["english"] = [
        ("en_short_1", "Hello, how are you doing today?", "Abdelrahman", "en", "short"),
        ("en_med_1", "Thank you for calling our support line, how can I help you today?",
         "Yasmin", "en", "medium"),
        ("en_long_1",
         "Welcome to our podcast. In today's episode we will talk about productivity, "
         "how to manage your time effectively, and a few simple habits that can make a big "
         "difference in your daily routine.", "Sayed", "en", "long"),
    ]

    # ---------------- Normalization stress tests ----------------
    # These exercise the Arabic normalization pipeline (numbers/dates/times/phone/email/...).
    EX["normalization"] = [
        ("norm_numbers", "كدة انا حسابي حوالي 450 جنيه و فلوس التوصيل 23 جنيه يبقا الاجمالي 473 جنيه عشان محدش ينصب عليا، هااا.",
         "Sayed", "arz", "أرقام وعملات"),
        ("norm_time", "المَعاد الساعة 3:30، والاجتماع التاني 7:45، فمتتأخرش عشان الموضوع مهم.",
         "Asmaa", "arz", "مواعيد"),
        ("norm_date", "الحلقة دي نزلت يوم 14/3/2024 وكان عليها مشاهدات كتير.",
         "Abdelrahman", "arz", "تواريخ"),
        ("norm_phone", "ابعتلي رسالة على الرقم ده: 01147450639، لو في اي حاجة مستعجِلَة.",
         "Hossam", "arz", "أرقام تليفون"),
        ("norm_email_url", "ابعتلي على mohamed.ali@gmail.com وادخُل على الموقع example.com/voicetut هتلاقي كل التفاصيل هناك.",
         "Esraa", "arz", "إيميل وروابط"),
        ("norm_percent", "في خصم 25% على كل المنتجات، والتقييم وصل 99.5% من العملاء.",
         "Yasmin", "arz", "نسب مئوية"),
    ]

    # ---------------- Per-speaker showcase (every built-in voice) ----------------
    # one representative code-switching line so each voice is heard on the same text.
    SHOWCASE = ("اهلا بيكم، انا الصوت بتاعي من VoiceTut، وممكن اقول لُكُم اي حاجة بالمصري وبال English كمان.")
    EX["speakers"] = [
        (f"spk_{name.lower()}", SHOWCASE, name, "arz", name)
        for name in SPEAKER_NAMES
    ]

    # ---------------- Customer-service use cases (VoiceTut only; male + female) ----------------
    CS_FEMALE = [
        ("cust_f_1",
         "مساء الخير مع حضرتك ياسمين من شركة VoiceTut، ممكن اتشرف باسم حضرتك؟", FEMALE_SPK),
        ("cust_f_2",
         "اهلا بحضرتك، شكرا لاتصالك بخدمة عملاء VoiceTut، ازاي اقدر اساعد حضرتك النهارده؟", FEMALE_SPK),
        ("cust_f_3",
         "تحت امرك، هسجل الشكوى وهيتم التواصل مع حضرتك خلال 24 ساعة على رقم حضرتك اللي متسجل عندنا.",
         FEMALE_SPK),
        ("cust_f_4",
         "للأسف الخدمة دي مش متاحة حاليا، بس ممكن احول حضرتك لل department المسئول يساعد حضرتك.",
         FEMALE_SPK),
    ]
    CS_MALE = [
        ("cust_m_1",
         "صباح الخير، معاك احمد من خدمة عملاء شركة VoiceTut، ممكن اعرف رقم الطلب بتاع حضرتك؟", MALE_SPK),
        ("cust_m_2",
         "تمام يا فندم، تم تأكيد الحجز، وهتوصلك رساله فيها كل التفاصيل على ال WhatsApp.", MALE_SPK),
        ("cust_m_3",
         "بعتذر لحضرتك عن التأخير، فريق الصيانه في الطريق وهيوصل خلال نص ساعة بالكتير.", MALE_SPK),
        ("cust_m_4",
         "لو حضرتك محتاج اي مساعده تَانْيَه، احنا موجودين 24 ساعة، وشكرا لتعاملك مع VoiceTut.",
         MALE_SPK),
    ]
    EX["customer_service"] = [(k, t, spk, "arz", "خدمة عملاء") for (k, t, spk) in (CS_FEMALE + CS_MALE)]

    # ---------------- Zero-shot voice cloning ----------------
    EX["cloning"] = [
        ("clone_1", "انا بجرب استنساخ الصوت ده، وان شاء الله يطلع حلو وطبيعي زي ما احنا عايزين.",
         None, "arz", "استنساخ"),
        ("clone_2", "ال demo ده بيوضح قد ايه ال model بقا شاطر في اللهجة المصرية وال code-switching كمان.",
         None, "arz", "استنساخ + code-switching"),
    ]
    return EX


# ==================================================================
# Generation
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--skip-base", action="store_true", help="skip base OmniVoice generation")
    ap.add_argument("--only", default="", help="comma list of sections to generate")
    ap.add_argument("--num-step", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- load VoiceTut ----
    from voicetut_tts import VoiceTutTTS, GenerationParams
    print(f"[VoiceTut] loading {VOICETUT_CKPT} ...")
    vt = VoiceTutTTS.from_pretrained(VOICETUT_CKPT)

    global SPEAKER_NAMES
    SPEAKER_NAMES = [s.speaker_name for s in vt.list_speakers()]
    SPK_BY_NAME = {s.speaker_name: s for s in vt.list_speakers()}
    params = GenerationParams(num_step=args.num_step)

    # ---- load base OmniVoice (optional) ----
    base = None
    if not args.skip_base:
        import torch
        from omnivoice.models.omnivoice import OmniVoice
        from omnivoice.utils.common import get_best_device
        print(f"[base] loading {BASE_REPO} ...")
        base = OmniVoice.from_pretrained(BASE_REPO, device_map=get_best_device(),
                                         dtype=torch.float16)

    EX = build_examples()
    sections = [s.strip() for s in args.only.split(",") if s.strip()] or list(EX.keys())

    def save(wav, name):
        p = os.path.join(args.out, name + ".wav")
        sf.write(p, wav, vt.sampling_rate)
        return os.path.basename(p)

    def gen_voicetut(text, speaker, lang, ref_audio=None, ref_text=None):
        if ref_audio:
            return vt.synthesize(text, ref_audio=ref_audio, ref_text=ref_text,
                                 language=lang, params=params)
        return vt.synthesize(text, speaker=speaker, language=lang, params=params)

    def gen_base(text, speaker, lang, ref_audio=None, ref_text=None):
        """Base OmniVoice: clone the SAME reference the built-in speaker uses, so the A/B
        is fair (same target voice, different model). For pure cloning, use the same ref."""
        # base needs a reference to match the voice; reuse the built-in speaker's ref clip
        if ref_audio is None and speaker is not None:
            spk = SPK_BY_NAME[speaker]
            ref_audio, ref_text = spk.audio_path, spk.reference_text
        audios = base.generate(text=text, language=("arz" if lang == "arz" else "en"),
                               ref_audio=ref_audio, ref_text=ref_text,
                               num_step=args.num_step)
        return audios[0]

    manifest = {"sections": []}
    t_start = time.time()

    for sec in sections:
        if sec not in EX:
            print(f"  (skip unknown section: {sec})"); continue
        print(f"\n=== section: {sec} ===")
        items = []
        for entry in EX[sec]:
            key, text, speaker, lang, note = entry
            ref_a = CLONE_REF_AUDIO if sec == "cloning" else None
            ref_t = CLONE_REF_TEXT if sec == "cloning" else None

            rec = {"key": key, "text": text, "speaker": speaker, "lang": lang, "note": note}
            # VoiceTut
            t0 = time.time()
            wav = gen_voicetut(text, speaker, lang, ref_audio=ref_a, ref_text=ref_t)
            rec["voicetut"] = save(wav, f"{key}_voicetut")
            dt = time.time() - t0
            print(f"  [{key}] VoiceTut {dt:.1f}s -> {rec['voicetut']}")

            # base OmniVoice (skip for customer_service — VoiceTut-only per request)
            if base is not None and sec != "customer_service":
                try:
                    wb = gen_base(text, speaker, lang, ref_audio=ref_a, ref_text=ref_t)
                    rec["base"] = save(wb, f"{key}_base")
                    print(f"        base    -> {rec['base']}")
                except Exception as e:
                    print(f"        base FAILED: {e}")
            # cloning: also expose the reference clip
            if sec == "cloning":
                rec["reference"] = os.path.basename(CLONE_REF_AUDIO)
            items.append(rec)
        manifest["sections"].append({"id": sec, "items": items})

    # write manifest
    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    total = time.time() - t_start
    n = sum(len(s["items"]) for s in manifest["sections"])
    print(f"\nDone: {n} examples across {len(manifest['sections'])} sections in {total/60:.1f} min")
    print(f"Manifest -> {mpath}")
    print("Commit docs/audio/*.wav + manifest.json and push to update the demo page.")


if __name__ == "__main__":
    SPEAKER_NAMES = []   # filled after model load
    main()
