"""Unit tests for text normalization and engine text-logic (no model/GPU needed).

Comparisons are diacritic-insensitive (via `_norm`) so they survive vowel-mark tweaks
to the built-in number/time word tables; we assert on the consonantal skeleton + word structure.
"""

import re
import pytest

from voicetut_tts.normalization import (
    ArabicNormalizer, number_to_arabic_words, _say_time, _say_phone_number,
)
from voicetut_tts.engine import split_sentences, resolve_language


def _norm(s: str) -> str:
    """Strip Arabic diacritics and unify hamza/alef forms for robust comparison."""
    s = re.sub(r"[ً-ْ]", "", s)            # harakat / shadda / sukun
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
           .replace("ة", "ه").replace("ى", "ي"))
    return s


# --------------------------------------------------------------- numbers
@pytest.mark.parametrize("n,expected", [
    (0, "صفر"), (7, "سبعه"), (15, "خمستاشر"), (21, "واحد وعشرين"),
    (100, "ميه"), (250, "متين وخمسين"), (1000, "الف"),
])
def test_number_words(n, expected):
    assert _norm(number_to_arabic_words(n)) == _norm(expected)


# --------------------------------------------------------------- normalizer
@pytest.fixture(scope="module")
def norm():
    return ArabicNormalizer()


def test_abbrev_not_inside_word(norm):
    # "م" must NOT expand inside "محمد"
    assert "محمد" in norm("محمد راجل كويس")


def test_currency(norm):
    assert "جنيه" in norm("الكتاب بـ 250 جنيه")
    assert "دولار" in norm("سعره 75$")


def test_percent(norm):
    assert "في المية" in norm("خصم 25%")


@pytest.mark.parametrize("h,m,expected", [
    (3, 30, "تلاتة و نص"),
    (4, 15, "أربعة و ربع"),
    (5, 20, "خمسة و تلت"),
    (5, 25, "خمسة و نص الا خمسة"),
    (6, 10, "ستة و عشرة"),
    (7, 45, "تمانية الا ربع"),
    (9, 50, "عشرة الا عشرة"),
    (11, 55, "اتناشر الا خمسة"),
    (1, 5, "واحدة و خمسة"),
    (12, 0, "اتناشر"),
])
def test_time_colloquial(h, m, expected):
    assert _norm(_say_time(h, m)) == _norm(expected)


@pytest.mark.parametrize("code,prefix", [
    ("010", "زيرو عشره"), ("011", "زيرو حداشر"),
    ("012", "زيرو اتناشر"), ("015", "زيرو خمستاشر"),
])
def test_phone_prefixes(code, prefix):
    out = _norm(_say_phone_number(code + "47450629"))
    assert out.strip().startswith(_norm(prefix))


def test_phone_pairs_as_tens():
    # 011 | 47 | 45 | 06 | 29  -> pairs as tens, leading-zero pair reads "زيرو ..."
    out = _norm(_say_phone_number("01147450629"))
    assert "حداشر" in out and "سبعه واربعين" in out and "تسعه وعشرين" in out


def test_phone_plain_and_leading_zero():
    out = _say_phone_number("01147450629")
    assert "،" not in out and "," not in out          # groups separated by spaces only
    assert not re.search(r"[ً-ْ]", out)     # generated words carry no diacritics
    assert "زيرو" in _norm(out).split("حداشر", 1)[1]  # "06" read as "زيرو سته", not bare "سته"


# --------------------------------------------------------------- generated words: no diacritics
@pytest.mark.parametrize("text", ["عندي 1250 جنيه", "الساعة 7:45", "خصم 25%", "يوم 14/3/2024", "3.5 كجم"])
def test_generated_words_have_no_diacritics(norm, text):
    assert not re.search(r"[ً-ْ]", norm(text))


# --------------------------------------------------------------- units only after numbers
def test_unit_word_without_number_is_kept(norm):
    assert norm("انا اخدت كم كبير من المعلومات") == "انا اخدت كم كبير من المعلومات"
    assert "سم" in norm("ده سم قاتل")


def test_unit_after_number_is_expanded(norm):
    out = norm("المسافة 12 كم والطول 30 سم")
    assert "كيلومتر" in out and "سنتيمتر" in out


# --------------------------------------------------------------- language of numbers
def test_numbers_follow_their_language_in_code_switching(norm):
    assert "three" in norm("عندي 3 tasks النهارده")
    assert "تلاتة" in norm("عندي 3 مشاريع")
    assert "dollars" in norm("ال project ده cost 50 dollars") or "fifty" in norm("ال project ده cost 50 dollars")


def test_english_language_reads_everything_in_english(norm):
    out = norm("The meeting is at 3:30 and costs $250 for 2 people", language="en")
    assert "three thirty" in out and "two hundred fifty dollars" in out and "two people" in out
    assert not re.search(r"[ء-ي]", out)


# --------------------------------------------------------------- pauses
def test_ellipsis_and_line_breaks_become_pauses(norm):
    out = norm("بعد النجاح...\nكان لازم")
    assert "..." not in out and "\n" not in out and "،" in out


def test_name_transliteration():
    n = ArabicNormalizer()
    assert "أحمد" in n("Ahmed")
    assert "محمد" in n("Mohamed")
    assert "منى" in n("Mona")


def test_custom_names():
    n = ArabicNormalizer()
    n.add_names({"Ziad": "زياد"})
    assert "زياد" in n("Ziad")


def test_email_url(norm):
    out = norm("ابعت على a.b@gmail.com")
    # email is spelled out: the "@" and "." are read aloud and the raw token is gone
    assert "@" not in out and "a.b@gmail.com" not in out
    assert ("at" in out or "آت" in out) and ("dot" in out or "نقطة" in out)


def test_keeps_english(norm):
    assert "meeting" in norm("عندي meeting بكرة")


def test_user_diacritics_preserved(norm):
    # diacritics are how users fix a mispronounced word -> they must reach the model
    assert "مْخطط" in norm("انا مْخطط للموضوع")
    assert "مُتَخيّل" in norm("مُتَخيّل كمان")


def test_doubled_diacritics_collapsed(norm):
    assert norm("مَََرحبا") == "مَرحبا"


def test_diacritized_letter_is_not_abbreviation(norm):
    # a lone diacritized "مَ" / "صُ" is part of a word, not the م/ص abbreviation
    assert "مساء" not in norm("مَ")
    assert "صباح" not in norm("صُ")


def test_strip_diacritics_option():
    n = ArabicNormalizer(keep_diacritics=False)
    assert n("مْخطط") == "مخطط"


# --------------------------------------------------------------- engine text logic
def test_language_resolution():
    assert resolve_language("ar") == "arz"
    assert resolve_language("English") == "en"
    with pytest.raises(ValueError):
        resolve_language("fr")


def test_sentence_split():
    chunks = split_sentences("جملة اولى؟ جملة تانية. جملة تالتة!")
    assert len(chunks) == 3


def test_long_sentence_wrap():
    long = "كلمة، " * 80
    chunks = split_sentences(long, max_chars=100)
    assert all(len(c) <= 110 for c in chunks)


# --------------------------------------------------------------- streaming chunk planner
from voicetut_tts.engine import plan_chunks, PAUSES


def test_plan_chunks_respects_written_pauses():
    chunks = plan_chunks("بعد النجاح اللي حققته النسخة الأولى...\nكان لازم النسخة التانية تكون أحسن.\n\nدعم التشكيل.")
    texts = [c for c, _ in chunks]
    assert all("..." not in t and "\n" not in t for t in texts)
    assert chunks[0][1] == PAUSES["ellipsis"] and chunks[1][1] == PAUSES["paragraph"] and chunks[-1][1] == 0.0
    assert texts[-1] == "دعم التشكيل."                 # a complete short sentence keeps its own pause


def test_plan_chunks_merges_fragments():
    chunks = plan_chunks("فـاا... رحبوا معايا بالنسخة الجديدة من الموديل.")
    assert len(chunks) == 1 and chunks[0][0].startswith("فـاا، رحبوا")


def test_plan_chunks_character_budget():
    long = "، ".join(["دي جملة طويلة شوية عشان نختبر التقسيم"] * 30) + "."
    chunks = plan_chunks(long, first_max=110, max_chars=220)
    assert len(chunks[0][0]) <= 110 and all(len(c) <= 220 for c, _ in chunks)
    assert "".join(c for c, _ in chunks).replace(" ", "").replace("،", "") == long.replace(" ", "").replace("،", "")
