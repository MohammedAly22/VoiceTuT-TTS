"""
Arabic / Egyptian text normalization for VoiceTut-TTS.

Turns "messy" real-world text into a clean, speakable Egyptian-Arabic string:
  * expands numbers, currencies, dates, times, percentages, phone numbers
  * reads out emails, URLs, @handles
  * expands common Egyptian/Arabic abbreviations
  * normalizes Arabic orthography (tatweel, digits) while keeping user diacritics
  * keeps Latin (English) words intact for code-switching
  * maps common English names to Arabic script for correct pronunciation

Design goals: deterministic, dependency-free, and each piece independently testable.

Run `python -m voicetut_tts.normalization` for a live demo over many examples.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NAMES_CSV = os.path.join(_HERE, "data", "names_en_ar.csv")

# --------------------------------------------------------------------------- diacritics
# Tanween, harakat, shadda, sukun (U+064B..U+0652) + superscript alef (U+0670).
# Diacritics written by the user are passed to the model untouched: they are how a
# mispronounced word gets fixed in place (e.g. "مْخطط" vs "مُخطط").
DIACRITICS = "ً-ْٰ"
_DIACRITICS_RE = re.compile(f"[{DIACRITICS}]")
_DOUBLED_MARK_RE = re.compile(f"([{DIACRITICS}])\\1+")


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics (harakat, tanween, shadda, sukun)."""
    return _DIACRITICS_RE.sub("", text)


# --------------------------------------------------------------------------- digits
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"
_AR2ASCII = {ord(a): d for a, d in zip(ARABIC_INDIC, ASCII_DIGITS)}

# --------------------------------------------------------------------------- number words (Arabic)
_ONES = ["", "وَاحِد", "اِتْنِين", "تَلَاتَة", "أَرْبَعَه", "خَمْسَه", "سِتَّه", "سَبْعَه", "تَمَانْيَه", "تِسْعَه"]
_TEENS = ["عَشَرَه", "حِدَاشَر", "اِتْنَاشَر", "تَلاتَّاشّر", "أَرْبَعْتَاشَر", "خَمَسْتَاشَر",
          "سِتَّاشَر", "سَبَعْتَاشَر", "تَمَنْتَاشَر", "تِسَعْتَاشَر"]
_TENS = ["", "عَشَرَه", "عِشْرين", "تلاتين", "أَرْبِعين", "خَمْسِين", "سِتِّين", "سَبْعين", "تَمانين", "تِسعين"]
_HUNDREDS = ["", "مِية", "مِتِين", "تُلْتٌمِية", "رُبْعُمِية", "خُمْسُمِية",
             "سُتُّمِية", "سُبْعُمِية", "تُمْنُمِية", "تُسْعُمِية"]
_SCALES = [(1_000_000_000, "مليار"), (1_000_000, "مليون"), (1_000, "ألف")]


def _three_digits_to_words(n: int) -> str:
    """0..999 -> Egyptian Arabic words."""
    parts: List[str] = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if rem:
        if rem < 10:
            parts.append(_ONES[rem])
        elif rem < 20:
            parts.append(_TEENS[rem - 10])
        else:
            t, o = divmod(rem, 10)
            parts.append((_ONES[o] + " و" + _TENS[t]) if o else _TENS[t])
    return " و".join(parts)


def number_to_arabic_words(n: int) -> str:
    """Convert a non-negative integer to Egyptian-Arabic words."""
    if n == 0:
        return "صفر"
    if n < 0:
        return "سالب " + number_to_arabic_words(-n)
    words: List[str] = []
    for value, name in _SCALES:
        if n >= value:
            count, n = divmod(n, value)
            if count == 1:
                words.append(name)
            elif count == 2:
                words.append(name + "ين" if name == "ألف" else "اتنين " + name)
            elif count <= 10:
                words.append(_three_digits_to_words(count) + " " + ("آلاف" if name == "ألف" else name))
            else:
                words.append(_three_digits_to_words(count) + " " + name)
    if n:
        words.append(_three_digits_to_words(n))
    return " و".join(w for w in words if w)


def _read_decimal(num_str: str) -> str:
    """'3.5' -> 'تلاتة فاصلة خمسة' (digit-by-digit fraction)."""
    if "." not in num_str:
        return number_to_arabic_words(int(num_str))
    intp, frac = num_str.split(".", 1)
    out = number_to_arabic_words(int(intp or "0")) + " فَاصْلَة "
    out += " ".join(_ONES[int(d)] if d != "0" else "صفر" for d in frac)
    return out


# --------------------------------------------------------------------------- time (colloquial Egyptian)
# Hours are read in the feminine ordinal-ish colloquial form after "الساعة".
_HOUR_FEM = {
    1: "وَاحْدَه", 2: "اتنين", 3: "تَلَاتَة", 4: "أَرْبَعَه", 5: "خَمْسَه", 6: "سِتَّه",
    7: "سَبْعَه", 8: "تَمَانْيَه", 9: "تِسْعَه", 10: "عَشَرَه", 11: "حِدَاشَر", 12: "اِتْنَاشَر",
}
# fraction words for common minute marks
_MIN_FRACTION = {5: "خَمْسَه", 10: "عَشَرَه", 15: "رُبع", 20: "تِلْت", 30: "نُصّ"}


def _hour_word(h: int) -> str:
    h = h % 12
    if h == 0:
        h = 12
    return _HOUR_FEM[h]


def _say_time(h: int, mn: int) -> str:
    """Colloquial Egyptian clock reading.

    3:30 -> تلاتة و نص      4:15 -> أربعة و ربع     5:20 -> خمسة و تلت
    6:10 -> ستة و عشرة      1:05 -> واحدة و خمسة
    5:25 -> خمسة و نص الا خمسة     5:35 -> خمسة و نص و خمسة
    7:45 -> تمانية الا ربع  9:50 -> عشرة الا عشرة    11:55 -> اتناشر الا خمسة
    """
    hour = _hour_word(h)
    if mn == 0:
        return hour
    # exact fraction (5,10,15,20,30) -> "hour و <fraction>"
    if mn in _MIN_FRACTION:
        return f"{hour} و {_MIN_FRACTION[mn]}"
    # around the half: 25 -> نص الا خمسة, 35 -> نص و خمسة, 40 -> نص و عشرة... but 40 reads as الا تلت
    if mn == 25:
        return f"{hour} و نٌصّ اِلَّا خَمْسَه"
    if mn == 35:
        return f"{hour} و نُصّ و خَمْسَه"
    # minutes past the half hour read as "الا" (to) the NEXT hour
    if mn > 30:
        nxt = _hour_word((h % 12) + 1 if (h % 12) != 0 else 1)
        rem = 60 - mn
        frac = _MIN_FRACTION.get(rem, number_to_arabic_words(rem))
        return f"{nxt} الا {frac}"
    # other minutes < 30 -> spoken number of minutes
    return f"{hour} و {number_to_arabic_words(mn)} دقيقة"


# --------------------------------------------------------------------------- Egyptian phone numbers
# Egyptian mobile numbers start with a 3-digit operator prefix; read it specially:
#   011 -> زيرو حداشر   010 -> زيرو عشرة   012 -> زيرو اتناشر   015 -> زيرو خمستاشر
PHONE_PREFIX = {
    "010": "زيرو عَشَرَه",
    "011": "زيرو حْدَاشَر",
    "012": "زيرو اِتْنَاشَر",
    "015": "زيرو خَمَسْتَاشَر",
}


def _say_phone_number(raw: str) -> str:
    """Read an Egyptian phone number: special prefix, then 2-digit groups as tens,
    with a pause (،) between groups so it isn't run together.

    01147450629 -> زيرو حْدَاشَر، سَبْعَه وأَرْبِعين، خَمْسَه وأَرْبِعين، زيرو سِتَّه، تِسْعَه وعِشْرين
    (groups: 011 | 47 | 45 | 06 | 29). A leading 0 inside a pair is read 'زيرو'.
    A trailing single digit is read on its own.
    """
    digits = re.sub(r"\D", "", raw)
    plus = raw.strip().startswith("+") or digits.startswith("20")
    if digits.startswith("20"):                       # +20 country code
        digits = "0" + digits[2:]
    out: List[str] = []
    rest = digits
    if digits[:3] in PHONE_PREFIX:
        out.append(PHONE_PREFIX[digits[:3]])
        rest = digits[3:]
    # read the remainder in 2-digit groups
    i = 0
    while i < len(rest):
        pair = rest[i:i + 2]
        if len(pair) == 2:
            if pair[0] == "0":                        # leading zero in a pair, e.g. "06" -> "زيرو ستة"
                out.append("زيرو " + number_to_arabic_words(int(pair[1])) if pair[1] != "0"
                           else "زيرو زيرو")
            else:
                out.append(number_to_arabic_words(int(pair)))
        else:                                          # trailing single digit
            out.append(number_to_arabic_words(int(pair)) if pair != "0" else "زيرو")
        i += 2
    # join groups with an Arabic comma so the TTS inserts a short pause between pairs
    spoken = "، ".join(out)
    return (" زائد " if plus else " ") + spoken + " "


# --------------------------------------------------------------------------- abbreviations
# Common Egyptian / Arabic / English abbreviations seen in podcasts & chat.
ABBREVIATIONS: Dict[str, str] = {
    "د.": "دكتور",
    "أ.": "أستاذ",
    "م.": "مهندس",
    "ج.م": "جنيه مصري",
    "ج.م.": "جنيه مصري",
    "كجم": "كيلو جرام",
    "كج": "كيلو",
    "كم": "كيلومتر",
    "سم": "سنتيمتر",
    "مم": "مليمتر",
    "ص": "صباحًا",
    "م": "مساءً",
    "ق.م": "قبل الميلاد",
    "إلخ": "إلى آخره",
    "الخ": "إلى آخره",
    # latin
    "Dr.": "دكتور",
    "Mr.": "مستر",
    "Eng.": "مهندس",
    "etc.": "إلى آخره",
    "e.g.": "على سبيل المثال",
    "i.e.": "أي",
    "vs.": "في مقابل",
}

# --------------------------------------------------------------------------- currencies / units
CURRENCY = {
    "ج.م": "جنيه", "جنيه": "جنيه", "EGP": "جنيه", "LE": "جنيه", "£": "جنيه",
    "$": "دولار", "USD": "دولار", "€": "يورو", "EUR": "يورو",
    "ر.س": "ريال", "SAR": "ريال", "د.إ": "درهم", "AED": "درهم",
}

# --------------------------------------------------------------------------- months / time
MONTHS = {
    1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
    7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر",
}

# --------------------------------------------------------------------------- symbol read-out
SYMBOL_WORDS = {"@": " آت ", "&": " اند ", "%": " في المية ", "+": " زائد ",
                "=": " يساوي ", "#": " هاشتاج ", "_": " ", "/": " ", "-": " "}

LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HANDLE_RE = re.compile(r"(?<!\w)@(\w+)")
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s-]{6,}\d)(?!\d)")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
NUMBER_RE = re.compile(r"(?<![\w@])(\d+(?:\.\d+)?)(?![\w@])")


@dataclass
class NormalizerConfig:
    names_csv: Optional[str] = DEFAULT_NAMES_CSV
    keep_diacritics: bool = True       # pass user diacritics through (False strips them)
    transliterate_names: bool = True   # map English names -> Arabic for correct pronunciation
    expand_numbers: bool = True
    expand_dates_times: bool = True
    expand_currency: bool = True
    expand_contacts: bool = True       # email / url / @handle / phone
    expand_abbreviations: bool = True
    keep_latin: bool = True            # leave other English words for code-switching


class ArabicNormalizer:
    """
    Normalize Egyptian-Arabic (and code-switched) text for TTS.

        norm = ArabicNormalizer()
        norm("عندي meeting الساعة 3:30 ومعايا 250 جنيه")

    Diacritics typed by the user are kept, so a mispronounced word can be fixed in place:
        norm("الموديل ده مْخطط له كويس")

    Add English-name overrides at runtime:
        norm.add_names({"Ziad": "زياد"})
    """

    def __init__(self, config: Optional[NormalizerConfig] = None, **kwargs):
        self.cfg = config or NormalizerConfig(**kwargs)
        # English -> Arabic name map (keys lowercased for case-insensitive match)
        self._names: Dict[str, str] = {}
        if self.cfg.transliterate_names and self.cfg.names_csv:
            self._load_names(self.cfg.names_csv)

    # ---------------------------------------------------------------- names
    def _load_names(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                en = (row.get("english") or "").strip()
                ar = (row.get("arabic") or "").strip()
                if en and ar:
                    self._names[en.lower()] = ar

    def add_names(self, mapping: Dict[str, str]) -> None:
        """Add/override English-name -> Arabic-form entries (e.g. {'Ziad': 'زياد'})."""
        self._names.update({k.strip().lower(): v.strip() for k, v in mapping.items()})

    def transliterate_names(self, text: str) -> str:
        """Replace known English names (whole word, case-insensitive) with Arabic forms."""
        def repl(m):
            return self._names.get(m.group(0).lower(), m.group(0))
        return re.sub(r"[A-Za-z]+", repl, text)

    # ---------------------------------------------------------------- atomic steps
    def normalize_orthography(self, text: str) -> str:
        text = text.translate(_AR2ASCII)                  # arabic-indic -> ascii digits
        text = re.sub("[ـ]", "", text)               # remove tatweel ـ
        if self.cfg.keep_diacritics:
            text = _DOUBLED_MARK_RE.sub(r"\1", text)      # "مَََ" -> "مَ"
        else:
            text = strip_diacritics(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def expand_contacts(self, text: str) -> str:
        text = URL_RE.sub(lambda m: self._say_url(m.group(0)), text)
        text = EMAIL_RE.sub(lambda m: self._say_email(m.group(0)), text)
        text = HANDLE_RE.sub(lambda m: " آت " + m.group(1) + " ", text)
        text = PHONE_RE.sub(lambda m: self._say_phone(m.group(1)), text)
        return text

    def expand_dates_times(self, text: str) -> str:
        def _date(m):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y = y + 2000 if y < 100 else y
            mo_name = MONTHS.get(mo, number_to_arabic_words(mo))
            return f"{number_to_arabic_words(d)} {mo_name} {number_to_arabic_words(y)}"

        def _time(m):
            return _say_time(int(m.group(1)), int(m.group(2)))

        text = DATE_RE.sub(_date, text)
        text = TIME_RE.sub(_time, text)
        return text

    def expand_currency(self, text: str) -> str:
        # "250 جنيه" / "250 EGP" / "$50" / "50$"
        for sym, word in sorted(CURRENCY.items(), key=lambda x: -len(x[0])):
            esym = re.escape(sym)
            # word-boundary only when the symbol ends in a letter (e.g. EGP, USD), so we
            # don't glue onto the next token; pure symbols ($, £) need no boundary.
            rb = r"\b" if sym[-1].isalnum() else ""
            text = re.sub(rf"(\d+(?:\.\d+)?)\s*{esym}{rb}",
                          lambda m: _read_decimal(m.group(1)) + " " + word, text)
            text = re.sub(rf"{esym}\s*(\d+(?:\.\d+)?)",
                          lambda m: _read_decimal(m.group(1)) + " " + word, text)
        return text

    def expand_numbers(self, text: str) -> str:
        text = PERCENT_RE.sub(lambda m: _read_decimal(m.group(1)) + " في المية", text)
        text = NUMBER_RE.sub(lambda m: _read_decimal(m.group(1)), text)
        return text

    def expand_abbreviations(self, text: str) -> str:
        # Only replace abbreviations as whole tokens, never as a substring inside an
        # Arabic word (e.g. "م" must NOT fire inside "محمد"). We require the match to be
        # surrounded by non-letter chars (start/end, space, digit, punctuation).
        # Diacritics count as part of a word, so "مَ" never fires the "م" abbreviation.
        AR = "ء-ي" + DIACRITICS
        for abbr, full in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
            e = re.escape(abbr)
            pat = rf"(?<![A-Za-z{AR}]){e}(?![A-Za-z{AR}])"
            text = re.sub(pat, " " + full + " ", text)
        return text

    # ---------------------------------------------------------------- helpers
    def _say_email(self, s: str) -> str:
        local, _, domain = s.partition("@")
        spell = lambda x: x.replace(".", " dot ").replace("-", " ").replace("_", " ")
        return f" {spell(local)} at {spell(domain)} "

    def _say_url(self, s: str) -> str:
        s = re.sub(r"^https?://", "", s).rstrip("/")
        s = s.replace("www.", "")
        return " " + s.replace(".", " dot ").replace("/", " slash ") + " "

    def _say_phone(self, s: str) -> str:
        return _say_phone_number(s)

    # ---------------------------------------------------------------- pipeline
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = self.normalize_orthography(text)
        if self.cfg.expand_abbreviations:
            text = self.expand_abbreviations(text)
        if self.cfg.expand_contacts:
            text = self.expand_contacts(text)
        if self.cfg.expand_dates_times:
            text = self.expand_dates_times(text)
        if self.cfg.expand_currency:
            text = self.expand_currency(text)
        if self.cfg.expand_numbers:
            text = self.expand_numbers(text)
        # read remaining standalone symbols
        for sym, word in SYMBOL_WORDS.items():
            if sym in "@&%+=#":
                text = text.replace(sym, word)
        if self.cfg.transliterate_names:           # English names -> Arabic (e.g. Ahmed -> أحمد)
            text = self.transliterate_names(text)
        return re.sub(r"\s+", " ", text).strip()

    __call__ = normalize


# --------------------------------------------------------------------------- demo / self-test
if __name__ == "__main__":
    norm = ArabicNormalizer()

    examples = [
        "عندي meeting الساعة 3:30 وهيكلفني حوالي 250 جنيه",
        "اتصل بيا على 01147450629 أو ابعتلي على ahmed.ali@gmail.com",
        "رقمي 01011624332 وكلمني الساعة 7:45",
        "الحلقة دي نزلت يوم 14/3/2024 على https://youtube.com/voicetut",
        "خصم 25% على كل المنتجات النهاردة بس!",
        "الكتاب بـ 1500 EGP والشحن 75$ بس",
        "Ahmed و Mohamed و Sarah كانوا في ال meeting",
        "المسافة 12 كم والوزن 3.5 كجم",
        "ازيك عامل ايه النهاردة؟ يا رب تكون كويس",
    ]
    print("=" * 70)
    for ex in examples:
        print("IN :", ex)
        print("OUT:", norm(ex))
        print("-" * 70)

    # time read-out (colloquial Egyptian)
    print("\n# time read-out")
    for h, m in [(3, 30), (4, 15), (5, 20), (5, 25), (6, 10), (7, 45), (9, 50), (11, 55)]:
        print(f"  {h}:{m:02d} -> {_say_time(h, m)}")

    # phone read-out (Egyptian prefix + 2-digit groups)
    print("\n# phone read-out")
    for p in ["01147450629", "01011624332", "+201234567890"]:
        print(f"  {p} -> {_say_phone_number(p).strip()}")

    # English -> Arabic name mapping
    print("\n# name transliteration")
    print("  IN : Ahmed and Mohamed and Mona")
    print("  OUT:", norm("Ahmed and Mohamed and Mona"))

    # user diacritics pass straight through -> fix a word's pronunciation in place
    print("\n# diacritics pass-through")
    print("IN :", "انا مْخطط للموضوع ده من بدري")
    print("OUT:", norm("انا مْخطط للموضوع ده من بدري"))

    # number sanity checks
    print("\n# number_to_arabic_words sanity")
    for n in [0, 7, 15, 21, 100, 250, 1500, 1000000, 2024]:
        print(f"  {n:>8} -> {number_to_arabic_words(n)}")
