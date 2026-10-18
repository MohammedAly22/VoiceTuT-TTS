"""
Arabic / Egyptian text normalization for VoiceTut-TTS.

Turns "messy" real-world text into a clean, speakable string:
  * expands numbers, currencies, dates, times, percentages, phone numbers, in the
    language of the surrounding words (Egyptian Arabic or English, also inside
    code-switched sentences)
  * reads out emails, URLs, @handles
  * expands common abbreviations; units (كم, سم, ...) only when they follow a number
  * normalizes orthography (tatweel, digits, ellipses) while keeping user diacritics
  * keeps Latin (English) words intact for code-switching
  * maps common English names to Arabic script for correct pronunciation

Generated words (numbers, times, phones) are written without diacritics. Diacritics the
user writes are kept, so a mispronounced word can still be fixed in place.

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

# --------------------------------------------------------------------------- Arabic number words
_ONES = ["", "واحد", "اتنين", "تلاتة", "اربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة"]
_TEENS = ["عشرة", "حداشر", "اتناشر", "تلتاشر", "اربعتاشر", "خمستاشر",
          "ستاشر", "سبعتاشر", "تمنتاشر", "تسعتاشر"]
_TENS = ["", "عشرة", "عشرين", "تلاتين", "اربعين", "خمسين", "ستين", "سبعين", "تمانين", "تسعين"]
_HUNDREDS = ["", "مية", "متين", "تلتمية", "ربعمية", "خمسمية", "ستمية", "سبعمية", "تمنمية", "تسعمية"]
_SCALES = [(1_000_000_000, "مليار"), (1_000_000, "مليون"), (1_000, "الف")]


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
    """Convert a non-negative integer to Egyptian-Arabic words (no diacritics)."""
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
                words.append("الفين" if name == "الف" else "اتنين " + name)
            elif count <= 10:
                words.append(_three_digits_to_words(count) + " " + ("آلاف" if name == "الف" else name))
            else:
                words.append(_three_digits_to_words(count) + " " + name)
    if n:
        words.append(_three_digits_to_words(n))
    return " و".join(w for w in words if w)


# --------------------------------------------------------------------------- English number words
_EN_SMALL = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]
_EN_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
           "nine": "ninth", "twelve": "twelfth"}


def _en_below_1000(n: int) -> str:
    parts = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_EN_SMALL[h] + " hundred")
    if rem:
        if rem < 20:
            parts.append(_EN_SMALL[rem])
        else:
            t, o = divmod(rem, 10)
            parts.append(_EN_TENS[t] + ("-" + _EN_SMALL[o] if o else ""))
    return " ".join(parts)


def number_to_english_words(n: int) -> str:
    """Convert a non-negative integer to English words: 1250 -> 'one thousand two hundred fifty'."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + number_to_english_words(-n)
    words = []
    for value, name in _EN_SCALES:
        if n >= value:
            count, n = divmod(n, value)
            words.append(_en_below_1000(count) + " " + name)
    if n:
        words.append(_en_below_1000(n))
    return " ".join(words)


def english_ordinal(n: int) -> str:
    words = number_to_english_words(n)
    head, sep, last = words.rpartition(" ") if " " in words else ("", "", words)
    stem, dash, unit = last.rpartition("-") if "-" in last else ("", "", last)
    if unit in _EN_ORD:
        unit = _EN_ORD[unit]
    elif unit.endswith("y"):
        unit = unit[:-1] + "ieth"
    else:
        unit += "th"
    last = stem + dash + unit
    return (head + sep + last) if head else last


def english_year(y: int) -> str:
    if 2000 <= y < 2010:
        return number_to_english_words(y)
    hi, lo = divmod(y, 100)
    if 1000 <= y < 10000 and lo:
        return f"{number_to_english_words(hi)} {'oh ' + _EN_SMALL[lo] if lo < 10 else number_to_english_words(lo)}"
    return number_to_english_words(y)


# --------------------------------------------------------------------------- language-dispatched readers
def say_number(n: int, lang: str) -> str:
    return number_to_english_words(n) if lang == "en" else number_to_arabic_words(n)


def say_decimal(num_str: str, lang: str) -> str:
    """'3.5' -> 'تلاتة فاصلة خمسة' / 'three point five' (fraction read digit by digit)."""
    if "." not in num_str:
        return say_number(int(num_str), lang)
    intp, frac = num_str.split(".", 1)
    if lang == "en":
        return number_to_english_words(int(intp or "0")) + " point " + " ".join(_EN_SMALL[int(d)] for d in frac)
    return number_to_arabic_words(int(intp or "0")) + " فاصلة " + " ".join(_ONES[int(d)] if d != "0" else "صفر" for d in frac)


# backwards-compatible name
def _read_decimal(num_str: str) -> str:
    return say_decimal(num_str, "ar")


# --------------------------------------------------------------------------- time
_HOUR_FEM = {1: "واحدة", 2: "اتنين", 3: "تلاتة", 4: "اربعة", 5: "خمسة", 6: "ستة", 7: "سبعة",
             8: "تمانية", 9: "تسعة", 10: "عشرة", 11: "حداشر", 12: "اتناشر"}
_MIN_FRACTION = {5: "خمسة", 10: "عشرة", 15: "ربع", 20: "تلت", 30: "نص"}


def _hour_word(h: int) -> str:
    h = h % 12
    return _HOUR_FEM[12 if h == 0 else h]


def _say_time(h: int, mn: int) -> str:
    """Colloquial Egyptian clock reading.

    3:30 -> تلاتة و نص      4:15 -> اربعة و ربع     5:20 -> خمسة و تلت
    6:10 -> ستة و عشرة      1:05 -> واحدة و خمسة
    5:25 -> خمسة و نص الا خمسة     5:35 -> خمسة و نص و خمسة
    7:45 -> تمانية الا ربع  9:50 -> عشرة الا عشرة    11:55 -> اتناشر الا خمسة
    """
    hour = _hour_word(h)
    if mn == 0:
        return hour
    if mn in _MIN_FRACTION:
        return f"{hour} و {_MIN_FRACTION[mn]}"
    if mn == 25:
        return f"{hour} و نص الا خمسة"
    if mn == 35:
        return f"{hour} و نص و خمسة"
    if mn > 30:
        nxt = _hour_word((h % 12) + 1)
        rem = 60 - mn
        return f"{nxt} الا {_MIN_FRACTION.get(rem, number_to_arabic_words(rem))}"
    return f"{hour} و {number_to_arabic_words(mn)} دقيقة"


def _say_time_en(h: int, mn: int) -> str:
    """3:30 -> 'three thirty', 9:05 -> 'nine oh five', 7:00 -> 'seven o'clock'."""
    hour = number_to_english_words(h % 12 or 12) if h != 0 else "twelve"
    if mn == 0:
        return f"{hour} o'clock"
    return f"{hour} {'oh ' + _EN_SMALL[mn] if mn < 10 else number_to_english_words(mn)}"


# --------------------------------------------------------------------------- phone numbers
# Egyptian mobile numbers start with a 3-digit operator prefix; read it specially:
#   011 -> زيرو حداشر   010 -> زيرو عشرة   012 -> زيرو اتناشر   015 -> زيرو خمستاشر
PHONE_PREFIX = {"010": "زيرو عشرة", "011": "زيرو حداشر", "012": "زيرو اتناشر", "015": "زيرو خمستاشر"}


def _say_phone_number(raw: str) -> str:
    """Read an Egyptian phone number: the operator prefix, then 2-digit groups read as tens.

    01147450629 -> زيرو حداشر سبعة واربعين خمسة واربعين زيرو ستة تسعة وعشرين
    (groups: 011 | 47 | 45 | 06 | 29). A leading 0 inside a pair is read 'زيرو'.
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
    for i in range(0, len(rest), 2):
        pair = rest[i:i + 2]
        if len(pair) == 2:
            if pair[0] == "0":
                out.append("زيرو " + (number_to_arabic_words(int(pair[1])) if pair[1] != "0" else "زيرو"))
            else:
                out.append(number_to_arabic_words(int(pair)))
        else:
            out.append(number_to_arabic_words(int(pair)) if pair != "0" else "زيرو")
    return (" زائد " if plus else " ") + " ".join(out) + " "


def _say_phone_number_en(raw: str) -> str:
    """Digit by digit: '+20 101 234' -> 'plus two zero one zero one two three four'."""
    digits = re.sub(r"\D", "", raw)
    return (" plus " if raw.strip().startswith("+") else " ") + " ".join(_EN_SMALL[int(d)] for d in digits) + " "


# --------------------------------------------------------------------------- abbreviations & units
# Expanded wherever they appear as whole tokens.
ABBREVIATIONS: Dict[str, str] = {
    "د.": "دكتور",
    "أ.": "أستاذ",
    "م.": "مهندس",
    "ج.م.": "جنيه مصري",
    "ج.م": "جنيه مصري",
    "ق.م": "قبل الميلاد",
    "إلخ": "إلى آخره",
    "الخ": "إلى آخره",
}
LATIN_ABBREVIATIONS: Dict[str, tuple] = {           # (Arabic context, English context)
    "Dr.": ("دكتور", "doctor"),
    "Mr.": ("مستر", "mister"),
    "Mrs.": ("مسز", "missus"),
    "Eng.": ("مهندس", "engineer"),
    "etc.": ("إلى آخره", "et cetera"),
    "e.g.": ("على سبيل المثال", "for example"),
    "i.e.": ("يعني", "that is"),
    "vs.": ("في مقابل", "versus"),
}
# Units: expanded ONLY right after a number ("12 كم" -> "12 كيلومتر"; "كم كبير" stays as is).
UNITS: Dict[str, tuple] = {                          # (Arabic, English)
    "كجم": ("كيلو جرام", "kilograms"), "كج": ("كيلو", "kilos"), "كم": ("كيلومتر", "kilometers"),
    "سم": ("سنتيمتر", "centimeters"), "مم": ("مليمتر", "millimeters"), "مل": ("مللي", "milliliters"),
    "kg": ("كيلو جرام", "kilograms"), "km": ("كيلومتر", "kilometers"), "cm": ("سنتيمتر", "centimeters"),
    "mm": ("مليمتر", "millimeters"), "ml": ("مللي", "milliliters"), "GB": ("جيجا", "gigabytes"),
    "MB": ("ميجا", "megabytes"),
}
# am / pm markers: only right after a time or an hour number.
DAY_PART: Dict[str, tuple] = {"ص": ("الصبح", "a m"), "م": ("بالليل", "p m"),
                              "am": ("الصبح", "a m"), "pm": ("بالليل", "p m")}

# --------------------------------------------------------------------------- currencies
CURRENCY: Dict[str, tuple] = {                       # (Arabic, English)
    "ج.م": ("جنيه", "pounds"), "جنيه": ("جنيه", "pounds"), "EGP": ("جنيه", "Egyptian pounds"),
    "LE": ("جنيه", "pounds"), "£": ("جنيه", "pounds"),
    "$": ("دولار", "dollars"), "USD": ("دولار", "dollars"), "€": ("يورو", "euros"), "EUR": ("يورو", "euros"),
    "ر.س": ("ريال", "riyals"), "SAR": ("ريال", "riyals"), "د.إ": ("درهم", "dirhams"), "AED": ("درهم", "dirhams"),
}

# --------------------------------------------------------------------------- months
MONTHS = {1: "يناير", 2: "فبراير", 3: "مارس", 4: "ابريل", 5: "مايو", 6: "يونيو",
          7: "يوليو", 8: "اغسطس", 9: "سبتمبر", 10: "اكتوبر", 11: "نوفمبر", 12: "ديسمبر"}
MONTHS_EN = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
             7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

# --------------------------------------------------------------------------- symbols
SYMBOL_WORDS = {"@": ("آت", "at"), "&": ("و", "and"), "+": ("زائد", "plus"),
                "=": ("يساوي", "equals"), "#": ("هاشتاج", "hashtag")}

AR_LETTERS = "ء-ي"
WORD_RE = re.compile(f"[A-Za-z]+|[{AR_LETTERS}]+")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HANDLE_RE = re.compile(r"(?<!\w)@(\w+)")
DOMAIN_RE = re.compile(r"\b([A-Za-z0-9-]+)\.(com|net|org|io|ai|eg|co|me|dev|app|edu|gov)\b", re.I)
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s-]{6,}\d)(?!\d)")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
NUMBER_RE = re.compile(r"(?<![\w@])(\d+(?:\.\d+)?)(?![\w@])")
ELLIPSIS_RE = re.compile(r"\s*(?:\.{2,}|…)\s*")


def context_language(text: str, start: int, end: int, default: str = "ar") -> str:
    """Language of the words around text[start:end]: the word right after it counts most
    (units/currencies: '250 جنيه' vs '250 dollars'), then the word before, then the next ones."""
    after = WORD_RE.findall(text[end:end + 60])[:2]
    before = WORD_RE.findall(text[max(0, start - 60):start])[-2:][::-1]
    nearest = after[:1] + before[:1] + after[1:] + before[1:]      # closest words first
    score = 0
    for weight, word in zip((4, 2, 1, 1), nearest):         # a number takes its noun's language
        score += weight if word[0].isascii() else -weight
    if score == 0:
        return default
    return "en" if score > 0 else "ar"


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
        norm("The meeting is at 3:30 and costs 250 dollars", language="en")

    Numbers are read in the language of the words around them, so code-switched text
    reads naturally: "عندي 3 tasks" -> English "three", "عندي 3 مشاريع" -> Arabic "تلاتة".

    Diacritics typed by the user are kept, so a mispronounced word can be fixed in place:
        norm("الموديل ده مْخطط له كويس")
    """

    def __init__(self, config: Optional[NormalizerConfig] = None, **kwargs):
        self.cfg = config or NormalizerConfig(**kwargs)
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
        return re.sub(r"[A-Za-z]+", lambda m: self._names.get(m.group(0).lower(), m.group(0)), text)

    # ---------------------------------------------------------------- atomic steps
    def normalize_orthography(self, text: str) -> str:
        text = text.translate(_AR2ASCII)                  # arabic-indic -> ascii digits
        text = text.replace("ـ", "")                      # tatweel
        if self.cfg.keep_diacritics:
            text = _DOUBLED_MARK_RE.sub(r"\1", text)      # "مَََ" -> "مَ"
        else:
            text = strip_diacritics(text)
        # line breaks and ellipses are pauses, not text: a comma gives a natural short pause
        text = re.sub(r"([^\s.!?؟،,؛;:])[ \t]*\n+\s*", r"\1، ", text)
        text = ELLIPSIS_RE.sub("، ", text)
        text = re.sub(r"([!?؟])\1+", r"\1", text)         # "!!!" -> "!"
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"^[،,\s]+", "", text).strip()

    def expand_abbreviations(self, text: str, lang: str = "ar") -> str:
        # Whole tokens only, never inside a word ("م" must NOT fire inside "محمد").
        # Diacritics count as part of a word, so "مَ" never fires either.
        AR = AR_LETTERS + DIACRITICS
        edge_l, edge_r = rf"(?<![A-Za-z{AR}])", rf"(?![A-Za-z{AR}])"
        for abbr, full in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
            text = re.sub(edge_l + re.escape(abbr) + edge_r, " " + full + " ", text)
        for abbr, (ar, en) in LATIN_ABBREVIATIONS.items():
            text = re.sub(edge_l + re.escape(abbr) + edge_r,
                          lambda m: " " + (en if lang == "en" else ar) + " ", text)
        return text

    def expand_units(self, text: str, lang: str = "ar") -> str:
        """Units and am/pm, only right after a number/time: '12 كم' -> '12 كيلومتر', 'كم كبير' unchanged."""
        AR = AR_LETTERS + DIACRITICS
        for unit, (ar, en) in sorted(UNITS.items(), key=lambda x: -len(x[0])):
            text = re.sub(rf"(\d+(?:\.\d+)?)\s*{re.escape(unit)}(?![A-Za-z{AR}])",
                          lambda m: f"{m.group(1)} {self._pick(text, m, lang, ar, en)}", text)
        for mark, (ar, en) in DAY_PART.items():
            text = re.sub(rf"(\b\d{{1,2}}(?::\d{{2}})?)\s*{re.escape(mark)}(?![A-Za-z{AR}])",
                          lambda m: f"{m.group(1)} {self._pick(text, m, lang, ar, en)}", text)
        return text

    def expand_contacts(self, text: str, lang: str = "ar") -> str:
        text = URL_RE.sub(lambda m: self._say_url(m.group(0)), text)
        text = EMAIL_RE.sub(lambda m: self._say_email(m.group(0)), text)
        text = DOMAIN_RE.sub(lambda m: f" {m.group(1)} dot {m.group(2)} ", text)   # "gmail.com" -> "gmail dot com"
        text = HANDLE_RE.sub(lambda m: (" at " if lang == "en" else " آت ") + m.group(1) + " ", text)
        text = PHONE_RE.sub(lambda m: _say_phone_number_en(m.group(1)) if self._lang(text, m, lang) == "en"
                            else _say_phone_number(m.group(1)), text)
        return text

    def expand_dates_times(self, text: str, lang: str = "ar") -> str:
        def _date(m):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y = y + 2000 if y < 100 else y
            if self._lang(text, m, lang) == "en":
                return f"{MONTHS_EN.get(mo, number_to_english_words(mo))} {english_ordinal(d)}, {english_year(y)}"
            return f"{number_to_arabic_words(d)} {MONTHS.get(mo, number_to_arabic_words(mo))} {number_to_arabic_words(y)}"

        def _time(m):
            h, mn = int(m.group(1)), int(m.group(2))
            return _say_time_en(h, mn) if self._lang(text, m, lang) == "en" else _say_time(h, mn)

        text = DATE_RE.sub(_date, text)
        return TIME_RE.sub(_time, text)

    def expand_currency(self, text: str, lang: str = "ar") -> str:
        # "250 جنيه" / "250 EGP" / "$50" / "50$"
        for sym, (ar, en) in sorted(CURRENCY.items(), key=lambda x: -len(x[0])):
            esym = re.escape(sym)
            rb = r"\b" if sym[-1].isalnum() else ""
            for pat in (rf"(\d+(?:\.\d+)?)\s*{esym}{rb}", rf"{esym}\s*(\d+(?:\.\d+)?)"):
                def repl(m, ar=ar, en=en, src=text):
                    l = self._lang(src, m, lang)
                    return say_decimal(m.group(1), l) + " " + (en if l == "en" else ar)
                text = re.sub(pat, repl, text)
        return text

    def expand_numbers(self, text: str, lang: str = "ar") -> str:
        text = PERCENT_RE.sub(lambda m: say_decimal(m.group(1), self._lang(text, m, lang)) +
                              (" percent" if self._lang(text, m, lang) == "en" else " في المية"), text)
        return NUMBER_RE.sub(lambda m: say_decimal(m.group(1), self._lang(text, m, lang)), text)

    def expand_symbols(self, text: str, lang: str = "ar") -> str:
        for sym, (ar, en) in SYMBOL_WORDS.items():
            text = text.replace(sym, f" {en if lang == 'en' else ar} ")
        return text

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _lang(text: str, m: re.Match, lang: str) -> str:
        """English-only text reads everything in English; otherwise follow the context."""
        return "en" if lang == "en" else context_language(text, m.start(), m.end(), "ar")

    def _pick(self, text, m, lang, ar, en):
        return en if self._lang(text, m, lang) == "en" else ar

    def _say_email(self, s: str) -> str:
        local, _, domain = s.partition("@")
        spell = lambda x: x.replace(".", " dot ").replace("-", " ").replace("_", " ")
        return f" {spell(local)} at {spell(domain)} "

    def _say_url(self, s: str) -> str:
        s = re.sub(r"^https?://", "", s).rstrip("/").replace("www.", "")
        return " " + s.replace(".", " dot ").replace("/", " slash ") + " "

    # ---------------------------------------------------------------- pipeline
    def normalize(self, text: str, language: str = "arz") -> str:
        """Normalize `text`. `language`: 'arz'/'ar' (Egyptian, numbers follow their context)
        or 'en' (read everything in English)."""
        if not text:
            return ""
        lang = "en" if str(language).lower().startswith("en") else "ar"
        text = self.normalize_orthography(text)
        if self.cfg.expand_abbreviations:
            text = self.expand_abbreviations(text, lang)
            text = self.expand_units(text, lang)
        if self.cfg.expand_contacts:
            text = self.expand_contacts(text, lang)
        if self.cfg.expand_dates_times:
            text = self.expand_dates_times(text, lang)
        if self.cfg.expand_currency:
            text = self.expand_currency(text, lang)
        if self.cfg.expand_numbers:
            text = self.expand_numbers(text, lang)
        text = self.expand_symbols(text, lang)
        if self.cfg.transliterate_names and lang != "en":   # English names -> Arabic (e.g. Ahmed -> أحمد)
            text = self.transliterate_names(text)
        return re.sub(r"\s+", " ", text).strip()

    __call__ = normalize


# --------------------------------------------------------------------------- demo / self-test
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    norm = ArabicNormalizer()
    examples = [
        ("عندي meeting الساعة 3:30 وهيكلفني حوالي 250 جنيه", "arz"),
        ("عندي 3 tasks النهارده و 2 مشاريع", "arz"),
        ("اتصل بيا على 01147450629 أو ابعتلي على ahmed.ali@gmail.com", "arz"),
        ("الحلقة دي نزلت يوم 14/3/2024 على https://youtube.com/voicetut", "arz"),
        ("خصم 25% على كل المنتجات النهاردة بس!", "arz"),
        ("المسافة 12 كم والوزن 3.5 كجم", "arz"),
        ("انا اخدت كم كبير من المعلومات", "arz"),
        ("بعد النجاح اللي حققته النسخة الأولى...\nكان لازم النسخة التانية تكون أحسن", "arz"),
        ("The meeting is at 3:30 on 14/3/2024 and costs $250, call +20 101 234 5678", "en"),
    ]
    for ex, lang in examples:
        print("IN :", ex.replace("\n", " ⏎ "))
        print("OUT:", norm(ex, language=lang))
        print("-" * 70)
