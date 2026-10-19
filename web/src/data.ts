export interface Mark {
  char: string;
  name: string;
}

// Combining marks shown on a dotted circle (◌) in the toolbar.
export const MARKS: Mark[] = [
  { char: "َ", name: "فتحة" },
  { char: "ُ", name: "ضمة" },
  { char: "ِ", name: "كسرة" },
  { char: "ْ", name: "سكون" },
  { char: "ّ", name: "شدة" },
  { char: "ً", name: "تنوين فتح" },
  { char: "ٌ", name: "تنوين ضم" },
  { char: "ٍ", name: "تنوين كسر" },
];

export const DIACRITICS_RE = /[ً-ْٰ]/g;
