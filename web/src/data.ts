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

export interface Fix {
  from: string;
  to: string;
  title: string;
  body: string;
  sentence: string;
}

export const FIXES: Fix[] = [
  {
    from: "مخطط",
    to: "مْخَطَّط",
    title: "حدّد الحركة اللي ناقصة",
    body: "لو الكلمة اتقرت «مُخَطَّط» (رسم بياني) وانت قصدك «مْخَطَّط» (ناوي)، سكّن أول حرف.",
    sentence: "انا مْخَطَّط للموضوع ده من بدري",
  },
  {
    from: "عشرة",
    to: "عِشْرة",
    title: "فرّق بين الكلمات المتشابهة",
    body: "«عَشَرة» الرقم غير «عِشْرة» العُمر. نفس الحروف، وحركتين بيحسموا المعنى.",
    sentence: "دي عِشْرة عمر يا صاحبي، مش يوم ولا اتنين",
  },
  {
    from: "الحب",
    to: "الحَبّ",
    title: "الشدة بتفرق",
    body: "«الحُبّ» غير «الحَبّ» (البذور). الشدة والفتحة بيوجّهوا النطق على طول.",
    sentence: "العصافير بتاكل الحَبّ من الأرض",
  },
];

export const TIPS = [
  {
    icon: "pen",
    title: "اكتب زي ما بتتكلم",
    body: "الموديل متدرب على كلام مصري طبيعي، فاكتب بالعامية: «مش عارف»، «عشان كده»، «إزيك».",
  },
  {
    icon: "languages",
    title: "الإنجليزي بحروفه",
    body: "اكتب الكلمات الإنجليزي بالحروف اللاتيني زي «ال meeting» و«ال deadline» وهتتنطق صح.",
  },
  {
    icon: "hash",
    title: "الأرقام والمواعيد",
    body: "الأرقام والساعة والتواريخ والفلوس والتليفونات بتتحول لكلام لوحدها. شوف «النص المقروء».",
  },
  {
    icon: "sparkles",
    title: "شكّل بس اللي محتاجه",
    body: "مش لازم تشكّل الجملة كلها. حركة واحدة على الحرف الصح كفاية تصلّح الكلمة.",
  },
] as const;
