import { AnimatePresence, motion } from "framer-motion";
import { Eraser, Eye, Shuffle, Trash2, Type } from "lucide-react";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { normalizeText, type Language } from "../api";
import { DIACRITICS_RE, MARKS } from "../data";
import { Segmented } from "./ui";

const LETTER = /[ء-ي]/;
const MARK = /[ً-ْٰ]/;
const SHADDA = "ّ";

export interface EditorHandle {
  focus: () => void;
}

function CountRing({ n, max }: { n: number; max: number }) {
  const f = Math.min(1, n / max);
  const r = 7;
  const c = 2 * Math.PI * r;
  return (
    <span className={`counter num${n > max ? " over" : ""}`}>
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
        <circle cx="9" cy="9" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="2.5" />
        <motion.circle
          cx="9"
          cy="9"
          r={r}
          fill="none"
          stroke={n > max ? "var(--danger)" : f > 0.85 ? "var(--warn)" : "var(--accent)"}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeDasharray={c}
          animate={{ strokeDashoffset: c * (1 - f) }}
          transform="rotate(-90 9 9)"
        />
      </svg>
      <span className="ltr">
        {n.toLocaleString("en")} / {max.toLocaleString("en")}
      </span>
    </span>
  );
}

const Editor = forwardRef<
  EditorHandle,
  {
    text: string;
    onText: (t: string) => void;
    language: Language;
    onLanguage: (l: Language) => void;
    maxChars: number;
    examples: string[];
    onSubmit: () => void;
    onToast: (kind: "error" | "info" | "ok", text: string) => void;
  }
>(function Editor({ text, onText, language, onLanguage, maxChars, examples, onSubmit, onToast }, ref) {
  const ta = useRef<HTMLTextAreaElement>(null);
  const [showNorm, setShowNorm] = useState(false);
  const [norm, setNorm] = useState<string>("");
  const [normBusy, setNormBusy] = useState(false);

  useImperativeHandle(ref, () => ({ focus: () => ta.current?.focus() }), []);

  // auto-grow
  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(460, Math.max(190, el.scrollHeight))}px`;
  }, [text]);

  // debounced "what the model reads" preview
  useEffect(() => {
    if (!showNorm) return;
    if (!text.trim()) {
      setNorm("");
      return;
    }
    setNormBusy(true);
    const id = setTimeout(() => {
      normalizeText(text, language)
        .then(setNorm)
        .catch(() => setNorm("—"))
        .finally(() => setNormBusy(false));
    }, 380);
    return () => clearTimeout(id);
  }, [text, showNorm, language]);

  const setWithCaret = (value: string, caret: number) => {
    onText(value);
    requestAnimationFrame(() => {
      const el = ta.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(caret, caret);
    });
  };

  /** Put `mark` on the letter right before the caret (vowels replace each other, shadda toggles). */
  const applyMark = (mark: string) => {
    const el = ta.current;
    if (!el) return;
    let pos = el.selectionEnd;
    while (pos < text.length && MARK.test(text[pos])) pos++; // caret inside a cluster -> its end
    let start = pos;
    while (start > 0 && MARK.test(text[start - 1])) start--;
    const base = start - 1;
    if (base < 0 || !LETTER.test(text[base])) {
      onToast("info", "حط المؤشر بعد الحرف اللي عايز تشكّله");
      el.focus();
      return;
    }
    const cluster = text.slice(start, pos);
    let shadda = cluster.includes(SHADDA);
    let vowel = [...cluster].find((c) => c !== SHADDA) ?? "";
    if (mark === SHADDA) shadda = !shadda;
    else vowel = vowel === mark ? "" : mark;
    const next = (shadda ? SHADDA : "") + vowel;
    setWithCaret(text.slice(0, start) + next + text.slice(pos), start + next.length);
  };

  const stripMarks = () => {
    const el = ta.current;
    if (!el) return;
    const { selectionStart: s, selectionEnd: e } = el;
    if (s !== e) {
      const mid = text.slice(s, e).replace(DIACRITICS_RE, "");
      setWithCaret(text.slice(0, s) + mid + text.slice(e), s + mid.length);
    } else if (MARK.test(text)) {
      setWithCaret(text.replace(DIACRITICS_RE, ""), 0);
      onToast("ok", "اتشال التشكيل من النص كله");
    }
  };

  const random = () => {
    if (!examples.length) return;
    let next = text;
    for (let i = 0; i < 6 && next === text; i++) next = examples[Math.floor(Math.random() * examples.length)];
    setWithCaret(next, next.length);
  };

  return (
    <div className="card editor">
      <div className="editor-top">
        <Segmented
          label="اللغة"
          value={language}
          onChange={onLanguage}
          options={[
            { value: "arz", label: "مصري" },
            { value: "en", label: "English" },
          ]}
        />
        <span className="grow" />
        <button className="tool" onClick={random} title="جملة عشوائية">
          <Shuffle size={15} />
          <span className="lbl">جملة عشوائية</span>
        </button>
        <button className="tool" aria-pressed={showNorm} onClick={() => setShowNorm((v) => !v)} title="النص زي ما الموديل هيقراه">
          <Eye size={15} />
          <span className="lbl">النص المقروء</span>
        </button>
        <button className="tool" onClick={() => setWithCaret("", 0)} title="امسح النص" aria-label="امسح النص">
          <Trash2 size={15} />
        </button>
      </div>

      <div className="marks" role="toolbar" aria-label="التشكيل">
        <span className="marks-label">
          <Type size={14} />
          شكّل
        </span>
        {MARKS.map((m) => (
          <motion.button
            key={m.char}
            className="mark"
            whileTap={{ scale: 0.88 }}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => applyMark(m.char)}
            aria-label={m.name}
          >
            {"◌" + m.char}
            <span className="tip">{m.name}</span>
          </motion.button>
        ))}
        <span className="marks-sep" />
        <motion.button
          className="mark text"
          whileTap={{ scale: 0.94 }}
          onMouseDown={(e) => e.preventDefault()}
          onClick={stripMarks}
          title="امسح التشكيل من الجزء المحدد (أو من النص كله)"
        >
          <Eraser size={14} />
          امسح التشكيل
        </motion.button>
      </div>

      <div className="textarea-wrap">
        <textarea
          ref={ta}
          className="textarea"
          dir="auto"
          spellCheck={false}
          value={text}
          maxLength={maxChars + 200}
          placeholder="اكتب النص هنا…"
          onChange={(e) => onText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
              e.preventDefault();
              onSubmit();
            }
          }}
        />
      </div>

      <AnimatePresence initial={false}>
        {showNorm && (
          <motion.div
            className="norm-box"
            initial={{ opacity: 0, height: 0, marginTop: 0 }}
            animate={{ opacity: 1, height: "auto", marginTop: 4 }}
            exit={{ opacity: 0, height: 0, marginTop: 0 }}
            transition={{ duration: 0.25 }}
          >
            <div className="section-label">
              <Eye size={13} />
              النص زي ما الموديل هيقراه {normBusy && "…"}
            </div>
            <p dir="auto">{norm || "—"}</p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="editor-foot">
        <CountRing n={text.length} max={maxChars} />
        <span className="grow" />
        <span className="kbd" style={{ color: "var(--muted)", background: "var(--surface-3)" }} title="ولّد الصوت">
          Ctrl + Enter
        </span>
      </div>
    </div>
  );
});

export default Editor;
