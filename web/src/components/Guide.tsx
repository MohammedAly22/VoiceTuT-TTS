import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, ChevronDown, Hash, Languages, PenLine, Sparkles, Wand2 } from "lucide-react";
import { forwardRef, useState } from "react";
import { FIXES, TIPS } from "../data";

const ICONS = { pen: PenLine, languages: Languages, hash: Hash, sparkles: Sparkles };

/** Usage tips, collapsed by default. */
const Guide = forwardRef<HTMLElement, { onTry: (sentence: string) => void }>(function Guide({ onTry }, ref) {
  const [open, setOpen] = useState(false);
  return (
    <section className="guide" ref={ref}>
      <button className="guide-toggle card" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="t-ico">
          <Wand2 size={16} />
        </span>
        <span className="guide-title">إزاي تصلّح النطق بالتشكيل</span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.25 }} style={{ display: "grid" }}>
          <ChevronDown size={18} />
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            className="guide-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1] }}
          >
            <div className="fix-grid">
              {FIXES.map((f, i) => (
                <motion.div
                  key={f.to}
                  className="card fix"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.05 + i * 0.05 }}
                >
                  <div className="fix-words">
                    <span className="from">{f.from}</span>
                    <ArrowLeft size={20} />
                    <span className="to">{f.to}</span>
                  </div>
                  <h3>{f.title}</h3>
                  <p>{f.body}</p>
                  <button className="btn sm" onClick={() => onTry(f.sentence)}>
                    <Wand2 size={14} />
                    <span className="try-text">جرّب: «{f.sentence}»</span>
                  </button>
                </motion.div>
              ))}
            </div>
            <div className="tips">
              {TIPS.map((t, i) => {
                const Icon = ICONS[t.icon];
                return (
                  <motion.div
                    key={t.title}
                    className="card tip-card"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: 0.15 + i * 0.04 }}
                  >
                    <span className="t-ico">
                      <Icon size={18} />
                    </span>
                    <h3>{t.title}</h3>
                    <p>{t.body}</p>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
});

export default Guide;
