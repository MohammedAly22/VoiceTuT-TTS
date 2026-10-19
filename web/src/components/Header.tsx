import { AnimatePresence, motion } from "framer-motion";
import { Cpu, Github, Moon, Sun } from "lucide-react";
import type { Info } from "../api";
import type { Theme } from "../hooks";

export function Header({
  info,
  offline,
  theme,
  onTheme,
}: {
  info: Info | null;
  offline: boolean;
  theme: Theme;
  onTheme: () => void;
}) {
  const dot = offline ? "off" : !info ? "warn" : info.mock ? "warn" : "on";
  const label = offline ? "غير متصل" : !info ? "بيحمّل…" : info.mock ? "وضع المعاينة" : info.device;
  const next = theme === "dark" ? "light" : "dark";
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <a className="brand" href="./" aria-label="VoiceTut-TTS">
          <span className="wordmark">
            VoiceTut<span>TTS</span>
          </span>
          <span className="ver">v1</span>
        </a>
        <div className="top-actions">
          <span className="pill device" title="الجهاز اللي شغال عليه الموديل">
            <i className={`dot ${dot}`} />
            <Cpu size={14} />
            <span className="lbl ltr">{label}</span>
          </span>
          <a
            className="icon-btn"
            href="https://github.com/MohammedAly22/VoiceTuT-TTS"
            target="_blank"
            rel="noopener"
            title="GitHub"
            aria-label="GitHub"
          >
            <Github size={17} />
          </a>
          {/* the icon shows the mode you switch TO: sun in dark mode, moon in light mode */}
          <button
            className="icon-btn theme-btn"
            onClick={onTheme}
            title={next === "light" ? "الوضع الفاتح" : "الوضع الغامق"}
            aria-label={next === "light" ? "التحويل للوضع الفاتح" : "التحويل للوضع الغامق"}
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.span
                key={theme}
                initial={{ rotate: -90, scale: 0.4, opacity: 0 }}
                animate={{ rotate: 0, scale: 1, opacity: 1 }}
                exit={{ rotate: 90, scale: 0.4, opacity: 0 }}
                transition={{ duration: 0.25 }}
                style={{ display: "grid" }}
              >
                {next === "light" ? <Sun size={17} /> : <Moon size={17} />}
              </motion.span>
            </AnimatePresence>
          </button>
        </div>
      </div>
    </header>
  );
}

export function Hero() {
  return (
    <section className="hero">
      <motion.h1 initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        اكتب بالمصري، <span className="grad">واسمعه بصوت طبيعي</span>
      </motion.h1>
    </section>
  );
}
