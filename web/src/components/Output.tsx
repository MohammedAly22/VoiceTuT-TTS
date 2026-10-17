import { animate, AnimatePresence, motion } from "framer-motion";
import { AudioLines, Clock, Download, Gauge, Pause, Play, Timer, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Metrics } from "../api";
import { engine, formatTime } from "../audio";
import { useEngine, useFrame } from "../hooks";
import Waveform from "./Waveform";

export interface HistoryItem {
  id: number;
  text: string;
  voice: string;
  duration: number;
  audio: ArrayBuffer;
  metrics: Metrics;
}

export type RunState = "idle" | "busy" | "ready" | "error";

function Count({ value, digits = 2 }: { value: number | null | undefined; digits?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const last = useRef(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (value == null || !isFinite(value)) {
      el.textContent = "—";
      return;
    }
    const c = animate(last.current, value, {
      duration: 0.7,
      ease: "easeOut",
      onUpdate: (v) => (el.textContent = v.toFixed(digits)),
    });
    last.current = value;
    return () => c.stop();
  }, [value, digits]);
  return <span ref={ref}>—</span>;
}

function Clock2() {
  const st = useEngine();
  const ref = useRef<HTMLSpanElement>(null);
  useFrame(() => {
    if (ref.current) ref.current.textContent = formatTime(engine.currentTime());
  });
  return (
    <span className="time num">
      <span ref={ref}>0:00</span>
      <span className="sep">/</span>
      <span>{formatTime(st.duration)}</span>
    </span>
  );
}

const STATUS: Record<RunState, string> = {
  idle: "جاهز",
  busy: "بيولّد…",
  ready: "خلص",
  error: "حصل خطأ",
};

export default function Output({
  run,
  busyLabel,
  metrics,
  voiceLabel,
  history,
  currentId,
  onPick,
}: {
  run: RunState;
  busyLabel: string;
  metrics: Metrics | null;
  voiceLabel: string;
  history: HistoryItem[];
  currentId: number | null;
  onPick: (h: HistoryItem) => void;
}) {
  const st = useEngine();
  const [dl, setDl] = useState(false);
  const canPlay = st.mode === "ready";

  return (
    <div className="card">
      <div className="card-head">
        <h2>
          <AudioLines size={17} />
          الصوت
        </h2>
        <span className="grow" />
        <span className="status" data-s={run}>
          <i className="dot" />
          {STATUS[run]}
        </span>
      </div>

      <Waveform busyLabel={busyLabel} />

      <div className="player">
        <motion.button
          className="play"
          disabled={!canPlay}
          whileTap={{ scale: 0.9 }}
          onClick={() => engine.toggle()}
          aria-label={st.playing ? "إيقاف مؤقت" : "تشغيل"}
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={st.playing ? "p" : "s"}
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.5, opacity: 0 }}
              transition={{ duration: 0.14 }}
              style={{ display: "grid" }}
            >
              {st.playing ? <Pause size={22} fill="currentColor" /> : <Play size={22} fill="currentColor" style={{ marginInlineEnd: -3 }} />}
            </motion.span>
          </AnimatePresence>
        </motion.button>
        <Clock2 />
        <span className="grow now-voice">{voiceLabel}</span>
        <a
          className="btn sm"
          href={st.url ?? undefined}
          download="voicetut.wav"
          aria-disabled={!canPlay}
          style={canPlay ? undefined : { opacity: 0.5, pointerEvents: "none" }}
          onClick={() => {
            setDl(true);
            setTimeout(() => setDl(false), 1200);
          }}
        >
          <motion.span animate={dl ? { y: [0, 3, 0] } : {}} transition={{ duration: 0.5 }} style={{ display: "grid" }}>
            <Download size={15} />
          </motion.span>
          تنزيل
        </a>
      </div>

      <dl className="metrics">
        <div className="metric">
          <dt>
            <Zap size={13} />
            أول صوت
          </dt>
          <dd className="num">
            <Count value={metrics?.ttfa} />
            <small>s</small>
          </dd>
        </div>
        <div className="metric">
          <dt>
            <Timer size={13} />
            الوقت الكلي
          </dt>
          <dd className="num">
            <Count value={metrics?.total} />
            <small>s</small>
          </dd>
        </div>
        <div className="metric">
          <dt>
            <Clock size={13} />
            مدة الصوت
          </dt>
          <dd className="num">
            <Count value={metrics?.audio} />
            <small>s</small>
          </dd>
        </div>
        <div className="metric">
          <dt>
            <Gauge size={13} />
            RTF
          </dt>
          <dd className="num">
            <Count value={metrics?.rtf} />
            <small>×</small>
          </dd>
        </div>
      </dl>

      <AnimatePresence initial={false}>
        {history.length > 0 && (
          <motion.div className="history" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}>
            <div className="section-label">آخر التوليدات</div>
            <ol>
              <AnimatePresence initial={false}>
                {history.map((h) => (
                  <motion.li key={h.id} layout initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                    <button onClick={() => onPick(h)} aria-current={h.id === currentId}>
                      <span className="h-ico">
                        <Play size={13} fill="currentColor" />
                      </span>
                      <span className="h-text" dir="auto">
                        {h.text}
                      </span>
                      <span className="h-meta num">
                        {h.voice} · {formatTime(h.duration)}
                      </span>
                    </button>
                  </motion.li>
                ))}
              </AnimatePresence>
            </ol>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
