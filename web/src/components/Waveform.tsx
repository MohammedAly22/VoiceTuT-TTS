import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { engine, formatTime } from "../audio";
import { useEngine, useFrame } from "../hooks";

// Time flows right -> left to match the RTL layout (like voice notes in Arabic apps).
const BAR = 3;
const GAP = 2.4;

interface Palette {
  a: string;
  b: string;
  muted: string;
  idle: string;
}

function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  return {
    a: s.getPropertyValue("--accent").trim(),
    b: s.getPropertyValue("--accent-2").trim(),
    muted: s.getPropertyValue("--wave-muted").trim(),
    idle: s.getPropertyValue("--wave-idle").trim(),
  };
}

function resample(peaks: Float32Array, n: number): Float32Array {
  const out = new Float32Array(n);
  if (!peaks.length) return out;
  const step = peaks.length / n;
  for (let i = 0; i < n; i++) {
    const s = Math.floor(i * step);
    const e = Math.max(s + 1, Math.floor((i + 1) * step));
    let m = 0;
    for (let j = s; j < e && j < peaks.length; j++) if (peaks[j] > m) m = peaks[j];
    out[i] = m;
  }
  return out;
}

const easeOut = (x: number) => 1 - Math.pow(1 - Math.max(0, Math.min(1, x)), 3);

export default function Waveform({ busyLabel }: { busyLabel: string }) {
  const st = useEngine();
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const palette = useRef<Palette | null>(null);
  const cache = useRef<{ v: number; n: number; bars: Float32Array }>({ v: -1, n: 0, bars: new Float32Array(0) });
  const level = useRef(0);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    palette.current = readPalette();
    const mo = new MutationObserver(() => (palette.current = readPalette()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => mo.disconnect();
  }, []);

  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => {
      const c = canvas.current!;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      c.width = Math.round(el.clientWidth * dpr);
      c.height = Math.round(el.clientHeight * dpr);
      cache.current.v = -1;
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useFrame((t) => {
    const c = canvas.current;
    const pal = palette.current;
    if (!c || !pal || !c.width) return;
    const g = c.getContext("2d")!;
    const dpr = c.width / (wrap.current?.clientWidth || 1);
    const W = c.width;
    const H = c.height;
    const bw = BAR * dpr;
    const step = (BAR + GAP) * dpr;
    const n = Math.max(8, Math.floor((W - 24 * dpr) / step));
    const x0 = (W - n * step + GAP * dpr) / 2;
    const mid = H / 2;
    const maxH = H * 0.78;
    g.clearRect(0, 0, W, H);

    const mode = st.mode;
    let bars: Float32Array;
    if (mode === "ready" || mode === "streaming") {
      if (cache.current.v !== st.peaksVersion || cache.current.n !== n) {
        cache.current = { v: st.peaksVersion, n, bars: resample(engine.peaks, n) };
      }
      bars = cache.current.bars;
    } else bars = new Float32Array(0);

    const prog = mode === "ready" || mode === "streaming" ? engine.progress() : 0;
    level.current += (engine.level() - level.current) * 0.25;
    const playIdx = prog * n;

    const grad = g.createLinearGradient(W, 0, 0, 0);
    grad.addColorStop(0, pal.a);
    grad.addColorStop(1, pal.b);

    for (let i = 0; i < n; i++) {
      // i = 0 is the right-most bar (start of the clip)
      const x = W - x0 - (i + 1) * step + GAP * dpr;
      let amp: number;
      let color: string | CanvasGradient;
      if (mode === "empty") {
        amp = 0.05 + 0.05 * (1 + Math.sin(i * 0.33 + t * 0.0013) * Math.sin(i * 0.071 - t * 0.0006));
        color = pal.idle;
      } else if (mode === "loading") {
        const phase = ((t / 1400) % 1.25) - 0.12;
        const d = i / n - phase;
        const bump = Math.exp(-(d * d) / 0.0065);
        amp = 0.06 + 0.62 * bump * (0.65 + 0.35 * Math.sin(i * 0.9 + t * 0.012)) + 0.04 * Math.sin(i * 0.5 + t * 0.004);
        g.globalAlpha = 0.35 + 0.65 * bump;
        color = grad;
      } else {
        const reveal = mode === "ready" ? easeOut((t - engine.revealAt - (i / n) * 520) / 320) : 1;
        amp = Math.max(0.035, bars[i]) * reveal;
        const near = Math.abs(i - playIdx);
        if (st.playing && near < 7) amp = Math.min(1, amp * (1 + level.current * 0.55 * (1 - near / 7)));
        color = i < playIdx ? grad : pal.muted;
        g.globalAlpha = 1;
      }
      const h = Math.max(2 * dpr, amp * maxH);
      g.fillStyle = color;
      g.beginPath();
      g.roundRect(x, mid - h / 2, bw, h, bw / 2);
      g.fill();
    }
    g.globalAlpha = 1;

    // playhead
    if ((mode === "ready" && (prog > 0 || st.playing)) || mode === "streaming") {
      const px = W - x0 - playIdx * step;
      g.fillStyle = pal.a;
      g.shadowColor = pal.a;
      g.shadowBlur = 12 * dpr;
      g.fillRect(px - dpr, H * 0.08, 2 * dpr, H * 0.84);
      g.shadowBlur = 0;
    }

    // hover cursor
    if (hover !== null && mode === "ready") {
      g.fillStyle = pal.muted;
      g.fillRect(hover * dpr - dpr / 2, 0, dpr, H);
    }
  });

  const seekable = st.mode === "ready";
  const fracAt = (clientX: number) => {
    const r = wrap.current!.getBoundingClientRect();
    return Math.max(0, Math.min(1, (r.right - clientX) / r.width));
  };

  return (
    <div
      ref={wrap}
      className={`wave${seekable ? " seekable" : ""}`}
      onPointerMove={(e) => seekable && setHover(e.clientX - wrap.current!.getBoundingClientRect().left)}
      onPointerLeave={() => setHover(null)}
      onClick={(e) => {
        if (!seekable) return;
        engine.seek(fracAt(e.clientX));
        if (!st.playing) void engine.play();
      }}
    >
      <canvas ref={canvas} aria-hidden />
      {hover !== null && seekable && (
        <span className="wave-tip num" style={{ left: hover }}>
          {formatTime(fracAt(hover + wrap.current!.getBoundingClientRect().left) * st.duration)}
        </span>
      )}
      <AnimatePresence>
        {(st.mode === "empty" || st.mode === "loading") && (
          <motion.div className="wave-msg" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}>
            <span>{st.mode === "empty" ? "الصوت هيظهر هنا بعد التوليد" : busyLabel}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
