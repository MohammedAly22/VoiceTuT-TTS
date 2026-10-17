import { motion } from "framer-motion";
import { useId, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";

const spring = { type: "spring", stiffness: 500, damping: 38 } as const;

export interface SegOption<T extends string> {
  value: T;
  label: ReactNode;
  icon?: ReactNode;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size,
  full,
  label,
}: {
  value: T;
  options: SegOption<T>[];
  onChange: (v: T) => void;
  size?: "sm";
  full?: boolean;
  label: string;
}) {
  const id = useId();
  return (
    <div className={`seg${size ? " " + size : ""}${full ? " full" : ""}`} role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={o.value === value} onClick={() => onChange(o.value)}>
          {o.value === value && <motion.span layoutId={`seg-${id}`} className="seg-thumb" transition={spring} />}
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Pointer + keyboard slider. Always left-to-right (numbers read LTR even in RTL pages). */
export function Slider({
  value,
  min,
  max,
  step,
  onChange,
  label,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  label: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState(false);
  const frac = (value - min) / (max - min);

  const fromX = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    const raw = min + f * (max - min);
    const snapped = Math.round(raw / step) * step;
    onChange(+Math.max(min, Math.min(max, snapped)).toFixed(4));
  };

  const down = (e: PointerEvent) => {
    (e.target as Element).setPointerCapture?.(e.pointerId);
    setDrag(true);
    fromX(e.clientX);
  };

  const key = (e: KeyboardEvent) => {
    const k = e.key;
    let v = value;
    if (k === "ArrowRight" || k === "ArrowUp") v += step;
    else if (k === "ArrowLeft" || k === "ArrowDown") v -= step;
    else if (k === "Home") v = min;
    else if (k === "End") v = max;
    else return;
    e.preventDefault();
    onChange(+Math.max(min, Math.min(max, v)).toFixed(4));
  };

  return (
    <div
      ref={ref}
      className={`slider${drag ? " drag" : ""}`}
      role="slider"
      tabIndex={0}
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={value}
      onPointerDown={down}
      onPointerMove={(e) => drag && fromX(e.clientX)}
      onPointerUp={() => setDrag(false)}
      onPointerCancel={() => setDrag(false)}
      onKeyDown={key}
    >
      <div className="track">
        <motion.div className="fill" style={{ left: 0 }} animate={{ width: `${frac * 100}%` }} transition={drag ? { duration: 0 } : spring} />
      </div>
      <motion.div className="thumb" animate={{ left: `${frac * 100}%` }} transition={drag ? { duration: 0 } : spring} />
    </div>
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button className="switch" role="switch" aria-checked={checked} aria-label={label} onClick={() => onChange(!checked)}>
      {/* RTL: off = knob at the start (right), on = knob at the end (left) */}
      <motion.span className="knob" animate={{ left: checked ? 2 : 20 }} transition={spring} />
    </button>
  );
}

/** ♀ / ♂ symbols drawn in the lucide stroke style. */
export function GenderIcon({ gender, size = 18 }: { gender: string; size?: number }) {
  const female = gender === "female";
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {female ? (
        <>
          <circle cx="12" cy="9" r="5.5" />
          <path d="M12 14.5V22M8.5 18.5h7" />
        </>
      ) : (
        <>
          <circle cx="10" cy="14" r="5.5" />
          <path d="M14 10l6.5-6.5M15 3.5h5.5V9" />
        </>
      )}
    </svg>
  );
}

export function Eq() {
  return (
    <span className="eq" aria-hidden>
      <i />
      <i />
      <i />
    </span>
  );
}

export function Ankh({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <defs>
        <linearGradient id="ankh-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ffd88f" />
          <stop offset="1" stopColor="#c8891a" />
        </linearGradient>
      </defs>
      <g stroke="url(#ankh-g)" strokeWidth="2.1" strokeLinecap="round">
        <ellipse cx="12" cy="6.6" rx="3.3" ry="4.1" />
        <path d="M12 10.8V21.5M6.6 12.4h10.8" />
      </g>
    </svg>
  );
}
