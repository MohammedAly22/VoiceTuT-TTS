import { RotateCcw, SlidersHorizontal } from "lucide-react";
import { Segmented, Slider, Switch } from "./ui";

export interface Params {
  numStep: number;
  guidance: number;
  speed: number;
  normalize: boolean;
}

export const DEFAULT_PARAMS: Params = { numStep: 32, guidance: 2.0, speed: 1.0, normalize: true };

type Preset = "fast" | "balanced" | "best" | "custom";
const PRESETS: Record<Exclude<Preset, "custom">, number> = { fast: 16, balanced: 32, best: 64 };

export default function Settings({ value, onChange }: { value: Params; onChange: (p: Params) => void }) {
  const preset: Preset =
    (Object.entries(PRESETS).find(([, n]) => n === value.numStep)?.[0] as Preset | undefined) ?? "custom";
  const set = (p: Partial<Params>) => onChange({ ...value, ...p });
  const isDefault = JSON.stringify(value) === JSON.stringify(DEFAULT_PARAMS);

  return (
    <div className="card">
      <div className="card-head">
        <h2>
          <SlidersHorizontal size={16} />
          الإعدادات
        </h2>
        <span className="grow" />
        <button className="link" onClick={() => onChange(DEFAULT_PARAMS)} disabled={isDefault} style={isDefault ? { opacity: 0.4 } : undefined}>
          <RotateCcw size={13} />
          الافتراضي
        </button>
      </div>
      <div className="side-body settings">
        <div>
          <div className="section-label">الجودة</div>
          <Segmented
            full
            size="sm"
            label="الجودة"
            value={preset}
            onChange={(p) => p !== "custom" && set({ numStep: PRESETS[p] })}
            options={[
              { value: "fast", label: "سريع" },
              { value: "balanced", label: "متوازن" },
              { value: "best", label: "أعلى جودة" },
            ]}
          />
        </div>

        <div className="slider-row">
          <div className="slider-head">
            <span>خطوات التوليد</span>
            <code>num_step</code>
            <output>{value.numStep}</output>
          </div>
          <Slider label="خطوات التوليد" min={8} max={64} step={1} value={value.numStep} onChange={(v) => set({ numStep: v })} />
          <div className="slider-hint">
            <span>أسرع</span>
            <span>أنضف</span>
          </div>
        </div>

        <div className="slider-row">
          <div className="slider-head">
            <span>قوة الالتزام بالنص</span>
            <code>guidance</code>
            <output>{value.guidance.toFixed(1)}</output>
          </div>
          <Slider label="قوة الالتزام بالنص" min={1} max={4} step={0.1} value={value.guidance} onChange={(v) => set({ guidance: v })} />
        </div>

        <div className="slider-row">
          <div className="slider-head">
            <span>السرعة</span>
            <code>speed</code>
            <output>{value.speed.toFixed(2)}×</output>
          </div>
          <Slider label="السرعة" min={0.6} max={1.6} step={0.05} value={value.speed} onChange={(v) => set({ speed: v })} />
          <div className="slider-hint">
            <span>أبطأ</span>
            <span>أسرع</span>
          </div>
        </div>

        <div
          className="switch-row"
          onClick={(e) => !(e.target as Element).closest(".switch") && set({ normalize: !value.normalize })}
        >
          <span className="txt">
            <b>تطبيع النص</b>
            <small>يحوّل الأرقام والمواعيد والإيميلات لكلام</small>
          </span>
          <Switch label="تطبيع النص" checked={value.normalize} onChange={(v) => set({ normalize: v })} />
        </div>
      </div>
    </div>
  );
}
