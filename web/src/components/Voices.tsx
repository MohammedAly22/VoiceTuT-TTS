import { AnimatePresence, motion } from "framer-motion";
import { Mic, Pause, Play, Search, ShieldCheck, Square, Upload, UserRound, Wand2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { assetUrl, type Speaker, type VoiceMode } from "../api";
import { formatTime, toReferenceWav } from "../audio";
import { stopPreview, usePreview } from "../hooks";
import { Eq, GenderIcon, Segmented } from "./ui";

export interface RefClip {
  wav: Blob;
  url: string;
  duration: number;
  peaks: Float32Array;
  name: string;
}

const MAX_REC = 15;

function MiniWave({ peaks }: { peaks: Float32Array }) {
  return (
    <span className="mini-wave" aria-hidden>
      {Array.from(peaks).map((p, i) => (
        <i key={i} style={{ height: `${Math.max(8, p * 100)}%` }} />
      ))}
    </span>
  );
}

function BuiltIn({
  speakers,
  selected,
  onSelect,
}: {
  speakers: Speaker[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const [q, setQ] = useState("");
  const [g, setG] = useState<"all" | "male" | "female">("all");
  const [playing, toggle] = usePreview();
  const list = useMemo(() => {
    const k = q.trim().toLowerCase();
    return speakers.filter(
      (s) =>
        (g === "all" || s.gender === g) &&
        (!k || s.speaker_name.toLowerCase().includes(k) || s.arabic_name.includes(k) || s.tags.some((t) => t.includes(k))),
    );
  }, [speakers, q, g]);
  const sel = speakers.find((s) => s.speaker_id === selected) ?? null;

  return (
    <>
      <div className="search">
        <Search size={16} />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="دوّر باسم الصوت أو طابعه…" aria-label="بحث" />
      </div>
      <Segmented
        size="sm"
        full
        label="النوع"
        value={g}
        onChange={setG}
        options={[
          { value: "all", label: `الكل (${speakers.length})` },
          { value: "male", label: "رجالي" },
          { value: "female", label: "حريمي" },
        ]}
      />
      <motion.div className="voices" role="listbox" aria-label="الأصوات" layout>
        <AnimatePresence initial={false}>
          {list.map((s) => {
            const on = s.speaker_id === selected;
            const isPlaying = playing === s.speaker_id;
            return (
              <motion.div
                key={s.speaker_id}
                layout
                initial={{ opacity: 0, scale: 0.94 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.94 }}
                transition={{ duration: 0.2 }}
                className="voice"
                role="option"
                aria-selected={on}
                tabIndex={0}
                onClick={() => onSelect(s.speaker_id)}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onSelect(s.speaker_id))}
              >
                {on && <motion.span layoutId="voice-ring" className="ring" transition={{ type: "spring", stiffness: 500, damping: 36 }} />}
                <span className={`avatar ${s.gender === "female" ? "female" : "male"}`}><GenderIcon gender={s.gender} /></span>
                <span className="voice-main">
                  <b>{s.arabic_name}</b>
                  <small>{s.tags.join(" · ") || s.speaker_name}</small>
                </span>
                <button
                  className="voice-play"
                  aria-label={`اسمع ${s.arabic_name}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggle(s.speaker_id, assetUrl(s.audio_url));
                  }}
                >
                  {isPlaying ? <Eq /> : <Play size={12} fill="currentColor" />}
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
        {!list.length && <div className="empty-note">مفيش صوت بالاسم ده</div>}
      </motion.div>

      <AnimatePresence mode="wait">
        {sel && (
          <motion.div
            key={sel.speaker_id}
            className="voice-detail"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
          >
            <div className="row">
              <span className={`avatar ${sel.gender === "female" ? "female" : "male"}`}><GenderIcon gender={sel.gender} /></span>
              <span className="voice-main">
                <b>{sel.arabic_name}</b>
                <small className="ltr" style={{ textAlign: "right" }}>
                  {sel.speaker_name}
                </small>
              </span>
              <div className="tags">
                <span className="tag g">{sel.gender === "female" ? "حريمي" : "رجالي"}</span>
                {sel.tags.map((t) => (
                  <span key={t} className="tag">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function Clone({
  clip,
  onClip,
  refText,
  onRefText,
  onToast,
}: {
  clip: RefClip | null;
  onClip: (c: RefClip | null) => void;
  refText: string;
  onRefText: (t: string) => void;
  onToast: (kind: "error" | "info" | "ok", text: string) => void;
}) {
  const file = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [rec, setRec] = useState(false);
  const [secs, setSecs] = useState(0);
  const [levels, setLevels] = useState<number[]>(() => Array(36).fill(0.05));
  const [playing, setPlaying] = useState(false);
  const recRef = useRef<{ stop: () => void } | null>(null);
  const player = useRef<HTMLAudioElement | null>(null);

  useEffect(() => () => recRef.current?.stop(), []);

  const accept = async (blob: Blob, name: string) => {
    try {
      const { wav, duration, peaks } = await toReferenceWav(blob, 20);
      if (duration < 2) {
        onToast("error", "الصوت قصير أوي — محتاجين ٣ ثواني على الأقل");
        return;
      }
      if (clip) URL.revokeObjectURL(clip.url);
      onClip({ wav, url: URL.createObjectURL(wav), duration, peaks, name });
      onToast("ok", "الصوت المرجعي جاهز");
    } catch {
      onToast("error", "مقدرناش نقرا الملف ده — جرّب wav أو mp3");
    }
  };

  const startRec = async () => {
    stopPreview();
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch {
      onToast("error", "محتاجين إذن الميكروفون");
      return;
    }
    const mr = new MediaRecorder(stream);
    const parts: Blob[] = [];
    const ac = new AudioContext();
    const an = ac.createAnalyser();
    an.fftSize = 256;
    ac.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.fftSize);
    const t0 = performance.now();
    let raf = 0;
    let last = 0;
    const tick = (t: number) => {
      const s = (t - t0) / 1000;
      if (t - last > 60) {
        last = t;
        an.getByteTimeDomainData(buf);
        let peak = 0;
        for (const v of buf) peak = Math.max(peak, Math.abs(v - 128) / 128);
        setLevels((l) => [...l.slice(1), Math.min(1, 0.06 + peak * 1.6)]);
        setSecs(s);
      }
      if (s >= MAX_REC) stop();
      else raf = requestAnimationFrame(tick);
    };
    const stop = () => {
      cancelAnimationFrame(raf);
      if (mr.state !== "inactive") mr.stop();
    };
    mr.ondataavailable = (e) => e.data.size && parts.push(e.data);
    mr.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      void ac.close();
      setRec(false);
      recRef.current = null;
      if (parts.length) void accept(new Blob(parts, { type: mr.mimeType }), "تسجيل من الميكروفون");
    };
    recRef.current = { stop };
    mr.start();
    setSecs(0);
    setRec(true);
    raf = requestAnimationFrame(tick);
  };

  const togglePlay = () => {
    if (!clip) return;
    if (!player.current) {
      player.current = new Audio();
      player.current.onended = () => setPlaying(false);
      player.current.onpause = () => setPlaying(false);
    }
    const p = player.current;
    if (playing) {
      p.pause();
      return;
    }
    stopPreview();
    p.src = clip.url;
    void p.play().then(() => setPlaying(true));
  };

  return (
    <>
      <AnimatePresence mode="wait" initial={false}>
        {clip ? (
          <motion.div key="clip" className="clip" initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.97 }}>
            <button className="voice-play" onClick={togglePlay} aria-label="اسمع الصوت المرجعي">
              {playing ? <Pause size={12} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
            </button>
            <span className="voice-main" style={{ maxWidth: 110 }}>
              <b>صوتك</b>
              <small className="num">
                {clip.name} · {formatTime(clip.duration)}
              </small>
            </span>
            <MiniWave peaks={clip.peaks.filter((_, i) => i % 3 === 0)} />
            <button
              className="icon-btn sm ghost"
              aria-label="امسح الصوت"
              onClick={() => {
                player.current?.pause();
                URL.revokeObjectURL(clip.url);
                onClip(null);
              }}
            >
              <X size={15} />
            </button>
          </motion.div>
        ) : (
          <motion.div key="pick" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div
              className={`drop${over ? " over" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => file.current?.click()}
              onKeyDown={(e) => e.key === "Enter" && file.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setOver(true);
              }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(false);
                const f = e.dataTransfer.files?.[0];
                if (f) void accept(f, f.name);
              }}
            >
              <motion.span className="drop-ico" animate={over ? { y: -4, scale: 1.06 } : { y: 0, scale: 1 }}>
                <Upload size={20} />
              </motion.span>
              <b>اسحب ملف صوتي هنا أو دوس للاختيار</b>
              <small>٥–١٥ ثانية لصوت واحد، نضيف ومن غير مزيكا</small>
              <input
                ref={file}
                type="file"
                accept="audio/*"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void accept(f, f.name);
                  e.target.value = "";
                }}
              />
            </div>
            <div className="or">أو</div>
            <div className="rec">
              <motion.button
                className={`rec-btn${rec ? "" : " idle"}`}
                whileTap={{ scale: 0.92 }}
                animate={rec ? { boxShadow: ["0 0 0 0 rgba(224,69,75,.45)", "0 0 0 12px rgba(224,69,75,0)"] } : {}}
                transition={rec ? { duration: 1.2, repeat: Infinity } : {}}
                onClick={() => (rec ? recRef.current?.stop() : void startRec())}
                aria-label={rec ? "وقّف التسجيل" : "سجّل"}
              >
                {rec ? <Square size={16} fill="currentColor" /> : <Mic size={19} />}
              </motion.button>
              {rec ? (
                <span className="rec-meter" aria-hidden>
                  {levels.map((l, i) => (
                    <motion.i key={i} animate={{ height: `${l * 100}%` }} transition={{ duration: 0.08 }} />
                  ))}
                </span>
              ) : (
                <span className="voice-main">
                  <b>سجّل بصوتك</b>
                  <small>لحد {MAX_REC} ثانية، اتكلم بشكل طبيعي</small>
                </span>
              )}
              {rec && <span className="rec-time num">{formatTime(secs)}</span>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="field">
        <label htmlFor="ref-text">النص اللي اتقال في الصوت (اختياري)</label>
        <textarea
          id="ref-text"
          dir="auto"
          value={refText}
          onChange={(e) => onRefText(e.target.value)}
          placeholder="لو كتبته النتيجة بتبقى أدق. لو سبته فاضي هيتكتب تلقائي."
        />
      </div>
      <div className="consent">
        <ShieldCheck size={15} />
        <span>استخدم صوتك أو صوت حد موافق. متقلدش صوت حد من غير إذنه.</span>
      </div>
    </>
  );
}

export default function Voices(props: {
  speakers: Speaker[];
  mode: VoiceMode;
  onMode: (m: VoiceMode) => void;
  speaker: string | null;
  onSpeaker: (id: string) => void;
  clip: RefClip | null;
  onClip: (c: RefClip | null) => void;
  refText: string;
  onRefText: (t: string) => void;
  onToast: (kind: "error" | "info" | "ok", text: string) => void;
}) {
  return (
    <div className="card">
      <div className="side-body">
        <Segmented
          full
          label="مصدر الصوت"
          value={props.mode}
          onChange={props.onMode}
          options={[
            { value: "speaker", label: "أصوات جاهزة", icon: <UserRound size={15} /> },
            { value: "clone", label: "استنسخ صوت", icon: <Wand2 size={15} /> },
          ]}
        />
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={props.mode}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 12 }}
            transition={{ duration: 0.2 }}
            style={{ display: "flex", flexDirection: "column", gap: 12 }}
          >
            {props.mode === "speaker" ? (
              <BuiltIn speakers={props.speakers} selected={props.speaker} onSelect={props.onSpeaker} />
            ) : (
              <Clone clip={props.clip} onClip={props.onClip} refText={props.refText} onRefText={props.onRefText} onToast={props.onToast} />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
