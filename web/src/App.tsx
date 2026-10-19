import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, CheckCircle2, Info as InfoIcon, Sparkles, Square, Waves } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { getInfo, streamSynth, synthesize, type Info, type Language, type Metrics, type SynthRequest, type VoiceMode } from "./api";
import { audioContext, concat, decode, encodeWav, engine, mono } from "./audio";
import Editor, { type EditorHandle } from "./components/Editor";
import { Header, Hero } from "./components/Header";
import Output, { type HistoryItem, type RunState } from "./components/Output";
import Settings, { DEFAULT_PARAMS, type Params } from "./components/Settings";
import { Segmented } from "./components/ui";
import Voices, { type RefClip } from "./components/Voices";
import { stopPreview, useTheme, useToasts } from "./hooks";

type GenMode = "generate" | "stream";

const FIRST_TEXT = "اتفقنا نعمل ال meeting بكرة الصبح، فياريت كل واحد يجهز ال presentation بتاعته.";
const FIRST_TEXT_EN = "The meeting has been moved to Thursday at 3:30, so please update your calendars.";

function loadParams(): Params {
  try {
    const p = JSON.parse(localStorage.getItem("voicetut-params") || "null");
    if (p && typeof p.numStep === "number") return { ...DEFAULT_PARAMS, ...p };
  } catch {
    /* ignore */
  }
  return DEFAULT_PARAMS;
}

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const { toasts, push } = useToasts();

  const [info, setInfo] = useState<Info | null>(null);
  const [offline, setOffline] = useState(false);

  const [text, setText] = useState(FIRST_TEXT);
  const [language, setLanguage] = useState<Language>("arz");
  const [voiceMode, setVoiceMode] = useState<VoiceMode>("speaker");
  const [speaker, setSpeaker] = useState<string | null>(null);
  const [clip, setClip] = useState<RefClip | null>(null);
  const [refText, setRefText] = useState("");
  const [params, setParams] = useState<Params>(loadParams);
  const [genMode, setGenMode] = useState<GenMode>("generate");

  const [run, setRun] = useState<RunState>("idle");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [voiceLabel, setVoiceLabel] = useState("");

  const abort = useRef<AbortController | null>(null);
  const editor = useRef<EditorHandle>(null);
  const editorWrap = useRef<HTMLDivElement>(null);

  // ---- server info (retries while a Space is waking up)
  useEffect(() => {
    let dead = false;
    let timer = 0;
    const load = () =>
      getInfo()
        .then((i) => {
          if (dead) return;
          setInfo(i);
          setOffline(false);
          setSpeaker((s) => s ?? i.speakers.find((x) => x.speaker_name === "Mohammed")?.speaker_id ?? i.speakers[0]?.speaker_id ?? null);
        })
        .catch(() => {
          if (dead) return;
          setOffline(true);
          timer = window.setTimeout(load, 4000);
        });
    load();
    return () => {
      dead = true;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("voicetut-params", JSON.stringify(params));
    } catch {
      /* ignore */
    }
  }, [params]);

  // space bar = play / pause (outside text fields)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.code !== "Space" || t.closest("textarea, input, button, [role=slider], [role=option], a")) return;
      const mode = engine.getState().mode;
      if (mode !== "ready" && mode !== "streaming") return;
      e.preventDefault();
      engine.toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const busy = run === "busy";
  const maxChars = info?.max_chars ?? 3000;

  const currentVoiceName = useCallback(() => {
    if (voiceMode === "clone") return "صوتك";
    return info?.speakers.find((s) => s.speaker_id === speaker)?.arabic_name ?? "";
  }, [voiceMode, speaker, info]);

  const addHistory = (audio: ArrayBuffer, m: Metrics, duration: number, voice: string, t: string) => {
    const id = Date.now();
    setHistory((h) => [{ id, text: t, voice, duration, audio, metrics: m }, ...h].slice(0, 6));
    setCurrentId(id);
  };

  const generate = async () => {
    if (busy) return;
    const t = text.trim();
    if (!t) return push("info", "اكتب النص الأول");
    if (t.length > maxChars) return push("error", `النص أطول من ${maxChars} حرف`);
    if (!info) return push("error", "السيرفر لسه مش جاهز، استنى ثواني");
    if (voiceMode === "speaker" && !speaker) return push("info", "اختار صوت");
    if (voiceMode === "clone" && !clip) return push("info", "ارفع أو سجّل صوت مرجعي الأول");

    stopPreview();
    audioContext(); // unlock audio inside the click gesture
    const req: SynthRequest = {
      text: t,
      language,
      voiceMode,
      speaker,
      refAudio: clip?.wav ?? null,
      refText,
      numStep: params.numStep,
      guidance: params.guidance,
      speed: params.speed,
      normalize: params.normalize,
    };
    const voice = currentVoiceName();
    setRun("busy");
    setMetrics(null);
    setVoiceLabel(voice);

    if (genMode === "generate") {
      engine.setLoading();
      try {
        const { audio, metrics: m } = await synthesize(req);
        await engine.load(audio);
        setMetrics(m);
        addHistory(audio, m, m.audio, voice, t);
        setRun("ready");
      } catch (e) {
        engine.reset();
        setRun("error");
        push("error", (e as Error).message);
      }
      return;
    }

    // streaming
    const ac = new AbortController();
    abort.current = ac;
    const parts: ArrayBuffer[] = [];
    let last: Metrics | null = null;
    engine.setLoading();
    try {
      let first = true;
      for await (const { audio, metrics: m } of streamSynth(req, ac.signal)) {
        if (first) {
          engine.beginStream();
          first = false;
        }
        parts.push(audio);
        last = m;
        setMetrics(m);
        await engine.pushChunk(audio);
      }
      engine.endStream({ cancelled: ac.signal.aborted });
      if (!parts.length) {
        if (!ac.signal.aborted) throw new Error("مفيش صوت رجع من السيرفر");
        engine.reset();
        setRun("idle");
        return;
      }
      setRun(ac.signal.aborted ? "idle" : "ready");
      // keep a stitched copy in the history list
      const decoded = await Promise.all(parts.map(decode));
      const sr = decoded[0].sampleRate;
      const all = concat(decoded.map(mono));
      const wav = await encodeWav(all, sr).arrayBuffer();
      addHistory(wav, last!, all.length / sr, voice, t);
    } catch (e) {
      engine.endStream({ cancelled: true });
      if (!parts.length) engine.reset();
      setRun("error");
      push("error", (e as Error).message);
    } finally {
      abort.current = null;
    }
  };

  const stop = () => {
    abort.current?.abort();
    engine.endStream({ cancelled: true });
  };

  const pick = async (h: HistoryItem) => {
    stopPreview();
    setCurrentId(h.id);
    setMetrics(h.metrics);
    setVoiceLabel(h.voice);
    setRun("ready");
    await engine.load(h.audio);
  };

  const busyLabel = genMode === "stream" ? "بيجهّز أول جملة…" : "بيولّد الصوت…";

  return (
    <>
      <div className="ambient" aria-hidden>
        <i />
        <i />
      </div>
      <div className="grain" aria-hidden />

      <Header info={info} offline={offline} theme={theme} onTheme={toggleTheme} />
      <div className="shell">
        <Hero />

        <motion.div className="layout" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.25 }}>
          <main className="col" ref={editorWrap} style={{ scrollMarginTop: 90 }}>
            <Editor
              ref={editor}
              text={text}
              onText={setText}
              language={language}
              onLanguage={setLanguage}
              maxChars={maxChars}
              examples={language === "en" ? info?.examples_en ?? [FIRST_TEXT_EN] : info?.examples ?? [FIRST_TEXT]}
              onSubmit={generate}
              onToast={push}
            />

            <div className="card gen-bar">
              <Segmented
                label="طريقة التوليد"
                value={genMode}
                onChange={setGenMode}
                options={[
                  { value: "generate", label: "توليد", icon: <Sparkles size={15} /> },
                  { value: "stream", label: "بث مباشر", icon: <Waves size={15} /> },
                ]}
              />
              <span className="grow" />
              <AnimatePresence mode="wait" initial={false}>
                {busy && genMode === "stream" ? (
                  <motion.button key="stop" className="btn" onClick={stop} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
                    <Square size={14} fill="currentColor" />
                    وقّف
                  </motion.button>
                ) : null}
              </AnimatePresence>
              <motion.button className={`btn-primary${busy ? " busy" : ""}`} onClick={generate} disabled={busy || !info} whileTap={{ scale: 0.98 }}>
                <span className="shine" />
                {busy ? <span className="spinner" /> : genMode === "stream" ? <Waves size={19} /> : <Sparkles size={19} />}
                {busy ? (genMode === "stream" ? "بيبث…" : "بيولّد…") : genMode === "stream" ? "ابدأ البث" : "ولّد الصوت"}
              </motion.button>
            </div>

            <Output
              run={run}
              busyLabel={busyLabel}
              metrics={metrics}
              voiceLabel={voiceLabel}
              history={history}
              currentId={currentId}
              onPick={pick}
            />
          </main>

          <aside className="col side">
            <Voices
              speakers={info?.speakers ?? []}
              mode={voiceMode}
              onMode={setVoiceMode}
              speaker={speaker}
              onSpeaker={setSpeaker}
              clip={clip}
              onClip={setClip}
              refText={refText}
              onRefText={setRefText}
              onToast={push}
            />
            <Settings value={params} onChange={setParams} />
          </aside>
        </motion.div>

        <footer className="footer">
          <span>
            VoiceTut-TTS <span className="ltr">v1{info ? ` · ${info.version}` : ""}</span>
          </span>
          <a href="https://huggingface.co/mohammedaly22/VoiceTut-TTS" target="_blank" rel="noopener">
            الموديل على HuggingFace
          </a>
          <a href="https://github.com/MohammedAly22/VoiceTuT-TTS" target="_blank" rel="noopener">
            GitHub
          </a>
          <a href="https://pypi.org/project/voicetut-tts/" target="_blank" rel="noopener">
            PyPI
          </a>
        </footer>
      </div>

      <div className="toasts" aria-live="polite">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              layout
              className={`toast ${t.kind}`}
              initial={{ opacity: 0, y: 16, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.96 }}
            >
              {t.kind === "error" ? <AlertCircle size={17} /> : t.kind === "ok" ? <CheckCircle2 size={17} /> : <InfoIcon size={17} />}
              <span>{t.text}</span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </>
  );
}
