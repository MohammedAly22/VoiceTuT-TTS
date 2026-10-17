// Audio helpers + a small playback engine shared by the whole UI.
//
// One-shot clips play through an <audio> element (seekable, downloadable). Streaming
// chunks are scheduled back-to-back on the Web Audio clock for gapless playback, and
// when the stream ends they are stitched into a single WAV that the <audio> element
// takes over. Both paths feed one AnalyserNode so the waveform can react to loudness.

let ctx: AudioContext | null = null;

export function audioContext(): AudioContext {
  if (!ctx) ctx = new AudioContext();
  if (ctx.state === "suspended") void ctx.resume();
  return ctx;
}

export async function decode(buf: ArrayBuffer): Promise<AudioBuffer> {
  return audioContext().decodeAudioData(buf.slice(0));
}

export function mono(b: AudioBuffer): Float32Array {
  if (b.numberOfChannels === 1) return b.getChannelData(0).slice();
  const out = new Float32Array(b.length);
  for (let c = 0; c < b.numberOfChannels; c++) {
    const d = b.getChannelData(c);
    for (let i = 0; i < d.length; i++) out[i] += d[i] / b.numberOfChannels;
  }
  return out;
}

/** Max-abs peaks, normalised to 0..1, `bins` values long. */
export function computePeaks(samples: Float32Array, bins = 1200): Float32Array {
  const out = new Float32Array(bins);
  if (!samples.length) return out;
  const step = samples.length / bins;
  let max = 0;
  for (let i = 0; i < bins; i++) {
    const s = Math.floor(i * step);
    const e = Math.min(samples.length, Math.floor((i + 1) * step) || s + 1);
    let m = 0;
    for (let j = s; j < e; j++) {
      const v = Math.abs(samples[j]);
      if (v > m) m = v;
    }
    out[i] = m;
    if (m > max) max = m;
  }
  if (max > 0) for (let i = 0; i < bins; i++) out[i] = Math.pow(out[i] / max, 0.8);
  return out;
}

export function concat(parts: Float32Array[]): Float32Array {
  const n = parts.reduce((a, p) => a + p.length, 0);
  const out = new Float32Array(n);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

export function encodeWav(samples: Float32Array, sr: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/** Any browser-decodable audio (upload or mic recording) -> mono 24 kHz WAV, trimmed. */
export async function toReferenceWav(
  blob: Blob,
  maxSeconds = 20,
  sr = 24000,
): Promise<{ wav: Blob; duration: number; peaks: Float32Array }> {
  const decoded = await decode(await blob.arrayBuffer());
  const length = Math.min(decoded.duration, maxSeconds);
  const off = new OfflineAudioContext(1, Math.ceil(length * sr), sr);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start(0, 0, length);
  const rendered = await off.startRendering();
  const data = rendered.getChannelData(0);
  return { wav: encodeWav(data, sr), duration: length, peaks: computePeaks(data, 160) };
}

export function formatTime(s: number): string {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, "0")}`;
}

// --------------------------------------------------------------------------- engine
export type EngineMode = "empty" | "loading" | "streaming" | "ready";

export interface EngineState {
  mode: EngineMode;
  playing: boolean;
  duration: number;
  url: string | null;
  peaksVersion: number;
}

type Listener = () => void;

class AudioEngine {
  readonly el: HTMLAudioElement;
  peaks: Float32Array = new Float32Array(0);
  revealAt = 0; // ms timestamp of the last "fresh waveform" (drives the grow-in animation)

  private analyser: AnalyserNode | null = null;
  private wired = false;
  private levels = new Uint8Array(256);
  private listeners = new Set<Listener>();
  private _state: EngineState = { mode: "empty", playing: false, duration: 0, url: null, peaksVersion: 0 };

  // streaming
  private streamSources: AudioBufferSourceNode[] = [];
  private streamParts: Float32Array[] = [];
  private streamStart = 0;
  private streamEnd = 0;
  private streamSr = 24000;
  private streamDone = false;

  constructor() {
    this.el = new Audio();
    this.el.preload = "auto";
    const sync = () => this.set({ playing: !this.el.paused && !this.el.ended });
    this.el.addEventListener("play", sync);
    this.el.addEventListener("pause", sync);
    this.el.addEventListener("ended", sync);
    this.el.addEventListener("loadedmetadata", () => {
      if (this._state.mode === "ready" && isFinite(this.el.duration)) this.set({ duration: this.el.duration });
    });
  }

  // ---- store plumbing (for useSyncExternalStore)
  subscribe = (l: Listener) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };
  getState = () => this._state;
  private set(p: Partial<EngineState>) {
    this._state = { ...this._state, ...p };
    this.listeners.forEach((l) => l());
  }

  private wire() {
    if (this.wired) return this.analyser!;
    const c = audioContext();
    this.analyser = c.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.78;
    this.analyser.connect(c.destination);
    c.createMediaElementSource(this.el).connect(this.analyser);
    this.wired = true;
    return this.analyser;
  }

  /** 0..1 loudness of what is playing right now. */
  level(): number {
    if (!this.analyser || !this._state.playing) return 0;
    this.analyser.getByteFrequencyData(this.levels);
    let sum = 0;
    for (let i = 2; i < 96; i++) sum += this.levels[i];
    return Math.min(1, sum / (94 * 180));
  }

  currentTime(): number {
    if (this._state.mode === "streaming") {
      const t = audioContext().currentTime - this.streamStart;
      return Math.max(0, Math.min(t, this.streamEnd - this.streamStart));
    }
    return this.el.currentTime || 0;
  }

  progress(): number {
    const d = this._state.duration;
    return d > 0 ? Math.min(1, this.currentTime() / d) : 0;
  }

  setLoading() {
    this.stopAll();
    this.set({ mode: "loading", playing: false });
  }

  reset() {
    this.stopAll();
    this.peaks = new Float32Array(0);
    this.set({ mode: "empty", playing: false, duration: 0, peaksVersion: this._state.peaksVersion + 1 });
  }

  /** Load a finished clip (WAV bytes). Autoplays by default. */
  async load(buf: ArrayBuffer, { autoplay = true } = {}) {
    this.wire();
    this.stopAll();
    const decoded = await decode(buf);
    const samples = mono(decoded);
    this.peaks = computePeaks(samples);
    this.revealAt = performance.now();
    this.setClip(new Blob([buf], { type: "audio/wav" }), decoded.duration);
    if (autoplay) void this.play();
  }

  private setClip(blob: Blob, duration: number) {
    if (this._state.url) URL.revokeObjectURL(this._state.url);
    const url = URL.createObjectURL(blob);
    this.el.src = url;
    this.el.currentTime = 0;
    this.set({ mode: "ready", url, duration, playing: false, peaksVersion: this._state.peaksVersion + 1 });
  }

  async play() {
    if (this._state.mode !== "ready") return;
    this.wire();
    audioContext();
    if (this.el.ended) this.el.currentTime = 0;
    try {
      await this.el.play();
    } catch {
      /* autoplay blocked: user can press play */
    }
  }

  pause() {
    this.el.pause();
  }

  toggle() {
    if (this._state.playing) this.pause();
    else void this.play();
  }

  seek(frac: number) {
    if (this._state.mode !== "ready") return;
    this.el.currentTime = Math.max(0, Math.min(1, frac)) * (this._state.duration || 0);
  }

  // ---- streaming
  beginStream() {
    this.wire();
    this.stopAll();
    this.streamParts = [];
    this.streamSources = [];
    this.streamDone = false;
    this.peaks = new Float32Array(0);
    this.revealAt = performance.now();
    const c = audioContext();
    this.streamStart = c.currentTime + 0.08;
    this.streamEnd = this.streamStart;
    this.set({ mode: "streaming", playing: true, duration: 0, peaksVersion: this._state.peaksVersion + 1 });
  }

  async pushChunk(buf: ArrayBuffer) {
    if (this._state.mode !== "streaming") return;
    const c = audioContext();
    const decoded = await decode(buf);
    if (this._state.mode !== "streaming") return;
    this.streamSr = decoded.sampleRate;
    const src = c.createBufferSource();
    src.buffer = decoded;
    src.connect(this.analyser!);
    const at = Math.max(this.streamEnd, c.currentTime + 0.03);
    if (!this.streamParts.length) this.streamStart = at;
    else if (at > this.streamEnd) {
      // network stall: insert the silence we actually waited for, so the stitched file matches
      this.streamParts.push(new Float32Array(Math.round((at - this.streamEnd) * this.streamSr)));
    }
    src.start(at);
    this.streamEnd = at + decoded.duration;
    src.onended = () => this.maybeFinishStream();
    this.streamSources.push(src);
    this.streamParts.push(mono(decoded));
    this.peaks = computePeaks(concat(this.streamParts));
    this.set({ duration: this.streamEnd - this.streamStart, peaksVersion: this._state.peaksVersion + 1 });
  }

  /** Server finished (or was cancelled). Playback continues until scheduled audio ends. */
  endStream({ cancelled = false } = {}) {
    if (this._state.mode !== "streaming") return;
    this.streamDone = true;
    if (cancelled || !this.streamParts.length) {
      this.streamSources.forEach((s) => {
        try {
          s.stop();
        } catch {
          /* already stopped */
        }
      });
    }
    this.maybeFinishStream(cancelled);
  }

  private maybeFinishStream(force = false) {
    if (this._state.mode !== "streaming" || !this.streamDone) return;
    const c = audioContext();
    if (!force && c.currentTime < this.streamEnd - 0.05) return;
    this.streamSources = [];
    if (!this.streamParts.length) {
      this.reset();
      return;
    }
    const all = concat(this.streamParts);
    this.setClip(encodeWav(all, this.streamSr), all.length / this.streamSr);
  }

  private stopAll() {
    this.el.pause();
    this.streamSources.forEach((s) => {
      s.onended = null;
      try {
        s.stop();
      } catch {
        /* not started */
      }
    });
    this.streamSources = [];
  }
}

export const engine = new AudioEngine();
