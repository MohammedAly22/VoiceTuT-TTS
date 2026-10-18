import { Client, handle_file } from "@gradio/client";

export interface Speaker {
  speaker_id: string;
  speaker_name: string;
  arabic_name: string;
  gender: "male" | "female" | string;
  tags: string[];
  reference_text: string;
  audio_url: string;
}

export interface Info {
  name: string;
  version: string;
  model_version: string;
  device: string;
  mock: boolean;
  sampling_rate: number;
  max_chars: number;
  speakers: Speaker[];
  examples: string[];
  examples_en?: string[];
}

export interface Metrics {
  total: number;
  audio: number;
  rtf: number | null;
  ttfa: number | null;
  chunks: number;
  vram_gb?: number | null;
  files?: string[]; // stream only: server paths of every chunk generated so far
}

export type Language = "arz" | "en";
export type VoiceMode = "speaker" | "clone";

export interface SynthRequest {
  text: string;
  language: Language;
  voiceMode: VoiceMode;
  speaker: string | null;
  refAudio: Blob | null;
  refText: string;
  numStep: number;
  guidance: number;
  speed: number;
  normalize: boolean;
}

interface FileOut {
  url?: string;
  path: string;
}

// ZeroGPU: let the huggingface.co page attach the visitor's login to GPU requests, so logged-in
// (e.g. PRO) users spend their own quota instead of the small anonymous one. Same handshake as
// Gradio's own frontend: announce support to the parent page, which answers with the same
// message; @gradio/client then fetches the auth headers from the parent for every queued call.
const ZEROGPU_HANDSHAKE = "supports-zerogpu-headers";
if (typeof window !== "undefined" && window.parent !== window) {
  window.addEventListener("message", (e) => {
    if (e.data === ZEROGPU_HANDSHAKE) (window as unknown as { supports_zerogpu_headers: boolean }).supports_zerogpu_headers = true;
  });
  const host = window.location.hostname;
  if (host.endsWith(".hf.space") || host.includes(".dev.")) {
    const origin = host.includes(".dev.") ? `https://moon-${host.split(".")[1]}.dev.spaces.huggingface.tech` : "https://huggingface.co";
    window.parent.postMessage(ZEROGPU_HANDSHAKE, origin);
  }
}

// Everything is resolved relative to the page, so the app works at "/" and under sub-paths.
const pageBase = new URL(".", window.location.href);
// the Gradio app is served on the same origin/path as the page
const API_ROOT = pageBase.href.replace(/\/$/, "");
export const assetUrl = (p: string) => new URL(p, pageBase).href;

/** URL of a file the server wrote (its output dir is in Gradio's allowed paths). */
const fileUrl = (path: string) => `${API_ROOT}/gradio_api/file=${encodeURI(path)}`;

let clientPromise: Promise<Client> | null = null;

function client(): Promise<Client> {
  if (!clientPromise) {
    clientPromise = Client.connect(API_ROOT, { events: ["data", "status"] }).catch((e) => {
      clientPromise = null;
      throw e;
    });
  }
  return clientPromise;
}

function errorText(e: unknown): string {
  if (!e) return "حصل خطأ غير متوقع";
  if (typeof e === "string") return e;
  const m = (e as { message?: unknown }).message;
  if (typeof m === "string" && /ZeroGPU quota/i.test(m)) {
    return "خلصت حصة الـ GPU المجانية. سجّل دخولك على HuggingFace وافتح الـ Space من صفحته على huggingface.co عشان تستخدم حصتك.";
  }
  if (typeof m === "string" && m) return m;
  return "حصل خطأ غير متوقع";
}

export class ApiError extends Error {}

function payload(r: SynthRequest): unknown[] {
  return [
    r.text,
    r.language,
    r.voiceMode,
    r.voiceMode === "speaker" ? r.speaker ?? "" : "",
    // a File (not a bare Blob): @gradio/client checks `instanceof Buffer` before Blob, which throws in browsers
    r.voiceMode === "clone" && r.refAudio ? handle_file(new File([r.refAudio], "reference.wav", { type: "audio/wav" })) : null,
    r.voiceMode === "clone" ? r.refText : "",
    r.numStep,
    r.guidance,
    r.speed,
    r.normalize,
  ];
}

async function fetchFile(f: FileOut): Promise<ArrayBuffer> {
  const res = await fetch(f.url ?? fileUrl(f.path));
  if (!res.ok) throw new ApiError(`تعذّر تحميل الصوت (${res.status})`);
  return res.arrayBuffer();
}

export async function getInfo(): Promise<Info> {
  try {
    const c = await client();
    const r = await c.predict("/info", []);
    return (r.data as unknown[])[0] as Info;
  } catch (e) {
    throw new ApiError(errorText(e));
  }
}

export async function normalizeText(text: string, language: Language): Promise<string> {
  try {
    const c = await client();
    const r = await c.predict("/normalize", [text, language]);
    return String((r.data as unknown[])[0] ?? "");
  } catch (e) {
    throw new ApiError(errorText(e));
  }
}

export async function synthesize(req: SynthRequest): Promise<{ audio: ArrayBuffer; metrics: Metrics }> {
  let out: unknown[];
  try {
    const c = await client();
    out = (await c.predict("/synthesize", payload(req))).data as unknown[];
  } catch (e) {
    throw new ApiError(errorText(e));
  }
  const audio = await fetchFile(out[0] as FileOut);
  return { audio, metrics: out[1] as Metrics };
}

/** Yields every audio chunk, in order, as the server finishes it.
 *
 * @gradio/client applies streamed diffs to ONE shared object, so by the time a queued
 * message is read it may already show a later chunk. Each message therefore carries the
 * cumulative `metrics.files` list, and we play whatever we haven't played yet, in order. */
export async function* streamSynth(
  req: SynthRequest,
  signal: AbortSignal,
): AsyncGenerator<{ audio: ArrayBuffer; metrics: Metrics }> {
  const c = await client().catch((e) => {
    throw new ApiError(errorText(e));
  });
  const job = c.submit("/stream", payload(req));
  const it = job[Symbol.asyncIterator]();
  let onAbort = () => {};
  const aborted = new Promise<IteratorReturnResult<undefined>>((resolve) => {
    onAbort = () => {
      void job.cancel();
      resolve({ done: true, value: undefined });
    };
  });
  signal.addEventListener("abort", onAbort, { once: true });
  let played = 0;
  try {
    while (!signal.aborted) {
      const step = await Promise.race([it.next(), aborted]);
      if (step.done) return;
      const msg = step.value;
      if (msg.type === "data") {
        const [file, metrics] = msg.data as [FileOut | null, Metrics];
        const paths = metrics?.files ?? (file ? [file.path] : []);
        while (played < paths.length && !signal.aborted) {
          const path = paths[played++];
          yield { audio: await fetchFile({ path }), metrics: { ...metrics, files: undefined } };
        }
      } else if (msg.type === "status") {
        if (msg.stage === "error") throw new ApiError(errorText(msg));
        if (msg.stage === "complete") return;
      }
    }
  } finally {
    signal.removeEventListener("abort", onAbort);
  }
}
