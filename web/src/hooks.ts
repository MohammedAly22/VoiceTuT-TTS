import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { engine } from "./audio";

export type Theme = "light" | "dark";

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(
    () => (document.documentElement.getAttribute("data-theme") as Theme) || "dark",
  );
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#07080d" : "#f3f4f8");
    try {
      localStorage.setItem("voicetut-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);
  return [theme, toggle];
}

export function useEngine() {
  return useSyncExternalStore(engine.subscribe, engine.getState);
}

/** Calls `fn` on every animation frame while `active`. */
export function useFrame(fn: (t: number) => void, active = true) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!active) return;
    let id = 0;
    const loop = (t: number) => {
      ref.current(t);
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [active]);
}

export interface Toast {
  id: number;
  kind: "error" | "ok" | "info";
  text: string;
}

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((kind: Toast["kind"], text: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-2), { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 6000 : 3200);
  }, []);
  return { toasts, push };
}

/** A single shared <audio> for previewing speaker reference clips. */
const preview = typeof Audio !== "undefined" ? new Audio() : null;
const previewListeners = new Set<() => void>();
let previewId: string | null = null;

function setPreview(id: string | null) {
  previewId = id;
  previewListeners.forEach((l) => l());
}

preview?.addEventListener("ended", () => setPreview(null));
preview?.addEventListener("pause", () => setPreview(null));

export function usePreview(): [string | null, (id: string, url: string) => void] {
  const id = useSyncExternalStore(
    (l) => {
      previewListeners.add(l);
      return () => previewListeners.delete(l);
    },
    () => previewId,
  );
  const toggle = useCallback((nextId: string, url: string) => {
    if (!preview) return;
    if (previewId === nextId) {
      preview.pause();
      return;
    }
    engine.pause();
    preview.src = url;
    void preview.play().then(
      () => setPreview(nextId),
      () => setPreview(null),
    );
  }, []);
  return [id, toggle];
}

export function stopPreview() {
  preview?.pause();
}
