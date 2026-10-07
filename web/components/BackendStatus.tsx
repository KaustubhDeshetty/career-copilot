"use client";

import { useEffect, useState } from "react";

type Status = "checking" | "ready" | "no-key" | "down";

/** Small indicator in the nav: is the Python backend up, and is GROQ_API_KEY set? */
export function BackendStatus() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        const data = (await res.json()) as { ok?: boolean; groq_key_set?: boolean };
        if (!cancelled) setStatus(!res.ok ? "down" : data.groq_key_set ? "ready" : "no-key");
      } catch {
        if (!cancelled) setStatus("down");
      }
    };
    check();
    const id = setInterval(check, 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const text: Record<Status, string> = {
    checking: "Checking backend…",
    ready: "Backend ready",
    "no-key": "GROQ_API_KEY not set",
    down: "Backend offline",
  };

  return (
    <span className={`status status-${status}`} title="Python FastAPI backend (api.py)">
      <span className="dot" aria-hidden /> {text[status]}
    </span>
  );
}
