"use client";

import { useEffect, useRef, useState } from "react";
import { AgentTimeline, type StageState } from "@/components/AgentTimeline";
import { ResultView } from "@/components/ResultView";
import { readEvents } from "@/lib/sse";
import type { Mode, PipelineResult } from "@/lib/types";
import { MAX_TASK_CHARS } from "@/lib/limits";

const EXAMPLES: { label: string; text: string }[] = [
  {
    label: "Apply",
    text:
      "Here's a job posting:\nGenerative AI Engineer\nAcme Corp\n\nRequirements:\n- 2-3 years of Python\n- Experience with LLMs, RAG and AI agents\n- Familiarity with React and REST APIs\n\nCan you check my fit and tailor my resume bullets?",
  },
  { label: "Interview", text: "Quiz me on precision and recall, hard difficulty." },
  { label: "Learn", text: "Help me review TF-IDF. I keep forgetting how it works." },
];

export default function Home() {
  const [task, setTask] = useState("");
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState<StageState[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [mode, setMode] = useState<Mode | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef(0);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setElapsed((Date.now() - startedAt.current) / 1000), 500);
    return () => clearInterval(id);
  }, [running]);

  async function run(runAnyway = false) {
    setRunning(true);
    setStages([]);
    setLogs([]);
    setMode(null);
    setResult(null);
    setError(null);
    setElapsed(0);
    startedAt.current = Date.now();

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task, runAnyway }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error((data as { error?: string }).error ?? `Request failed (${res.status}).`);
      }

      for await (const event of readEvents(res)) {
        if (event.type === "stage") {
          const key = `${event.stage}-${event.round ?? 0}`;
          setStages((prev) => {
            const next = prev.filter((s) => s.key !== key);
            const existingIndex = prev.findIndex((s) => s.key === key);
            const entry: StageState = {
              key,
              stage: event.stage,
              round: event.round,
              status: event.status,
              seconds: event.seconds,
              detail: event.detail,
            };
            if (existingIndex === -1) return [...prev, entry];
            next.splice(existingIndex, 0, entry);
            return next;
          });
          if (event.stage === "router" && event.status === "done" && event.detail?.startsWith("mode = ")) {
            setMode(event.detail.slice(7) as Mode);
          }
        } else if (event.type === "log") {
          setLogs((prev) => [...prev.slice(-199), event.line]);
        } else if (event.type === "result") {
          setResult(event.data);
          setMode(event.data.mode);
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setElapsed((Date.now() - startedAt.current) / 1000);
      setRunning(false);
    }
  }

  const skipped = result?.archetype?.verdict === "SKIP";

  return (
    <div className="stack">
      <section className="hero">
        <h1>One assistant, three modes</h1>
        <p className="muted">
          Paste a job posting to check your fit, ask to be quizzed, or ask to review a concept from
          your notes. The Router decides which mode to run.
        </p>
      </section>

      <section className="card">
        <label htmlFor="task" className="sr-only">
          Your request
        </label>
        <textarea
          id="task"
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="e.g. Quiz me on precision and recall, hard"
          rows={task.length > 200 ? 12 : 4}
          maxLength={MAX_TASK_CHARS}
          disabled={running}
        />
        <div className="row spread">
          <div className="row">
            {EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                className="chip"
                onClick={() => setTask(ex.text)}
                disabled={running}
                type="button"
              >
                Try {ex.label}
              </button>
            ))}
          </div>
          <div className="row">
            <span className="muted small">
              {task.length.toLocaleString()} / {MAX_TASK_CHARS.toLocaleString()}
            </span>
            <button className="btn" onClick={() => run()} disabled={running || task.trim().length < 3}>
              {running ? "Running…" : "Run"}
            </button>
          </div>
        </div>
      </section>

      {error && (
        <section className="card error-card" role="alert">
          <strong>Something went wrong.</strong> {error}
        </section>
      )}

      {(running || stages.length > 0) && (
        <AgentTimeline stages={stages} mode={mode} running={running} elapsed={elapsed} />
      )}

      {skipped && !running && (
        <section className="card warn-card">
          The Selector thinks this posting is the wrong <em>kind</em> of role, so it skipped the
          full 4–5 minute assessment. That&apos;s a judgement about job category, not about whether you
          are qualified.{" "}
          <button className="btn-secondary" onClick={() => run(true)}>
            Run the full assessment anyway
          </button>
        </section>
      )}

      {result && <ResultView result={result} />}

      {logs.length > 0 && (
        <details className="card">
          <summary>Backend log ({logs.length} lines)</summary>
          <pre className="log">{logs.join("\n")}</pre>
        </details>
      )}
    </div>
  );
}
