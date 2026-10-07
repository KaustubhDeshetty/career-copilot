"use client";

import type { Mode, StageId } from "@/lib/types";

export interface StageState {
  key: string; // e.g. "writer-2"
  stage: StageId;
  round: number | null;
  status: "start" | "done" | "error";
  seconds?: number;
  detail?: string | null;
}

const LABELS: Record<StageId, { name: string; blurb: string }> = {
  router: { name: "Router", blurb: "forced tool call picks the mode" },
  planner: { name: "Planner", blurb: "breaks the task down (runs in parallel with the Router)" },
  selector: { name: "Selector", blurb: "matches the posting to a role archetype" },
  researcher: { name: "Researcher", blurb: "semantic + keyword retrieval" },
  writer: { name: "Writer", blurb: "drafts the answer" },
  critic: { name: "Critic", blurb: "verifies every claim, fails closed" },
};

export function AgentTimeline({
  stages,
  mode,
  running,
  elapsed,
}: {
  stages: StageState[];
  mode: Mode | null;
  running: boolean;
  elapsed: number;
}) {
  const visible = stages.filter(
    (s) => !(s.stage === "selector" && mode && mode !== "apply"),
  );

  return (
    <section className="card timeline" aria-live="polite">
      <header className="timeline-head">
        <h2>Agent pipeline</h2>
        <span className="muted">
          {running ? "running" : "finished"} · {formatSeconds(elapsed)}
        </span>
      </header>
      <ol>
        {visible.map((s) => (
          <li key={s.key} className={`step step-${s.status}`}>
            <span className="dot" aria-hidden />
            <div>
              <strong>
                {LABELS[s.stage].name}
                {s.round ? ` · round ${s.round}` : ""}
              </strong>
              <span className="muted"> {s.detail || LABELS[s.stage].blurb}</span>
            </div>
            <span className="muted step-time">
              {s.status === "start" ? "working…" : s.seconds !== undefined ? `${s.seconds}s` : ""}
            </span>
          </li>
        ))}
        {running && visible.length === 0 && (
          <li className="step step-start">
            <span className="dot" aria-hidden />
            <div>
              <strong>Starting</strong>
              <span className="muted"> loading models and indexes…</span>
            </div>
          </li>
        )}
      </ol>
      {mode === "apply" && running && (
        <p className="muted small">
          Apply mode runs two Writer ⇄ Critic rounds over a full resume and posting, which usually
          takes about 4–5 minutes on Groq&apos;s free tier.
        </p>
      )}
    </section>
  );
}

function formatSeconds(total: number): string {
  const m = Math.floor(total / 60);
  const s = Math.floor(total % 60);
  return m ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
}
