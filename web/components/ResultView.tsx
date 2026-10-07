"use client";

import { useState } from "react";
import type { PipelineResult } from "@/lib/types";
import { FitPanel } from "./FitPanel";
import { Markdown } from "./Markdown";
import { QuizCards } from "./QuizCards";

const MODE_LABEL = { apply: "Apply", interview: "Interview", learn: "Learn" } as const;
type Tab = "plan" | "research" | "critique" | "sources";

export function ResultView({ result }: { result: PipelineResult }) {
  const [tab, setTab] = useState<Tab>("critique");

  return (
    <div className="result">
      <div className="row">
        {result.mode && <span className="pill pill-mode">{MODE_LABEL[result.mode]} mode</span>}
        <span className={`pill ${result.approved ? "pill-strong" : "pill-partial"}`}>
          {result.approved
            ? `Critic approved · ${result.revision_count} round(s)`
            : `Not approved after ${result.revision_count} round(s) · showing best draft`}
        </span>
      </div>

      {result.mode === "apply" && result.fit && <FitPanel fit={result.fit} archetype={result.archetype} />}

      {result.mode === "interview" && result.qa_pairs.length > 0 ? (
        <QuizCards pairs={result.qa_pairs} />
      ) : (
        <section className="card">
          <Markdown>{result.draft || "_No draft produced: the Writer returned nothing._"}</Markdown>
        </section>
      )}

      <SaveToTracker result={result} />

      <section className="card">
        <div className="tabs" role="tablist">
          {(["critique", "plan", "research", "sources"] as Tab[]).map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              className={tab === t ? "tab tab-active" : "tab"}
              onClick={() => setTab(t)}
            >
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
        <div className="tab-body">
          {tab === "critique" && <Markdown>{result.critique || "_No critique._"}</Markdown>}
          {tab === "plan" && <Markdown>{result.plan || "_No plan._"}</Markdown>}
          {tab === "research" && <pre className="research">{result.research || "No research text."}</pre>}
          {tab === "sources" &&
            (result.sources.length ? (
              <ul>
                {result.sources.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            ) : (
              <p className="muted">No sources: Apply mode reads the posting and your resume directly.</p>
            ))}
        </div>
      </section>
    </div>
  );
}

function SaveToTracker({ result }: { result: PipelineResult }) {
  const [state, setState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [message, setMessage] = useState("");

  if (result.mode !== "apply" && result.mode !== "interview") return null;

  const label =
    result.mode === "apply"
      ? `${result.fit?.role ?? "Unknown role"} at ${result.fit?.company ?? "Unknown company"}`
      : `${result.topic ?? "Unknown topic"} · ${result.difficulty ?? "medium"} · ${result.qa_pairs.length} question(s)`;

  async function save() {
    setState("saving");
    const url = result.mode === "apply" ? "/api/tracker/applications" : "/api/tracker/interviews";
    const body =
      result.mode === "apply"
        ? {
            company: result.fit?.company ?? "Unknown company",
            role: result.fit?.role ?? "Unknown role",
            fit_score: result.fit?.fit_score ?? "Unknown",
            recommendation: result.fit?.recommendation ?? "Unknown",
            approved: result.approved,
            revision_count: result.revision_count,
          }
        : {
            topic: result.topic ?? "Unknown topic",
            difficulty: ["easy", "medium", "hard"].includes(result.difficulty ?? "")
              ? result.difficulty
              : "medium",
            num_questions: result.qa_pairs.length,
            approved: result.approved,
            revision_count: result.revision_count,
          };
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.ok || res.status === 409) {
      setState("saved");
      setMessage(res.ok ? "Saved to the tracker." : "Already saved to the tracker today.");
    } else {
      const data = await res.json().catch(() => ({}));
      setState("error");
      setMessage((data as { error?: string }).error ?? "Could not save.");
    }
  }

  return (
    <section className="card row save">
      <span>
        <strong>Detected:</strong> {label}
      </span>
      {state === "saved" ? (
        <span className="muted">{message}</span>
      ) : (
        <button className="btn-secondary" onClick={save} disabled={state === "saving"}>
          {state === "saving" ? "Saving…" : result.mode === "apply" ? "Save to application tracker" : "Save to practice log"}
        </button>
      )}
      {state === "error" && <span className="error small">{message}</span>}
    </section>
  );
}
