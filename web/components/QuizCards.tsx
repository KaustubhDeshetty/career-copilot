"use client";

import { useState } from "react";
import type { QaPair } from "@/lib/types";
import { Markdown } from "./Markdown";

export function QuizCards({ pairs }: { pairs: QaPair[] }) {
  const [revealed, setRevealed] = useState<Set<number>>(new Set());

  const toggle = (i: number) =>
    setRevealed((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  return (
    <div className="quiz">
      {pairs.map((qa, i) => (
        <article key={i} className="card quiz-card">
          <span className="eyebrow">Question {i + 1}</span>
          <Markdown>{qa.question}</Markdown>
          <button className="btn-secondary" onClick={() => toggle(i)} aria-expanded={revealed.has(i)}>
            {revealed.has(i) ? "Hide answer" : "Show answer"}
          </button>
          {revealed.has(i) && (
            <div className="answer">
              <Markdown>{qa.answer}</Markdown>
            </div>
          )}
        </article>
      ))}
    </div>
  );
}
