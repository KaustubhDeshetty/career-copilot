import type { ArchetypeResult, FitSummary } from "@/lib/types";

function verdictClass(met: number | null, total: number | null): string {
  if (!met || !total) return "pill-partial";
  const ratio = met / total;
  if (ratio >= 0.75) return "pill-strong";
  if (ratio >= 0.4) return "pill-partial";
  return "pill-weak";
}

function shortVerdict(recommendation: string): string {
  const m = recommendation.match(/\s*((?:strong|partial|weak|good|poor)[a-z ]*match)/i);
  if (m) return m[1].trim().replace(/\b\w/g, (c) => c.toUpperCase());
  return recommendation.length > 41 ? `${recommendation.slice(0, 38)}…` : recommendation;
}

const ARCHETYPE_NAMES: Record<string, string> = {
  genai: "Generative AI",
  ml_engineer: "ML Engineer",
  mlops: "MLOps",
  cloud_ai: "Cloud AI",
  data_platform: "Data & ML Platform",
  architect: "AI/ML Architect",
};

export function FitPanel({ fit, archetype }: { fit: FitSummary; archetype: ArchetypeResult | null }) {
  const top = archetype?.ranking.slice(0, 3) ?? [];
  const maxScore = Math.max(1, ...top.map(([, v]) => v));

  return (
    <section className="fit-grid">
      <div className="card stat">
        <span className="muted small">Requirements met</span>
        <span className="stat-value">
          {fit.met !== null && fit.total !== null ? `${fit.met} / ${fit.total}` : "N/A"}
        </span>
        <span className="muted small">
          {fit.role} · {fit.company}
        </span>
      </div>

      <div className="card">
        {fit.recommendation && fit.recommendation !== "Unknown" ? (
          <>
            <span className={`pill ${verdictClass(fit.met, fit.total)}`}>
              {shortVerdict(fit.recommendation)}
            </span>
            {fit.recommendation !== shortVerdict(fit.recommendation) && (
              <p className="small">{fit.recommendation}</p>
            )}
          </>
        ) : (
          <>
            <span className="muted small">Recommendation</span>
            <p className="small">No one-line recommendation detected. See the full assessment below.</p>
          </>
        )}
      </div>

      {archetype && top.length > 0 && (
        <div className="card">
          <span className="muted small">
            Role archetype{archetype.is_tie ? " · near tie, your call" : ""}
          </span>
          <ul className="bars">
            {top.map(([key, score]) => (
              <li key={key}>
                <span>{ARCHETYPE_NAMES[key] ?? key}</span>
                <span className="bar">
                  <span style={{ width: `${(score / maxScore) * 100}%` }} />
                </span>
                <span className="muted small">{Math.round(score)}</span>
              </li>
            ))}
          </ul>
          {archetype.gaps && <p className="small muted">Hardest asks: {archetype.gaps}</p>}
        </div>
      )}
    </section>
  );
}
