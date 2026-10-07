// Shared contract between the Python backend (api.py -> _serialise) and this UI.
// Mirrors PipelineState in state.py; if you add a field there, add it here too.

export type Mode = "apply" | "interview" | "learn";

export type StageId = "router" | "planner" | "selector" | "researcher" | "writer" | "critic";

export interface QaPair {
  question: string;
  answer: string;
}

export interface FitSummary {
  fit_score: string;
  recommendation: string;
  met: number | null;
  total: number | null;
  company: string;
  role: string;
}

export interface ArchetypeResult {
  verdict: string | null;
  best: string | null;
  out_of_space: number;
  ranking: [string, number][];
  is_tie: boolean;
  why: string | null;
  gaps: string | null;
}

export interface PipelineResult {
  mode: Mode | null;
  task: string;
  plan: string;
  research: string;
  sources: string[];
  draft: string;
  critique: string;
  approved: boolean;
  revision_count: number;
  topic: string | null;
  difficulty: string | null;
  archetype: ArchetypeResult | null;
  fit: FitSummary | null;
  qa_pairs: QaPair[];
}

// Server-Sent Events emitted by POST /api/run/stream
export type StreamEvent =
  | { type: "started" }
  | {
      type: "stage";
      stage: StageId;
      status: "start" | "done" | "error";
      round: number | null;
      seconds?: number;
      detail?: string | null;
    }
  | { type: "log"; line: string }
  | { type: "result"; data: PipelineResult }
  | { type: "error"; message: string };

export interface TrackerSheet {
  headers: string[];
  rows: string[][];
}

export interface TrackerData {
  applications: TrackerSheet;
  interviews: TrackerSheet;
}
