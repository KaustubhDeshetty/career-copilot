"""
api.py -- FastAPI backend for the Career Copilot web front end (web/).

Wraps the existing pipeline without changing it. orchestrator.py, agents.py,
router.py and selector.py are untouched; the Streamlit app keeps working.

Run:
    uvicorn api:app --reload --port 8000

Endpoints
    GET  /api/health                      liveness + whether GROQ_API_KEY is set
    POST /api/run                         run the pipeline, return the result as JSON
    POST /api/run/stream                  same, but streams agent progress as Server-Sent Events
    GET  /api/tracker                     both tracker sheets (applications + interview practice)
    POST /api/tracker/applications        save an Apply-mode result (409 if already saved)
    POST /api/tracker/interviews          save an Interview-mode session (409 if already saved)

How live progress works without touching the agents
    orchestrator.py imports each agent function into its own namespace, so
    run_pipeline looks them up there at call time. While a streamed run is in
    progress we swap those names for thin wrappers that emit "stage started /
    finished" events, then restore the originals. stdout is also teed so the
    existing [Router] / [Researcher tool call] prints appear in the UI.

    Both tricks are process-global, so runs are serialised with a lock. That is
    the right trade here anyway: on Groq's free tier (8,000 TPM) two concurrent
    Apply runs would just rate-limit each other.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import orchestrator
from tracker_utils import (
    application_exists,
    extract_company_role,
    extract_fit_summary,
    interview_session_exists,
    load_applications,
    load_interview_sessions,
    save_application,
    save_interview_session,
)

MAX_TASK_CHARS = 20_000

app = FastAPI(title="Career Copilot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_run_lock = threading.Lock()


# ---------------------------------------------------------------- schemas

class RunRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=MAX_TASK_CHARS)
    run_anyway: bool = False


class SaveApplicationRequest(BaseModel):
    company: str = Field(..., max_length=200)
    role: str = Field(..., max_length=200)
    fit_score: str = Field("Unknown", max_length=100)
    recommendation: str = Field("Unknown", max_length=2000)
    approved: bool = False
    revision_count: int = 0


class SaveInterviewRequest(BaseModel):
    topic: str = Field(..., max_length=300)
    difficulty: str = Field("medium", max_length=20)
    num_questions: int = 0
    approved: bool = False
    revision_count: int = 0


# ---------------------------------------------------------------- helpers

def _parse_qa_pairs(draft: str) -> list[dict[str, str]]:
    """Same delimiter format the Streamlit app parses (===Q=== / ===A=== / ===END===)."""
    pattern = r"===Q===\s*(.*?)\s*===A===\s*(.*?)\s*===END==="
    return [{"question": q.strip(), "answer": a.strip()}
            for q, a in re.findall(pattern, draft or "", re.DOTALL)]


def _serialise(state: dict[str, Any]) -> dict[str, Any]:
    """Turn PipelineState into the JSON contract the front end expects
    (mirrored by `PipelineResult` in web/lib/types.ts)."""
    mode = state.get("mode")
    draft = state.get("draft", "") or ""
    out: dict[str, Any] = {
        "mode": mode,
        "task": state.get("task", ""),
        "plan": state.get("plan", ""),
        "research": state.get("research", ""),
        "sources": list(state.get("sources") or []),
        "draft": draft,
        "critique": state.get("critique", ""),
        "approved": bool(state.get("approved")),
        "revision_count": int(state.get("revision_count") or 0),
        "topic": state.get("topic"),
        "difficulty": state.get("difficulty"),
        "archetype": None,
        "fit": None,
        "qa_pairs": [],
    }
    if state.get("archetype_ranking"):
        detail = state.get("archetype_detail") or {}
        out["archetype"] = {
            "verdict": state.get("archetype_verdict"),
            "best": state.get("archetype_choice"),
            "out_of_space": float(state.get("archetype_out_of_space") or 0),
            "ranking": [[k, float(v)] for k, v in state["archetype_ranking"]],
            "is_tie": bool(detail.get("is_tie")),
            "why": detail.get("why"),
            "gaps": detail.get("gaps"),
        }
    if mode == "apply":
        fit_score, recommendation = extract_fit_summary(draft)
        match = re.search(r"(\d+)\s+of\s+(\d+)", fit_score or "")
        role, company = extract_company_role(state.get("job_posting_text", "") or "")
        out["fit"] = {
            "fit_score": fit_score,
            "recommendation": recommendation,
            "met": int(match.group(1)) if match else None,
            "total": int(match.group(2)) if match else None,
            "company": company,
            "role": role,
        }
    if mode == "interview":
        out["qa_pairs"] = _parse_qa_pairs(draft)
    return out


class _Tee(io.TextIOBase):
    """Writes to the real stdout and forwards complete lines to a callback."""

    def __init__(self, original, on_line: Callable[[str], None]):
        self._original = original
        self._on_line = on_line
        self._buffer = ""

    def write(self, s: str) -> int:
        self._original.write(s)
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._on_line(line.rstrip())
        return len(s)

    def flush(self) -> None:
        self._original.flush()


# name in orchestrator's namespace -> stage id shown in the UI
_STAGES = {
    "build_initial_state": "router",
    "planner_agent": "planner",
    "selector_agent": "selector",
    "researcher_agent": "researcher",
    "writer_agent": "writer",
    "critic_agent": "critic",
}


@contextmanager
def _instrumented(emit: Callable[[dict], None]) -> Iterator[None]:
    """Temporarily wrap the agent functions orchestrator.run_pipeline calls."""
    originals = {name: getattr(orchestrator, name) for name in _STAGES}
    rounds = {"writer": 0, "critic": 0}

    def wrap(name: str, fn: Callable) -> Callable:
        stage = _STAGES[name]

        def wrapper(*args, **kwargs):
            if stage in rounds:
                rounds[stage] += 1
            rnd = rounds.get(stage)
            emit({"type": "stage", "stage": stage, "status": "start", "round": rnd})
            started = time.time()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                emit({"type": "stage", "stage": stage, "status": "error", "round": rnd,
                      "detail": str(exc)})
                raise
            detail = None
            if isinstance(result, dict):
                if stage == "router":
                    detail = f"mode = {result.get('mode')}"
                elif stage == "selector":
                    detail = (f"{result.get('archetype_verdict')}: {result.get('archetype_choice')}"
                              if result.get("archetype_choice") else "not needed for this mode")
                elif stage == "researcher":
                    detail = f"{len(result.get('sources') or [])} source(s)"
                elif stage == "critic":
                    detail = "approved" if result.get("approved") else "revision requested"
            emit({"type": "stage", "stage": stage, "status": "done", "round": rnd,
                  "seconds": round(time.time() - started, 1), "detail": detail})
            return result

        return wrapper

    for name, fn in originals.items():
        setattr(orchestrator, name, wrap(name, fn))
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(orchestrator, name, fn)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


# ---------------------------------------------------------------- routes

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "groq_key_set": bool(os.environ.get("GROQ_API_KEY", "").strip()),
            "busy": _run_lock.locked()}


@app.post("/api/run")
def run(req: RunRequest) -> dict[str, Any]:
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "A run is already in progress. Try again when it finishes.")
    try:
        state = orchestrator.run_pipeline(req.task, run_anyway=req.run_anyway)
        return _serialise(state)
    finally:
        _run_lock.release()


@app.post("/api/run/stream")
def run_stream(req: RunRequest) -> StreamingResponse:
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "A run is already in progress. Try again when it finishes.")

    events: "queue.Queue[dict | None]" = queue.Queue()

    def worker() -> None:
        original_stdout = sys.stdout
        try:
            sys.stdout = _Tee(original_stdout, lambda line: events.put({"type": "log", "line": line}))
            with _instrumented(events.put):
                state = orchestrator.run_pipeline(req.task, run_anyway=req.run_anyway)
            events.put({"type": "result", "data": _serialise(state)})
        except Exception as exc:  # surface the failure to the UI instead of a hung stream
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            sys.stdout = original_stdout
            _run_lock.release()
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream() -> Iterator[str]:
        yield _sse({"type": "started"})
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"  # comment line; stops proxies closing an idle stream
                continue
            if event is None:
                break
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/tracker")
def tracker() -> dict[str, Any]:
    app_headers, app_rows = load_applications()
    int_headers, int_rows = load_interview_sessions()
    clean = lambda rows: [[("" if v is None else str(v)) for v in row] for row in rows]
    return {
        "applications": {"headers": app_headers, "rows": clean(app_rows)},
        "interviews": {"headers": int_headers, "rows": clean(int_rows)},
    }


@app.post("/api/tracker/applications", status_code=201)
def save_app(req: SaveApplicationRequest) -> dict[str, Any]:
    if application_exists(req.company, req.role):
        raise HTTPException(409, "Already saved to the tracker today.")
    notes = "" if req.approved else (
        f"UNVERIFIED -- Critic still had objections after {req.revision_count} revision round(s)")
    try:
        save_application(req.company, req.role, req.fit_score, req.recommendation, notes=notes)
    except PermissionError:
        raise HTTPException(423, "applications.xlsx is open in Excel. Close it and try again.")
    return {"saved": True}


@app.post("/api/tracker/interviews", status_code=201)
def save_interview(req: SaveInterviewRequest) -> dict[str, Any]:
    if interview_session_exists(req.topic, req.difficulty):
        raise HTTPException(409, "Already saved to the practice log today.")
    verdict = "Approved" if req.approved else "Revise"
    try:
        save_interview_session(req.topic, req.difficulty, req.num_questions, verdict, req.revision_count)
    except PermissionError:
        raise HTTPException(423, "applications.xlsx is open in Excel. Close it and try again.")
    return {"saved": True}
