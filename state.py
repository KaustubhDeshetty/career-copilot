"""
Shared state passed between every agent in the pipeline.

Using a single typed dict (rather than each agent having its own
input/output signature) is deliberate: it's exactly the shape LangGraph
expects for its StateGraph, so migrating later means wrapping these same
functions as nodes - not redesigning them.

Extended for the Career Copilot: mode/user_profile route between the
three pipelines (apply / interview / learn), which all reuse the same
planner_agent / researcher_agent / writer_agent / critic_agent functions
from agents.py -- only their system prompts and MAX_REVISIONS differ.
"""

from typing import TypedDict, List


class PipelineState(TypedDict, total=False):
    task: str              # the original user request
    plan: str               # Planner's breakdown of the task
    research: str            # Researcher's RAG-grounded findings
    sources: List[str]        # titles of documents used for research
    draft: str               # Writer's current draft
    critique: str             # Critic's most recent feedback
    approved: bool            # whether the Critic approved the draft
    revision_count: int        # how many Writer <-> Critic loops have run

    # --- Career Copilot additions ---
    mode: str               # "apply" | "interview" | "learn"
    user_profile: dict       # persistent facts about the user, carried across modes

    # Mode-specific fields (only the relevant one(s) populated per run)
    job_posting_text: str   # Apply mode: the pasted job posting
    topic: str                # Interview/Learn mode: what to quiz/teach on
    difficulty: str            # Interview mode: "easy" | "medium" | "hard"
