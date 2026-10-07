"""
router.py
Reads the user's request, picks exactly one mode (apply / interview / learn)
via forced tool-calling, and builds the initial PipelineState.

Reuses the SAME planner_agent / researcher_agent / writer_agent / critic_agent
functions from agents.py for every mode -- state["task"] is what those
functions actually read, so setting it here means zero changes are needed
to those four functions to make them mode-aware at the routing layer.
"""

import json
from state import PipelineState
from agents import _create_completion_with_retry

ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "enter_apply_mode",
            "description": (
                "Use when the user wants to apply to a job, tailor a resume "
                "or cover letter, or has pasted a job posting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_posting_text": {
                        "type": "string",
                        "description": "The job posting text, pasted or summarized by the user."
                    }
                },
                "required": ["job_posting_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enter_interview_mode",
            "description": (
                "Use when the user wants mock interview practice or wants "
                "to be quizzed like a DS/AIML interviewer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to be interviewed on, e.g. 'decision trees'."
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                        "description": "Requested difficulty. Default to 'medium' if unspecified."
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enter_learn_mode",
            "description": (
                "Use when the user wants to learn, review, or be taught a "
                "concept from their ML/NLP coursework."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The concept to study, e.g. 'TF-IDF' or 'ROC curves'."
                    }
                },
                "required": ["topic"]
            }
        }
    }
]

ROUTER_SYSTEM_PROMPT = (
    "You are the router for a career copilot with three modes: apply, "
    "interview, and learn. Read the user's request and call exactly ONE "
    "tool with arguments extracted from their message. Never respond with "
    "plain text -- always pick a mode."
)


def route_request(user_request: str, client, model="openai/gpt-oss-120b") -> tuple[str, dict]:
    """Classifies the request into a mode and extracts starting arguments.

    Returns (mode, args). tool_choice="required" guarantees a tool call --
    the router must never fall through to a plain-text answer.

    Note: `client` is kept as a parameter for backward compatibility with
    existing callers (test_router.py, build_initial_state), but is no
    longer used directly -- _create_completion_with_retry uses agents.py's
    own client internally, and adds the same rate-limit retry/fail-fast
    protection every other agent already has.
    """
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_request}
    ]

    # For Interview/Learn, the router's output is tiny (a topic string).
    # For Apply mode, job_posting_text must reproduce the ENTIRE job posting
    # verbatim as a JSON string argument -- a fixed small budget (this used
    # to be a flat 512) truncates mid-JSON on any real-length JD, producing
    # invalid JSON that Groq rejects outright. Scale to the input instead of
    # guessing a single number, so short requests don't over-reserve budget
    # against the daily quota while long JDs still get enough room.
    estimated_tokens = len(user_request.split()) * 2 + 200
    router_max_tokens = max(512, min(estimated_tokens, 8000))

    response = _create_completion_with_retry(
        model=model,
        messages=messages,
        tools=ROUTER_TOOLS,
        tool_choice="required",
        max_completion_tokens=router_max_tokens,
    )
    message = response.choices[0].message

    tool_call = message.tool_calls[0]
    mode = tool_call.function.name.replace("enter_", "").replace("_mode", "")
    args = json.loads(tool_call.function.arguments)

    print(f"[Router] -> mode={mode}, args={args}")
    return mode, args


def build_initial_state(user_request: str, client, user_profile: dict | None = None) -> PipelineState:
    """Builds the starting PipelineState after routing.

    "task" is set to the raw user request -- planner_agent and researcher_agent
    read state["task"] directly and need no changes to work across all three
    modes. Mode-specific fields are stashed alongside for later use by
    mode-aware system prompts (e.g. Critic checking ATS keywords in Apply
    mode vs. scoring a rubric in Interview mode).
    """
    mode, args = route_request(user_request, client)

    state: PipelineState = {
        "task": user_request,
        "mode": mode,
        "user_profile": user_profile or {},
        "revision_count": 0,
    }

    if mode == "apply":
        state["job_posting_text"] = args.get("job_posting_text", "")
    elif mode == "interview":
        state["topic"] = args.get("topic", "")
        state["difficulty"] = args.get("difficulty", "medium")
    elif mode == "learn":
        state["topic"] = args.get("topic", "")

    return state
