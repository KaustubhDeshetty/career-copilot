"""
Plain-loop orchestrator. This is the piece that gets replaced by a
LangGraph StateGraph later - the agent functions it calls stay the same.

Now routes every request through router.py first, loads/saves user_profile
via memory_utils.py, and applies a per-mode revision cap -- Apply/Interview
keep the full 2-round Writer/Critic loop, Learn stays fast with 1 round.
"""

from concurrent.futures import ThreadPoolExecutor
from state import PipelineState
from agents import planner_agent, researcher_agent, writer_agent, critic_agent, client
from router import build_initial_state
from memory_utils import load_user_profile, save_user_profile
from selector import selector_agent

MAX_REVISIONS_BY_MODE = {
    "apply": 2,
    "interview": 2,
    "learn": 1,
}


def _skip_result(state: PipelineState) -> PipelineState:
    """Fill the state for a run the Selector short-circuited.

    Every downstream reader -- app.py, the tracker, the CLI below -- expects
    draft / research / critique / approved / sources to exist. Returning early
    with them missing turns a cost saving into a KeyError somewhere else, so
    the skip populates them with an honest account of why nothing ran.
    """
    detail = state.get("archetype_detail") or {}
    ranking = state.get("archetype_ranking") or []
    ranked = " | ".join(f"{k} {v:.0f}" for k, v in ranking)

    state["research"] = ""
    state["sources"] = []
    state["critique"] = "Not run -- the Selector returned SKIP before the pipeline started."
    state["approved"] = False
    state["revision_count"] = 0
    state["draft"] = (
        "SKIPPED -- this posting looks like the wrong KIND of role, so the full "
        "assessment was not run.\n\n"
        f"Nearest archetype: {state.get('archetype_choice')}\n"
        f"Ranking: {ranked}\n"
        f"Out-of-space score: {detail.get('out_of_space', 0):.0f}"
        f"  (higher than every archetype above, which is what triggered the skip)\n"
        f"Reason: {detail.get('why') or 'not given'}\n"
        f"Hardest asks in the posting: {detail.get('gaps') or 'not given'}\n\n"
        "This is a judgement about job CATEGORY, not about whether you are "
        "qualified. To run the full assessment anyway, pass run_anyway=True "
        "(CLI: add --run-anyway)."
    )
    return state


def run_pipeline(task: str, run_anyway: bool = False) -> PipelineState:
    user_profile = load_user_profile()

    # Router and Planner both only depend on the raw task text -- neither
    # needs the other's output -- so run them concurrently instead of
    # sequentially. Saves one full network round-trip with zero change to
    # either function's own logic; results are merged once both finish.
    with ThreadPoolExecutor(max_workers=2) as executor:
        route_future = executor.submit(build_initial_state, task, client, user_profile=user_profile)
        plan_future = executor.submit(planner_agent, {"task": task})

        state = route_future.result()
        plan_state = plan_future.result()

    state["plan"] = plan_state["plan"]

    # Apply mode only -- a no-op in Interview and Learn. Classifies the posting
    # against the six archetypes plus the out_of_space reject class, and swaps
    # user_profile["resume_text"] to the matching resume variant. Because it
    # sets that one field, the Researcher, Writer and both Critics need no
    # changes at all.
    state = selector_agent(state)

    # A SKIP saves the full ~4.5-minute Apply run. It is never silent and never
    # final: the reasoning is written into the draft and run_anyway overrides
    # it. The parser is built so a truncated or malformed Selector reply can
    # never reach this branch -- only a reply where out_of_space genuinely
    # outscored every archetype can.
    if state.get("archetype_verdict") == "SKIP" and not run_anyway:
        state = _skip_result(state)
        save_user_profile(state["user_profile"])
        return state

    state = researcher_agent(state)

    max_revisions = MAX_REVISIONS_BY_MODE.get(state.get("mode"), 2)

    while True:
        state = writer_agent(state)
        state = critic_agent(state)
        if state["approved"] or state["revision_count"] >= max_revisions:
            break

    save_user_profile(state["user_profile"])
    return state


if __name__ == "__main__":
    import sys

    run_anyway = "--run-anyway" in sys.argv
    sys.argv = [a for a in sys.argv if a != "--run-anyway"]

    if len(sys.argv) > 1 and sys.argv[1].endswith(".txt"):
        # Read a job posting (or any long task text) from a file instead of a
        # quoted CLI argument -- sidesteps PowerShell mangling long, punctuation
        # -heavy pasted text (colons, +, blank lines all risk breaking quoting).
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            jd_text = f.read()
        task = f"Here's a job posting:\n{jd_text}\n\nCan you help me tailor my resume bullets for this role?"
    else:
        task = " ".join(sys.argv[1:]) or "Can you help me review TF-IDF today? I keep forgetting how it works."

    result = run_pipeline(task, run_anyway=run_anyway)

    print("\n=== MODE ===\n", result.get("mode"))
    if result.get("archetype_ranking"):
        print("\n=== ARCHETYPE ===")
        print(" verdict:", result.get("archetype_verdict"),
              "| best:", result.get("archetype_choice"),
              "| out_of_space:", f"{result.get('archetype_out_of_space', 0):.0f}")
        print(" ranking:", " | ".join(f"{k} {v:.0f}"
                                      for k, v in result["archetype_ranking"]))
        detail = result.get("archetype_detail") or {}
        # A near-tie is information, not noise: the ranking is reported to be
        # overruled by a human, and a 5-point gap between the top two is the
        # case where that matters most. The parser already computes this --
        # printing the ranking without it hid the one thing worth arguing with.
        if detail.get("is_tie"):
            a, b = result["archetype_ranking"][0][0], result["archetype_ranking"][1][0]
            print(f" NEAR TIE: {a} vs {b}, {detail.get('gap', 0):.0f} points apart"
                  " -- your call, not the model's")
            if detail.get("tie"):
                print(f"   {detail['tie']}")
        if detail.get("gaps"):
            print(" gaps:", detail["gaps"])
        if not detail.get("complete", True):
            print(" !! Selector reply was incomplete -- verdict forced away from SKIP")
    print("\n=== PLAN ===\n", result["plan"])
    print("\n=== SOURCES USED ===\n", result["sources"])
    print("\n=== RESEARCH ===\n", result["research"])
    header = "=== FINAL DRAFT ===" if result.get("approved") else (
        f"=== BEST DRAFT (NOT APPROVED after {result.get('revision_count', 0)} "
        f"round(s) -- see the Critic verdict below) ===")
    print("\n" + header + "\n", result["draft"])
    print("\n=== CRITIC VERDICT (round", result["revision_count"], ") ===\n", result["critique"])
