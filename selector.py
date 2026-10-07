"""
Archetype Selector -- Apply mode, stage 1.

Classifies a job posting against the six role archetypes in archetypes.json
BEFORE the expensive Fit Assessment runs, and picks which flavoured resume to
send. See PHASE2_DESIGN.md.

Deliberately a separate module rather than a fifth function in agents.py:
agents.py holds the debugged core whose md5 is the project's integrity anchor
(PROJECT_STATE.md), and this is new, unproven code. Merging it in later is a
copy-paste; un-merging it after it breaks something is not. The function
signature is the same state-in/state-out shape as every other agent, so it
drops into the pipeline -- and into a future LangGraph node -- unchanged.

Two lessons from the bug log are load-bearing here:

  Bug 3  -- an agent cannot tell which source a fact came from unless the
            prompt teaches it what the section labels MEAN. The rubrics
            describe the JOB, the JD describes the JOB, and the resume is not
            in this call at all. Said explicitly in the prompt.

  Bug 10 -- a reply that runs out of room before its verdict must fail closed.
            Here "closed" is inverted relative to the Critic: the dangerous
            action is SKIPPING a job, so an unparseable reply must never
            produce SKIP. It degrades to STRETCH and the full run proceeds.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHETYPES_PATH = os.path.join(HERE, "archetypes.json")
RESUMES_PATH = os.path.join(HERE, "resumes.json")

VERDICT_MARKER = "VERDICT:"
VALID_VERDICTS = ("APPLY", "STRETCH", "SKIP")
TIE_THRESHOLD = 5          # points; below this the top two are reported as a tie
SKIP_CEILING = 60          # a top score at or above this can never be SKIP


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

SELECTOR_SYSTEM_PROMPT = (
    "OUTPUT CONTRACT -- read this first and obey it exactly:\n"
    "Your reply MUST begin with the literal line 'VERDICT: APPLY' or "
    "'VERDICT: STRETCH' or 'VERDICT: SKIP'. That line comes FIRST, before "
    "any other text. Then, in this order and nothing else:\n"
    "  BEST: <one archetype key>\n"
    "  RANKING: <key> <score> | ... all six archetypes AND out_of_space, "
    "highest first\n"
    "  TIE: <key> vs <key> -- <the question that decides it, one line>\n"
    "  GAPS: <short comma-separated list, or 'none apparent'>\n"
    "  WHY: <one sentence, max 25 words>\n"
    "Omit the TIE line entirely unless the top two scores are within 5 points. "
    "Do NOT show your working. Do NOT explain the rubrics back to me. Do NOT "
    "add headings, preamble, or a closing summary. Seven scores, one line.\n\n"

    "You are classifying a JOB POSTING against six role archetypes -- the six "
    "kinds of job this candidate targets -- plus a seventh option, "
    "out_of_space, for postings that are none of them. You are NOT assessing "
    "whether the candidate is qualified; a later stage does that, and you will "
    "not see a resume. Your only question is: what KIND of role is this?\n\n"

    "The rubrics below describe the JOB SIDE. 'use_when' is what the posting "
    "must be about for that archetype to fit. 'avoid_when' rules it out. "
    "'signals' are vocabulary that tends to appear -- weak evidence, not a "
    "checklist, and a posting that means the same thing in different words "
    "counts just as much. 'vs' names the archetype most often confused with "
    "this one and the question that separates them; when two archetypes both "
    "fit, answer that question and let it decide.\n\n"

    "AVOID_WHEN IS A CAP, NOT A DISCOUNT. If a posting matches an "
    "archetype's avoid_when, that archetype CANNOT score above 40 -- however "
    "much of its use_when and signals also match. Apply the cap first, then "
    "score what is left.\n\n"

    "SCORE THE WORK DESCRIBED, NOT THE JOB TITLE. Titles are unreliable: a "
    "posting titled 'Machine Learning Engineer' that never trains a model on "
    "the company's own data is not an ml_engineer role; a posting titled "
    "'AI Architect' whose body is hands-on building is not an architect role; "
    "a posting titled 'Data Scientist' over dashboards and SQL reporting is "
    "out_of_space. Where title and body disagree, the body wins, every time."
    "\n\n"

    "SCORING SCALE -- absolute, not relative. Do NOT make the scores sum to "
    "100, and do NOT spread them out to look decisive:\n"
    "  80-100  the posting is squarely this\n"
    "  60-79   strong match; this is most of the job\n"
    "  45-59   partial; real overlap but not the centre of the role\n"
    "  20-44   incidental; mentioned or implied, not the work\n"
    "  0-19    essentially absent\n"
    "Several archetypes scoring high at once is a legitimate result -- it "
    "means a broad role, usually a small company where one person does "
    "everything. That is NOT a no-match, and out_of_space should be LOW "
    "there.\n\n"

    "OUT_OF_SPACE competes on the same scale as the six. Score it by its own "
    "rubric, honestly, and do not deflate it to be helpful: a demanding "
    "robotics or quant or analytics role that shares vocabulary with the six "
    "is still a different discipline, and saying so is the single most useful "
    "thing you can do here.\n\n"

    "VERDICT:\n"
    "  SKIP     out_of_space is the HIGHEST score -- wrong kind of job\n"
    "  APPLY    an archetype leads and its score is 60 or above\n"
    "  STRETCH  an archetype leads with a score of 45-59, or the posting is "
    "too thin to classify\n"
    "SKIP means 'the wrong KIND of job', never 'this looks hard' and never "
    "'I am unsure'. When a posting is too vague to classify -- a few generic "
    "lines with no substance -- score everything in the 30s and 40s, keep "
    "out_of_space low, and emit STRETCH: there is nothing there to rule the "
    "job out with. NEVER emit SKIP while an archetype outscores "
    "out_of_space.\n\n"

    "BEST is always one of the six archetypes, never out_of_space -- on a "
    "SKIP it names the nearest miss.\n\n"

    "GAPS: name the requirements in the posting that look like the hardest "
    "asks, in a few words each. If nothing stands out, write 'none apparent'. "
    "Do not invent gaps to seem thorough."
)


def _render_rubrics(arch):
    """Render the six rubrics for the prompt.

    `summary` is deliberately left out: it describes the candidate's resume
    flavour, not the kind of job, and this call is only about the job. Skipping
    it saves roughly 500 tokens per request and removes the one field that
    could tempt the model into assessing the candidate.
    """
    out = []
    for key, r in arch["archetypes"].items():
        lines = [
            f"[{key}]  {r['label']}",
            f"  use_when:   {r['use_when']}",
            f"  avoid_when: {r['avoid_when']}",
            f"  signals:    {', '.join(r['signals'])}",
        ]
        for v in r.get("vs", []):
            lines.append(f"  vs {v['archetype']}: {v['question']}")
        out.append("\n".join(lines))

    rc = arch.get("reject_class")
    if rc:
        out.append("\n".join([
            f"[{rc['key']}]  {rc['label']}   <- the reject option",
            f"  use_when:   {rc['use_when']}",
            f"  avoid_when: {rc['avoid_when']}",
            f"  signals:    {', '.join(rc['signals'])}",
        ]))
    return "\n\n".join(out)


def reject_key(arch):
    return (arch.get("reject_class") or {}).get("key", "out_of_space")


def build_selector_messages(jd_text, arch):
    keys = ", ".join(list(arch["archetypes"]) + [reject_key(arch)])
    user = (
        f"ARCHETYPE RUBRICS (these describe kinds of job):\n\n"
        f"{_render_rubrics(arch)}\n\n"
        f"{'=' * 60}\n"
        f"JOB POSTING (classify this):\n\n{jd_text.strip()}\n"
        f"{'=' * 60}\n\n"
        f"Score all seven of: {keys}. Verdict line first."
    )
    return SELECTOR_SYSTEM_PROMPT, user


# --------------------------------------------------------------------------
# parsing -- fails closed, and "closed" here means "do not skip"
# --------------------------------------------------------------------------

def _has_verdict(text):
    return bool(text) and VERDICT_MARKER in text.upper()


def parse_selector_output(text, keys, reject="out_of_space"):
    """Parse the Selector's reply into a dict.

    Returns keys: verdict, best, scores, ranking, tie, gaps, why, complete.

    `complete` is False when the reply did not carry a verdict line or did not
    score all six archetypes -- i.e. it was truncated or ignored the contract.
    An incomplete reply is NEVER allowed to produce SKIP: the whole point of
    the short-circuit is to save four and a half minutes, and doing that on a
    reply we could not read is exactly the silent door-closing PHASE2_DESIGN.md
    warns against. It degrades to STRETCH and the full run proceeds.
    """
    result = {
        "verdict": "STRETCH", "best": None, "scores": {k: 0.0 for k in keys},
        "ranking": [], "tie": None, "gaps": "", "why": "", "complete": False,
        "retried": False, "out_of_space": 0.0,
        "raw": text or "",
    }
    # No early return on empty text: every caller reads result["ranking"] and
    # result["is_tie"], and an early exit that skipped the tail of this
    # function left those keys missing -- turning a bad model reply into a
    # KeyError several frames away. Empty text falls through and produces an
    # all-zero, incomplete, non-SKIP result, which is the honest answer.
    if not text:
        result["why"] = "empty reply from the model"
    text = text or ""
    upper = text.upper()

    # scores: any "<key> <number>" pair anywhere in the reply, so a stray
    # markdown table or a reordered RANKING line still parses.
    found = {}
    for k in list(keys) + [reject]:
        m = re.search(rf"\b{re.escape(k)}\b\D{{0,4}}(\d{{1,3}})", text, re.IGNORECASE)
        if m:
            found[k] = min(float(m.group(1)), 100.0)
    # `scores` stays the six real archetypes so callers can rank them; the
    # reject class is not an archetype and must never appear in a ranking or
    # be selected as `best`. It is a verdict input, nothing else.
    result["scores"].update({k: v for k, v in found.items() if k in keys})
    result["out_of_space"] = found.get(reject, 0.0)

    for label, field in (("TIE:", "tie"), ("GAPS:", "gaps"), ("WHY:", "why")):
        m = re.search(rf"{label}\s*(.+)", text, re.IGNORECASE)
        if m:
            result[field] = m.group(1).strip()

    m = re.search(r"BEST:\s*([a-z_]+)", text, re.IGNORECASE)
    if m and m.group(1).lower() in keys:
        result["best"] = m.group(1).lower()

    result["ranking"] = sorted(
        keys, key=lambda k: (-result["scores"][k], keys.index(k))
    )
    if not result["best"]:
        result["best"] = result["ranking"][0]

    result["complete"] = _has_verdict(text) and len(found) == len(keys) + 1

    if result["complete"]:
        section = upper.split(VERDICT_MARKER)[1][:40]
        stated = next((v for v in VALID_VERDICTS if v in section), None)
        lead = result["ranking"][0]
        top = result["scores"][lead]
        oos = result["out_of_space"]

        if oos > top:
            # The reject class outscored every archetype. This is the ONE path
            # where SKIP is authoritative no matter how well an archetype did
            # -- and it is the whole reason the reject class exists. On the
            # first sweep a quantitative-research posting scored ml_engineer 85
            # and came back APPLY, because a forced choice among six had no way
            # to say "none of these". Now it does.
            result["verdict"] = "SKIP"
            if stated != "SKIP":
                result["why"] = (f"[out_of_space {oos:.0f} beat {lead} {top:.0f}; "
                                 f"forced to SKIP] " + result["why"])
        elif stated == "SKIP":
            # An archetype outscored the reject class, so SKIP contradicts the
            # model's own numbers. Trust the numbers: a false SKIP closes a
            # door on a job and is the expensive mistake.
            result["verdict"] = "APPLY" if top >= SKIP_CEILING else "STRETCH"
            result["why"] = (f"model said SKIP but {lead} {top:.0f} beat "
                             f"out_of_space {oos:.0f}; overridden. " + result["why"])
        elif stated:
            result["verdict"] = stated
    else:
        missing = [k for k in list(keys) + [reject] if k not in found]
        result["why"] = (
            "incomplete reply (no verdict line)" if not _has_verdict(text)
            else f"incomplete reply (unscored: {', '.join(missing)})"
        ) + " -- not treated as SKIP"

    # Tie detection is ours, not the model's -- it is arithmetic, and the model
    # forgetting the TIE line should not hide a real tie.
    a, b = result["ranking"][0], result["ranking"][1]
    gap = result["scores"][a] - result["scores"][b]
    result["is_tie"] = gap <= TIE_THRESHOLD
    result["gap"] = gap
    return result


# --------------------------------------------------------------------------
# scorer entry point used by test_selector.py
# --------------------------------------------------------------------------

SELECTOR_MAX_TOKENS = 2500       # was 1200; see the note below
SELECTOR_RETRY_EFFORT = "low"


def score_jd(jd_text, arch, call_llm=None,
             max_completion_tokens=SELECTOR_MAX_TOKENS, allow_retry=True):
    """Classify one posting. Returns the full parsed dict.

    Retries ONCE at reasoning_effort="low" when the first reply comes back
    incomplete.

    Why: gpt-oss-120b does hidden chain-of-thought before its visible answer,
    and that reasoning is charged against the same max_completion_tokens
    budget. On the hardest posting in the test set -- the founding-generalist
    role where every archetype fires at once -- the model spent the entire
    1,200-token budget deliberating and emitted nothing at all. Zero scores,
    finish_reason='length'. Same family as Bug 10: not a wrong answer, an
    answer that never got out.

    Raising the ceiling alone is the obvious fix and the wrong one on its own,
    because deliberation expands to fill whatever it is given. Dropping
    reasoning_effort for every call is also wrong -- the near-tie cases are
    exactly where the deliberation earns its keep, and #14 was called
    correctly at medium effort when the keyword baseline got it wrong.

    So: give it real room at medium effort, and if it still talks itself out
    of an answer, ask again in a mode that cannot. The retry costs a second
    call only on postings that failed, which in the smoke test was one in four.
    """
    if call_llm is None:
        from agents import _call_llm as call_llm
    system, user = build_selector_messages(jd_text, arch)
    keys = list(arch["archetypes"])
    rej = reject_key(arch)

    reply = call_llm(system, user, max_completion_tokens=max_completion_tokens)
    result = parse_selector_output(reply, keys, rej)

    if result["complete"] or not allow_retry:
        return result

    retry = call_llm(system, user,
                     max_completion_tokens=max_completion_tokens,
                     reasoning_effort=SELECTOR_RETRY_EFFORT)
    retried = parse_selector_output(retry, keys, rej)
    retried["retried"] = True
    if retried["complete"]:
        retried["why"] = f"[recovered on low-effort retry] {retried['why']}"
        return retried

    # Both attempts failed. Return the retry (still non-SKIP by construction)
    # and say so, rather than pretending the first attempt was better.
    retried["why"] = ("incomplete on BOTH medium and low reasoning effort -- "
                      "not treated as SKIP. " + retried["why"])
    return retried


# --------------------------------------------------------------------------
# the agent
# --------------------------------------------------------------------------

def load_archetypes(path=ARCHETYPES_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_resumes(path=RESUMES_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["resumes"]


def selector_agent(state):
    """Apply mode only. Sets the archetype fields and, crucially, swaps
    user_profile['resume_text'] to the chosen flavour -- which is the whole
    reason the Researcher, Writer and both Critics need no changes."""
    if state.get("mode") != "apply":
        return state

    jd = state.get("job_posting_text") or state.get("task", "")
    if not jd.strip():
        return state

    arch = load_archetypes()
    result = score_jd(jd, arch)

    state["archetype_ranking"] = [(k, result["scores"][k]) for k in result["ranking"]]
    state["archetype_choice"] = result["best"]
    state["archetype_verdict"] = result["verdict"]
    state["archetype_out_of_space"] = result["out_of_space"]
    state["archetype_detail"] = result

    try:
        resumes = load_resumes()
        chosen = resumes.get(result["best"])
        if chosen and chosen.get("text"):
            profile = state.setdefault("user_profile", {})
            profile["resume_text"] = chosen["text"]
            profile["resume_flavour"] = result["best"]
    except FileNotFoundError:
        # No resumes.json -- keep whatever resume_text the profile already had
        # rather than failing the run. The classification is still useful.
        pass

    return state


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "jd.txt"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    r = score_jd(text, load_archetypes())
    print(f"\nVERDICT: {r['verdict']}   BEST: {r['best']}"
          f"{'   (TIE, gap %.0f)' % r['gap'] if r['is_tie'] else ''}")
    print("RANKING: " + " | ".join(f"{k} {r['scores'][k]:.0f}" for k in r["ranking"])
          + f"   || out_of_space {r['out_of_space']:.0f}")
    if r["tie"]:
        print(f"TIE:  {r['tie']}")
    print(f"GAPS: {r['gaps']}\nWHY:  {r['why']}")
    if not r["complete"]:
        print("\n!! incomplete reply -- verdict forced away from SKIP")
