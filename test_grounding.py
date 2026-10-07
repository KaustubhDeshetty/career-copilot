"""
test_grounding.py
Adversarial regression tests for the three Apply-mode bugs that were found and
fixed by hand. Each case is a resume/posting pair engineered to tempt one
specific failure -- if a future prompt tweak reintroduces the bug, the matching
case starts failing instead of the regression going unnoticed until someone
happens to re-read a draft line by line.

    Bug 1  Fabrication    the Writer invented metrics the resume never contained
    Bug 2  Grounding      the Critic was never shown the research it was
                          supposed to verify claims against
    Bug 3  Attribution    the posting's stated minimum ("5+ years") was written
                          up as the candidate's own tenure (really "11+ years"),
                          and a company-wide platform stat from the posting's
                          About section was presented as a personal achievement

WHAT THIS COSTS. Each case makes 3 Groq calls (1 Writer + Apply mode's 2
parallel Critic calls). Router and Planner are deliberately skipped -- their
output is not what is under test, and skipping them saves 2 calls per case.
The fixtures are also deliberately short (a compact synthetic resume, not the
real 4,500-token one), which keeps each case comfortably under the 8,000 TPM
per-request ceiling and leaves the daily quota mostly intact.

    python test_grounding.py                # all cases, one run each
    python test_grounding.py --case fabrication
    python test_grounding.py --runs 3       # repeat: these are LLM calls, and
                                            # one clean pass is weak evidence

A NOTE ON WHAT A PASS MEANS. These assertions are deterministic, but what they
inspect is not: the model can phrase the same correct answer many ways, and a
single green run is evidence, not proof. Treat a failure as a real signal worth
reading; treat one pass as "no regression seen this run", and use --runs when
it matters.
"""

import argparse
import os
import re
import sys

# --- fixtures ---------------------------------------------------------------
# Kept short on purpose: every token here is charged three times per case.

# No percentages, no multipliers, no throughput figures anywhere. Any such
# number appearing in a draft built from this resume was invented outright.
RESUME_NO_NUMBERS = """\
Jordan Avery
Senior AI/ML Engineer

Summary
11+ years of experience building and deploying machine learning systems.

Experience
Senior AI/ML Engineer, Northwind Data (2019 - present)
- Built retrieval-augmented generation systems over enterprise document sets.
- Fine-tuned open-weight language models using LoRA and QLoRA.
- Deployed inference services with FastAPI, packaged in Docker and run on
  Kubernetes.
- Reduced model memory footprint through quantization and distillation.
- Wrote performance-critical components in Rust.

ML Engineer, Corvid Analytics (2014 - 2019)
- Trained gradient-boosted models for tabular prediction tasks.
- Built ETL pipelines in Python and PySpark.
"""

POSTING_WITH_TRAPS = """\
Senior Machine Learning Engineer - LLM Platform
Vantage Systems

About Vantage Systems: our platform serves 50 million inference requests daily
for enterprise customers, and we have cut serving costs by 40% over the last
two years.

Required Qualifications:
- 5+ years of professional experience deploying ML systems in production
- Hands-on PyTorch experience, including LoRA or QLoRA fine-tuning
- Experience with Docker and Kubernetes
- Experience building model-serving APIs with FastAPI
- Track record of mentoring junior engineers and leading technical initiatives
"""


def _sentences_about_candidate(draft: str) -> list:
    """Lines that read as claims about the candidate, excluding the ones that
    are quoting the posting's requirements back.

    Without this, a perfectly correct draft fails: a Fit Assessment table row
    legitimately says "5+ years production ML | MET", quoting the employer's
    ask. What matters is whether the candidate is *described* as having 5
    years, not whether the string appears anywhere at all."""
    quoting = re.compile(
        r"require|qualification|posting|employer|they (?:ask|want|seek)|"
        r"job description|\bJD\b|asks for|looking for",
        re.IGNORECASE,
    )
    kept = []
    for line in draft.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Fit Assessment table rows restate the employer's requirements with a
        # MET/NOT MET status -- "| 5+ years production ML | MET |" is the draft
        # doing its job, not claiming the candidate has five years.
        if stripped.startswith("|"):
            continue
        if quoting.search(stripped):
            continue
        kept.append(line)
    return kept


# --- checks -----------------------------------------------------------------
# Each returns (passed, message).

def check_no_invented_statistics(draft: str, **_):
    """Bug 1. The fixture resume contains no percentage, no multiplier, and no
    throughput figure -- so any of those in the draft came from nowhere.

    Numbers inside a [quantify: ...] placeholder are exempt: that placeholder
    is the prescribed correct behaviour, not a fabrication."""
    without_placeholders = re.sub(r"\[quantify:[^\]]*\]", "", draft, flags=re.IGNORECASE)

    invented = []
    invented += re.findall(r"\b\d+(?:\.\d+)?\s?%", without_placeholders)
    invented += re.findall(r"\b\d+(?:\.\d+)?\s?[x×]\b", without_placeholders)
    invented += re.findall(r"\b\d+\s?k?\s?(?:RPS|QPS|req/s)\b", without_placeholders,
                           flags=re.IGNORECASE)

    # The posting's own figures are allowed if the draft is quoting the company,
    # which check_no_company_stat_stolen tests separately and more precisely.
    invented = [n for n in invented if n.strip() not in {"50", "40%", "40 %"}]

    if invented:
        return False, f"invented statistics in draft: {sorted(set(invented))}"
    return True, "no invented percentages, multipliers, or throughput figures"


ACTIVITY_CLAIM = re.compile(
    r"\bmentor(?:ed|ing|s)?\b|\bcoach(?:ed|ing)?\b|\bjunior engineer|"
    r"\bled the team\b|\bmanaged a team\b|\bteam of \d+",
    re.IGNORECASE,
)


def check_no_unevidenced_activity(draft: str, **_):
    """Bug 16. The fixture resume contains no mentoring, no leadership and no
    team of any size; the posting asks for all of it. Any bullet describing the
    candidate as having mentored anyone was invented to match the ask.

    This is Bug 3's family -- a posting requirement re-attributed to the
    candidate -- but it slipped through Bug 3's fix because every anti-invention
    rule in the Writer prompt named a category the mentoring claim was not in:
    'metrics, percentages, numbers', then 'technologies, tools, hardware,
    frameworks, technical methods'. Mentoring is an activity, so the rules
    were followed and the sentence still got written.

    A [not evidenced: ...] placeholder is the prescribed correct behaviour and
    is exempt, exactly as [quantify: ...] is in check_no_invented_statistics."""
    without_placeholders = re.sub(r"\[not evidenced:[^\]]*\]", "", draft,
                                  flags=re.IGNORECASE)

    denial = re.compile(r"not met|no evidence|does not|doesn't|lacks|absent|gap",
                        re.IGNORECASE)
    offenders = []
    for line in _sentences_about_candidate(without_placeholders):
        if denial.search(line):
            continue
        if ACTIVITY_CLAIM.search(line):
            offenders.append(line.strip())

    if offenders:
        return False, ("claims mentoring/leadership the resume never shows: "
                       + " | ".join(o[:90] for o in offenders[:2]))
    return True, "no unevidenced mentoring or leadership claim"


# "11+ years", "11 + years", "11 plus years" -- the model spaces the plus
# however it likes, and markdown renderers encourage the spaced form. The
# original patterns allowed whitespace only AFTER the plus, which made this
# check flaky in both directions: it failed a correct draft that wrote
# "11 + years", and it would have PASSED a real Bug 3 violation written as
# "5 + years". The second half is the dangerous one -- a grounding test that
# silently misses the violation it exists to catch is worse than no test.
_YEARS = r"{n}\s*(?:\+|plus)?\s*years"


def check_tenure_not_taken_from_posting(draft: str, **_):
    """Bug 3a. Resume says 11+ years; posting requires 5+. The draft must
    describe the candidate with the resume's number."""
    if not re.search(_YEARS.format(n="11"), draft, re.IGNORECASE):
        return False, "draft never states the candidate's real tenure (11+ years)"

    for line in _sentences_about_candidate(draft):
        if re.search(r"\b" + _YEARS.format(n="5"), line, re.IGNORECASE):
            return False, f"posting's 5+ year minimum presented as candidate tenure: {line.strip()!r}"
    return True, "candidate tenure stated as 11+ years, not the posting's 5+ minimum"


def check_no_company_stat_stolen(draft: str, **_):
    """Bug 3b. '50 million requests daily' and the '40% cost reduction' belong to
    the employer's About section. Neither may show up as the candidate's work."""
    stolen = []
    for line in _sentences_about_candidate(draft):
        if re.search(r"50\s*(?:million|m\b)", line, re.IGNORECASE):
            stolen.append(line.strip())
        if re.search(r"\b40\s?%", line):
            stolen.append(line.strip())
    if stolen:
        return False, f"employer's own statistics attributed to the candidate: {stolen}"
    return True, "employer's platform statistics not claimed by the candidate"


def check_critic_saw_the_research(critique: str, **_):
    """Bug 2. The Critic used to receive only the task and the draft, and said
    so out loud -- 'No external research text was provided.' If that sentence
    ever comes back, the research has been unplugged from its prompt again."""
    blind = re.search(
        r"no (?:external )?(?:research|source|reference)(?: text| material)?"
        r"[^.]{0,40}(?:was )?(?:provided|supplied|given|available)",
        critique, re.IGNORECASE,
    )
    if blind:
        return False, f"Critic reports it was given no research: {blind.group(0)!r}"
    if not critique.strip():
        return False, "Critic returned nothing at all"
    return True, "Critic did not report missing research"


def check_critic_reached_a_verdict(critique: str, **_):
    """Not one of the three bugs, but the failure that masks all of them: a
    truncated Critic call produces no verdict, and critic_agent flags that
    inline. If this fires, the token budgets need raising again."""
    if "[INCOMPLETE" in critique:
        return False, "a Critic call was cut off before reaching a verdict (raise its token budget)"
    return True, "both Critic calls reached a verdict"


CASES = {
    "fabrication": {
        "why": "Bug 1 -- resume has no metrics; the Writer must not invent any",
        "resume": RESUME_NO_NUMBERS,
        "posting": POSTING_WITH_TRAPS,
        "checks": [check_no_invented_statistics, check_critic_reached_a_verdict],
    },
    "grounding": {
        "why": "Bug 2 -- the Critic must actually receive the research",
        "resume": RESUME_NO_NUMBERS,
        "posting": POSTING_WITH_TRAPS,
        "checks": [check_critic_saw_the_research, check_critic_reached_a_verdict],
    },
    "activity": {
        "why": "Bug 16 -- posting asks for mentoring; the resume has none, so "
               "the Writer must omit it or mark it [not evidenced: ...]",
        "resume": RESUME_NO_NUMBERS,
        "posting": POSTING_WITH_TRAPS,
        "checks": [check_no_unevidenced_activity, check_critic_reached_a_verdict],
    },
    "attribution": {
        "why": "Bug 3 -- posting's minimum and the employer's stats are not candidate facts",
        "resume": RESUME_NO_NUMBERS,
        "posting": POSTING_WITH_TRAPS,
        "checks": [check_tenure_not_taken_from_posting, check_no_company_stat_stolen,
                   check_critic_reached_a_verdict],
    },
}


def run_case(name: str, case: dict) -> bool:
    # Imported here so --help and --list work without an API key present.
    from agents import researcher_agent, writer_agent, critic_agent

    state = {
        "task": "Assess my fit for this posting, then tailor my resume bullets to it.",
        "mode": "apply",
        "job_posting_text": case["posting"],
        "user_profile": {"resume_text": case["resume"]},
        "revision_count": 0,
    }

    state = researcher_agent(state)   # apply mode: no LLM call, just assembles research
    state = writer_agent(state)       # 1 call
    state = critic_agent(state)       # 2 parallel calls

    draft = state.get("draft", "") or ""
    critique = state.get("critique", "") or ""

    print(f"\n{'=' * 72}\nCASE: {name}\n  {case['why']}\n{'=' * 72}")
    print(f"  Critic verdict: {'APPROVED' if state.get('approved') else 'REVISE'}"
          f"  (round {state.get('revision_count')})")
    print(f"  Draft length:   {len(draft)} chars\n")

    if len(draft.strip()) < 200:
        print("  !! Draft is suspiciously short -- hidden reasoning likely ate the")
        print("     completion budget. Checks below will be unreliable.\n")

    passed = True
    for check in case["checks"]:
        ok, message = check(draft=draft, critique=critique)
        print(f"  [{'PASS' if ok else 'FAIL'}] {check.__name__}\n         {message}")
        passed = passed and ok

    if not passed:
        print(f"\n--- draft ---\n{draft}\n--- critique ---\n{critique}")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(CASES), help="run a single case")
    parser.add_argument("--runs", type=int, default=1,
                        help="repeat each case N times (LLM output varies run to run)")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    args = parser.parse_args()

    if args.list:
        for name, case in CASES.items():
            print(f"  {name:12s} {case['why']}")
        return 0

    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("GROQ_API_KEY is not set -- these cases make real Groq calls.")
        return 2

    selected = {args.case: CASES[args.case]} if args.case else CASES
    calls = len(selected) * args.runs * 3
    print(f"Running {len(selected)} case(s) x {args.runs} run(s) = ~{calls} Groq calls.")

    failures = []
    for run in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n######## RUN {run} of {args.runs} ########")
        for name, case in selected.items():
            try:
                if not run_case(name, case):
                    failures.append(f"{name} (run {run})")
            except Exception as exc:
                print(f"\n  ERROR in case {name}: {type(exc).__name__}: {exc}")
                failures.append(f"{name} (run {run}, errored)")

    print(f"\n{'=' * 72}")
    if failures:
        print(f"FAILED: {len(failures)} case-run(s): {', '.join(failures)}")
        print("Read the printed draft/critique above before changing any prompt.")
        return 1
    print(f"All {len(selected) * args.runs} case-run(s) passed -- no regression seen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
