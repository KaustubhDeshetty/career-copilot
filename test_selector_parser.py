"""
Parser regression tests for the archetype Selector. No API key, no tokens.

Two invariants, pulling in opposite directions:

  NEVER SKIP ON A BROKEN REPLY.  SKIP short-circuits the 4.5-minute Apply run,
  so a false SKIP silently closes a door on a job. A truncated, empty or
  malformed reply must degrade to STRETCH. (The Critic fails closed to NOT
  APPROVED; the Selector fails closed to NOT SKIP -- same idea, opposite
  polarity.)

  ALWAYS SKIP WHEN THE REJECT CLASS WINS.  The first full sweep scored a
  quantitative-research posting ml_engineer 85 and returned APPLY, because a
  forced choice among six archetypes had no way to say "none of these". The
  seventh option, out_of_space, is that way -- and when it outscores every
  archetype the SKIP is authoritative, however high the runner-up scored.

    python test_selector_parser.py
"""

from selector import parse_selector_output, SKIP_CEILING, TIE_THRESHOLD

KEYS = ["genai", "ml_engineer", "mlops", "cloud_ai", "data_platform", "architect"]

WELL_FORMED = """VERDICT: APPLY
BEST: genai
RANKING: genai 78 | mlops 71 | ml_engineer 52 | cloud_ai 40 | data_platform 22 | architect 15 | out_of_space 5
GAPS: distributed training at scale
WHY: Retrieval grounding and evaluation are the core deliverable."""

NEAR_TIE = """VERDICT: APPLY
BEST: mlops
RANKING: mlops 74 | genai 71 | cloud_ai 45 | ml_engineer 30 | data_platform 25 | architect 12 | out_of_space 8
TIE: mlops vs genai -- does the role change what the model produces, or only run it?
GAPS: vLLM internals
WHY: Serving systems work in generative vocabulary."""

TRUNCATED = """BEST: genai
RANKING: genai 78 | mlops 71 | ml_engineer 52 | cloud_ai 40 | data_pl"""

SKIP_CONTRADICTING_SCORES = """VERDICT: SKIP
BEST: genai
RANKING: genai 82 | mlops 66 | ml_engineer 40 | cloud_ai 35 | data_platform 20 | architect 10 | out_of_space 12
GAPS: none apparent
WHY: Outside target space."""

PARTIAL_SCORES = """VERDICT: SKIP
BEST: mlops
RANKING: mlops 30 | genai 22
GAPS: none apparent"""

LEGITIMATE_SKIP = """VERDICT: SKIP
BEST: ml_engineer
RANKING: out_of_space 82 | ml_engineer 28 | mlops 15 | genai 12 | data_platform 10 | cloud_ai 8 | architect 5
GAPS: C++, SLAM, factor graphs
WHY: Robotics perception, not in the target space."""

# The exact shape that failed on the first sweep.
REJECT_BEATS_STRONG_ARCHETYPE = """VERDICT: SKIP
BEST: ml_engineer
RANKING: out_of_space 88 | ml_engineer 85 | data_platform 30 | architect 25 | mlops 15 | genai 10 | cloud_ai 5
GAPS: PhD, buy-side research experience
WHY: Quantitative finance research, a different discipline."""

REJECT_WINS_BUT_MODEL_SAID_APPLY = """VERDICT: APPLY
BEST: data_platform
RANKING: out_of_space 78 | data_platform 62 | mlops 30 | architect 25 | genai 10 | ml_engineer 8 | cloud_ai 5
GAPS: none apparent
WHY: dbt and Looker reporting work."""

OLD_FORMAT_NO_REJECT = """VERDICT: SKIP
BEST: ml_engineer
RANKING: ml_engineer 28 | mlops 15 | genai 12 | data_platform 10 | cloud_ai 8 | architect 5
GAPS: none apparent
WHY: Pre-reject-class reply shape."""

MARKDOWN_TABLE = """**VERDICT: APPLY**
**BEST:** `mlops`
| archetype | score |
|---|---|
| mlops | 81 |
| cloud_ai | 62 |
| ml_engineer | 44 |
| data_platform | 30 |
| genai | 18 |
| architect | 12 |
| out_of_space | 6 |
GAPS: none apparent
WHY: Lifecycle ownership end to end."""

#         name                                        text                        verdict  complete best  tie
CASES = [
    ("well-formed reply parses",                 WELL_FORMED,                "APPLY",   True,  "genai",       False),
    ("near-tie is flagged by arithmetic",        NEAR_TIE,                   "APPLY",   True,  "mlops",       True),
    ("truncated before verdict is NOT a SKIP",   TRUNCATED,                  "STRETCH", False, "genai",       None),
    ("SKIP against its own high scores is void", SKIP_CONTRADICTING_SCORES,  "APPLY",   True,  "genai",       False),
    ("verdict present, scores missing: no SKIP", PARTIAL_SCORES,             "STRETCH", False, "mlops",       None),
    ("a real SKIP survives",                     LEGITIMATE_SKIP,            "SKIP",    True,  "ml_engineer", None),
    ("reject class beats a strong archetype",    REJECT_BEATS_STRONG_ARCHETYPE, "SKIP", True,  "ml_engineer", None),
    ("reject wins, model said APPLY: forced",    REJECT_WINS_BUT_MODEL_SAID_APPLY, "SKIP", True, "data_platform", None),
    ("pre-reject reply shape is incomplete",     OLD_FORMAT_NO_REJECT,       "STRETCH", False, "ml_engineer", None),
    ("markdown decoration hides nothing",        MARKDOWN_TABLE,             "APPLY",   True,  "mlops",       False),
    ("empty reply is not a SKIP",                "",                         "STRETCH", False, None,          None),
    ("whitespace is not a SKIP",                 "   \n  ",                  "STRETCH", False, None,          None),
    ("prose with no contract is not a SKIP",     "Probably MLOps.",          "STRETCH", False, None,          None),
]


def main():
    passed = 0
    for name, text, want_verdict, want_complete, want_best, want_tie in CASES:
        r = parse_selector_output(text, KEYS)
        ok = r["verdict"] == want_verdict and r["complete"] == want_complete
        if want_best is not None:
            ok = ok and r["best"] == want_best
        if want_tie is not None:
            ok = ok and r["is_tie"] == want_tie
        # out_of_space must never be rankable or selectable
        ok = ok and "out_of_space" not in r["scores"] and r["best"] != "out_of_space"
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        got verdict={r['verdict']} complete={r['complete']} "
                  f"best={r['best']} is_tie={r['is_tie']} oos={r['out_of_space']}")

    for name, text, want_verdict, *_ in CASES:
        r = parse_selector_output(text, KEYS)
        if want_verdict != "SKIP":
            assert r["verdict"] != "SKIP", f"{name} produced a SKIP -- door closed silently"
        else:
            assert r["verdict"] == "SKIP", f"{name} lost a SKIP the reject class earned"

    print(f"\n{passed}/{len(CASES)} passed  "
          f"(skip ceiling {SKIP_CEILING}, tie threshold {TIE_THRESHOLD})")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
