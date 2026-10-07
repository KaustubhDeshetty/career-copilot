"""
test_selector.py -- regression harness for the Apply-mode archetype Selector.

Plays the role for the Selector that test_grounding.py plays for the Critic:
a fixed set of job postings with gold labels, and a score you have to beat.

Two scorers are wired in:

  keyword    pure substring matching over the `signals` lists in archetypes.json.
             No API key, no tokens, runs in a second. This is the FLOOR -- an
             LLM Selector that cannot beat it is not earning its 1,300 tokens.

  llm        the real Selector. Not implemented yet; see selector_agent below.

Usage:
    python test_selector.py                 # keyword baseline
    python test_selector.py --scorer llm    # once the Selector exists
    python test_selector.py --verbose       # per-JD detail
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TESTSET = os.path.join(HERE, "jd_testset")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_all():
    with open(os.path.join(HERE, "archetypes.json"), encoding="utf-8") as f:
        arch = json.load(f)
    with open(os.path.join(TESTSET, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    jds = {}
    for name in sorted(labels["labels"]):
        with open(os.path.join(TESTSET, name), encoding="utf-8") as f:
            jds[name] = f.read()
    return arch, labels, jds


# --------------------------------------------------------------------------
# scorers -- each returns {archetype_key: float}, higher = stronger match
# --------------------------------------------------------------------------

def keyword_baseline(jd_text, arch):
    """Count distinct rubric signals present in the posting.

    Deliberately dumb. Its job is to be the number the LLM has to beat, and to
    expose signal-list leakage: if a JD with no LLM content scores high on
    genai, a term is in the wrong list (see jd 13, tagged signal_trap).
    """
    low = jd_text.lower()
    out = {}
    for key, rub in arch["archetypes"].items():
        hits = [s for s in rub["signals"] if s.lower() in low]
        out[key] = float(len(hits))
    return out


def llm_selector(jd_text, arch):
    """The real Selector. Returns its own verdict rather than letting the
    harness derive one from score level -- deriving it is exactly what scored
    40% precision on the keyword baseline."""
    from selector import score_jd
    r = score_jd(jd_text, arch)
    return r["scores"], r["verdict"], r


def _keyword_wrapper(jd_text, arch):
    return keyword_baseline(jd_text, arch), None, None


SCORERS = {"keyword": _keyword_wrapper, "llm": llm_selector}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def ranking_from_scores(scores, order):
    """Highest score first. Ties broken by the canonical archetype order so a
    run is reproducible rather than dict-order dependent."""
    return sorted(order, key=lambda k: (-scores[k], order.index(k)))


def spearman(pred, gold):
    """Rank correlation over a permutation of the same 6 items. -1 .. 1."""
    n = len(gold)
    d2 = sum((pred.index(k) - gold.index(k)) ** 2 for k in gold)
    return 1 - (6 * d2) / (n * (n * n - 1))


def flatness(scores):
    """Gap between best and worst, as a fraction of best. 0 = perfectly flat.

    Low flatness plus a LOW top score is the no-match signal. Low flatness plus
    a HIGH top score is a generalist role -- a different thing entirely, and the
    reason jds 18 and 20 are both in the set.
    """
    vals = sorted(scores.values(), reverse=True)
    if vals[0] == 0:
        return 0.0
    return (vals[0] - vals[-1]) / vals[0]


def derive_verdict(scores, skip_below=3.0):
    """Crude verdict for the keyword baseline: if nothing scores above the
    floor, the posting is outside the target space. The real Selector emits its
    own verdict and should ignore this."""
    top = max(scores.values())
    if top < skip_below:
        return "SKIP"
    return "APPLY"


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run(scorer_name, verbose=False, only=None, progress=False, assume_yes=False):
    arch, labels, jds = load_all()
    order = labels["_archetypes"]
    scorer = SCORERS[scorer_name]

    top1 = top2 = 0
    rhos = []
    skip_hit = skip_total = skip_called = 0
    false_skip = []
    no_signal = []
    excluded_unparseable = []
    excluded_nosignal = []
    excluded_lowconf = []
    confusion = Counter()
    in_space = 0
    rows = []

    items = list(labels["labels"].items())
    if only:
        items = [(n, g) for n, g in items if any(o in n for o in only)]

    if scorer_name == "llm":
        # Groq free tier is 8,000 TPM / 200,000 TPD and the Selector call is
        # not small. Say what this run costs BEFORE spending it -- a full
        # sweep is roughly half a day's quota and takes ten-plus minutes of
        # rate-limit backoff, which is a surprise worth not having.
        # Measured, not guessed: render the real prompt for the longest JD in
        # the selected set and add the actual output budget.
        try:
            import selector as _sel
            longest = max((jds[n] for n, _ in items), key=len)
            sys_p, usr_p = _sel.build_selector_messages(longest, arch)
            per = (len(sys_p) + len(usr_p)) // 4 + _sel.SELECTOR_MAX_TOKENS
        except Exception:
            per = 5900
        total = per * len(items)
        print(f"\n{len(items)} JD(s) x ~{per:,} tokens = ~{total:,} tokens "
              f"(~{100*total/200000:.0f}% of a 200,000 TPD free-tier day).")
        print(f"At 8,000 TPM this paces out to roughly "
              f"{total/8000:.0f} minute(s) of wall clock with backoff.")
        print("A posting whose first reply comes back incomplete costs a "
              "second call (low-effort retry).")
        if not assume_yes:
            if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted.")
                return 0, 0
    for name, gold in items:
        scores, own_verdict, detail = scorer(jds[name], arch)
        pred = ranking_from_scores(scores, order)
        g = gold["rank"]
        is_nomatch = "no_match" in gold["tests"]

        verdict = own_verdict or derive_verdict(scores)
        gold_skip = gold["verdict"] == "SKIP"
        if gold_skip:
            skip_total += 1
            if verdict == "SKIP":
                skip_hit += 1
        if verdict == "SKIP":
            skip_called += 1
            if gold["verdict"] == "APPLY":
                false_skip.append(name)
        if max(scores.values()) == 0 and not (detail and detail["complete"]):
            no_signal.append(name)

        # A ranking metric is only meaningful when there IS a ranking and the
        # gold order means something. Three exclusions, all declared before
        # this run rather than chosen after seeing the scores:
        #
        #   no_match    labels.json says their order is arbitrary by design
        #   confidence  a label marked "low" says the same thing per-row (#16,
        #               #18 -- recruiter spam and a frontend role; the note on
        #               each already reads "the ranking is not the point")
        #   no ranking  the scorer produced nothing to rank -- an unparseable
        #               reply, or all six scores at zero
        #
        # That last one is the important fix. In the first LLM smoke test a
        # reply truncated to nothing scored zero across the board, canonical
        # tie-break put genai first, gold was genai, and the harness recorded
        # a PASS. A total failure to answer was being counted as a correct
        # answer, which inflates exactly the number this file exists to
        # protect.
        unparseable = (detail is not None and not detail["complete"])
        # All-zero scores from a COMPLETE reply is a real answer -- the model
        # saying "none of these, at all", which is the correct read of a React
        # frontend posting. Only treat all-zero as a non-answer when the reply
        # did not parse. (The keyword scorer has no completeness signal, so
        # all-zero there stays a non-answer, which is what it is.)
        no_signal_row = max(scores.values()) == 0 and not (
            detail is not None and detail["complete"])
        low_conf = gold.get("confidence") == "low"

        if unparseable:
            excluded_unparseable.append(name)
        elif no_signal_row and not is_nomatch:
            excluded_nosignal.append(name)
        elif low_conf and not is_nomatch:
            excluded_lowconf.append(name)

        if not is_nomatch and not (unparseable or no_signal_row or low_conf):
            in_space += 1
            if pred[0] == g[0]:
                top1 += 1
            else:
                confusion[(g[0], pred[0])] += 1
            top2 += len(set(pred[:2]) & set(g[:2])) / 2.0
            rhos.append(spearman(pred, g))

        rows.append((name, gold, scores, pred, verdict, is_nomatch))
        if progress:
            mark = "--" if is_nomatch else ("ok" if pred[0] == g[0] else "XX")
            tie = f" TIE(gap {detail['gap']:.0f})" if detail and detail["is_tie"] else ""
            inc = "  !!INCOMPLETE" if detail and not detail["complete"] else ""
            if detail and detail.get("retried"):
                inc += "  (retried)"
            print(f"  [{mark}] {name:40} {pred[0]:14} {verdict:8}{tie}{inc}", flush=True)

    print(f"\nscorer: {scorer_name}   |   {len(items)} JDs "
          f"({in_space} in-space, {skip_total} no-match)")
    if not labels.get("_ratified"):
        print("  ** labels are UNRATIFIED and the JDs are synthetic -- "
              "internal consistency only, not validation **")
    print("-" * 74)
    n_ex = len(excluded_unparseable) + len(excluded_nosignal) + len(excluded_lowconf)
    if n_ex:
        print(f"  scored on {in_space} of {in_space + n_ex} in-space JDs; "
              f"{n_ex} excluded from ranking metrics:")
        for lbl, lst in (("unparseable reply", excluded_unparseable),
                         ("no signal at all", excluded_nosignal),
                         ("label confidence low", excluded_lowconf)):
            for n in lst:
                print(f"      {n:44} ({lbl})")
        print()
    print(f"  top-1 accuracy (in-space)   {top1}/{in_space}  "
          f"{100.0*top1/max(in_space,1):5.1f}%")
    print(f"  top-2 set overlap           {top2:.1f}/{in_space}  "
          f"{100.0*top2/max(in_space,1):5.1f}%")
    print(f"  mean Spearman rho           {sum(rhos)/len(rhos):+.3f}" if rhos else "  mean Spearman rho           n/a")
    print(f"  SKIP recall (no-match)      {skip_hit}/{skip_total}")
    prec = (100.0 * skip_hit / skip_called) if skip_called else 0.0
    print(f"  SKIP precision              {skip_hit}/{skip_called}  {prec:5.1f}%"
          f"   <- recall alone is not a result")
    print(f"  FALSE SKIP on a gold APPLY  {len(false_skip)}"
          f"   <- the dangerous error class")
    for n in false_skip:
        print(f"      {n}")
    if no_signal:
        print(f"\n  !! {len(no_signal)} JD(s) scored ZERO on all six archetypes:")
        for n in no_signal:
            print(f"      {n}")

    if confusion:
        print("\n  confusions (gold -> predicted):")
        for (g, p), n in confusion.most_common():
            print(f"    {g:14} -> {p:<14} x{n}")

    print("\n  flat-distribution check (18 should be flat+LOW, 20 flat+HIGH):")
    for name, gold, scores, pred, verdict, _ in rows:
        if "low_information" in gold["tests"] or "flat_high" in gold["tests"]:
            print(f"    {name:34} top={max(scores.values()):4.0f}  "
                  f"spread={flatness(scores):.2f}  -> {verdict}")

    if verbose:
        print("\n" + "=" * 74)
        for name, gold, scores, pred, verdict, is_nm in rows:
            ok = "--" if is_nm else ("PASS" if pred[0] == gold["rank"][0] else "FAIL")
            print(f"\n[{ok}] {name}   tests={','.join(gold['tests'])}")
            print(f"      gold {' > '.join(gold['rank'])}")
            print(f"      pred {' > '.join(pred)}")
            print(f"      scores " + "  ".join(
                f"{k}={scores[k]:.0f}" for k in pred))
            print(f"      verdict {verdict} (gold {gold['verdict']})")
            print(f"      {gold['note']}")

    return top1, in_space


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="keyword", choices=list(SCORERS))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", nargs="*", help="substring match on filenames")
    ap.add_argument("--yes", action="store_true", help="skip the cost prompt")
    ap.add_argument("--progress", action="store_true",
                    help="print each JD as it completes (the llm scorer is slow)")
    a = ap.parse_args()
    try:
        run(a.scorer, a.verbose, a.only, a.progress or a.scorer == 'llm', a.yes)
    except NotImplementedError as e:
        print(f"\n{e}\n")
        sys.exit(1)
