"""
Consistency checks on archetypes.json. No API key, no tokens.

Exists because a real bug got through: `dbt` was listed under BOTH
data_platform.signals and out_of_space.signals. Those two are mutually
exclusive by construction -- one is lakehouse infrastructure, the other is the
reject class -- so a posting full of dbt was handing the model contradictory
evidence about itself. The original zero-overlap check covered the six
archetypes only, and adding the reject class later slipped past it.

The lesson is the one from the bug log: a check that isn't automated is a
check that stops running the moment the file changes.

    python test_archetypes.py
"""

import json
import os
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ("label", "summary", "use_when", "avoid_when", "signals", "vs", "resume_key")

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
    print(f"  {'PASS' if condition else 'FAIL'}  {message}")


def main():
    with open(os.path.join(HERE, "archetypes.json"), encoding="utf-8") as f:
        doc = json.load(f)
    arch = doc["archetypes"]
    reject = doc["reject_class"]

    check(len(arch) == 6, f"six archetypes defined (found {len(arch)})")

    for key, r in arch.items():
        missing = [f for f in REQUIRED if f not in r]
        check(not missing, f"{key}: all required fields present"
                           + (f" -- missing {missing}" if missing else ""))

    # signal lists must not contradict each other
    sig = {k: {s.lower() for s in v["signals"]} for k, v in arch.items()}
    sig[reject["key"]] = {s.lower() for s in reject["signals"]}
    for a, b in combinations(sig, 2):
        overlap = sorted(sig[a] & sig[b])
        check(not overlap, f"{a} / {b}: no shared signals"
                           + (f" -- shares {overlap}" if overlap else ""))

    # vs must point at archetypes that exist, and never at itself
    for key, r in arch.items():
        targets = [v["archetype"] for v in r["vs"]]
        check(all(t in arch for t in targets), f"{key}: vs targets all exist")
        check(key not in targets, f"{key}: vs does not point at itself")

    # the reject class is not an archetype and has no resume
    check(reject["key"] not in arch, "reject class is not one of the six")

    # every archetype must resolve to a resume
    resumes_path = os.path.join(HERE, "resumes.json")
    if os.path.exists(resumes_path):
        with open(resumes_path, encoding="utf-8") as f:
            resumes = json.load(f)["resumes"]
        for key, r in arch.items():
            check(r["resume_key"] in resumes, f"{key}: resume_key resolves")
        check(reject["key"] not in resumes, "reject class has no resume, correctly")
    else:
        print("  SKIP  resumes.json not present -- merge checks not run")

    print(f"\n{'all checks passed' if not failures else str(len(failures)) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
