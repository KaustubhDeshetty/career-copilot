"""
test_tracker.py
Unit tests for the pure-parsing half of tracker_utils.py.

Why these exist: extract_fit_summary has three fallback tiers of decreasing
robustness, and until now all three were only ever exercised indirectly --
by running the whole pipeline and squinting at whatever landed in
applications.xlsx. That is how the Tier-3 bug survived long enough to write
two "Unknown" rows into the tracker: the fallback required an "Overall fit:"
prefix the Writer prompt never actually asks for, so it never once fired.

These tests need no API key and no network -- they feed fixed draft text
straight into the parsers. Run either way:

    pytest test_tracker.py -v
    python test_tracker.py
"""

from tracker_utils import extract_fit_summary, extract_company_role


# ---------------------------------------------------------------------------
# Tier 1 -- bolded status markers (the common case)
# ---------------------------------------------------------------------------

TIER1_DRAFT = """
## Fit Assessment

| Requirement | Status |
| --- | --- |
| 5+ years production ML | **MET** |
| PyTorch fine-tuning | **MET** |
| LoRA / QLoRA | **MET** |
| Kubernetes + Docker | **PARTIALLY MET** |
| RTOS / embedded | **NOT MET** |

3 of 5 core requirements met (1 partially met)

**Recommendation:** Partial match -- worth applying if you can speak to the gaps.
"""


def test_tier1_counts_bolded_markers():
    fit, _ = extract_fit_summary(TIER1_DRAFT)
    assert fit == "3 of 5 core requirements met (1 partially met)"


def test_tier1_met_does_not_match_inside_not_met():
    """The whole Tier-1 count depends on '**MET**' being unable to match inside
    '**NOT MET**' or '**PARTIALLY MET**' -- the characters before MET there are
    'T ' and 'Y ', not '**'. If that ever stopped holding, every NOT MET row
    would silently be counted as a MET row and a weak match would read strong."""
    draft = "| A | **NOT MET** |\n| B | **PARTIALLY MET** |"
    fit, _ = extract_fit_summary(draft)
    assert fit == "0 of 2 core requirements met (1 partially met)"


def test_tier1_all_met():
    draft = "| A | **MET** |\n| B | **MET** |"
    fit, _ = extract_fit_summary(draft)
    assert fit == "2 of 2 core requirements met"
    assert "partially" not in fit


# ---------------------------------------------------------------------------
# Tier 2 -- unbolded markers inside a table (the model skips bold sometimes;
# nothing in the Writer prompt actually requires it)
# ---------------------------------------------------------------------------

TIER2_DRAFT = """
## Fit Assessment

| Requirement | Status |
| --- | --- |
| Python | MET |
| PyTorch | MET |
| RTOS | NOT MET |

Recommendation: Weak match -- the core requirements aren't reflected here.
"""


def test_tier2_counts_unbolded_table_markers():
    fit, _ = extract_fit_summary(TIER2_DRAFT)
    assert fit == "2 of 3 core requirements met"


def test_tier2_ignores_metrics_and_prose():
    """Tier 2's regex has to be tight enough that 'METRICS' and 'has MET the
    bar' don't get counted as requirement rows -- that's why it demands a
    table cell or a bullet/colon on one side and a delimiter on the other."""
    draft = """
    - METRICS were improved across the board
    Note: the candidate has MET the bar in several areas
    Overall this is a reasonable fit.
    """
    fit, _ = extract_fit_summary(draft)
    assert fit == "Unknown", f"prose should not be counted as requirements, got {fit!r}"


# ---------------------------------------------------------------------------
# Tier 3 -- no table at all, fall back to the summary sentence
# ---------------------------------------------------------------------------

def test_tier3_summary_sentence_fallback():
    """This is the tier that was broken: it used to require a literal
    'Overall fit:' prefix that the Writer prompt never asks for, so it never
    fired and the tracker logged 'Unknown'."""
    draft = """
    After reviewing the posting against the resume, 3 of 4 core requirements met.

    Recommendation: Strong match -- worth applying.
    """
    fit, rec = extract_fit_summary(draft)
    assert fit == "3 of 4 core requirements met"
    # Trailing '.' is retained: strip(" *") removes spaces and asterisks, not
    # sentence punctuation. Harmless in an Excel cell, and correct English for
    # a recommendation that is a complete sentence -- documented, not a bug.
    assert rec == "Strong match -- worth applying."


def test_no_fit_information_at_all():
    fit, rec = extract_fit_summary("Here are some tailored bullets with no assessment.")
    assert fit == "Unknown"
    assert rec == "Unknown"


def test_empty_draft():
    assert extract_fit_summary("") == ("Unknown", "Unknown")
    assert extract_fit_summary(None) == ("Unknown", "Unknown")


# ---------------------------------------------------------------------------
# Recommendation extraction
# ---------------------------------------------------------------------------

def test_recommendation_strips_bold_markers():
    """'**Recommendation:**' must not leave stray asterisks in the cell. The
    trailing period stays (see test_tier3 note) -- only the markdown goes."""
    _, rec = extract_fit_summary(TIER1_DRAFT)
    assert rec == "Partial match -- worth applying if you can speak to the gaps."
    assert "*" not in rec


def test_recommendation_survives_abbreviations():
    """Regression: splitting on any '.' truncated the recommendation mid-word at
    'e.g.', cutting it off right after '(e'. The split has to require a real
    sentence boundary -- period, space, capital letter."""
    draft = ("Recommendation: Partial match (e.g. you would need to address "
             "the Kubernetes gap). Consider applying anyway.")
    _, rec = extract_fit_summary(draft)
    assert rec == "Partial match (e.g. you would need to address the Kubernetes gap)"


def test_recommendation_takes_only_first_sentence():
    draft = "Recommendation: Strong match. You should apply immediately."
    _, rec = extract_fit_summary(draft)
    assert rec == "Strong match"


# ---------------------------------------------------------------------------
# Company / role extraction
# ---------------------------------------------------------------------------

def test_company_role_standard_two_line_posting():
    role, company = extract_company_role(
        "Senior Machine Learning Engineer — LLM Platform\nMeridian AI\n\nAbout us: ..."
    )
    assert role == "Senior Machine Learning Engineer — LLM Platform"
    assert company == "Meridian AI"


def test_company_role_long_second_line_is_not_a_company():
    """A posting that dives straight into prose shouldn't have its first
    paragraph stored as the company name -- hence the 60-char ceiling."""
    long_line = "We are a fast-growing company building enterprise AI " \
                "infrastructure for Fortune 500 clients worldwide."
    role, company = extract_company_role(f"ML Engineer\n{long_line}")
    assert role == "ML Engineer"
    assert company == "Unknown company"


def test_company_role_single_line():
    role, company = extract_company_role("ML Engineer")
    assert role == "ML Engineer"
    assert company == "Unknown company"


def test_company_role_empty():
    role, company = extract_company_role("")
    assert role == "Unknown role"
    assert company == "Unknown company"


def test_company_role_skips_leading_blank_lines():
    role, company = extract_company_role("\n\n   \nML Engineer\nAcme Corp\n")
    assert role == "ML Engineer"
    assert company == "Acme Corp"


# ---------------------------------------------------------------------------
# Standalone runner, so this works without pytest installed
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
