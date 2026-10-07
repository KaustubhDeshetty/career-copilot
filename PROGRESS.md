# Career Copilot — Progress Log

This documents the pivot from the Day 1 single-pipeline scaffold (Planner →
Researcher → Writer → Critic over SciFact) to a three-mode "Career Copilot"
system, and the debugging session that followed while getting Apply mode
working end-to-end. Written in the order things actually happened, including
the dead ends — the failed attempts are as informative as the fix that
finally worked.

## 1. Architecture decision: hybrid, not pure sequential or pure hierarchical

The system needed to support three distinct user intents — tailoring a
resume to a job posting (**Apply**), mock interview practice (**Interview**),
and reviewing course concepts (**Learn**) — without writing three separate
pipelines from scratch.

**Chosen design:** a hierarchical **Router** picks exactly one mode per
request, then each mode runs the *same* sequential Researcher → Writer →
Critic chain from the original scaffold, just with mode-specific system
prompts and (for Interview/Learn) a different corpus. This reuses debugged
code instead of duplicating it three times, while still giving real routing
flexibility at the top.

## 2. Shared state, extended not replaced

`state.py` kept every original field (`task`, `plan`, `research`, `sources`,
`draft`, `critique`, `approved`, `revision_count`) untouched, and added:

- `mode` — `"apply"` | `"interview"` | `"learn"`
- `user_profile` — persisted facts about the user (resume text, later:
  learning history)
- `job_posting_text`, `topic`, `difficulty` — mode-specific starting fields

Kept flat rather than nested, to match the existing schema's style and avoid
touching `agents.py`'s field access patterns unnecessarily.

## 3. Router: forced tool-calling, not `tool_choice="auto"`

`router.py` classifies the user's request into a mode using three tool
schemas (`enter_apply_mode`, `enter_interview_mode`, `enter_learn_mode`) and
`tool_choice="required"` — deliberately different from the class's
`tool_choice="auto"` pattern, since the Router's only job is to pick a mode;
it should never be allowed to fall through to a plain-text answer.

Verified standalone with `test_router.py` against clear-cut prompts, a
deliberately ambiguous one ("help me with data science"), and a dual-intent
one ("interview next week for a job I found") — all routed correctly,
including the LLM correctly disambiguating "job" appearing in an
interview-prep request rather than an apply request.

## 4. Multi-corpus RAG, with Apply mode explicitly excluded

`rag_utils.py` was generalized from a single hardcoded SciFact corpus to a
named-corpus registry (`scifact`, `interview_qa`, `class_notes`), each with
its own loader and persisted Chroma index. Apply mode was deliberately left
out of this — it doesn't search a corpus at all; the "research" is just the
job posting plus the user's resume, both already in hand. `researcher_agent`
in `agents.py` branches on `state["mode"]` accordingly.

## 5. Resume persistence

`memory_utils.py` adapted the class's Day 3 `PersistentMemoryAgent` pattern
(disk-backed JSON profile) to this project's plain-function style. Resume
text is loaded once via `python memory_utils.py resume.txt` and read back
automatically by `researcher_agent` on every Apply-mode run.

## 6. First live Apply-mode run: it worked, but the Writer fabricated numbers

Wiring `router.py` + `memory_utils.py` into `orchestrator.py` and running a
real Apply-mode request produced a complete, well-organized draft — Critic
approved it on round 1. Looked correct on the surface.

It wasn't. Comparing the draft against the source resume line by line
turned up multiple invented statistics with no basis in the resume at all
(a claimed "70% VRAM reduction," "3× throughput boost," "10k RPS" — none of
which appear anywhere in the source). The generic Critic prompt only checked
"accuracy, clarity, completeness" in the abstract — it had no instruction to
verify numbers against anything.

This is the same failure class already documented from the original
Researcher bug: **an agent can produce confident, plausible output while
being completely ungrounded, and it looks fine until you check.**

## 7. Fix attempt 1 — mode-aware prompts

Added `WRITER_SYSTEM_PROMPTS` / `CRITIC_SYSTEM_PROMPTS` dicts keyed by mode,
so Apply mode gets stricter instructions (never invent numbers; use
`[quantify: ...]` placeholders) while Interview/Learn keep the original
generic prompt, deferred until those modes are actually tested.

**Result:** Writer stopped fabricating. Critic swung the other way —
started flagging real, verbatim-sourced numbers ("11+ years," "GPT-5.4,"
"2×") as unsourced, apparently pattern-matching on "contains a digit"
rather than actually re-reading the resume.

## 8. Fix attempt 2 — tighter critic scope

Explicitly excluded years-of-experience, version-numbered tool/model names,
and non-numeric qualitative language from the check; taught it that
paraphrases ("2x" = "doubling") count as the same claim.

**Result:** Still flagging real numbers as unsourced, on a second job
posting test. The prompt wording wasn't the actual problem.

## 9. Root cause found — the Critic never received the resume at all

`critic_agent`'s user message was:

```python
user = f"Task: {state['task']}\n\nDraft:\n{state['draft']}"
```

`state['research']` was never included. The Critic had been asked to verify
numbers against a document it had never once been shown — present in the
original class scaffold from the start, invisible until Apply mode's Critic
was asked to actually check something specific against it. One line of the
Critic's own output on the next failed run stated this outright: *"No
external research text was provided."* It wasn't hallucinating — it was
telling the truth about a real gap.

## 10. Fix 3 — give the Critic its data, force quote-based verification

```python
user = f"Task: {state['task']}\n\nResearch:\n{state.get('research', '')}\n\nDraft:\n{state['draft']}"
```

Combined with restructuring the Critic's prompt into an explicit three-step
process (list claimed numbers → quote the matching source sentence for each
→ end on a parseable `Final verdict:` line), plus a lightweight regex-based
safety net that auto-approves if every flagged number turns out to be a
verbatim substring of the research text (defense-in-depth, not a
replacement for the LLM check).

**Result:** Approved on round 1, with real quoted evidence for every number.

## 11. Infrastructure detours along the way

Several unrelated environment issues surfaced during testing and were fixed
as they came up:

- Missing `rag_utils.py` in the working folder → `ModuleNotFoundError`
- Missing pip packages (`datasets`, `langchain-community`, `chromadb`,
  `sentence-transformers`)
- Windows Smart App Control blocking numpy's compiled DLL (`mtrand.pyd`) —
  resolved by disabling Smart App Control (still in Evaluation mode, so
  reversible) rather than downgrading Python
- A stray newline in the `GROQ_API_KEY` environment variable from a bad
  `setx` paste, breaking the Bearer auth header
- Groq's free-tier rate limit (8,000 TPM) being hit mid-pipeline once the
  Critic started carrying the full resume in its prompt too — fixed with
  automatic retry-with-backoff in `_call_llm`, parsing the suggested wait
  time straight from Groq's own error message

## 12. PowerShell quoting breaks on long pasted job descriptions

Testing with a full multi-paragraph synthetic JD (not just a one-line
summary) revealed that pasting long, punctuation-heavy text as a quoted CLI
argument is fragile — PowerShell's parser broke partway through a real
paste, executing the rest of the JD's lines as separate (invalid) commands.
Fixed by adding file-based input to `orchestrator.py`
(`python orchestrator.py jd.txt`), the same pattern already used for the
resume — sidesteps the shell parser entirely.

With that fixed, confirmed the Router preserves a full ~400-word JD
verbatim through its tool-call argument extraction — no truncation or
summarization, which had been an open question.

## 13. Fix 4 — job posting requirements vs. candidate facts

With a longer, more realistic JD in play, a new and subtler bug appeared:
the Writer stated the candidate's experience as "5+ years" — the *job
posting's* minimum requirement — instead of the resume's actual "11+
years." The Critic approved it, because "5+ years" *did* technically appear
somewhere in the combined research text (in the job posting section), even
though it wasn't a fact about the candidate at all.

Fixed by teaching both Writer and Critic that `research` has two sources
with different meanings — job posting (employer's ask, not facts about the
candidate) vs. resume (the candidate's actual claims) — and that only
resume-sourced numbers count as valid claims about the candidate. The
safety net was updated to check only the Resume section, so it couldn't
inadvertently rescue a misattributed number the Critic had correctly
flagged.

**Result:** confirmed correct on the next run — draft now states "11+
years" accurately, and even frames it as exceeding the posting's stated
minimum.

## Bugs found and fixed, summarized

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | Writer fabricated performance metrics | No instruction against inventing numbers | Mode-specific prompt forbidding fabrication, `[quantify: ...]` placeholders |
| 2 | Critic flagged real sourced numbers as fake | Prompt wording too blunt ("contains a digit") | Tighter scope excluding years/versions/qualitative language |
| 3 | Critic still flagged real numbers | **Critic never received `state['research']` at all** | Added research to Critic's context; explicit quote-based verification |
| 4 | Occasional rate-limit crash | Larger prompts (resume in both Writer and Critic now) hitting Groq's 8,000 TPM cap | Automatic retry-with-backoff, parsing wait time from the error |
| 5 | PowerShell mangled long pasted JD | Quoting fragility with punctuation-heavy multi-paragraph CLI args | File-based input (`orchestrator.py jd.txt`) |
| 6 | Candidate's real experience replaced with the job posting's minimum | Critic didn't distinguish which section of `research` a match came from | Explicit Job-posting-vs-Resume distinction in both prompts; safety net restricted to Resume section only |
| 7 | Interview difficulty had no effect on the quiz | `state["difficulty"]` was extracted by the Router and displayed in the UI, but never passed into the Writer's prompt | `_mode_context_for_prompt()` surfaces it as a labelled line; DIFFICULTY CALIBRATION block added to the Interview Writer prompt |
| 8 | Duplicate tracker rows after a page refresh | Duplicate guard lived only in `st.session_state`, which Streamlit clears on refresh | `tracker_utils.row_exists()` checks the workbook itself, scoped to same-entry-same-day |

## 14. Post-demo-prep hardening pass

A later working session closed out the four "next steps" this document
originally ended on, and added a test layer the project had been running
without.

**Items 1–4 from the original next-steps list are all done.** Interview and
Learn were tested end-to-end (including the zero-coverage and weak-match
adversarial cases), rate-limit retry was extended to `researcher_agent`'s
tool-calling loop via the shared `_create_completion_with_retry` wrapper, and
mode-specific Writer/Critic prompts now exist for all three modes rather than
just Apply.

**Bug 7 — Interview difficulty was extracted and then silently dropped.**
`router.py` has always pulled `easy`/`medium`/`hard` out of the request into
`state["difficulty"]`, and `app.py` displays it and logs it to the tracker —
but nothing ever passed it to the Writer. `writer_agent` built its user message
from `_task_for_prompt(state)` and `research` only, so the difficulty never
reached the model: "quiz me on decision trees" and "quiz me on decision trees,
make it hard" produced the same quiz from the same prompt.

This is the same *class* of bug as #3 (the Critic never receiving
`state['research']`): a field that is faithfully populated upstream, visibly
present in state, and never actually read by the thing that needs it. It stayed
invisible because the output still looked plausible — a medium-difficulty quiz
is a perfectly reasonable answer to a request for a hard one, and nothing about
the result announces that a parameter was ignored.

Fixed with `_mode_context_for_prompt()`, which surfaces difficulty and topic as
labelled lines in the Writer's user message, plus a DIFFICULTY CALIBRATION block
in `WRITER_SYSTEM_PROMPTS["interview"]` defining what each level means in terms
of question depth and count. Unrecognised values fall back to `medium` rather
than reaching the prompt unhandled, since state can be hand-built by tests or a
direct orchestrator call and bypass the Router's enum.

**Duplicate tracker rows survived a page refresh.** `app.py`'s guard lived only
in `st.session_state`, which Streamlit clears on refresh — so refresh, re-run,
Save wrote a second identical row. `tracker_utils.row_exists()` now checks the
workbook itself, scoped to same-entry-same-day so that legitimately re-running a
posting next week still logs a new row.

**Test layer added.** The project previously had `test_router.py` and nothing
else; correctness of the Writer/Critic fixes was verified by re-running the
pipeline and reading drafts by eye.

- `test_tracker.py` — 16 unit tests over `extract_fit_summary`'s three fallback
  tiers and `extract_company_role`. No API key needed. These cover the parsing
  logic that produced the two `Unknown` rows in `applications.xlsx`.
- `test_grounding.py` — adversarial fixtures that tempt each of bugs 1, 2 and 3
  (a resume containing no metrics at all, a posting whose stated minimum
  conflicts with the resume's real tenure, and an About section carrying a
  company-wide statistic), with deterministic assertions on the resulting draft
  and critique. Skips Router and Planner to keep each case to 3 Groq calls.

**Also added:** `requirements.txt` (the dependency list previously existed only
as prose in section 11 of this document) and `VERIFICATION.md`, a staged
pre-demo checklist.

## Current state

- **All three modes working and tested end-to-end**, including adversarial
  cases: zero corpus coverage produces an honest refusal rather than
  fabrication, and a severe resume/posting mismatch produces an honest
  weak-match verdict with no softening.
- **Streamlit UI** wraps the pipeline with no duplicated backend logic, plus an
  Excel tracker written entirely by parsing output the pipeline already
  produced — no additional LLM calls.
- **Performance work is real, not theoretical:** Router and Planner run
  concurrently; Apply's Critic runs as two parallel checks at 1,400 / 1,800
  tokens so both can use full reasoning effort under the 8,000 TPM ceiling.

## Known-open

1. The Interview-mode Critic's reasoning-vs-recall fix still wants one clean
   confirmation run — see `VERIFICATION.md` stage 2a.
2. The redesigned UI has never been watched end-to-end across all three modes in
   one sitting — stage 4.
3. Apply mode's Critic still runs with a genuinely tight token margin under the
   8,000-token per-request ceiling. Documented and instrumented (an incomplete
   verification is flagged inline in the critique rather than silently
   resembling a normal REVISE), but not eliminated.
4. **The `Interview Sessions` sheet does not exist in the current
   `applications.xlsx`**, which means `save_interview_session()` has never
   actually run against a real workbook. The sheet-creation path is unit-tested
   but not yet exercised through the UI — stage 4.
5. Model swap to `llama-3.3-70b-versatile` remains deliberately not done. See
   `HANDOFF.md` for the full reasoning; the short version is that
   `reasoning_effort` is GPT-OSS-specific and every token budget here was
   measured against GPT-OSS's behaviour rather than derived from a general rule.
