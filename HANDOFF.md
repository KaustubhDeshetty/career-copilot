# Session handoff -- Career Copilot

**Context:** demo is tomorrow. Full technical history is in `PROGRESS.md`; this
note covers only what is current and still open. Updated after the hardening
pass (see `PROGRESS.md` section 14).

## Current state

- All three modes (Apply / Interview / Learn) built and passing individual
  tests, including edge cases: zero corpus coverage, severe resume/JD mismatch,
  partial match.
- Streamlit UI is enterprise-styled: custom theme, Fit Assessment as a metric,
  reveal-on-click quiz cards for Interview, Excel application tracker.
- Presentation deck (18 slides, `career_copilot_deck.pptx`) covering
  architecture, debugging journey and lessons, with speaker notes on every
  slide.
- Router + Planner run concurrently; Apply's Critic runs as two parallel calls
  (Fit check 1,400 tokens + Claims check 1,800). Both are working
  optimisations, not theoretical.

## Added in the hardening pass

- **Interview difficulty now actually works.** It was previously extracted by
  the Router, shown in the UI, logged to the tracker -- and never passed to the
  Writer, so every difficulty produced the same quiz. Same class of bug as the
  Critic never receiving `research`.
- **Duplicate tracker rows fixed.** The guard was session-state only and reset
  on page refresh; it now checks the workbook.
- **Test layer:** `test_tracker.py` (16 unit tests, no API key) and
  `test_grounding.py` (adversarial fixtures for bugs 1-3, ~3 Groq calls per
  case).
- **`requirements.txt`** and **`VERIFICATION.md`** (staged pre-demo checklist).

## Not yet verified -- do this before presenting

Work through `VERIFICATION.md` top to bottom. The three items that predate this
pass, still open:

1. The Interview-mode Critic's reasoning-vs-recall fix has never been
   re-confirmed on a clean run (stage 2a).
2. The redesigned UI has never been watched end-to-end across all three modes in
   one sitting (stage 4).
3. Apply's Critic still runs on a tight token margin under the 8,000-token
   per-request ceiling -- documented and instrumented, not eliminated (stage 3).

Plus one surfaced during the pass:

4. **`save_interview_session()` has never run against a real workbook** -- the
   `Interview Sessions` sheet does not exist in the current
   `applications.xlsx`. Unit-tested, not yet exercised through the UI. This is
   the least-tested code path in the project.

## Checked and clear

- **Learn mode's corpus is real, not empty.** `class_notes/` holds 7 notebooks
  (python foundations, ML, deep learning, NLP, numpy/statistics, RAG-with-agent,
  agentic AI + RAG) and `chroma_db_class_notes/` is a built 4.8 MB index. The
  interview index is built too (2.5 MB). `chroma_db_scifact/` is absent, which
  is expected -- nothing routes there unless `orchestrator.py` is run with no
  mode at all, and it would rebuild on demand.

## Bug 10 -- FIXED: a truncated Critic check could pass silently

Found during verification, fixed the same night, parser tested offline.

The Claims critic was ignoring its "reply with ONLY the verdict" instruction
and emitting a full `Step 1 / Step 2` enumeration of every sourced claim. On a
technology-dense resume that runs to thirty-plus items, blew the 1,800-token
budget, and was severed mid-word -- twice in one run, confirmed by
`finish_reason='length'` in the terminal.

Two defects, not one:

1. **The INCOMPLETE guard tested length, not completeness.** It flagged a reply
   under 10 characters. The real truncation is the opposite shape: long,
   diligent, and missing only its verdict line. It sailed straight through.
2. **`_verdict_from_text` failed open.** With no `Final verdict:` marker,
   `split` returns the whole reply, and the scan then matched the word
   "approved" occurring anywhere in the Critic's prose. A truncated check
   containing e.g. "the 2x figure was approved on the prior round" was read as
   APPROVED, with no warning. Verified against the old code.

Fixes: both Apply critic prompts now open with an explicit OUTPUT CONTRACT
putting the verdict on the FIRST line and forbidding shown working, so
truncation can no longer eat the verdict; `_has_verdict()` flags INCOMPLETE on
a missing marker rather than on length; `_verdict_from_text` fails closed
without the marker.

**Budgets were deliberately NOT raised.** Measured: research (~4,200 tokens for
the real resume plus JD) + draft (~2,000) + prompt (~300) + the 1,800 output
already totals ~8,300 against an 8,000 per-request ceiling. There is no
headroom -- the only fix available was to make the output shorter, which is
what the output contract does. This is HANDOFF item 3 finally biting.

Still to confirm on a live run: that the Claims check now emits a bare verdict
and stops. If it still truncates, the next lever is trimming what the Critic is
sent, not raising what it may write.

## Future scope -- Bug 9: the Researcher's own query is inside `research`

Found during verification, not yet fixed. Same failure family as Bug 3.

`researcher_agent` builds research with:

```python
collected_results.append(f"[{name}({args})]\n{result}")
```

`args` is the query string **the model composed itself**. It lands in
`research` alongside genuinely retrieved documents, and neither the Writer nor
the Critic distinguishes the two.

Observed on the zero-coverage case ("Quiz me on blockchain consensus
mechanisms"). Retrieval returned "No results found", but the quiz still asked
what problem Proof-of-Work, Proof-of-Stake, DPoS, PBFT, Raft, Tendermint and
Proof-of-Authority are meant to solve -- a list that appears nowhere in the
corpus and came only from the model's own search query. The Critic approved it,
correctly by its own contract: that text *was* present in the research it was
shown.

Severity: lower than bugs 1-3. It did not fabricate blockchain facts, and it
was transparent that the search failed. But it is a real grounding hole -- the
query section is strictly less trustworthy than a job posting, because the
model authored it.

Demo note: slide 13 claims the system "offers a quiz built only from what's
actually indexed". This run instead produced questions *about the search
failing*, which is a different (weaker) behaviour than the slide describes.

Likely fix when revisited: label the two sections distinctly when assembling
research (retrieved content vs. the query that produced it), the same way the
Apply-mode fix taught both agents to separate Job posting from Resume. A
narrower alternative is to stop interpolating `args` into `collected_results`
at all and keep the query only in the printed trace.

## Future scope -- known design asymmetry, deliberately not acted on

**The Learn critic is stricter than the Interview critic, by prompt, not by
accident.** `CRITIC_SYSTEM_PROMPTS["interview"]` carries an explicit
DO NOT FLAG carve-out for standard, well-established, textbook-level
elaboration. `CRITIC_SYSTEM_PROMPTS["learn"]` has no equivalent -- it approves
"only if the explanation is fully grounded" and checks that every fact,
formula, numeric range and code example traces to the research. The
reasoning-vs-recall fix was applied to Interview only and never mirrored.

Observed on a word-embeddings run: the Learn critic returned REVISE flagging
the distributional hypothesis, cosine similarity, and the
king - man + woman = queen analogy as ungrounded. All standard NLP knowledge,
all correctly flagged *per the prompt as written*.

Two of the five flags on that run were unambiguously correct and should survive
any future change: a "50-300 dimensions" range invented from a research snippet
that only mentioned 100-dimensional GloVe (the Learn Writer prompt explicitly
forbids invented numeric ranges), and a described procedure for loading GloVe
vectors into `nn.Embedding` that the source notebook never shows.

Not changed because: strictness is arguably *correct* for a tool reviewing the
user's own course notes -- the point is to be anchored to that material rather
than fill gaps with general knowledge the user has not studied. Learn's 1-round
cap also means a REVISE still ships the best draft rather than blocking. If it
is ever revisited, the decision to make is whether Learn should share
Interview's textbook-elaboration carve-out or keep its own stricter contract --
not simply whether it is "over-flagging."

## Open question, deliberately not acted on

Switching from `openai/gpt-oss-120b` to `llama-3.3-70b-versatile`. Investigated
in detail:

- Real trade-off, not a strict upgrade -- Llama gives 12,000 TPM (vs 8,000) but
  only 100,000 TPD (vs 200,000).
- `reasoning_effort` -- the fix for the Apply Critic going blank -- is
  GPT-OSS-specific and does not transfer.
- Every token budget here was measured against GPT-OSS's specific behaviour,
  not derived from a general rule, so a swap invalidates all of them at once.
- **Decision: do not switch before the demo.** Worth exploring afterward as a
  documented, tested alternate config -- not a last-minute swap.

## Files

`PROGRESS.md`, `VERIFICATION.md`, `requirements.txt`, `agents.py`, `app.py`,
`router.py`, `orchestrator.py`, `rag_utils.py`, `state.py`, `memory_utils.py`,
`tracker_utils.py`, `test_router.py`, `test_tracker.py`, `test_grounding.py` --
current versions from the actual project folder, not regenerated from memory.
