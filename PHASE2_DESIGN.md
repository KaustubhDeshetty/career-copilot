# Phase 2 — JD archetype matching for Apply mode

Design decisions from the 2026-09-02 brainstorm. Nothing here is built yet.
Context: the professor wants to take the project toward an enterprise-level
application. Enterprise scoping was **deliberately deferred**; this phase is
the concrete feature agreed on first.

---

## The finding that shaped everything

Six "flavoured resumes" were provided — Solutions Architect, Cloud AI,
Data & ML Platform, GenAI, ML Engineer, MLOps.

**They are one resume, reordered.** Measured, not assumed:

- TF-IDF cosine between every pair: **0.95–0.985, mean 0.971**
- All six are within 200 characters of each other in length
- Only three things vary: the **title**, the **summary paragraph**, and the
  **ordering** of experience bullets and skills-table rows
- **Not one bullet is unique to any flavour.** Every claim appears in all six

Two consequences:

1. **Embedding / similarity ranking is not viable.** Cosine scores would differ
   in the third decimal — noise presented as confidence. Confirmed by
   `analyze_resume_overlap.py`.
2. **The fit verdict is flavour-independent.** The Fit Assessment evaluates
   requirements against the experience section, which is constant. Choosing a
   different flavour cannot change whether the candidate qualifies.

The user's response to this reframed the feature usefully: stop treating the
flavours as different qualifications, and treat them as **role archetypes** —
the six *kinds of job* being targeted.

## The design

**The flavours become a JD classification taxonomy, not a resume-selection
problem.** This is why the resumes being near-identical stops mattering.

Two stages answering different questions:

**Stage 1 — archetype match (new, cheap).**
JD + six hand-written rubrics → one LLM call → match strength per archetype.
Roughly 1,300 tokens in, a few seconds. Well under a tenth of a full Apply run.

**Stage 2 — fit check (existing, unchanged).**
The current Fit Assessment, requirement by requirement against experience.

Combined output:

```
ROLE TYPE   GenAI Engineer      71
            MLOps / AI          68
            ML Engineer         54
            Cloud AI            41
            Data & ML Platform  38
            Solutions Architect 33
            → send the GenAI resume

FIT         5 of 6 core requirements met
VERDICT     Apply
```

### Show all six scores, not just the winner

Decided yes. The Selector already reasons across all six in one call, so
emitting per-archetype scores costs ~50 extra output tokens.

- **Near-ties are real** — MLOps and Cloud AI will often land within a few
  points, and the user should break that tie, not the model.
- **A flat distribution is the no-match signal** — everything in the forties
  means the JD is outside the target space entirely.
- **It makes failures debuggable** — "said MLOps 82 / GenAI 79, you'd have
  picked GenAI" points at a rubric margin, not a rewrite.

**Caveat to design around:** these are *relative match strengths*, not
calibrated probabilities. They will move between runs on the same JD (no
temperature is set). Do not normalise to 100 or label them "confidence" — lean
on ordering and gaps.

### The capability this actually adds

The **no-match case**. Today, hand Apply mode a Robotics Perception JD and it
will dutifully produce a fit percentage, because it only asks "do requirements
match experience." It has no concept of the job being in the wrong category.

The archetype layer catches that before spending the 4.5-minute run.

### Teaching the system the distinctions

The distinction between flavours does not exist in the resume text — it exists
in the user's head. So it has to be supplied explicitly. **Same shape as Bug 3:**
the `Job posting:` / `Resume:` labels already existed; what was missing was
their *meaning*, supplied by prompt.

**Rubrics go in the prompt** — six entries, hand-written, stored as metadata:

```json
"genai": {
  "label": "Senior Generative AI Engineer",
  "use_when": "JD leads with LLMs, RAG, agents, prompt/context engineering, fine-tuning, inference optimisation",
  "avoid_when": "LLMs are incidental; role is really platform ops or classical ML",
  "signals": ["LangGraph", "RAG", "LoRA", "vector database", "agentic", "prompt"],
  "summary": "<existing hand-written summary>"
}
```

**Labelled examples go in a test set, not the prompt.** 15–20 real JDs tagged
with the archetype the user would pick. Zero per-request token cost, tells you
exactly where the rubric is wrong, and acts as a regression test for the
Selector — the role `test_grounding.py` plays for the Critic. Refine the rubric
until it matches the labels.

### Where it sits

```
[Router ∥ Planner] → Selector (apply only) → Researcher → Writer ⇄ Critic
```

**Keep it backward compatible:** `researcher_agent` reads
`user_profile["resume_text"]`. If the Selector's job is simply to *set* that
field, the Researcher, Writer and both Critics need **zero changes**.

```json
{
  "resumes": {
    "genai": { "label": "...", "text": "...", "rubric": {...} },
    "mlops": { ... }
  },
  "resume_text": "<selected one, set at runtime by the Selector>"
}
```

New state fields: `archetype_ranking`, `archetype_choice`, `archetype_verdict`.

**Output contract — verdict first** (the Bug 10 lesson applies):

```
BEST: genai
VERDICT: APPLY | STRETCH | SKIP

RANKING:
  genai 71 | mlops 68 | ml_engineer 54 | cloud_ai 41 | data_platform 38 | architect 33

GAPS: distributed training at scale, on-call ownership
```

### Open question: who owns the skip verdict?

If the Selector's SKIP is **authoritative**, the whole 4.5-minute run is
skipped — a real cost and latency win, and arguably the most useful feature
here. But a summary-level judgement is vetoing a job.

If it is **provisional**, you always pay for the full assessment.

Leaning: Selector may short-circuit, but must show reasoning and offer a
"run it anyway" override. Cheap by default, never silently closing a door.

### One correctness detail

The Fit Assessment currently sees the whole resume including the summary — the
one thing that genuinely varies between flavours. So resume choice could nudge
the fit verdict, and it should not. **A summary is a claim about yourself; the
bullets are evidence.** Have the Fit Assessment read experience only.

---

## Other Apply-mode improvements proposed (not yet decided)

Ranked by value as argued in the brainstorm.

**1. Close the Apply → Interview loop.** Slide 3 of the deck states the problem
as *"none of these tools remember what the others already know about you"* —
and the three modes currently do not talk. Apply already produces exactly what
Interview needs: every requirement marked NOT MET or PARTIALLY MET is a topic
you would be grilled on. Write those to the profile and let Interview mode
offer them as practice. **This delivers the deck's stated premise, which the
architecture currently does not.**

**2. Batch triage.** The archetype classifier is cheap enough to classify ten
JDs in seconds, then spend the expensive assessment only on the top two or
three. Changes the product from "help me apply to this job" to "which of these
ten deserve my evening."

**3. Extract requirements as structured data before assessing them.** The
Writer currently decides what counts as a core requirement in the same breath
as evaluating it — which is why the Meridian JD came back 7-of-7 when it lists
eight required bullets. The denominator is editorial and non-reproducible.
Splitting extraction from assessment makes the fit table stable across runs,
comparable between jobs, and storable as structured data. **This may be close
to what the professor meant by "a decoupled version for Apply mode."**

**4. Stop sending the full resume to every Critic call.** Directly addresses the
token ceiling that caused Bug 10. The resume is static — extract a compact
"verified claims index" once (every number, technology and achievement with its
source line), cache it, and send that to the Claims critic instead of 3,400
tokens of prose.

**Cheaper items:** ATS keyword gap check (pure string matching, zero LLM cost);
duplicate-application warnings from the existing tracker; progressive rendering
so the Fit Assessment appears at round one rather than after four minutes.

## Deferred

**Enterprise scoping.** Not yet answered: whether "enterprise-level" means a
portfolio-grade showcase, an internal tool for a real group, or a multi-tenant
product. Also unanswered: timeline, team, and which constraints are hard
(API cost, latency, PII handling, stack lock-in). These change the roadmap
substantially and should be settled before any infrastructure work.

Known gaps for any of those paths: single-user local-file state, no auth, no
observability, no CI, no evals at scale, 4.5-minute latency, free-tier rate
limits, and resumes are PII sitting in plaintext JSON.
