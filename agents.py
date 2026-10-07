"""
Each agent is a plain function: takes the shared state, returns the
updated state. This is intentional - it's the same signature LangGraph
nodes use, so upgrading later is a wiring change, not a rewrite.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from groq import Groq, RateLimitError
from state import PipelineState
from rag_utils import semantic_search, keyword_search, MODE_TO_CORPUS

client = Groq(api_key=os.environ.get("GROQ_API_KEY", "").strip())
MODEL = "openai/gpt-oss-120b"  # reliable native tool-calling; llama-3.3-70b-versatile is deprecated on Groq


def _get_retry_wait_seconds(error: Exception, default: float = 10.0) -> float:
    """Groq's rate-limit errors include a suggested wait time in the message
    text itself -- short waits look like '382ms' or '5.265s'; longer waits
    (e.g. daily quota resets) look like '1h1m34.464s'. Parse whichever form
    is present so waits of any length are handled correctly, rather than
    silently falling back to a default that's wrong for long waits."""
    message = str(error)

    # Milliseconds first and separately -- unambiguous, and must not be
    # confused with the "m" (minutes) group in the compound pattern below.
    ms_match = re.search(r"try again in ([\d.]+)ms\b", message)
    if ms_match:
        return max(float(ms_match.group(1)) / 1000, 0.5) + 0.5

    # Compound "XhYmZ.WWWs" format, any subset of h/m/s present.
    compound = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?\b", message)
    if compound and any(compound.groups()):
        hours = float(compound.group(1) or 0)
        minutes = float(compound.group(2) or 0)
        seconds = float(compound.group(3) or 0)
        total = hours * 3600 + minutes * 60 + seconds
        if total > 0:
            return total + 0.5

    return default


MAX_AUTO_RETRY_WAIT_SECONDS = 60  # don't silently block the terminal longer than this


def _create_completion_with_retry(max_retries: int = 3, **kwargs):
    """Wraps client.chat.completions.create with automatic retry-with-backoff
    on Groq rate limit errors. Shared by _call_llm and researcher_agent's
    tool-calling loop, since both hit the same per-minute token cap --
    researcher_agent needs tools=/tool_choice=, which _call_llm doesn't
    support, so it can't just call _call_llm directly.

    Also sets an explicit max_completion_tokens rather than relying on
    Groq's implicit default -- gpt-oss-120b does hidden chain-of-thought
    reasoning before its visible answer (reasoning_effort defaults to
    "medium"), which eats into the token budget invisibly. Combined with
    today's longer Critic prompts (Fit Assessment + technology checks +
    full quote verification), a response can run out of budget mid-
    generation without this, cutting off before reaching a final verdict.

    Waits longer than MAX_AUTO_RETRY_WAIT_SECONDS usually mean the DAILY
    quota is exhausted, not the per-minute one -- retrying won't help
    within a normal session, so this fails fast with a clear message
    instead of silently blocking for an hour+."""
    kwargs.setdefault("max_completion_tokens", 2048)
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            finish_reason = resp.choices[0].finish_reason
            if finish_reason == "length":
                print(f"  [WARNING: response cut off by max_completion_tokens={kwargs['max_completion_tokens']} -- finish_reason='length']")
            return resp
        except RateLimitError as e:
            wait_seconds = _get_retry_wait_seconds(e)
            if wait_seconds > MAX_AUTO_RETRY_WAIT_SECONDS:
                raise RuntimeError(
                    f"Groq rate limit requires a {wait_seconds / 60:.0f}-minute wait "
                    f"(this is almost certainly the DAILY token quota, not a "
                    f"per-minute one -- retrying won't help right now). Try again "
                    f"after that time, or check current usage at "
                    f"https://console.groq.com/settings/billing"
                ) from e
            if attempt == max_retries:
                raise
            wait_seconds = _get_retry_wait_seconds(e)
            print(f"  [Rate limit hit -- waiting {wait_seconds:.1f}s before retry {attempt + 1}/{max_retries}]")
            time.sleep(wait_seconds)


def _call_llm(system: str, user: str, max_completion_tokens: int = 2560, reasoning_effort: str = "medium") -> str:
    resp = _create_completion_with_retry(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )
    # The SDK types `content` as Optional[str], and it really does come back
    # None when hidden reasoning eats the whole budget -- the same state the
    # finish_reason=="length" warning above watches for. Every caller does
    # string work on this (.strip(), re.findall(...)), so a None here becomes
    # an AttributeError/TypeError several frames away, and inside the Apply
    # critic's ThreadPoolExecutor it takes the whole run down. Coerce once,
    # here: _verdict_from_text already treats near-empty text as NOT approved,
    # so "" degrades into an honest failed-verification instead of a crash.
    return resp.choices[0].message.content or ""


def planner_agent(state: PipelineState) -> PipelineState:
    system = "You are a planning agent. Break the user's request into a short, numbered research plan."
    state["plan"] = _call_llm(system, state["task"])
    return state


# --- Researcher agent: real tool use, not a hardcoded call ---
# The model decides whether to search, which tool to use, and with what
# query - the same pattern as a weather/calculator tool-use agent, just
# pointed at a document corpus instead of external APIs.

RESEARCHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search the corpus by meaning, for conceptual or natural-language questions.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Natural language search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Search the corpus for an exact word or phrase. Use this for precise terms, names, or methods rather than broad concepts.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string", "description": "Exact word or phrase to look for"}},
                "required": ["keyword"],
            },
        },
    },
]


def researcher_agent(state: PipelineState) -> PipelineState:
    mode = state.get("mode")

    if mode == "apply":
        # Apply mode has no corpus to search -- "research" is just the job
        # posting plus the user's resume, both already sitting in state.
        resume_text = state.get("user_profile", {}).get("resume_text", "[resume not yet wired in]")
        state["research"] = (
            f"Job posting:\n{state.get('job_posting_text', '')}\n\nResume:\n{resume_text}"
        )
        state["sources"] = ["job posting", "resume"]
        return state

    # Interview/Learn search their own corpus; anything else (e.g. running
    # orchestrator.py directly with no mode set) falls back to SciFact so
    # your original Day-1 CLI demo still works unchanged.
    corpus = MODE_TO_CORPUS.get(mode, "scifact")

    sources_used: list[str] = []
    collected_results: list[str] = []  # raw tool outputs, for the fallback synthesis call

    def run_tool(name: str, args: dict) -> str:
        if name == "semantic_search":
            docs = semantic_search(args["query"], corpus=corpus)
        elif name == "keyword_search":
            docs = keyword_search(args["keyword"], corpus=corpus)
        else:
            return "Unknown tool."
        print(f"  [Researcher tool call] {name}({args})")
        sources_used.extend(d.metadata.get("title", "untitled") for d in docs)
        result = "\n\n".join(d.page_content for d in docs) or "No results found."
        collected_results.append(f"[{name}({args})]\n{result}")
        return result

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research agent with two tools: semantic_search for conceptual "
                "questions, and keyword_search for exact terms. Use at least one tool to gather "
                "facts relevant to the plan below before answering. Summarize ONLY what your "
                "searches return - do not add outside information."
            ),
        },
        {"role": "user", "content": f"Plan:\n{state['plan']}"},
    ]

    MAX_TOOL_ROUNDS = 3
    round_num = 0

    while round_num < MAX_TOOL_ROUNDS:
        response = _create_completion_with_retry(
            model=MODEL,
            messages=messages,
            tools=RESEARCHER_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            break

        round_num += 1
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = run_tool(call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Research is always built from the RAW retrieved content, never from an
    # LLM's own paraphrase of it (whether the model stopped early or hit the
    # round cap) -- summarizing risks silently dropping specifics (code,
    # exact formulas) while still claiming they exist, which then invites
    # the Writer to fabricate a replacement rather than flag the gap. This
    # also sidesteps the earlier tool_choice="none" crash entirely: there's
    # no longer a "final text turn" being forced on a tool-primed conversation
    # at all, since we don't need the model to produce one.
    if collected_results:
        state["research"] = "\n\n".join(collected_results)
    else:
        state["research"] = "No research produced."
    state["sources"] = sources_used or ["No sources retrieved."]
    return state


WRITER_SYSTEM_PROMPTS = {
    "learn": (
        "You are a learning coach explaining a concept for review. Explain "
        "clearly and helpfully, but every fact, formula, numeric range, and "
        "example must come from the research provided -- including code. If "
        "the research contains or describes actual code, use or closely "
        "adapt THAT code rather than writing a new example from scratch. If "
        "the research only mentions that a code example exists without "
        "giving its content, say so explicitly rather than inventing a "
        "replacement.\n\n"
        "Do not invent numeric ranges or thresholds that aren't in the "
        "research (e.g. don't state a score range like '3-5' unless the "
        "research actually gives that range). If the research is thin on "
        "some sub-topic, say so honestly rather than filling the gap with "
        "outside knowledge. After the explanation, end with 1-2 short quiz "
        "questions grounded in the same research."
    ),
    "interview": (
        "You are a mock-interview quiz agent. Build interview questions and "
        "explanations using material the research step actually returned as "
        "the factual anchor. Do NOT invent specific numbers, named "
        "techniques/formulas, datasets, proofs, or worked examples beyond "
        "what the research contains -- even if you know the general topic "
        "well from training.\n\n"
        "QUESTION STYLE MATTERS: a real interviewer tests whether the "
        "candidate UNDERSTANDS and can REASON about a concept, not whether "
        "they can recall a name or label. If the retrieved material states a "
        "simple fact (e.g. 'the Titanic dataset predicts survival'), do NOT "
        "turn it into a recall question like 'which dataset predicts "
        "survival?'. Instead, ask something that requires reasoning about "
        "that same fact -- e.g. 'why is predicting Titanic survival a "
        "supervised problem, and what would change if survival labels were "
        "missing?', or 'what could go wrong if this dataset were "
        "imbalanced?'. Prefer 'why', 'how', 'what happens if', and 'compare' "
        "framings over 'what is' or 'which ___ is used for'.\n\n"
        "Answering these 'why/how' questions correctly REQUIRES explaining "
        "standard, well-established implications of the stated fact, even "
        "when the research doesn't spell that implication out verbatim -- "
        "this is expected and fine, not a grounding violation. Example: if "
        "the research states 'KNN stores all training instances,' it is "
        "correct and appropriate to explain that this means prediction gets "
        "slower and more memory-hungry as the dataset grows, even though "
        "the research didn't say that explicitly -- this is the standard, "
        "textbook consequence of the stated fact, not an invented one. The "
        "line to hold is: never introduce a NEW specific number, named "
        "technique, dataset, or claim that isn't standard/well-established "
        "and could plausibly be wrong -- but DO explain the well-known "
        "reasoning and implications behind what the research states.\n\n"
        "If the research is thin, tangential, or doesn't cover the requested "
        "topic/difficulty in depth (e.g. a keyword search came back empty, or "
        "the results are about a related-but-different topic), say so "
        "honestly and build a SMALLER or SIMPLER quiz from what IS actually "
        "available, rather than fabricating additional advanced content to "
        "fill the gap. A short, honest, well-grounded quiz beats an elaborate "
        "fabricated one every time.\n\n"
        "DIFFICULTY CALIBRATION: the requested level is given on a "
        "'Difficulty:' line in the task block. Difficulty changes WHAT you "
        "ask and how many questions you ask -- it never loosens any "
        "grounding rule above. A hard quiz is still built only from the "
        "research; it is harder because the questions demand more of the "
        "candidate, not because you invented more material.\n"
        "- easy: 2-3 questions. Single-step reasoning about one stated "
        "fact. 'Why does X behave this way?' Accept a short answer.\n"
        "- medium: 3-4 questions. Multi-step reasoning, or connecting two "
        "facts from the research. 'How would X change if Y were "
        "different?'\n"
        "- hard: 4-5 questions. Trade-off analysis, failure modes, edge "
        "cases, or comparing two approaches the research covers. 'When "
        "would you NOT use X, and what breaks first?' At least one "
        "question should require the candidate to reason about a scenario "
        "the research does not state outright but that follows directly "
        "from what it does state.\n"
        "The question counts above are targets, not quotas. If the "
        "research is too thin to support the requested level honestly, "
        "drop to fewer questions at a lower level -- following the "
        "thin-research rule above -- rather than padding with invented "
        "material to hit a count.\n\n"
        "OUTPUT FORMAT: structure the ENTIRE quiz using exactly this "
        "delimiter block, repeated once per question, with nothing else "
        "outside these blocks (no intro paragraph, no numbered list "
        "wrapper, no closing summary):\n\n"
        "===Q===\n"
        "<the question text -- markdown formatting inside is fine>\n"
        "===A===\n"
        "<the answer and explanation -- markdown formatting inside is fine>\n"
        "===END===\n\n"
        "Repeat this block for each question, one after another, with no "
        "other text before, between, or after the blocks."
    ),
    "apply": (
        "You are a resume-tailoring writing agent. Write clear, well-organized, "
        "achievement-focused resume bullets and application materials using ONLY "
        "the research provided. The research has two distinct sources: a 'Job "
        "posting' section (the EMPLOYER's stated requirements -- NOT facts about "
        "the candidate) and a 'Resume' section (the candidate's actual, verified "
        "experience).\n\n"
        "BEFORE writing any bullets, open with a 'Fit Assessment' section: "
        "list each core/required qualification stated in the Job posting "
        "(not preferred/nice-to-have ones -- focus on what's explicitly "
        "required). For each one, mark it MET, PARTIALLY MET, or NOT MET "
        "based solely on what the Resume section actually shows -- be "
        "honest and direct, do not stretch a tangential skill into a MET. "
        "Then give ONE clear line: how many core requirements are met (e.g. "
        "'3 of 4 core requirements met'), followed by a direct, honest "
        "recommendation -- not a vague hedge. Use judgment and calibrate "
        "the tone to the actual fit, e.g. 'Strong match -- worth applying' "
        "/ 'Partial match -- worth applying if you can speak to the gaps' "
        "/ 'Weak match -- the core requirements aren't reflected in this "
        "resume, consider whether this is the right target.'\n\n"
        "AFTER the Fit Assessment, proceed with the best-effort tailored "
        "bullets as before, using only real, sourced experience. Even if "
        "the fit is weak, still make the best honest case possible from "
        "what IS real -- but the Fit Assessment must come first, and must "
        "NOT be softened just to make the bullets that follow look more "
        "justified.\n\n"
        "When stating the candidate's own years of experience, "
        "skills, or achievements, use ONLY numbers from the Resume section -- "
        "never substitute the job posting's stated requirement (e.g. '5+ years "
        "required') as if it were the candidate's own tenure. State the "
        "candidate's real experience accurately even when it exceeds what the "
        "posting asks for (e.g. if the Resume says '11+ years', write '11+ "
        "years', not the posting's '5+ years').\n\n"
        "The Resume's own summary IS evidence. Where the Resume states a "
        "total -- '11+ years of experience' -- use that figure. Do NOT "
        "recompute tenure from individual role date ranges, and never let an "
        "arithmetic estimate of your own contradict a figure the Resume "
        "states outright.\n\n"
        "UNDER-stating the candidate is as much an error as over-stating "
        "them. The rules above forbid inventing experience; they do not ask "
        "you to withhold experience that is on the page. Marking a "
        "requirement PARTIALLY MET or NOT MET when the Resume plainly "
        "supports it is a false negative, and a false negative here costs "
        "the candidate an application they were qualified for. Be accurate "
        "in both directions.\n\n"
        "CRITICAL: never invent metrics, percentages, or numbers that are not "
        "present in the Resume section. If the Resume has no number for an "
        "achievement, either omit the number entirely or insert a bracketed "
        "placeholder like [quantify: e.g. % improvement] -- do not fabricate a "
        "plausible-sounding statistic.\n\n"
        "This same rule applies to EVERY claim about what the candidate has "
        "DONE -- not only numbers, and not only technologies. Technologies, "
        "tools, hardware, frameworks and methods are covered, and so is "
        "everything about the shape of the work: mentoring, leading, team "
        "size, seniority, on-call ownership, client- or executive-facing "
        "work, industry and domain exposure, certifications, clearances and "
        "education. A responsibility the posting would like to see is not "
        "evidence that the candidate has ever done it.\n\n"
        "Never attach a real, sourced number to a fabricated context (e.g. "
        "taking a real 'sub-100ms inference' claim and inventing that it ran "
        "on 'ARM-based accelerator cards' when the Resume never mentions "
        "ARM), and never reframe an unrelated skill as an equivalent (e.g. "
        "don't offer 'Rust is directly comparable to C/C++' in place of the "
        "actual requirement).\n\n"
        "When the posting asks for something the Resume does not show, you "
        "have exactly TWO honest options: omit it, or name the gap with a "
        "bracketed placeholder -- [not evidenced: mentoring junior "
        "engineers]. Writing it as though it happened is never one of them. "
        "Prefer the placeholder when the posting treats the item as "
        "important: a visible gap the candidate can prepare for is worth "
        "more than a silent omission, and far more than a sentence they "
        "cannot defend in the interview."
    ),
}
DEFAULT_WRITER_PROMPT = "You are a writing agent. Write a clear, well-organized response using the research provided."


def _task_for_prompt(state: PipelineState) -> str:
    """Apply mode's task text IS the job posting -- the user pastes the posting
    and their request together, so state["task"] carries the whole JD inline.
    But `research` already contains that same posting verbatim, which means
    sending state["task"] as-is puts the entire JD into every Writer and Critic
    prompt TWICE.

    On this account's 8,000-token ceiling that duplication isn't a rounding
    error: a full-length posting is ~800 tokens, charged three times per round
    (Writer + two Critic calls), across up to two rounds. Swapping the pasted
    copy for a pointer keeps the user's actual instruction intact while giving
    that budget back to the model's visible output.

    Interview/Learn task text is a single short sentence, so it's returned
    untouched."""
    task = state.get("task", "") or ""
    if state.get("mode") != "apply":
        return task

    posting = state.get("job_posting_text", "") or ""
    if len(posting) > 200 and posting in task:
        return task.replace(posting, "[full job posting is in the Research section below]")

    # The Router's extracted copy didn't match the pasted text verbatim (it
    # normalises whitespace sometimes). Only fall back to a generic instruction
    # when the task is long enough that it's clearly carrying a posting inline
    # -- never discard a genuinely short instruction.
    if len(task) > 1500:
        return ("Assess my fit for the job posting in the Research section below, "
                "then tailor my resume bullets to it.")
    return task


VALID_DIFFICULTIES = ("easy", "medium", "hard")


def _mode_context_for_prompt(state: PipelineState) -> str:
    """Per-mode context the Writer needs that isn't carried by task or research.

    Interview mode's difficulty is extracted by the Router into
    state["difficulty"] and, until now, was never read again by anything --
    writer_agent only ever saw the raw task text, so "quiz me on decision
    trees, make it hard" and "quiz me on decision trees" produced the same
    quiz. The Router's own enum already constrains this to easy/medium/hard,
    but state can also be hand-built (tests, a direct orchestrator call), so
    anything unrecognised falls back to "medium" rather than reaching the
    prompt as an unhandled level.

    Topic is included alongside it because _task_for_prompt can hand back a
    generic instruction, which would otherwise leave the difficulty line
    with no subject attached to it."""
    if state.get("mode") != "interview":
        return ""

    difficulty = str(state.get("difficulty") or "").strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "medium"

    context = f"\nDifficulty: {difficulty}"
    topic = str(state.get("topic") or "").strip()
    if topic:
        context += f"\nTopic: {topic}"
    return context


def writer_agent(state: PipelineState) -> PipelineState:
    system = WRITER_SYSTEM_PROMPTS.get(state.get("mode"), DEFAULT_WRITER_PROMPT)
    feedback = f"\n\nAddress this critique from the last round:\n{state['critique']}" if state.get("critique") else ""
    user = f"Task: {_task_for_prompt(state)}{_mode_context_for_prompt(state)}\n\nResearch:\n{state['research']}{feedback}"
    # Apply mode's input (full resume + JD) alone runs ~4500+ tokens on this
    # account's 8000 TPM per-request ceiling -- 4096 pushed total requests
    # over the limit outright (confirmed: a real request hit 8622). With the
    # duplicated posting now stripped out by _task_for_prompt, ~800 tokens of
    # input came back, so this can afford 2500 rather than 2200 and still sit
    # lower overall than it did before.
    max_tokens = 2500 if state.get("mode") == "apply" else 4096
    draft = _call_llm(system, user, max_completion_tokens=max_tokens)

    # gpt-oss-120b spends hidden chain-of-thought out of this SAME completion
    # budget before it writes anything visible. On a task as long as Apply's
    # (assess every requirement, then write bullets) that reasoning can consume
    # the entire allowance, so `content` comes back empty or as a stub -- which
    # is exactly how a missing Fit Assessment shows up downstream: the UI reads
    # "Requirements Met: N/A" and the Critic REVISEs a draft that was never
    # really written. Retry once with reasoning dialled down, which spends the
    # budget on the answer instead. Only fires on an already-failed call, so
    # the normal path keeps full reasoning quality.
    if state.get("mode") == "apply" and len(draft.strip()) < 200:
        print(f"  [Writer returned {len(draft.strip())} chars -- retrying with reasoning_effort='low']")
        draft = _call_llm(system, user, max_completion_tokens=max_tokens, reasoning_effort="low")

    state["draft"] = draft
    return state


CRITIC_SYSTEM_PROMPTS = {
    "learn": (
        "You are a critic agent reviewing a learning explanation for "
        "GROUNDING, not just clarity or completeness. Check whether every "
        "fact, formula, numeric range, and code example in the draft is "
        "actually traceable to the research -- especially code: if the "
        "draft includes a code example, verify it matches (or is a close "
        "adaptation of) code that appears in the research, rather than "
        "being newly written from general knowledge. Also flag any "
        "invented numeric ranges or thresholds (e.g. a stated score range "
        "like '3-5') that don't appear in the research.\n\n"
        "Reply 'APPROVED' on the first line only if the explanation is "
        "fully grounded; otherwise reply 'REVISE' followed by exactly what "
        "is ungrounded and why."
    ),
    "interview": (
        "You are a critic agent reviewing a mock-interview quiz for GROUNDING, "
        "not just internal correctness. Check each question for genuine "
        "fabrication -- NOT for whether every word traces verbatim to the "
        "research, since these are 'why/how' reasoning questions that are "
        "SUPPOSED to explain implications beyond the bare stated fact.\n\n"
        "FLAG (fabrication): a NEW specific number, named technique/formula, "
        "dataset, proof, or worked example that does not appear in the "
        "research and is not standard, well-established domain knowledge -- "
        "this could plausibly be wrong, so it must trace to the research.\n\n"
        "DO NOT FLAG (expected elaboration): a standard, well-established, "
        "textbook-level explanation of WHY a stated fact matters or what it "
        "implies, even if the research doesn't spell that reasoning out "
        "verbatim. Example: if the research states 'KNN stores all training "
        "instances,' explaining that this means slower predictions and "
        "higher memory use as the dataset grows is correct, standard "
        "knowledge -- not fabrication. If in doubt, ask: could this specific "
        "claim plausibly be WRONG or is it universally agreed-upon domain "
        "knowledge? Only the former should be flagged.\n\n"
        "Reply 'APPROVED' on the first line if no genuine fabrication is "
        "present; otherwise reply 'REVISE' followed by exactly which "
        "questions contain a fabricated specific (not just unexplained "
        "reasoning) and why."
    ),
}
DEFAULT_CRITIC_PROMPT = (
    "You are a critic agent. Review the draft for accuracy, clarity, and completeness. "
    "Reply with 'APPROVED' on the first line if it's good, otherwise 'REVISE' "
    "followed by specific, actionable feedback."
)

# Apply mode's critic runs as two smaller, focused calls instead of one
# combined check. The combined version needed reasoning_effort="low" to fit
# the token budget, which caused it to miss a real error -- a company-wide
# platform statistic from the job posting's About section ("50 million
# daily requests") got attributed to the candidate's personal achievement,
# and the weakened reviewer approved it. Splitting the job means each call
# can run at full "medium" reasoning while still fitting the per-request
# ceiling, since each has a narrower task and needs less reasoning to do it
# thoroughly.

FIT_ASSESSMENT_CRITIC_PROMPT = (
    "OUTPUT CONTRACT -- read this first and obey it exactly:\n"
    "Your reply MUST begin with the literal line 'Final verdict: APPROVED' "
    "or 'Final verdict: REVISE'. That line comes FIRST, before any other "
    "text. Do NOT show your working. Do NOT write 'Step 1', 'Step 2', or "
    "enumerate the items you checked. Do the checking silently and report "
    "only the conclusion. After a REVISE verdict, list ONLY the rows that "
    "are actually wrong, one short line each -- never the ones that were "
    "fine.\n\n"
    "You are a critic checking ONLY the 'Fit Assessment' section of a "
    "resume draft -- ignore everything else in the draft. The research has "
    "two sources: 'Job posting' (employer's ask, NOT facts about the "
    "candidate) and 'Resume' (candidate's real, verified experience).\n\n"
    "For each requirement marked MET or PARTIALLY MET, check: is it "
    "genuinely supported by the Resume section specifically (not just "
    "mentioned in the Job posting)? Does the overall count match the "
    "number of MET/PARTIALLY MET rows? Does the recommendation's tone "
    "match the evidence (a weak match shouldn't sound upbeat)? Missing a "
    "Fit Assessment section entirely is itself a problem.\n\n"
    "Remember: verdict line first, no working shown, problems only."
)

CLAIMS_CRITIC_PROMPT = (
    "OUTPUT CONTRACT -- read this first and obey it exactly:\n"
    "Your reply MUST begin with the literal line 'Final verdict: APPROVED' "
    "or 'Final verdict: REVISE'. That line comes FIRST, before any other "
    "text. Do NOT show your working, and do NOT list the claims you checked "
    "and found fine -- a technology-dense resume has thirty or more valid "
    "claims, and naming them is the single most common way this reply runs "
    "out of room before reaching a verdict. Verify silently; report ONLY "
    "problems, one short line each.\n\n"
    "You check the claims in a resume draft -- ignore the Fit Assessment "
    "section. The research has two sources: 'Job posting' (the employer's "
    "ask, NOT facts about the candidate -- and sometimes facts about the "
    "EMPLOYER, e.g. 'our platform serves 50 million requests daily') and "
    "'Resume' (the candidate's real, verified experience).\n\n"
    "For every claim the draft attributes to THE CANDIDATE personally -- "
    "numbers, named technologies, and equally the things they are said to "
    "have DONE: mentoring, leading, team size, seniority, on-call, "
    "client-facing or industry experience, certifications -- check: is it in "
    "the Resume section specifically, exactly or as an obvious equivalent "
    "(e.g. '2x' = 'doubling')? A related-but-different technology does NOT "
    "count. Watch for: (a) a real number glued to a fabricated context, "
    "(b) a company-wide or platform-wide statistic from the Job posting "
    "presented as the candidate's own, (c) a responsibility the Job posting "
    "asks for, written up as something the candidate has already done.\n\n"
    "A bracketed placeholder -- [not evidenced: ...] or [quantify: ...] -- is "
    "the Writer correctly marking a gap. Never flag one as a problem.\n\n"
    "Remember: verdict line first, no working shown, problems only."
)


def _approved_from_prefix(text: str) -> bool:
    """Verdict parser for the Interview/Learn critics, which are asked to put
    APPROVED or REVISE on the first line.

    A bare `text.strip().upper().startswith("APPROVED")` is too strict for
    this model: it bolds and headers its verdicts constantly, so "**APPROVED**"
    and "### APPROVED" both read as NOT approved. The cost of that false
    negative is a full extra Writer+Critic round plus a UI pill claiming the
    draft was rejected when it wasn't.

    So: look only near the top (a verdict buried 500 chars down isn't a
    verdict), strip everything that isn't a letter or space so markdown
    decoration can't hide the word, and require a standalone APPROVED with no
    REVISE alongside it. Blank/truncated text is NOT approved -- same posture
    as _verdict_from_text."""
    if not text or len(text.strip()) < 3:
        return False
    head = re.sub(r"[^A-Z ]", " ", text.strip().upper()[:200])
    head = f" {head} "
    return " APPROVED " in head and " REVISE " not in head


VERDICT_MARKER = "FINAL VERDICT:"


def _has_verdict(text: str) -> bool:
    """Did this Critic call actually reach a verdict line?

    The Apply critics are contractually required to emit 'Final verdict: ...'.
    A reply without it did not finish -- either it was truncated by
    max_completion_tokens, or it ignored the output contract. Both mean the
    verification is incomplete, which is a different and more serious state
    than a verification that ran and found problems.

    This replaces a length test (`< 10 chars`) that only ever caught a blank
    reply. The real truncation mode is the opposite shape: the Claims check
    enumerating thirty sourced items on a technology-dense resume and being
    severed mid-word at the token ceiling, hundreds of characters long and
    carrying no verdict at all. That sailed straight past a length check."""
    return bool(text) and VERDICT_MARKER in text.upper()


def _verdict_from_text(text: str) -> bool:
    """Shared verdict parser for the 'Final verdict: APPROVED/REVISE' line
    format used by both split Apply-mode calls.

    Fails closed: no verdict marker means NOT approved, treated as a genuine
    failure to verify rather than a silent pass. That matters because the
    marker is also what bounds the search -- without it, `split` returns the
    whole reply and the scan below would match the word 'approved' occurring
    anywhere in the Critic's prose, including inside a sentence describing a
    problem ('this claim was approved incorrectly')."""
    if not _has_verdict(text):
        return False
    verdict_section = text.upper().split(VERDICT_MARKER)[-1]
    return "APPROVED" in verdict_section and "REVISE" not in verdict_section


def critic_agent(state: PipelineState) -> PipelineState:
    if state.get("mode") == "apply":
        user = f"Task: {_task_for_prompt(state)}\n\nResearch:\n{state.get('research', '')}\n\nDraft:\n{state['draft']}"

        # Budgets raised from 700/1000 -- both were hit by real truncation in
        # testing (the Fit check came back completely blank), showing this
        # model's actual output length for even a narrow verification task
        # has more variance than earlier estimates accounted for.
        # The two calls are independent (neither needs the other's result),
        # so run them concurrently instead of sequentially -- same total
        # tokens used, roughly half the wall-clock time for this step.
        with ThreadPoolExecutor(max_workers=2) as executor:
            fit_future = executor.submit(
                _call_llm, FIT_ASSESSMENT_CRITIC_PROMPT, user, max_completion_tokens=1400, reasoning_effort="medium"
            )
            claims_future = executor.submit(
                _call_llm, CLAIMS_CRITIC_PROMPT, user, max_completion_tokens=1800, reasoning_effort="medium"
            )
            fit_result = fit_future.result()
            claims_result = claims_future.result()

        # Make a truncated/empty result visible in the critique text itself,
        # rather than letting it silently resemble a normal REVISE -- an
        # incomplete verification is a different, more serious problem than
        # a verification that ran and found issues.
        # Flag on the ABSENCE OF A VERDICT, not on length. A reply can be
        # 1,800 tokens of diligent enumeration and still never reach its
        # verdict line -- that is the truncation mode actually observed on a
        # technology-dense resume, and a length test misses it entirely,
        # letting an unfinished check read downstream like a normal REVISE.
        incomplete = " [INCOMPLETE -- no verdict line; the check was cut off before finishing]"
        fit_note = "" if _has_verdict(fit_result) else incomplete
        claims_note = "" if _has_verdict(claims_result) else incomplete

        state["critique"] = (
            f"[Fit Assessment Check]{fit_note}\n{fit_result}\n\n"
            f"[Numbers & Technical Claims Check]{claims_note}\n{claims_result}"
        )
        state["approved"] = _verdict_from_text(fit_result) and _verdict_from_text(claims_result)
    else:
        system = CRITIC_SYSTEM_PROMPTS.get(state.get("mode"), DEFAULT_CRITIC_PROMPT)
        user = f"Task: {_task_for_prompt(state)}\n\nResearch:\n{state.get('research', '')}\n\nDraft:\n{state['draft']}"
        result = _call_llm(system, user, max_completion_tokens=4096, reasoning_effort="medium")
        state["critique"] = result
        state["approved"] = _approved_from_prefix(result)

    state["revision_count"] = state.get("revision_count", 0) + 1
    return state
