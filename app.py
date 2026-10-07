"""
app.py
Streamlit wrapper around orchestrator.run_pipeline() -- enterprise-styled
version. Still zero new backend logic: everything here is presentation.

Run with:  streamlit run app.py
"""

import re
import pandas as pd
import streamlit as st
from orchestrator import run_pipeline
from tracker_utils import (
    extract_company_role,
    extract_fit_summary,
    save_application,
    save_interview_session,
    load_applications,
    load_interview_sessions,
    application_exists,
    interview_session_exists,
)

st.set_page_config(page_title="Career Copilot", page_icon=None, layout="centered")

# ---------- Styling ----------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding-bottom: 16px;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 28px;
    }
    .main-header .mark {
        width: 40px; height: 40px; border-radius: 9px;
        background: linear-gradient(135deg, #2563EB, #1E3A8A);
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 18px; flex-shrink: 0;
    }
    .main-header h1 { font-size: 22px; font-weight: 700; color: #0F172A; margin: 0; }
    .main-header p { font-size: 13px; color: #64748B; margin: 2px 0 0 0; }

    .mode-pill, .status-pill {
        display: inline-block; padding: 5px 14px; border-radius: 999px;
        font-size: 12px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.04em; margin-bottom: 4px;
    }
    .mode-apply { background: #DBEAFE; color: #1E40AF; }
    .mode-interview { background: #EDE9FE; color: #5B21B6; }
    .mode-learn { background: #CCFBF1; color: #0F766E; }

    .status-strong { background: #DCFCE7; color: #166534; }
    .status-partial { background: #FEF3C7; color: #92400E; }
    .status-weak { background: #FEE2E2; color: #991B1B; }
    .status-approved { background: #DCFCE7; color: #166534; }
    .status-revise { background: #FEF3C7; color: #92400E; }

    .result-card {
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 24px; margin-top: 12px; margin-bottom: 12px;
    }

    .qa-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 18px 22px; margin-bottom: 14px;
    }
    .qa-number {
        display: inline-block; background: #EDE9FE; color: #5B21B6;
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; padding: 3px 10px; border-radius: 999px;
        margin-bottom: 10px;
    }
    .qa-question { font-size: 15px; font-weight: 500; color: #0F172A; margin-bottom: 10px; }

    div.stButton > button {
        background: #2563EB; color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.5rem;
    }
    div.stButton > button:hover { background: #1E40AF; color: white; }
    </style>

    <div class="main-header">
        <div class="mark">CC</div>
        <div>
            <h1>Career Copilot</h1>
            <p>Multi-agent system for job applications, interview prep, and concept review</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        """
**Apply** -- paste a full job posting, get a Fit Assessment plus tailored
resume bullets. *(Requires a resume already saved via
`python memory_utils.py resume.txt`.)*

**Interview** -- ask to be quizzed on a topic, e.g. *"Quiz me on
precision and recall."*

**Learn** -- ask to review a concept from your class notes, e.g.
*"Explain TF-IDF."*
"""
    )
    st.caption(
        "Measured run times: Interview ~70-85s, Learn ~1-2.5 min (the first "
        "run also loads the embedding model), Apply ~4-5 min because it makes "
        "two Writer/Critic rounds over a full resume and posting."
    )

# ---------- Helpers ----------

def _parse_qa_pairs(draft: str):
    """Splits an Interview-mode draft into (question, answer) pairs, using
    the ===Q===/===A===/===END=== delimiter format the Writer is instructed
    to produce. Returns an empty list if the format isn't found, so callers
    can fall back to showing the raw draft instead of showing nothing."""
    pattern = r"===Q===\s*(.*?)\s*===A===\s*(.*?)\s*===END==="
    matches = re.findall(pattern, draft, re.DOTALL)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in matches]


def _parse_fit_ratio(fit_score: str):
    match = re.search(r"(\d+)\s+of\s+(\d+)", fit_score)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _short_verdict(recommendation: str) -> str:
    """The status pill is a rounded, uppercase, letter-spaced lozenge sized for
    a two-word label -- but a real recommendation from the Writer runs well
    over 100 characters and wraps into a large blob. Pull just the leading
    verdict phrase for the pill; the full sentence still shows underneath."""
    match = re.match(r"\s*((?:strong|partial|weak|good|poor)[a-z ]*match)", recommendation, re.IGNORECASE)
    if match:
        return match.group(1).strip().title()
    return (recommendation[:38] + "...") if len(recommendation) > 41 else recommendation


def _fit_status_class(met: int, total: int) -> str:
    if total == 0:
        return "status-partial"
    ratio = met / total
    if ratio >= 0.75:
        return "status-strong"
    if ratio >= 0.4:
        return "status-partial"
    return "status-weak"


def _show_table(headers, rows, empty_msg: str):
    """Renders a tracker sheet as a table, or an honest empty state."""
    if not rows:
        st.caption(empty_msg)
        return
    df = pd.DataFrame(rows, columns=headers)
    # width="stretch" superseded use_container_width. The old kwarg still works
    # but prints a two-line deprecation notice on EVERY rerun -- twice per call,
    # once per tracker tab -- which buries the [Router] and [Researcher tool
    # call] lines in the terminal. Newest API first, then the old one, then
    # bare: renders on any version, noisy on none.
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        try:
            st.dataframe(df, use_container_width=True, hide_index=True)
        except TypeError:
            # hide_index landed in later Streamlit versions; fall back rather
            # than crashing the page on an older install.
            st.dataframe(df)


MODE_LABELS = {"apply": "Apply", "interview": "Interview", "learn": "Learn"}
MODE_CLASSES = {"apply": "mode-apply", "interview": "mode-interview", "learn": "mode-learn"}

# ---------- Main input ----------

task = st.text_area(
    "What do you need help with?",
    height=150,
    placeholder=(
        "e.g. \"Here's a job posting: ... Can you help me tailor my resume bullets?\"\n"
        "or \"Quiz me on decision trees, make it hard\"\n"
        "or \"Can you help me review TF-IDF today?\""
    ),
)

run_clicked = st.button("Run", type="primary", disabled=not task.strip())

if run_clicked:
    with st.spinner("Running the pipeline..."):
        try:
            st.session_state["result"] = run_pipeline(task)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

if "result" in st.session_state:
    result = st.session_state["result"]
    mode = result.get("mode", "unknown")

    mode_class = MODE_CLASSES.get(mode, "mode-learn")
    mode_label = MODE_LABELS.get(mode, mode.title())
    st.markdown(f'<span class="mode-pill {mode_class}">{mode_label} Mode</span>', unsafe_allow_html=True)

    fit_score, recommendation = None, None
    if mode == "apply":
        fit_score, recommendation = extract_fit_summary(result.get("draft", ""))
        ratio = _parse_fit_ratio(fit_score)
        status_class = _fit_status_class(*ratio) if ratio else "status-partial"

        if fit_score == "Unknown" and recommendation == "Unknown":
            st.warning(
                "No Fit Assessment found in the draft. That usually means the "
                "Writer's completion budget was consumed by hidden reasoning "
                "before it wrote anything visible -- check the terminal for a "
                "`finish_reason='length'` warning, and the Research tab to "
                "confirm the full job posting actually made it through."
            )

        col1, col2 = st.columns([1, 2])
        with col1:
            if ratio:
                st.metric("Requirements Met", f"{ratio[0]} / {ratio[1]}")
            else:
                st.metric("Requirements Met", "N/A")
        with col2:
            st.markdown(
                f'<div style="padding-top: 8px;"><span class="status-pill {status_class}">'
                f'{_short_verdict(recommendation)}</span></div>',
                unsafe_allow_html=True,
            )
            if recommendation and recommendation != _short_verdict(recommendation):
                st.caption(recommendation)
        st.markdown("<br>", unsafe_allow_html=True)

    approval_class = "status-approved" if result.get("approved") else "status-revise"
    approval_label = (
        f"Approved after {result.get('revision_count', 1)} round(s)"
        if result.get("approved")
        else f"Not approved after {result.get('revision_count', 0)} round(s) -- showing best draft"
    )
    st.markdown(f'<span class="status-pill {approval_class}">{approval_label}</span>', unsafe_allow_html=True)

    qa_pairs = []
    if mode == "interview":
        qa_pairs = _parse_qa_pairs(result.get("draft", ""))
        if qa_pairs:
            for i, qa in enumerate(qa_pairs, start=1):
                # The question text is markdown (the Writer prompt explicitly
                # allows it), and markdown is NOT parsed inside a block-level
                # HTML element -- so interpolating it into the card div renders
                # "**bold**" and backticks literally. Emit the card chrome as
                # HTML, then let Streamlit render the question itself.
                with st.container(border=True):
                    st.markdown(
                        f'<span class="qa-number">Question {i}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(qa["question"])
                    with st.expander("Show Answer"):
                        st.markdown(qa["answer"])
        else:
            # Writer didn't follow the delimiter format this run -- fall
            # back to the plain draft rather than showing nothing.
            with st.container(border=True):
                st.markdown(result.get("draft") or "_No draft produced -- the Writer returned nothing._")
    else:
        # Each st.markdown call is its own block, so an opening <div> in one
        # call and a closing </div> in another never wrap the content between
        # them -- Streamlit auto-closes the orphan and the card styling is
        # silently lost. st.container(border=True) actually draws the card.
        with st.container(border=True):
            st.markdown(result.get("draft") or "_No draft produced -- the Writer returned nothing._")

    tab_labels = ["Plan", "Research", "Critique", "Sources"]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        st.markdown(result.get("plan", ""))
    with tabs[1]:
        st.text(result.get("research", ""))
    with tabs[2]:
        st.markdown(result.get("critique", ""))
    with tabs[3]:
        st.write(result.get("sources", []))

    if mode == "apply":
        with st.container(border=True):
            role, company = extract_company_role(result.get("job_posting_text", ""))
            st.markdown(f"**Detected:** {role} at {company}")
            # Guard against writing the same application twice. session_state
            # alone caught the double-click but reset on every page refresh, so
            # refresh-then-Save still wrote a duplicate; the workbook check is
            # what actually survives a refresh or a brand-new session.
            saved_key = f"saved::{company}::{role}"
            already_saved = st.session_state.get(saved_key) or application_exists(company, role)
            if already_saved:
                st.success("Already saved to applications.xlsx")
            elif st.button("Save to application tracker"):
                try:
                    # The Critic's verdict has to travel with the row. The
                    # loop can exit at the revision cap with objections still
                    # outstanding, and the draft shown is then the best one --
                    # not an approved one. Logging its recommendation bare
                    # turns "the Critic rejected this" into a tracker row that
                    # reads exactly like a verified match. Same shape as Bug 2:
                    # the verification result existed and never reached the
                    # place that needed it.
                    save_application(
                        company, role, fit_score, recommendation,
                        notes="" if result.get("approved") else (
                            f"UNVERIFIED -- Critic still had objections after "
                            f"{result.get('revision_count', 0)} revision round(s)"
                        ),
                    )
                    st.session_state[saved_key] = True
                    st.success("Saved to applications.xlsx")
                except PermissionError:
                    st.error(
                        "Can't write applications.xlsx -- it's open in Excel. "
                        "Close the file and click Save again."
                    )

    if mode == "interview":
        with st.container(border=True):
            topic = result.get("topic") or "Unknown topic"
            difficulty = result.get("difficulty") or "medium"
            verdict = "Approved" if result.get("approved") else "Revise"
            rounds = result.get("revision_count", 0)
            st.markdown(
                f"**Detected:** {topic} · {difficulty} · "
                f"{len(qa_pairs)} question(s) · Critic: {verdict}"
            )
            # Same duplicate guard as Apply -- one practice session, one row.
            # Note the workbook check deliberately ignores `rounds`: the same
            # topic re-run at the same difficulty is the same practice session
            # even if the Critic happened to take a different number of rounds.
            practice_key = f"logged::{topic}::{difficulty}::{rounds}"
            already_logged = (
                st.session_state.get(practice_key)
                or interview_session_exists(topic, difficulty)
            )
            if already_logged:
                st.success("Already saved to applications.xlsx")
            elif st.button("Save to practice log"):
                try:
                    save_interview_session(topic, difficulty, len(qa_pairs), verdict, rounds)
                    st.session_state[practice_key] = True
                    st.success("Saved to applications.xlsx")
                except PermissionError:
                    st.error(
                        "Can't write applications.xlsx -- it's open in Excel. "
                        "Close the file and click Save again."
                    )

# ---------- Tracker (always visible, independent of the current run) ----------
# Reads straight off the workbook rather than session state, so it shows the
# full history from previous sessions too -- and refreshes on the rerun that
# follows a Save click, so a new row appears immediately.

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Tracker", expanded=False):
    tracker_tabs = st.tabs(["Applications", "Interview practice"])
    with tracker_tabs[0]:
        headers, rows = load_applications()
        _show_table(headers, rows, "No applications saved yet.")
    with tracker_tabs[1]:
        headers, rows = load_interview_sessions()
        _show_table(headers, rows, "No practice sessions saved yet.")
