"""
analyze_resume_overlap.py
Measures how different the flavoured resumes actually are from each other.

WHY THIS EXISTS
The Selector has to rank N resumes against a job posting. The cheapest possible
ranker is embedding similarity -- and the project already has Chroma and
sentence-transformers wired up, so it would cost nothing to reuse.

But that only works if the resumes are actually far apart in vector space. If
they are the same career reframed five ways -- same employers, same timeline,
different emphasis -- they will sit almost on top of each other and the ranking
will be noise dressed up as a score. This script measures that before any
design decision gets made on a guess.

WHAT IT REPORTS
  - Lexical overlap  (TF-IDF cosine): do they use the same words?
  - Semantic overlap (embeddings, if sentence-transformers is installed):
    do they mean the same thing?
  - Distinctive terms per resume: what does each one actually foreground?
  - A verdict on whether similarity-based ranking is viable.

USAGE
    python analyze_resume_overlap.py resumes/*.txt
    python analyze_resume_overlap.py mlops.txt genai.txt cloud_ai.txt

No API key needed. Nothing leaves the machine.
"""

import sys
import os
import re
from itertools import combinations


# --- thresholds ------------------------------------------------------------
# Calibrated on the practical question "can a ranker tell these apart?" rather
# than on any statistical convention. A pair above HIGH is effectively a
# coin-flip for a similarity ranker; below LOW the signal is real.
HIGH_SIMILARITY = 0.80   # too close for similarity ranking to discriminate
LOW_SIMILARITY = 0.55    # comfortably distinct


def load(paths):
    docs = {}
    for p in paths:
        if not os.path.isfile(p):
            print(f"  skipping {p} (not a file)")
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if not text:
            print(f"  skipping {p} (empty)")
            continue
        docs[os.path.splitext(os.path.basename(p))[0]] = text
    return docs


def lexical_matrix(docs):
    """TF-IDF cosine. Answers: do these resumes use the same vocabulary?"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    names = list(docs)
    # sublinear_tf dampens the effect of a term repeated many times in one
    # resume, which matters here: a skills table can list "Python" five times
    # and swamp the signal from everything else.
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
    X = vec.fit_transform([docs[n] for n in names])
    return names, cosine_similarity(X), vec, X


def semantic_matrix(docs):
    """Embedding cosine, using the same model the RAG pipeline uses so the
    numbers are directly comparable to what a Chroma-based ranker would see."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return None, None

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    names = list(docs)
    emb = model.encode([docs[n] for n in names], normalize_embeddings=True)
    return names, np.asarray(emb) @ np.asarray(emb).T


def distinctive_terms(names, vec, X, top=10):
    """The highest-TF-IDF terms per document -- i.e. what this resume
    foregrounds that the others do not. This is the qualitative half: two
    resumes can score 0.9 similar overall and still lead with entirely
    different skills, which is exactly what an LLM ranker can see and a
    cosine score cannot."""
    import numpy as np
    features = np.array(vec.get_feature_names_out())
    out = {}
    for i, n in enumerate(names):
        row = X[i].toarray().ravel()
        out[n] = [features[j] for j in row.argsort()[::-1][:top] if row[j] > 0]
    return out


def print_matrix(title, names, M):
    print(f"\n{title}")
    w = max(len(n) for n in names) + 2
    print(" " * w + "".join(f"{n[:10]:>12}" for n in names))
    for i, n in enumerate(names):
        cells = "".join(f"{M[i][j]:>12.2f}" for j in range(len(names)))
        print(f"{n:<{w}}{cells}")


def verdict(names, M, label):
    pairs = [(names[i], names[j], M[i][j]) for i, j in combinations(range(len(names)), 2)]
    pairs.sort(key=lambda p: -p[2])

    print(f"\n{label} — pairwise, most similar first:")
    for a, b, s in pairs:
        flag = "  <-- too close to rank apart" if s >= HIGH_SIMILARITY else (
               "  <-- distinct" if s < LOW_SIMILARITY else "")
        print(f"  {a:<18} vs {b:<18} {s:.3f}{flag}")

    worst = pairs[0][2] if pairs else 0.0
    mean = sum(p[2] for p in pairs) / len(pairs) if pairs else 0.0
    print(f"\n  highest pair: {worst:.3f}   mean: {mean:.3f}")
    return worst, mean


def main(paths):
    docs = load(paths)
    if len(docs) < 2:
        print("Need at least two resume files to compare.")
        return 1

    print(f"Loaded {len(docs)} resumes:")
    for n, t in docs.items():
        print(f"  {n:<20} {len(t):>6} chars  ~{len(t)//4:>5} tokens")

    names, L, vec, X = lexical_matrix(docs)
    print_matrix("LEXICAL similarity (TF-IDF cosine) — shared vocabulary", names, L)
    lex_worst, lex_mean = verdict(names, L, "Lexical")

    sem_worst = None
    snames, S = semantic_matrix(docs)
    if S is not None:
        print_matrix("SEMANTIC similarity (all-MiniLM-L6-v2) — shared meaning", snames, S)
        sem_worst, _ = verdict(snames, S, "Semantic")
    else:
        print("\n(sentence-transformers not installed here — semantic scores skipped.")
        print(" Run this on the project machine to get them; that is the number")
        print(" that actually predicts whether a Chroma-based ranker would work.)")

    print("\n" + "=" * 68)
    print("WHAT EACH RESUME FOREGROUNDS (highest TF-IDF terms)")
    print("=" * 68)
    for n, terms in distinctive_terms(names, vec, X).items():
        print(f"  {n:<20} {', '.join(terms)}")

    # The semantic number is the one that matters for an embedding ranker;
    # fall back to lexical only when it is unavailable.
    deciding = sem_worst if sem_worst is not None else lex_worst
    basis = "semantic" if sem_worst is not None else "lexical (semantic unavailable)"

    print("\n" + "=" * 68)
    print(f"VERDICT  (on {basis} similarity, closest pair = {deciding:.3f})")
    print("=" * 68)
    if deciding >= HIGH_SIMILARITY:
        print("""
  SIMILARITY RANKING IS NOT VIABLE on its own.

  The closest pair is near-identical to a vector ranker. Cosine scores would
  differ in the third decimal place, which is noise, not signal -- the ranker
  would look confident and be arbitrary.

  Use an LLM Selector reading short per-resume summaries instead. It can
  reason about emphasis and framing, which is precisely what differs here and
  precisely what cosine similarity cannot see.
""")
    elif deciding >= LOW_SIMILARITY:
        print("""
  SIMILARITY RANKING IS A USABLE PRE-FILTER, NOT A DECIDER.

  There is real separation, but not enough to trust a top-1 pick. Reasonable
  design: use similarity to drop the clearly-irrelevant resumes, then have the
  LLM Selector choose among the survivors.
""")
    else:
        print("""
  SIMILARITY RANKING IS VIABLE.

  These resumes are genuinely distinct. Embedding similarity alone would give
  a defensible ranking, and the LLM Selector becomes an explainability layer
  on top rather than the thing doing the work.
""")

    print("  Either way the LLM Selector is still worth building: the")
    print("  apply / stretch / skip verdict and the portfolio gap analysis")
    print("  need reasoning, not a distance metric.\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1:]))
