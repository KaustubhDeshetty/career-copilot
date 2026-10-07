"""
RAG setup, isolated from the agent logic on purpose.

Extended for the Career Copilot: three separate corpora, one per mode
that actually needs retrieval (Interview and Learn -- Apply mode doesn't
search a corpus at all, see agents.py). The Researcher agent picks a
corpus by passing corpus="interview_qa" | "class_notes" | "scifact" into
semantic_search/keyword_search -- everything else about those two
functions' shape is unchanged, so agents.py barely has to know this
happened.
"""

import os
import json
from functools import lru_cache
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCIFACT_SIZE = 5183
NOTES_DIR = "./class_notes"  # drop your .ipynb files here for Learn mode


# ---------- corpus-specific document loaders ----------

@lru_cache(maxsize=1)
def _load_scifact_docs():
    print(f"Loading {SCIFACT_SIZE} documents from SciFact...")
    raw = load_dataset("BeIR/scifact", "corpus", split="corpus")
    subset = raw.select(range(min(SCIFACT_SIZE, len(raw))))
    return [
        Document(
            page_content=f"{row.get('title', '')}\n{row.get('text', '')}",
            metadata={"title": row.get("title", "untitled")},
        )
        for row in subset
    ]


@lru_cache(maxsize=1)
def _load_interview_qa_docs():
    print("Loading interview Q&A corpus (mjphayes/machine_learning_questions)...")
    raw = load_dataset("mjphayes/machine_learning_questions", split="train")
    return [
        Document(
            page_content=f"Q: {row['question']}\nA: {row['answer']}",
            metadata={"title": row["question"][:60]},
        )
        for row in raw
    ]


def _parse_notebook_into_chunks(path: str) -> list[dict]:
    """Splits a .ipynb file into chunks, one per markdown header section --
    each chunk carries that header's markdown text plus any code cells up
    to the next header. No nbformat dependency: .ipynb is just JSON."""
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    chunks = []
    current_header = "Untitled"
    current_text: list[str] = []

    def flush():
        if current_text:
            chunks.append({"title": current_header, "text": "\n".join(current_text).strip()})

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            for line in source.split("\n"):
                if line.strip().startswith("#"):
                    flush()
                    current_header = line.strip("# ").strip()
                    current_text = []
                else:
                    current_text.append(line)
        elif cell.get("cell_type") == "code":
            current_text.append(f"```python\n{source}\n```")

    flush()
    return chunks


@lru_cache(maxsize=1)
def _load_class_notes_docs():
    if not os.path.isdir(NOTES_DIR):
        print(f"No class notes found at {NOTES_DIR} -- Learn mode corpus will be empty until you add .ipynb files there.")
        return []

    docs = []
    for fname in sorted(os.listdir(NOTES_DIR)):
        if not fname.endswith(".ipynb"):
            continue
        for chunk in _parse_notebook_into_chunks(os.path.join(NOTES_DIR, fname)):
            if chunk["text"]:
                docs.append(Document(
                    page_content=f"{chunk['title']}\n{chunk['text']}",
                    metadata={"title": chunk["title"], "source": fname},
                ))
    print(f"Loaded {len(docs)} chunks from {NOTES_DIR}")
    return docs


_CORPUS_LOADERS = {
    "scifact": _load_scifact_docs,
    "interview_qa": _load_interview_qa_docs,
    "class_notes": _load_class_notes_docs,
}

# Which corpus each mode searches. "apply" is deliberately absent --
# Apply mode's Researcher reads state directly instead, see agents.py.
MODE_TO_CORPUS = {
    "interview": "interview_qa",
    "learn": "class_notes",
}


# ---------- retrieval, parameterized by corpus ----------

@lru_cache(maxsize=8)
def _get_retriever(corpus: str, k: int = 4):
    if corpus not in _CORPUS_LOADERS:
        raise ValueError(f"Unknown corpus '{corpus}'. Options: {list(_CORPUS_LOADERS)}")

    persist_dir = f"./chroma_db_{corpus}"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Loading existing '{corpus}' vector index from disk (instant)...")
        store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return store.as_retriever(search_kwargs={"k": k})

    docs = _CORPUS_LOADERS[corpus]()
    if not docs:
        return None
    print(f"Embedding {len(docs)} '{corpus}' documents (only happens once - future runs load instantly)...")
    store = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    print(f"'{corpus}' index built and saved to disk.")
    return store.as_retriever(search_kwargs={"k": k})


def semantic_search(query: str, corpus: str = "scifact", limit: int = 4) -> list[Document]:
    """Vector similarity search - good for conceptual, natural-language questions."""
    retriever = _get_retriever(corpus, k=limit)
    return retriever.invoke(query) if retriever else []


def keyword_search(keyword: str, corpus: str = "scifact", limit: int = 4) -> list[Document]:
    """Plain substring search over titles/text - good for exact terms or names.
    No embeddings involved - a genuinely different retrieval strategy,
    not just semantic_search under a different name."""
    docs = _CORPUS_LOADERS[corpus]()
    needle = keyword.lower()
    matches = [d for d in docs if needle in d.page_content.lower()]
    return matches[:limit]
