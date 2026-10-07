import json
import math
import re
from collections import Counter
from pathlib import Path
import time

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# =========================================================
# SETTINGS
# =========================================================

SUPPORTED_SUBJECTS = [
    "Physics",
    "Chemistry",
    "Biology",
    "Computer",
]

SUBJECT_KEYS = {
    "Physics": "phy",
    "Chemistry": "chem",
    "Biology": "bio",
    "Computer": "computer",
}

SUBJECT_LABELS = {
    "phy": "Physics",
    "chem": "Chemistry",
    "bio": "Biology",
    "computer": "Computer",
}

SUBJECT_META = {
    "Physics": {
        "icon": "⚛️",
        "color": "#6366F1",
        "prompts": [
            "Explain Newton's laws of motion with examples",
            "What is the work-energy theorem?",
            "State Kepler's laws of planetary motion",
        ],
    },
    "Chemistry": {
        "icon": "🧪",
        "color": "#EC4899",
        "prompts": [
            "What are the postulates of Bohr's atomic model?",
            "Explain periodic trends in ionization enthalpy",
            "What is Hess's law of constant heat summation?",
        ],
    },
    "Biology": {
        "icon": "🧬",
        "color": "#10B981",
        "prompts": [
            "Explain the fluid mosaic model of plasma membrane",
            "What are the key stages of mitosis?",
            "Describe the light reaction in photosynthesis",
        ],
    },
    "Computer": {
        "icon": "💻",
        "color": "#06B6D4",
        "prompts": [
            "Explain binary search algorithm and its complexity",
            "What are tuples vs lists in Python?",
            "How does bubble sort work step-by-step?",
        ],
    },
}

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 6

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Small + fast model used ONLY to understand/rewrite the student's question.
# If it is unavailable the app silently falls back to rule-based rewriting.
QUERY_REWRITE_MODEL = "llama-3.1-8b-instant"

# Optional cross-encoder reranker (downloaded automatically on first run).
# If it cannot be loaded the app silently falls back to hybrid ranking.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 30

MAX_CONTEXT_CHARS = 14000

REFUSAL_PHRASE = "couldn't find this information"

STOP_WORDS = {
    "what", "what's", "whats", "is", "are", "was", "were", "the", "a", "an",
    "of", "to", "in", "on", "for", "from", "how", "why", "when", "where",
    "which", "who", "whom", "does", "do", "did", "can", "could", "would",
    "should", "explain", "define", "definition", "describe", "state", "tell",
    "me", "about", "with", "and", "or", "please", "give", "meaning", "mean",
    "means", "write", "short", "note", "notes", "brief", "briefly", "detail",
    "details", "detailed", "understand", "know", "you", "your", "i", "my",
    "it", "its", "this", "that", "these", "those", "be", "been", "by", "as",
    "at", "any", "some", "also", "between", "difference", "name", "list",
}


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant | Class 11",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# =========================================================
# OPTIONAL RERANKER (cross-encoder)
# =========================================================

@st.cache_resource
def load_reranker():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(RERANK_MODEL, max_length=512)
    except Exception:
        return None


reranker = load_reranker()


# =========================================================
# GROQ
# =========================================================

@st.cache_resource
def load_groq_client():

    api_key = st.secrets["GROQ_API_KEY"]

    return Groq(api_key=api_key)


groq_client = load_groq_client()


# =========================================================
# LOAD VECTORSTORE
# =========================================================

@st.cache_resource
def load_subject_index(subject):

    subject_dir = VECTORSTORE_DIR / subject

    index_path = subject_dir / "index.faiss"
    metadata_path = subject_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


# =========================================================
# GET PAGE NUMBER
# Compatible with old + new metadata formats
# =========================================================

def get_page_number(item):

    # New Computer metadata format
    if "page" in item:
        return item["page"]

    # Existing metadata format
    metadata = item.get("metadata", {})

    if isinstance(metadata, dict):
        return metadata.get("page", "Unknown")

    return "Unknown"


# =========================================================
# TEXT HELPERS
# =========================================================

def _normalize_words(text):
    """Return normalized alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def _stem(word):
    """Very light plural stemming (forces -> force, laws -> law)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"

    if (
        len(word) > 3
        and word.endswith("s")
        and not word.endswith("ss")
        and not word.endswith("us")
        and not word.endswith("is")
    ):
        return word[:-1]

    return word


def _tokenize(text):
    """Tokens for lexical search: lowercase, stop words removed, stemmed."""
    return [
        _stem(w)
        for w in _normalize_words(text)
        if w not in STOP_WORDS and len(w) >= 2
    ]


def _concept_words(question):
    """Important words of a question (stop words removed)."""
    return [
        w
        for w in _normalize_words(question)
        if len(w) >= 3 and w not in STOP_WORDS
    ]


# =========================================================
# BM25 LEXICAL INDEX (built once per subject)
# =========================================================

@st.cache_resource
def load_bm25(subject):

    _, metadata = load_subject_index(subject)

    docs = [_tokenize(item.get("text", "") or "") for item in metadata]

    doc_freq = Counter()
    for doc in docs:
        doc_freq.update(set(doc))

    total_len = sum(len(doc) for doc in docs)
    avg_len = (total_len / len(docs)) if docs and total_len else 1.0

    return {
        "tfs": [Counter(doc) for doc in docs],
        "lengths": [len(doc) for doc in docs],
        "doc_freq": doc_freq,
        "avg_len": avg_len,
        "n_docs": len(docs),
    }


def bm25_scores(bm25, query_terms, k1=1.5, b=0.75):
    """Return {doc_index: bm25_score} for every doc that matches a term."""

    scores = {}
    n_docs = bm25["n_docs"]

    for term in query_terms:

        df = bm25["doc_freq"].get(term, 0)

        if df == 0:
            continue

        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        for idx, tf_counter in enumerate(bm25["tfs"]):

            tf = tf_counter.get(term, 0)

            if tf == 0:
                continue

            length = bm25["lengths"][idx]

            denom = tf + k1 * (1 - b + b * length / bm25["avg_len"])

            scores[idx] = scores.get(idx, 0.0) + idf * (tf * (k1 + 1)) / denom

    return scores


# =========================================================
# QUERY UNDERSTANDING
# Turns ANY phrasing into several meaning-based search queries
# =========================================================

@st.cache_data(show_spinner=False, ttl=3600)
def understand_query(question, subject_label):
    """
    Returns a dict:
      standalone_query      - short, clean search query
      keywords              - important terms + synonyms / related terms
      hypothetical_passage  - short textbook-style answer (HyDE)

    Uses a small LLM when available, and always has a rule-based fallback,
    so retrieval never depends on the student's exact wording.
    """

    concept_words = _concept_words(question)
    concept_query = " ".join(concept_words).strip()

    result = {
        "standalone_query": concept_query or question,
        "keywords": concept_words,
        "hypothetical_passage": "",
    }

    prompt = f"""You are a search-query optimizer for a Class 11 {subject_label} textbook retrieval system.

Student question: "{question}"

Return ONLY a JSON object (no markdown, no explanation) with exactly these keys:
- "standalone_query": a short search query (2-8 words) capturing the core concept asked about.
- "keywords": a list of 4-10 important textbook terms: the main concept, its synonyms, closely related terms, and formula/law names a textbook would use.
- "hypothetical_passage": 2-3 sentences written in textbook style that would answer the question.
"""

    try:
        response = groq_client.chat.completions.create(
            model=QUERY_REWRITE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )

        raw = (response.choices[0].message.content or "").strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            data = json.loads(match.group(0))

            standalone = str(data.get("standalone_query", "")).strip()
            keywords = data.get("keywords", [])
            passage = str(data.get("hypothetical_passage", "")).strip()

            if standalone:
                result["standalone_query"] = standalone

            if isinstance(keywords, list):
                clean_keywords = []
                for kw in keywords:
                    clean_keywords.extend(_normalize_words(str(kw)))
                result["keywords"] = list(
                    dict.fromkeys(concept_words + clean_keywords)
                )

            if passage:
                result["hypothetical_passage"] = passage

    except Exception:
        pass

    return result


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(
    question,
    subject,
    top_k=TOP_K
):
    """
    Advanced meaning-based hybrid retrieval.

    Pipeline:
    1. Query understanding  -> several rewrites of the question
       (original, concept-only, definition style, LLM standalone query,
       keyword query, hypothetical textbook passage).
    2. Semantic search (FAISS) for ALL rewrites. Each chunk is scored by
       its best AND average similarity, so a strong match for the bare
       concept ("momentum") is never lost because the longer wording
       ("what is momentum?") ranked differently.
    3. BM25 lexical search using the concept + synonym keywords.
    4. Reciprocal Rank Fusion of semantic and lexical rankings.
    5. Optional cross-encoder reranking of the best candidates.
    """

    index, metadata = load_subject_index(subject)

    if index.ntotal == 0 or not metadata:
        return []

    subject_label = SUBJECT_LABELS.get(subject, subject)

    # -----------------------------------------------------
    # 1. Query understanding
    # -----------------------------------------------------

    understood = understand_query(question, subject_label)

    concept_words = _concept_words(question)
    concept_query = " ".join(concept_words).strip()

    standalone_query = understood.get("standalone_query") or concept_query or question
    keywords = understood.get("keywords") or concept_words
    hypothetical = understood.get("hypothetical_passage", "")

    query_texts = [question]

    def add_query(text):
        text = (text or "").strip()
        if text and text not in query_texts:
            query_texts.append(text)

    add_query(concept_query)
    add_query(standalone_query)

    if concept_query:
        add_query(f"definition explanation concept of {concept_query}")

    if keywords:
        add_query(" ".join(keywords[:8]))

    add_query(hypothetical)

    # -----------------------------------------------------
    # 2. Semantic search over all rewrites
    # -----------------------------------------------------

    candidate_k = min(
        max(top_k * 10, 60),
        index.ntotal
    )

    embeddings = embedding_model.encode(
        query_texts,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    all_scores, all_indices = index.search(
        embeddings,
        candidate_k
    )

    score_maps = []

    for row_scores, row_indices in zip(all_scores, all_indices):

        score_map = {}

        for score, idx in zip(row_scores, row_indices):

            if idx == -1:
                continue

            idx = int(idx)

            if idx >= len(metadata):
                continue

            score_map[idx] = float(score)

        if score_map:
            score_maps.append(score_map)

    if not score_maps:
        return []

    floors = [min(m.values()) for m in score_maps]

    semantic_candidates = set()
    for m in score_maps:
        semantic_candidates.update(m)

    semantic_final = {}
    semantic_best = {}

    for idx in semantic_candidates:

        values = []
        present = []

        for m, floor in zip(score_maps, floors):
            if idx in m:
                values.append(m[idx])
                present.append(m[idx])
            else:
                values.append(floor)

        best = max(present) if present else 0.0
        mean = sum(values) / len(values)

        semantic_best[idx] = best
        semantic_final[idx] = 0.5 * best + 0.5 * mean

    # -----------------------------------------------------
    # 3. BM25 lexical search
    # -----------------------------------------------------

    lexical_terms = []

    for term in list(concept_words) + list(keywords):
        for token in _tokenize(term):
            if token not in lexical_terms:
                lexical_terms.append(token)

    bm25 = load_bm25(subject)
    lexical_raw = bm25_scores(bm25, lexical_terms)

    max_lexical = max(lexical_raw.values(), default=0.0)

    lexical_norm = {
        idx: (score / max_lexical if max_lexical > 0 else 0.0)
        for idx, score in lexical_raw.items()
    }

    # -----------------------------------------------------
    # 4. Reciprocal Rank Fusion
    # -----------------------------------------------------

    semantic_ranking = sorted(
        semantic_final,
        key=semantic_final.get,
        reverse=True
    )

    lexical_ranking = sorted(
        lexical_raw,
        key=lexical_raw.get,
        reverse=True
    )[:candidate_k]

    RRF_K = 60

    fused = {}

    for rank, idx in enumerate(semantic_ranking):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, idx in enumerate(lexical_ranking):
        fused[idx] = fused.get(idx, 0.0) + 0.7 / (RRF_K + rank + 1)

    if not fused:
        return []

    # -----------------------------------------------------
    # Relevance gate (very forgiving - the LLM makes the final call)
    # -----------------------------------------------------

    MIN_SEMANTIC_SCORE = 0.15

    best_semantic = max(semantic_best.values(), default=0.0)
    best_lexical = max(lexical_norm.values(), default=0.0)

    if best_semantic < MIN_SEMANTIC_SCORE and best_lexical <= 0.0:
        return []

    pool_indices = sorted(
        fused,
        key=fused.get,
        reverse=True
    )[:max(RERANK_POOL, top_k)]

    # -----------------------------------------------------
    # 5. Optional cross-encoder reranking
    # -----------------------------------------------------

    rerank_scores = {}

    if reranker is not None:

        try:
            rerank_query = standalone_query or question

            pairs = [
                (rerank_query, metadata[idx].get("text", "") or "")
                for idx in pool_indices
            ]

            ce_scores = reranker.predict(pairs)

            for idx, ce in zip(pool_indices, ce_scores):
                rerank_scores[idx] = float(ce)

        except Exception:
            rerank_scores = {}

    if rerank_scores:

        ce_ranking = sorted(
            rerank_scores,
            key=rerank_scores.get,
            reverse=True
        )

        hybrid_ranking = pool_indices

        final_fusion = {}

        for rank, idx in enumerate(ce_ranking):
            final_fusion[idx] = (
                final_fusion.get(idx, 0.0)
                + 1.0 / (RRF_K + rank + 1)
            )

        for rank, idx in enumerate(hybrid_ranking):
            final_fusion[idx] = (
                final_fusion.get(idx, 0.0)
                + 0.6 / (RRF_K + rank + 1)
            )

        final_order = sorted(
            final_fusion,
            key=final_fusion.get,
            reverse=True
        )

    else:
        final_order = pool_indices

    # -----------------------------------------------------
    # Build result objects
    # -----------------------------------------------------

    results = []

    for idx in final_order[:top_k]:

        result = metadata[idx].copy()

        result["score"] = float(fused.get(idx, 0.0))
        result["semantic_score"] = float(semantic_best.get(idx, 0.0))
        result["keyword_score"] = float(lexical_norm.get(idx, 0.0))

        if idx in rerank_scores:
            result["rerank_score"] = float(rerank_scores[idx])

        result["_index"] = int(idx)

        results.append(result)

    return results


# =========================================================
# CONTEXT BUILDER (adds neighbouring chunks for completeness)
# =========================================================

def _page_as_int(page):
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def build_context(results, subject, neighbor_count=3):
    """
    Build textbook context from retrieved chunks.
    The top chunks also pull in their immediate neighbours, because a
    definition is often split across two consecutive chunks.
    """

    _, metadata = load_subject_index(subject)

    selected = {}

    for result in results:
        idx = result.get("_index")
        if idx is not None:
            selected[idx] = metadata[idx]

    for result in results[:neighbor_count]:

        idx = result.get("_index")

        if idx is None:
            continue

        base_page = _page_as_int(get_page_number(metadata[idx]))

        for neighbor in (idx - 1, idx + 1):

            if neighbor < 0 or neighbor >= len(metadata):
                continue

            if neighbor in selected:
                continue

            neighbor_page = _page_as_int(get_page_number(metadata[neighbor]))

            if (
                base_page is not None
                and neighbor_page is not None
                and abs(neighbor_page - base_page) > 1
            ):
                continue

            selected[neighbor] = metadata[neighbor]

    context_parts = []
    total_chars = 0

    for idx in sorted(selected):

        item = selected[idx]

        text = (item.get("text", "") or "").strip()

        if not text:
            continue

        part = f"SOURCE PAGE: {get_page_number(item)}\n{text}"

        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n\n".join(context_parts)


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    subject,
    answer_mode,
    expanded=False
):

    top_k = TOP_K * 2 if expanded else TOP_K

    results = retrieve_context(
        question,
        subject,
        top_k
    )

    if not results:

        return (
            "I couldn't find this information in the selected textbook."
        ), []

    # -----------------------------------------------------
    # BUILD TEXTBOOK CONTEXT
    # -----------------------------------------------------

    context = build_context(
        results,
        subject,
        neighbor_count=6 if expanded else 3
    )

    subject_label = SUBJECT_LABELS.get(subject, subject)

    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

    system_prompt = """
You are an AI Study Assistant for Class 11 students.

STRICT RULES:

1. Answer ONLY from the supplied textbook context.

2. Do NOT use outside knowledge.

3. Do NOT use internet knowledge.

4. Do NOT invent facts, definitions, formulas,
examples, explanations, or conclusions.

5. Use ONLY the selected subject's textbook.

6. The student may phrase the question differently from the
textbook (for example "momentum", "what is momentum", "define
momentum" or "momentum kya hota hai" all mean the same thing).
Match by MEANING, not by exact wording. If the context contains
relevant, partial or related information about the topic, answer
using whatever the context supports.

7. ONLY if the supplied context contains nothing relevant to the
topic at all, say exactly:

"I couldn't find this information in the selected textbook."

8. Stay faithful to the textbook.

9. Answer at a clear Class 11 student level.

10. Use simple and clean formatting.

11. DO NOT use LaTeX.

12. DO NOT use commands such as \\mathbf, \\frac,
\\begin, \\end, or other LaTeX commands.

13. Write formulas in plain text.
Example: p = mv

14. ALWAYS end every answer with this exact heading:

### In Short:

(The only exception is when you reply with the
"I couldn't find this information in the selected textbook."
sentence.)

15. The "In Short" section must be detailed but concise.

16. The "In Short" section should normally contain
4-6 sentences or short bullet points.

17. The "In Short" section should include the main
definition, important formula(s), key concept(s),
important law(s), and essential relationships
covered in the answer.

18. NEVER omit the "### In Short:" section when you answer.

19. The "In Short" section must use ONLY information
supported by the supplied textbook context.
"""

    # =====================================================
    # ANSWER MODE
    # =====================================================

    if answer_mode == "Explanation":

        mode_instruction = """
Explain the answer clearly and step-by-step.

Use definitions, concepts, formulas and examples
only when supported by the textbook context.

Keep formatting simple and readable.

Write formulas in plain text.
For example:
p = mv

At the end, ALWAYS add:

### In Short:

Give a detailed but concise revision summary.

Include the main definition, important formula(s),
key concept(s), important law(s), and essential
relationships or conclusions from the answer.

Use 4-6 sentences or short bullet points.

Do not introduce any information not present
in the textbook context.
"""

    elif answer_mode == "Summary":

        mode_instruction = """
Give a concise revision-oriented answer.

Focus on important definitions, concepts,
formulas, laws and facts supported by the textbook.

Keep formatting simple and readable.

Write formulas in plain text.

At the end, ALWAYS add:

### In Short:

Give a detailed revision summary covering the main
definitions, formulas, laws, concepts and important
relationships discussed in the answer.

Use 4-6 sentences or short bullet points.

Use ONLY information from the textbook context.
"""

    else:

        mode_instruction = """
Create a quiz ONLY from the supplied textbook context.

Include:
- MCQs
- Short questions
- Conceptual questions

Provide answers after the questions.

Do not introduce outside information.

Keep formatting simple and readable.

At the end, ALWAYS add:

### In Short:

Summarize the important concepts, definitions,
formulas and laws covered in the quiz.

Use 4-6 sentences or short bullet points.

Use ONLY information from the textbook context.
"""

    # =====================================================
    # USER PROMPT
    # =====================================================

    user_prompt = f"""
SELECTED SUBJECT:
{subject_label}

ANSWER MODE:
{answer_mode}

STUDENT QUESTION:
{question}

TEXTBOOK CONTEXT:
{context}

MODE INSTRUCTIONS:
{mode_instruction}

FINAL REQUIREMENTS:

1. Answer ONLY using the supplied textbook context.

2. Match the question to the context by meaning, not by exact words.

3. Use simple, clean and readable formatting.

4. Do NOT use LaTeX.

5. Write formulas in plain text.

6. Your response MUST end with:

### In Short:

7. After "### In Short:", provide a detailed but
concise revision summary covering the important
points of the answer.

8. The In Short section should normally contain
4-6 sentences or short bullet points.

9. Do not use information outside the textbook context.
"""

    # =====================================================
    # GROQ REQUEST
    # =====================================================

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1
    )

    answer = (response.choices[0].message.content or "").strip()

    # -----------------------------------------------------
    # SAFETY NET:
    # If the model refused even though chunks were retrieved,
    # retry once with a wider search + more neighbouring context.
    # -----------------------------------------------------

    if (
        not expanded
        and answer
        and REFUSAL_PHRASE in answer.lower()
        and len(answer) < 300
    ):
        return generate_answer(
            question,
            subject,
            answer_mode,
            expanded=True
        )

    return answer, results


# =========================================================
# PREMIUM ANIMATED UI STYLES
# =========================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-dark: #0a0e17;
    --card-surface: rgba(18, 24, 38, 0.7);
    --card-border: rgba(255, 255, 255, 0.08);
    --primary-indigo: #6366f1;
    --primary-purple: #8b5cf6;
    --primary-cyan: #06b6d4;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
}

/* Background & Core Layout */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(139, 92, 246, 0.10) 0%, transparent 35%),
        radial-gradient(circle at 50% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 45%);
    background-attachment: fixed;
}

/* Center and frame main container */
.block-container {
    max-width: 920px !important;
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
    margin: 0 auto !important;
}

/* Hide default streamlit header/footer decorations */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer {
    visibility: hidden;
}

/* Keyframe Animations */
@keyframes fadeInSlideUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulseGlow {
    0%, 100% {
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
    }
    50% {
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.45);
    }
}

@keyframes badgeFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-4px); }
}

/* Header Container */
.hero-wrapper {
    text-align: center;
    padding: 2.2rem 1.5rem 1.8rem 1.5rem;
    margin-bottom: 2rem;
    background: rgba(17, 24, 39, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: 24px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    animation: fadeInSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #a5b4fc;
    margin-bottom: 1rem;
    animation: badgeFloat 4s ease-in-out infinite;
}

.hero-title {
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0.2rem 0 0.8rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 40%, #818cf8 70%, #a855f7 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 8s ease infinite;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 620px;
    margin: 0 auto 1.4rem auto;
    line-height: 1.6;
    font-weight: 400;
}

.feature-strip {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
}

.feature-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: #cbd5e1;
    font-size: 0.78rem;
    font-weight: 500;
}

/* Step Card Headers */
.step-header-box {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.65rem;
    margin-top: 0.4rem;
}

.step-num-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 700;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4);
}

.step-label {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
}

/* Streamlit Widget Customization */
div[data-testid="stSelectbox"] > div {
    background: rgba(18, 24, 38, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stSelectbox"] > div:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15) !important;
}

div[data-testid="stTextArea"] textarea {
    background: rgba(18, 24, 38, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    color: #f8fafc !important;
    font-size: 1rem !important;
    padding: 14px 16px !important;
    line-height: 1.6 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25), 0 8px 24px rgba(0, 0, 0, 0.4) !important;
}

/* Custom Prompt Suggestion Chips */
.chips-wrapper {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 0.8rem;
    margin-top: 0.2rem;
}

.chip-title {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* Primary Button Styling */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%) !important;
    background-size: 200% 200% !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.85rem 1.5rem !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    animation: gradientFlow 6s ease infinite;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.6) !important;
}

div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(1px) !important;
}

/* Secondary Chip Buttons */
div[data-testid="stButton"] > button:not([kind="primary"]) {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Radio Button formatting */
div[data-testid="stRadio"] > div {
    gap: 16px !important;
}

div[data-testid="stRadio"] label {
    background: rgba(18, 24, 38, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 10px 18px !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stRadio"] label:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
}

/* Answer Presentation Card */
.answer-container {
    background: rgba(17, 24, 39, 0.8);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 1.8rem;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 20px rgba(99, 102, 241, 0.15);
    animation: fadeInSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

.answer-top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    padding-bottom: 1rem;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.answer-badge-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.answer-badge {
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.badge-subject {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
}

.badge-mode {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

.badge-source {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fcd34d;
}

.in-short-banner {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 14px;
    padding: 1.2rem;
    margin-top: 1.5rem;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
}

.in-short-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #c7d2fe;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Source Citation Card */
.source-citation-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    transition: all 0.25s ease;
}

.source-citation-card:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateX(4px);
}

.source-page-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    margin-bottom: 0.4rem;
}

.source-text-snippet {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
    font-style: italic;
}

/* Expander Styling */
div[data-testid="stExpander"] {
    background: rgba(18, 24, 38, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(10, 14, 23, 0.8);
}
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.35);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.65);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# STATE INITIALIZATION
# =========================================================

if "question_input_text" not in st.session_state:
    st.session_state.question_input_text = ""


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-pill">
            <span>✨</span> AI-POWERED TEXTBOOK INTELLIGENCE · CLASS 11
        </div>
        <h1 class="hero-title">AI Study Assistant</h1>
        <p class="hero-subtitle">
            Ask questions directly from your Class 11 textbooks. Get instant, verified,
            and zero-hallucination answers powered by high-speed Groq LPU inference.
        </p>
        <div class="feature-strip">
            <div class="feature-badge">⚡ Instant Groq LPU Engine</div>
            <div class="feature-badge">🔍 Hybrid Semantic + Keyword Search</div>
            <div class="feature-badge">🛡️ Strict Textbook Verification</div>
            <div class="feature-badge">📄 Page-Accurate Citations</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# STEP 1 & STEP 2 — CLASS & SUBJECT (2-COLUMN GRID)
# =========================================================

col_step1, col_step2 = st.columns([1, 2], gap="medium")

with col_step1:
    st.markdown(
        """
        <div class="step-header-box">
            <span class="step-num-badge">1</span>
            <span class="step-label">Academic Class</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Select Class",
        ["Class 11"],
        label_visibility="collapsed",
    )

with col_step2:
    st.markdown(
        """
        <div class="step-header-box">
            <span class="step-num-badge">2</span>
            <span class="step-label">Select Subject</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_subject_name = st.selectbox(
        "Select Subject",
        SUPPORTED_SUBJECTS,
        format_func=lambda s: f"{SUBJECT_META.get(s, {}).get('icon', '📖')}  {s}",
        label_visibility="collapsed",
    )

    selected_subject = SUBJECT_KEYS[selected_subject_name]


# =========================================================
# STEP 3 — QUESTION INPUT & QUICK SUGGESTION CHIPS
# =========================================================

st.markdown(
    """
    <div class="step-header-box" style="margin-top: 1.2rem;">
        <span class="step-num-badge">3</span>
        <span class="step-label">Ask your Question</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggested question chips for active subject
active_meta = SUBJECT_META.get(selected_subject_name, {})
prompts = active_meta.get("prompts", [])

st.markdown(
    f'<div class="chip-title">💡 Popular {selected_subject_name} Inquiries (Click to autofill):</div>',
    unsafe_allow_html=True,
)

chip_cols = st.columns(len(prompts))
for idx, prompt_text in enumerate(prompts):
    with chip_cols[idx]:
        if st.button(
            f"📌 {prompt_text[:28]}...",
            key=f"chip_{selected_subject_name}_{idx}",
            help=prompt_text,
            use_container_width=True,
        ):
            st.session_state.question_input_text = prompt_text
            st.rerun()

question = st.text_area(
    "Enter your question:",
    value=st.session_state.question_input_text,
    placeholder=f"Example: {prompts[0] if prompts else 'Explain photosynthesis'}",
    height=120,
    label_visibility="collapsed",
)


# =========================================================
# STEP 4 — ANSWER FORMAT
# =========================================================

st.markdown(
    """
    <div class="step-header-box" style="margin-top: 1.2rem;">
        <span class="step-num-badge">4</span>
        <span class="step-label">Answer Format</span>
    </div>
    """,
    unsafe_allow_html=True,
)

answer_mode = st.radio(
    "Choose answer format:",
    [
        "Explanation",
        "Summary",
        "Quiz"
    ],
    horizontal=True,
    format_func=lambda mode: {
        "Explanation": "💡 Explanation (Step-by-Step)",
        "Summary": "⚡ Summary (High-Yield Revision)",
        "Quiz": "🎯 Quiz (Practice & MCQs)",
    }.get(mode, mode),
    label_visibility="collapsed",
)


# =========================================================
# ASK BUTTON
# =========================================================

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

ask_button = st.button(
    "🤖 Ask AI Study Assistant",
    type="primary",
    use_container_width=True
)


# =========================================================
# PROCESS
# =========================================================

if ask_button:

    if not question.strip():

        st.warning(
            "⚠️ Please enter a question or select a prompt suggestion first."
        )

    else:

        with st.spinner(
            f"🔍 Searching official {selected_subject_name} textbook and synthesizing answer..."
        ):

            try:
                start_time = time.time()

                answer, sources = generate_answer(
                    question.strip(),
                    selected_subject,
                    answer_mode
                )

                elapsed_time = round(time.time() - start_time, 2)

                # Split answer and 'In Short' summary if present
                main_answer = answer
                in_short_summary = None

                if "### In Short:" in answer:
                    parts = answer.split("### In Short:")
                    main_answer = parts[0].strip()
                    in_short_summary = parts[1].strip()

                # Render Answer Container
                st.markdown(
                    f"""
                    <div class="answer-container">
                        <div class="answer-top-bar">
                            <div class="answer-badge-group">
                                <span class="answer-badge badge-subject">
                                    {active_meta.get('icon', '📚')} {selected_subject_name}
                                </span>
                                <span class="answer-badge badge-mode">
                                    {answer_mode} Mode
                                </span>
                                <span class="answer-badge badge-source">
                                    ⚡ {elapsed_time}s
                                </span>
                            </div>
                            <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">
                                Model: {GROQ_MODEL}
                            </span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Main Answer Markdown
                st.markdown(main_answer)

                # High-Yield Revision Callout
                if in_short_summary:
                    st.markdown(
                        f"""
                        <div class="in-short-banner">
                            <div class="in-short-title">
                                <span>⚡</span> In Short: High-Yield Revision Summary
                            </div>
                            <div style="color: #e0e7ff; font-size: 0.95rem; line-height: 1.6;">
                                {in_short_summary}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

                # -------------------------------------------------
                # TEXTBOOK SOURCES
                # -------------------------------------------------

                if sources:

                    with st.expander(
                        f"📖 Verified Textbook Sources ({len(sources)} Citations Found)",
                        expanded=True
                    ):

                        for source in sources:

                            page = get_page_number(
                                source
                            )

                            snippet = source.get("text", "").strip()
                            clean_snippet = (
                                snippet[:300] + "..."
                                if len(snippet) > 300
                                else snippet
                            )

                            st.markdown(
                                f"""
                                <div class="source-citation-card">
                                    <span class="source-page-tag">📄 PAGE {page}</span>
                                    <div class="source-text-snippet">
                                        "{clean_snippet}"
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            except Exception as e:

                st.error(
                    f"❌ Something went wrong: {str(e)}"
                )
