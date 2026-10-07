import json
import re
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
TOP_K = 5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
# RETRIEVAL HELPERS
# =========================================================

def _normalize_words(text):
    """Return normalized alphanumeric tokens for lexical search."""
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def _keyword_score(query_words, text):
    """Score lexical overlap, with a small phrase bonus."""
    text_words = set(
        _normalize_words(text)
    )

    if not query_words or not text_words:
        return 0

    matches = sum(
        1
        for word in query_words
        if word in text_words
    )

    phrase = " ".join(query_words)

    phrase_bonus = (
        2
        if phrase and phrase in text.lower()
        else 0
    )

    return matches + phrase_bonus


# =========================================================
# ADVANCED HYBRID RAG RETRIEVAL
# =========================================================

def retrieve_context(
    question,
    subject,
    top_k=TOP_K
):
    """
    Advanced Hybrid RAG.

    Uses multiple retrieval signals:

    1. Full natural-language semantic search
    2. Condensed concept semantic search
    3. Definition-oriented semantic search
    4. Keyword matching
    5. Phrase matching
    6. Reciprocal Rank Fusion (RRF)

    Exact question wording is NOT required.

    Example:

        "momentum"
        "what is momentum?"
        "define momentum"
        "explain momentum"

    can retrieve the same relevant textbook content.
    """

    index, metadata = load_subject_index(subject)

    if index.ntotal == 0 or not metadata:
        return []


    # =====================================================
    # QUERY NORMALIZATION
    # =====================================================

    stop_words = {
        "what",
        "what's",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "does",
        "do",
        "did",
        "can",
        "could",
        "would",
        "should",
        "explain",
        "define",
        "definition",
        "describe",
        "state",
        "tell",
        "me",
        "about",
        "with",
        "and",
        "or",
        "please",
        "give",
        "meaning",
    }

    raw_words = _normalize_words(
        question
    )

    query_words = [
        word
        for word in raw_words
        if len(word) >= 3
        and word not in stop_words
    ]

    concept_query = " ".join(
        query_words
    ).strip()

    # Example:
    # "What is momentum?" -> "momentum"
    #
    # This is NOT used as a requirement.
    # It is simply an additional retrieval query.

    if concept_query:
        definition_query = (
            f"definition and explanation of {concept_query}"
        )
    else:
        definition_query = question


    # =====================================================
    # SEMANTIC SEARCH HELPER
    # =====================================================

    def semantic_search(
        query_text,
        candidate_k
    ):
        """
        Run cosine-style semantic search using the
        normalized SentenceTransformer embeddings.
        """

        embedding = embedding_model.encode(
            [query_text],
            normalize_embeddings=True
        )

        embedding = np.asarray(
            embedding,
            dtype="float32"
        )

        scores, indices = index.search(
            embedding,
            candidate_k
        )

        rank_map = {}
        score_map = {}

        for rank, (score, idx) in enumerate(
            zip(
                scores[0],
                indices[0]
            ),
            start=1
        ):

            if idx == -1:
                continue

            idx = int(idx)

            # Safety check if metadata/index counts
            # ever become inconsistent.
            if idx >= len(metadata):
                continue

            rank_map[idx] = rank
            score_map[idx] = float(score)

        return rank_map, score_map


    # =====================================================
    # LARGE SEMANTIC CANDIDATE POOL
    # =====================================================

    semantic_k = min(
        max(top_k * 8, 40),
        index.ntotal
    )


    # =====================================================
    # SEMANTIC SEARCH 1
    # FULL QUESTION
    # =====================================================

    full_rank, full_scores = semantic_search(
        question,
        semantic_k
    )


    # =====================================================
    # SEMANTIC SEARCH 2
    # CONDENSED CONCEPT
    # =====================================================

    concept_rank, concept_scores = semantic_search(
        concept_query if concept_query else question,
        semantic_k
    )


    # =====================================================
    # SEMANTIC SEARCH 3
    # DEFINITION / EXPLANATION FORM
    # =====================================================

    definition_rank, definition_scores = semantic_search(
        definition_query,
        semantic_k
    )


    # =====================================================
    # KEYWORD + PHRASE SEARCH
    # =====================================================

    lexical_scores = {}
    phrase_scores = {}

    original_question = (
        question.strip().lower()
    )

    concept_phrase = concept_query.lower()


    for idx, item in enumerate(metadata):

        text = item.get(
            "text",
            ""
        )

        if not text:
            continue

        text_lower = text.lower()

        text_words = set(
            _normalize_words(text)
        )


        # -------------------------------------------------
        # KEYWORD MATCHES
        # -------------------------------------------------

        matches = 0

        for word in query_words:

            if word in text_words:

                matches += 1
                continue


            # Basic singular/plural support
            if len(word) > 3:

                if (
                    word.endswith("ies")
                    and word[:-3] + "y"
                    in text_words
                ):
                    matches += 1

                elif (
                    word.endswith("es")
                    and word[:-2]
                    in text_words
                ):
                    matches += 1

                elif (
                    word.endswith("s")
                    and word[:-1]
                    in text_words
                ):
                    matches += 1


        if matches > 0:
            lexical_scores[idx] = matches


        # -------------------------------------------------
        # PHRASE MATCH
        # -------------------------------------------------

        phrase_bonus = 0


        if (
            original_question
            and original_question
            in text_lower
        ):

            phrase_bonus += 2


        if (
            concept_phrase
            and len(query_words) >= 2
            and concept_phrase
            in text_lower
        ):

            phrase_bonus += 2


        if phrase_bonus > 0:
            phrase_scores[idx] = phrase_bonus


    # -----------------------------------------------------
    # SORT LEXICAL RESULTS
    # -----------------------------------------------------

    lexical_sorted = sorted(
        lexical_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    lexical_rank = {
        idx: rank
        for rank, (idx, _) in enumerate(
            lexical_sorted,
            start=1
        )
    }


    # =====================================================
    # RECIPROCAL RANK FUSION
    # =====================================================

    RRF_K = 60

    candidates = set()

    candidates.update(
        full_rank.keys()
    )

    candidates.update(
        concept_rank.keys()
    )

    candidates.update(
        definition_rank.keys()
    )

    candidates.update(
        lexical_rank.keys()
    )

    candidates.update(
        phrase_scores.keys()
    )


    fused_scores = {}


    for idx in candidates:

        score = 0.0


        # -------------------------------------------------
        # Full-question semantic ranking
        # Strongest signal
        # -------------------------------------------------

        if idx in full_rank:

            score += (
                0.40
                /
                (
                    RRF_K
                    +
                    full_rank[idx]
                )
            )


        # -------------------------------------------------
        # Condensed concept semantic ranking
        # -------------------------------------------------

        if idx in concept_rank:

            score += (
                0.25
                /
                (
                    RRF_K
                    +
                    concept_rank[idx]
                )
            )


        # -------------------------------------------------
        # Definition semantic ranking
        # -------------------------------------------------

        if idx in definition_rank:

            score += (
                0.20
                /
                (
                    RRF_K
                    +
                    definition_rank[idx]
                )
            )


        # -------------------------------------------------
        # Keyword ranking
        # Secondary only
        # -------------------------------------------------

        if idx in lexical_rank:

            score += (
                0.10
                /
                (
                    RRF_K
                    +
                    lexical_rank[idx]
                )
            )


        # -------------------------------------------------
        # Exact phrase bonus
        # Small bonus only
        # -------------------------------------------------

        if idx in phrase_scores:

            score += (
                0.05
                *
                min(
                    phrase_scores[idx],
                    2
                )
                /
                2
            )


        fused_scores[idx] = score


    # =====================================================
    # FINAL SORT
    # =====================================================

    ranked_indices = sorted(
        fused_scores.keys(),
        key=lambda idx: fused_scores[idx],
        reverse=True
    )


    if not ranked_indices:
        return []


    # =====================================================
    # BUILD FINAL RESULTS
    # =====================================================

    results = []


    for idx in ranked_indices[:top_k]:

        result = metadata[idx].copy()


        result["score"] = float(
            fused_scores[idx]
        )


        result["semantic_score"] = float(
            full_scores.get(
                idx,
                0.0
            )
        )


        result["concept_semantic_score"] = float(
            concept_scores.get(
                idx,
                0.0
            )
        )


        result["definition_semantic_score"] = float(
            definition_scores.get(
                idx,
                0.0
            )
        )


        result["keyword_score"] = float(
            lexical_scores.get(
                idx,
                0
            )
        )


        result["phrase_score"] = float(
            phrase_scores.get(
                idx,
                0
            )
        )


        result["_index"] = int(
            idx
        )


        results.append(
            result
        )


    # =====================================================
    # RELEVANCE GATE
    # =====================================================

    # FAISS always returns nearest neighbours.
    # So this prevents totally unrelated questions
    # from being sent to Groq.
    #
    # This threshold is intentionally permissive
    # because our vectorstore is semantic/page-level.

    best = results[0]


    best_full = best[
        "semantic_score"
    ]


    best_concept = best[
        "concept_semantic_score"
    ]


    best_definition = best[
        "definition_semantic_score"
    ]


    best_keyword = best[
        "keyword_score"
    ]


    MIN_SEMANTIC_SCORE = 0.20
    MIN_DEFINITION_SCORE = 0.20


    if (
        best_full < MIN_SEMANTIC_SCORE
        and
        best_concept < MIN_SEMANTIC_SCORE
        and
        best_definition < MIN_DEFINITION_SCORE
        and
        best_keyword <= 0
    ):

        return []


    return results


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    subject,
    answer_mode
):

    results = retrieve_context(
        question,
        subject,
        TOP_K
    )


    if not results:

        return (
            "I couldn't find this information in the selected textbook."
        ), []


    # -----------------------------------------------------
    # BUILD TEXTBOOK CONTEXT
    # -----------------------------------------------------

    context_parts = []


    for result in results:

        page = get_page_number(
            result
        )


        text = result.get(
            "text",
            ""
        )


        context_parts.append(
            f"SOURCE PAGE: {page}\n{text}"
        )


    context = "\n\n".join(
        context_parts
    )


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

6. If the supplied context does not contain
enough information to answer the question, say:

"I couldn't find this information in the selected textbook."

7. Stay faithful to the textbook.

8. Answer at a clear Class 11 student level.

9. If the question is not supported by the context,
do not guess.

10. Use simple and clean formatting.

11. DO NOT use LaTeX.

12. DO NOT use commands such as \\mathbf, \\frac,
\\begin, \\end, or other LaTeX commands.

13. Write formulas in plain text.
Example: p = mv

14. ALWAYS end every answer with this exact heading:

### In Short:

15. The "In Short" section must be detailed but concise.

16. The "In Short" section should normally contain
4-6 sentences or short bullet points.

17. The "In Short" section should include the main
definition, important formula(s), key concept(s),
important law(s), and essential relationships
covered in the answer.

18. NEVER omit the "### In Short:" section.

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
{subject}

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

2. Use simple, clean and readable formatting.

3. Do NOT use LaTeX.

4. Write formulas in plain text.

5. Your response MUST end with:

### In Short:

6. After "### In Short:", provide a detailed but
concise revision summary covering the important
points of the answer.

7. The In Short section should normally contain
4-6 sentences or short bullet points.

8. Do not use information outside the textbook context.
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


    answer = response.choices[0].message.content.strip()


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


st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True
)


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

        <h1 class="hero-title">
            AI Study Assistant
        </h1>

        <p class="hero-subtitle">
            Ask questions directly from your Class 11 textbooks. Get instant, verified,
            and zero-hallucination answers powered by high-speed Groq LPU inference.
        </p>

        <div class="feature-strip">
            <div class="feature-badge">
                ⚡ Instant Groq LPU Engine
            </div>

            <div class="feature-badge">
                🔍 Hybrid Semantic + Keyword Search
            </div>

            <div class="feature-badge">
                🛡️ Strict Textbook Verification
            </div>

            <div class="feature-badge">
                📄 Page-Accurate Citations
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# STEP 1 & STEP 2 — CLASS & SUBJECT
# =========================================================

col_step1, col_step2 = st.columns(
    [1, 2],
    gap="medium"
)

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
        format_func=lambda s: (
            f"{SUBJECT_META.get(s, {}).get('icon', '📖')}  {s}"
        ),
        label_visibility="collapsed",
    )

    selected_subject = SUBJECT_KEYS[
        selected_subject_name
    ]


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
active_meta = SUBJECT_META.get(
    selected_subject_name,
    {}
)

prompts = active_meta.get(
    "prompts",
    []
)


st.markdown(
    f'<div class="chip-title">💡 Popular {selected_subject_name} Inquiries (Click to autofill):</div>',
    unsafe_allow_html=True,
)


chip_cols = st.columns(
    len(prompts)
)

for idx, prompt_text in enumerate(
    prompts
):

    with chip_cols[idx]:

        if st.button(
            f"📌 {prompt_text[:28]}...",
            key=f"chip_{selected_subject_name}_{idx}",
            help=prompt_text,
            use_container_width=True,
        ):

            st.session_state.question_input_text = (
                prompt_text
            )

            st.rerun()


question = st.text_area(
    "Enter your question:",
    value=st.session_state.question_input_text,
    placeholder=(
        f"Example: "
        f"{prompts[0] if prompts else 'Explain photosynthesis'}"
    ),
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
    }.get(
        mode,
        mode
    ),
    label_visibility="collapsed",
)


# =========================================================
# ASK BUTTON
# =========================================================

st.markdown(
    "<div style='height: 12px;'></div>",
    unsafe_allow_html=True
)


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

                elapsed_time = round(
                    time.time() - start_time,
                    2
                )


                # Split answer and 'In Short' summary if present
                main_answer = answer
                in_short_summary = None


                if "### In Short:" in answer:

                    parts = answer.split(
                        "### In Short:"
                    )

                    main_answer = parts[0].strip()
                    in_short_summary = parts[1].strip()


                # =================================================
                # RENDER ANSWER CONTAINER
                # =================================================

                st.markdown(
                    f"""
                    <div class="answer-container">

                        <div class="answer-top-bar">

                            <div class="answer-badge-group">

                                <span class="answer-badge badge-subject">
                                    {active_meta.get('icon', '📚')}
                                    {selected_subject_name}
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


                # =================================================
                # MAIN ANSWER
                # =================================================

                st.markdown(
                    main_answer
                )


                # =================================================
                # HIGH-YIELD REVISION CALLOUT
                # =================================================

                if in_short_summary:

                    st.markdown(
                        f"""
                        <div class="in-short-banner">

                            <div class="in-short-title">
                                <span>⚡</span>
                                In Short: High-Yield Revision Summary
                            </div>

                            <div style="color: #e0e7ff; font-size: 0.95rem; line-height: 1.6;">
                                {in_short_summary}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


                # =================================================
                # TEXTBOOK SOURCES
                # =================================================

                if sources:

                    with st.expander(
                        f"📖 Verified Textbook Sources ({len(sources)} Citations Found)",
                        expanded=True
                    ):

                        for source in sources:

                            page = get_page_number(
                                source
                            )

                            snippet = source.get(
                                "text",
                                ""
                            ).strip()

                            clean_snippet = (
                                snippet[:300] + "..."
                                if len(snippet) > 300
                                else snippet
                            )

                            st.markdown(
                                f"""
                                <div class="source-citation-card">

                                    <span class="source-page-tag">
                                        📄 PAGE {page}
                                    </span>

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
