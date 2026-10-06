import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PATHS & CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

SUPPORTED_SUBJECTS = [
    "Physics",
    "Chemistry",
    "Biology",
    "Computer",
]

SUBJECT_METADATA = {
    "Physics": {
        "key": "phy",
        "icon": "⚛️",
        "tag": "Mechanics & Waves",
        "color": "#6366F1",
        "quick_questions": [
            "Explain Newton's laws of motion with examples",
            "What is work-energy theorem and its derivation?",
            "State and explain Kepler's laws of planetary motion",
            "What is simple harmonic motion and its characteristics?",
        ],
    },
    "Chemistry": {
        "key": "chem",
        "icon": "🧪",
        "tag": "Atomic & Molecular",
        "color": "#EC4899",
        "quick_questions": [
            "What are the postulates of Bohr's atomic model?",
            "Explain periodic trends in ionization enthalpy and electron gain",
            "What is Hess's law of constant heat summation?",
            "Differentiate between sigma and pi bonds with examples",
        ],
    },
    "Biology": {
        "key": "bio",
        "icon": "🧬",
        "tag": "Cell & Life Systems",
        "color": "#10B981",
        "quick_questions": [
            "Explain the fluid mosaic model of plasma membrane",
            "What are the key stages of mitosis and their significance?",
            "Describe the mechanism of light reaction in photosynthesis",
            "Explain the structure and function of a nephron",
        ],
    },
    "Computer": {
        "key": "computer",
        "icon": "💻",
        "tag": "Data Structures & Python",
        "color": "#06B6D4",
        "quick_questions": [
            "Explain binary search algorithm and its time complexity",
            "What are tuples vs lists in Python with code examples?",
            "Explain basic cyber safety and digital footprints",
            "How does bubble sort work step-by-step?",
        ],
    },
}

SUBJECT_KEYS = {name: data["key"] for name, data in SUBJECT_METADATA.items()}

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant | Class 11",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# ADVANCED CUSTOM CSS & ANIMATIONS
# =========================================================

CUSTOM_CSS = """
<style>
/* Import Premium Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-dark: #090D16;
    --card-bg: rgba(18, 24, 38, 0.72);
    --card-border: rgba(255, 255, 255, 0.08);
    --primary-indigo: #6366F1;
    --primary-purple: #8B5CF6;
    --primary-cyan: #06B6D4;
    --primary-pink: #EC4899;
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
}

/* Global App Container */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    background-image: 
        radial-gradient(circle at 10% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 90% 25%, rgba(139, 92, 246, 0.10) 0%, transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(6, 182, 212, 0.08) 0%, transparent 45%);
    background-attachment: fixed;
}

/* Limit max width and center layout */
.main .block-container {
    max-width: 980px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

/* Animated Gradient Hero Title */
.hero-container {
    text-align: center;
    padding: 2.2rem 1.5rem 1.5rem 1.5rem;
    position: relative;
    margin-bottom: 1.5rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #A5B4FC;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    animation: pulseBadge 3s ease-in-out infinite;
}

@keyframes pulseBadge {
    0%, 100% { transform: scale(1); box-shadow: 0 0 15px rgba(99, 102, 241, 0.25); }
    50% { transform: scale(1.03); box-shadow: 0 0 25px rgba(139, 92, 246, 0.45); }
}

.hero-title {
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 30%, #818CF8 70%, #C084FC 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shineGradient 6s linear infinite;
    margin-bottom: 0.5rem !important;
}

@keyframes shineGradient {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 620px;
    margin: 0 auto 1.5rem auto;
    line-height: 1.6;
}

/* Feature Pills in Hero */
.features-strip {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 0.75rem;
}

.feature-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: #CBD5E1;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Glassmorphism Card Containers */
.glass-panel {
    background: var(--card-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 16px 36px -12px rgba(0, 0, 0, 0.45);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.glass-panel:hover {
    border-color: rgba(99, 102, 241, 0.25);
    box-shadow: 0 20px 40px -10px rgba(99, 102, 241, 0.12);
}

/* Section Header Badges */
.step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.1rem;
}

.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 7px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: #FFFFFF;
    font-size: 0.75rem;
    font-weight: 700;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
}

.step-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.01em;
}

/* Streamlit Widget Styling Overrides */
div[data-testid="stSelectbox"] > div {
    background-color: rgba(15, 23, 42, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    transition: all 0.2s ease;
}

div[data-testid="stSelectbox"] > div:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
}

/* Text Area Enhancements */
div[data-testid="stTextArea"] textarea {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 14px !important;
    color: #F8FAFC !important;
    font-size: 0.98rem !important;
    line-height: 1.55 !important;
    padding: 14px 16px !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25), inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

/* Radio Button Group (Format Picker) */
div[role="radiogroup"] {
    display: flex;
    gap: 14px;
    background: rgba(15, 23, 42, 0.5);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

div[role="radiogroup"] label {
    flex: 1;
    background: transparent;
    padding: 8px 14px !important;
    border-radius: 10px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    cursor: pointer;
}

div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.04);
}

/* Primary Button Styling with Animated Gradient */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%) !important;
    background-size: 200% auto !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.75rem 2rem !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.5) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
}

div.stButton > button[kind="primary"]:hover {
    background-position: right center !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 32px -4px rgba(139, 92, 246, 0.65) !important;
}

div.stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
}

/* Prompt Chips (Quick Questions) */
.chip-btn-wrap {
    margin-top: 8px;
    margin-bottom: 12px;
}

.chip-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

/* Answer Output Container */
.answer-card {
    background: rgba(18, 24, 38, 0.85);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 18px;
    padding: 1.75rem;
    margin-top: 1.5rem;
    box-shadow: 0 20px 45px -15px rgba(0, 0, 0, 0.6);
    position: relative;
    overflow: hidden;
    animation: fadeInSlideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.answer-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899, #06B6D4);
    background-size: 300% 100%;
    animation: gradientShift 6s ease infinite;
}

@keyframes fadeInSlideUp {
    from {
        opacity: 0;
        transform: translateY(14px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.answer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    padding-bottom: 0.9rem;
    margin-bottom: 1.2rem;
}

.answer-title-group {
    display: flex;
    align-items: center;
    gap: 10px;
}

.answer-badge {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #A5B4FC;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
}

/* Highlight / "In Short" Revision Box */
.in-short-container {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(49, 46, 129, 0.4) 100%);
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-top: 1.4rem;
    position: relative;
    box-shadow: 0 8px 24px -6px rgba(99, 102, 241, 0.25);
}

.in-short-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1rem;
    font-weight: 700;
    color: #C7D2FE;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Source Citation Card */
.source-pill-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    transition: all 0.2s ease;
}

.source-pill-card:hover {
    background: rgba(15, 23, 42, 0.85);
    border-color: rgba(99, 102, 241, 0.3);
}

.source-page-badge {
    background: #1E293B;
    border: 1px solid #334155;
    color: #38BDF8;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.source-preview-text {
    color: #94A3B8;
    font-size: 0.85rem;
    line-height: 1.45;
    margin-top: 6px;
}

/* Hide default streamlit decoration */
header[data-testid="stHeader"] {
    background: transparent !important;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# MODELS & CLIENTS
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_groq_client():
    # Support both Streamlit secrets and environment variables
    api_key = None
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    else:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


embedding_model = load_embedding_model()
groq_client = load_groq_client()


# =========================================================
# LOAD VECTORSTORE
# =========================================================

@st.cache_resource(show_spinner=False)
def load_subject_index(subject):
    subject_dir = VECTORSTORE_DIR / subject
    index_path = subject_dir / "index.faiss"
    metadata_path = subject_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index file not found at: {index_path}. "
            "Please ensure the vectorstore index has been built."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata JSON not found at: {metadata_path}."
        )

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


# =========================================================
# GET PAGE NUMBER
# =========================================================

def get_page_number(item):
    if "page" in item:
        return item["page"]
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("page", "Unknown")
    return "Unknown"


# =========================================================
# RETRIEVAL (HYBRID KEYWORD + FAISS)
# =========================================================

def retrieve_context(question, subject, top_k=TOP_K):
    index, metadata = load_subject_index(subject)

    question_words = [
        word.lower().strip(".,?!:;()[]{}")
        for word in question.split()
        if len(word.strip(".,?!:;()[]{}")) > 3
    ]

    # Keyword search pass
    keyword_results = []
    for idx, item in enumerate(metadata):
        text = item.get("text", "").lower()
        matches = sum(1 for word in question_words if word in text)
        if matches > 0:
            result = item.copy()
            result["score"] = float(matches)
            result["match_type"] = "Keyword Match"
            result["keyword_matches"] = matches
            result["_index"] = idx
            keyword_results.append(result)

    keyword_results.sort(key=lambda x: x["keyword_matches"], reverse=True)

    if keyword_results:
        return keyword_results[:top_k]

    # Semantic search fallback
    query_embedding = embedding_model.encode([question], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = metadata[idx].copy()
        result["score"] = float(score)
        result["match_type"] = "Semantic Search"
        result["_index"] = int(idx)
        results.append(result)

    return results


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question, subject, answer_mode):
    if groq_client is None:
        raise ValueError(
            "GROQ_API_KEY is not configured. Please add it to Streamlit Secrets or your environment."
        )

    results = retrieve_context(question, subject, TOP_K)

    if not results:
        return (
            "I couldn't find this information in the selected textbook.",
            [],
        )

    context_parts = []
    for result in results:
        page = get_page_number(result)
        text = result.get("text", "")
        context_parts.append(f"SOURCE PAGE: {page}\n{text}")

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an AI Study Assistant for Class 11 students.

STRICT RULES:
1. Answer ONLY from the supplied textbook context.
2. Do NOT use outside knowledge.
3. Do NOT use internet knowledge.
4. Do NOT invent facts, definitions, formulas, examples, explanations, or conclusions.
5. Use ONLY the selected subject's textbook.
6. If the supplied context does not contain enough information to answer the question, say:
"I couldn't find this information in the selected textbook."
7. Stay faithful to the textbook.
8. Answer at a clear Class 11 student level.
9. If the question is not supported by the context, do not guess.
10. Use simple and clean formatting.
11. DO NOT use LaTeX.
12. DO NOT use commands such as \\mathbf, \\frac, \\begin, \\end, or other LaTeX commands.
13. Write formulas in plain text. Example: p = mv
14. ALWAYS end every answer with this exact heading:
### In Short:
15. The "In Short" section must be detailed but concise.
16. The "In Short" section should normally contain 4-6 sentences or short bullet points.
17. The "In Short" section should include the main definition, important formula(s), key concept(s), important law(s), and essential relationships covered in the answer.
18. NEVER omit the "### In Short:" section.
19. The "In Short" section must use ONLY information supported by the supplied textbook context.
"""

    if answer_mode == "Explanation":
        mode_instruction = """
Explain the answer clearly and step-by-step.
Use definitions, concepts, formulas and examples only when supported by the textbook context.
Keep formatting simple and readable.
Write formulas in plain text (e.g., p = mv).

At the end, ALWAYS add:
### In Short:
Give a detailed but concise revision summary (4-6 sentences or short bullet points).
Include the main definition, important formula(s), key concept(s), important law(s), and essential relationships.
"""
    elif answer_mode == "Summary":
        mode_instruction = """
Give a concise revision-oriented answer.
Focus on important definitions, concepts, formulas, laws and facts supported by the textbook.
Write formulas in plain text.

At the end, ALWAYS add:
### In Short:
Give a detailed revision summary (4-6 sentences or short bullet points) covering main definitions, formulas, laws, and concepts.
"""
    else:
        mode_instruction = """
Create a quiz ONLY from the supplied textbook context.
Include:
- MCQs
- Short questions
- Conceptual questions
Provide answers after the questions.
Write formulas in plain text.

At the end, ALWAYS add:
### In Short:
Summarize the important concepts, definitions, formulas and laws covered in the quiz (4-6 sentences or short bullet points).
"""

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
3. Do NOT use LaTeX commands.
4. Write formulas in plain text.
5. Your response MUST end with:
### In Short:
6. After "### In Short:", provide a detailed but concise revision summary covering the important points of the answer.
7. The In Short section should normally contain 4-6 sentences or short bullet points.
8. Do not use information outside the textbook context.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content.strip()
    return answer, results


# =========================================================
# UI STATE MANAGEMENT
# =========================================================

if "question_text" not in st.session_state:
    st.session_state.question_text = ""


def set_quick_question(text):
    st.session_state.question_text = text


# =========================================================
# HERO HEADER SECTION
# =========================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">
            <span>✨</span> Next-Gen RAG Study Intelligence
        </div>
        <h1 class="hero-title">AI Study Assistant</h1>
        <p class="hero-subtitle">
            Instant, zero-hallucination answers powered directly by your official 
            Class 11 curriculum textbooks and Groq ultra-low latency inference.
        </p>
        <div class="features-strip">
            <div class="feature-pill">⚡ Groq LPU Accelerated</div>
            <div class="feature-pill">🔍 Hybrid Keyword + FAISS Search</div>
            <div class="feature-pill">🛡️ Strict Textbook Grounding</div>
            <div class="feature-pill">📝 Instant High-Yield Revision</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MAIN INTERACTION PANEL
# =========================================================

# Step 1 & 2 in side-by-side columns
col_class, col_sub = st.columns([1, 2])

with col_class:
    st.markdown(
        """
        <div class="step-header">
            <span class="step-num">1</span>
            <span class="step-title">Academic Level</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Select Academic Level",
        ["Class 11 (Standard NCERT / CBSE)"],
        label_visibility="collapsed",
    )

with col_sub:
    st.markdown(
        """
        <div class="step-header">
            <span class="step-num">2</span>
            <span class="step-title">Select Subject</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    subject_options = [
        f"{SUBJECT_METADATA[name]['icon']} {name}" for name in SUPPORTED_SUBJECTS
    ]
    selected_subject_raw = st.selectbox(
        "Choose Subject",
        subject_options,
        index=0,
        label_visibility="collapsed",
    )
    selected_subject_name = selected_subject_raw.split(" ")[-1]
    selected_subject = SUBJECT_KEYS[selected_subject_name]
    subject_info = SUBJECT_METADATA[selected_subject_name]


# =========================================================
# STEP 3 — QUESTION INPUT & QUICK PROMPTS
# =========================================================

st.markdown(
    """
    <div class="step-header" style="margin-top: 1rem;">
        <span class="step-num">3</span>
        <span class="step-title">Ask Your Question</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Quick Question Prompt Chips for the active subject
st.markdown(
    f'<div class="chip-label">💡 Recommended {selected_subject_name} Inquiries (Click to autofill):</div>',
    unsafe_allow_html=True,
)

chip_cols = st.columns(len(subject_info["quick_questions"]))
for idx, q_prompt in enumerate(subject_info["quick_questions"]):
    with chip_cols[idx]:
        if st.button(
            f"📌 {q_prompt[:32]}...",
            key=f"chip_{selected_subject_name}_{idx}",
            help=q_prompt,
            use_container_width=True,
        ):
            st.session_state.question_text = q_prompt
            st.rerun()

# Question Input Area
question_input = st.text_area(
    "Enter Question",
    value=st.session_state.question_text,
    placeholder=f"E.g., {subject_info['quick_questions'][0]}",
    height=125,
    label_visibility="collapsed",
)


# =========================================================
# STEP 4 — ANSWER FORMAT SELECTOR
# =========================================================

st.markdown(
    """
    <div class="step-header" style="margin-top: 1rem;">
        <span class="step-num">4</span>
        <span class="step-title">Response Mode</span>
    </div>
    """,
    unsafe_allow_html=True,
)

answer_mode = st.radio(
    "Choose answer format:",
    [
        "Explanation",
        "Summary",
        "Quiz",
    ],
    format_func=lambda x: {
        "Explanation": "💡 Comprehensive Explanation (Step-by-Step)",
        "Summary": "⚡ High-Yield Revision (Core Formulas & Laws)",
        "Quiz": "🎯 Self-Test Practice (MCQs & Conceptual)",
    }[x],
    horizontal=True,
    label_visibility="collapsed",
)


# =========================================================
# ASK AI BUTTON
# =========================================================

st.write("")
ask_button = st.button(
    "⚡ Generate Textbook Answer",
    type="primary",
    use_container_width=True,
)


# =========================================================
# PROCESS & DISPLAY
# =========================================================

if ask_button:
    current_question = question_input.strip()

    if not current_question:
        st.warning("⚠️ Please enter a question or click one of the suggested topics above.")
    else:
        if groq_client is None:
            st.error(
                "🔑 **Groq API Key Not Found**: Please configure `GROQ_API_KEY` in your `.streamlit/secrets.toml` "
                "or as an environment variable to activate answers."
            )
        else:
            with st.spinner(
                f"🔎 Searching official {selected_subject_name} textbook and synthesizing answer..."
            ):
                start_time = time.time()
                try:
                    answer, sources = generate_answer(
                        current_question,
                        selected_subject,
                        answer_mode,
                    )
                    latency = round(time.time() - start_time, 2)

                    # Extract "### In Short:" section if present
                    main_body = answer
                    in_short_body = None

                    if "### In Short:" in answer:
                        parts = answer.split("### In Short:")
                        main_body = parts[0].strip()
                        in_short_body = parts[1].strip()

                    # Render Response Container
                    st.markdown(
                        f"""
                        <div class="answer-card">
                            <div class="answer-header">
                                <div class="answer-title-group">
                                    <span style="font-size: 1.4rem;">{subject_info['icon']}</span>
                                    <div>
                                        <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">
                                            {selected_subject_name} Answer
                                        </h3>
                                        <span style="font-size: 0.8rem; color: #94A3B8;">
                                            Model: {GROQ_MODEL} • Generated in {latency}s
                                        </span>
                                    </div>
                                </div>
                                <span class="answer-badge">{answer_mode} Mode</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Render Main Answer Markdown
                    st.markdown(main_body)

                    # Render "In Short" Revision Callout if available
                    if in_short_body:
                        st.markdown(
                            f"""
                            <div class="in-short-container">
                                <div class="in-short-title">
                                    <span>⚡</span> High-Yield Revision Summary (In Short)
                                </div>
                                <div style="color: #E0E7FF; font-size: 0.95rem; line-height: 1.6;">
                                    {in_short_body}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

                    # -------------------------------------------------
                    # TEXTBOOK SOURCES ACCORDION
                    # -------------------------------------------------
                    if sources:
                        with st.expander(
                            f"📚 Textbook Source Verification ({len(sources)} Verified Citations)",
                            expanded=True,
                        ):
                            for idx, source in enumerate(sources, 1):
                                page = get_page_number(source)
                                match_type = source.get("match_type", "Textbook Excerpt")
                                text_snippet = source.get("text", "").strip()

                                display_snippet = (
                                    text_snippet[:340] + "..."
                                    if len(text_snippet) > 340
                                    else text_snippet
                                )

                                st.markdown(
                                    f"""
                                    <div class="source-pill-card">
                                        <div style="flex: 1;">
                                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                                <span class="source-page-badge">PAGE {page}</span>
                                                <span style="font-size: 0.78rem; color: #818CF8; font-weight: 600;">
                                                    {match_type}
                                                </span>
                                            </div>
                                            <div class="source-preview-text">
                                                "{display_snippet}"
                                            </div>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                except FileNotFoundError as fnf_err:
                    st.error(
                        f"⚠️ **Vectorstore Index Not Found:** {str(fnf_err)}\n\n"
                        "Please verify your `vectorstore/` directory contains `index.faiss` and `metadata.json` for this subject."
                    )
                except Exception as e:
                    st.error(f"❌ **Error generating response:** {str(e)}")
