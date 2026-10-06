import json
from pathlib import Path

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

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_app_styles():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        .stApp {
            background: linear-gradient(165deg, #f8fafc 0%, #e0f2fe 45%, #f1f5f9 100%);
            background-attachment: fixed;
        }

        .block-container {
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Hero */
        .cloud-hero {
            text-align: center;
            padding: 2.25rem 1.5rem 2rem;
            margin-bottom: 1.75rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #0ea5e9 100%);
            box-shadow: 0 20px 50px -12px rgba(37, 99, 235, 0.45);
            animation: heroFade 0.8s ease-out both;
            position: relative;
            overflow: hidden;
        }

        .cloud-hero::before {
            content: "";
            position: absolute;
            top: -50%;
            right: -20%;
            width: 60%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
            pointer-events: none;
        }

        .cloud-hero-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #dbeafe;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.25);
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            margin-bottom: 0.85rem;
            backdrop-filter: blur(8px);
        }

        .cloud-hero h1 {
            color: #ffffff !important;
            font-size: 2.15rem !important;
            font-weight: 700 !important;
            margin: 0 0 0.5rem 0 !important;
            letter-spacing: -0.02em;
        }

        .cloud-hero p {
            color: #e0f2fe;
            font-size: 1.05rem;
            margin: 0;
            line-height: 1.55;
            max-width: 520px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Step cards */
        .step-card {
            animation: slideUp 0.55s ease-out both;
        }

        .step-card-delay-1 { animation-delay: 0.08s; }
        .step-card-delay-2 { animation-delay: 0.16s; }
        .step-card-delay-3 { animation-delay: 0.24s; }
        .step-card-delay-4 { animation-delay: 0.32s; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            background: rgba(255, 255, 255, 0.82) !important;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 24px -4px rgba(15, 23, 42, 0.08);
            transition: box-shadow 0.25s ease, transform 0.25s ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 12px 32px -8px rgba(37, 99, 235, 0.15);
        }

        .step-header {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 0.25rem;
        }

        .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 8px;
            background: linear-gradient(135deg, #2563eb, #0ea5e9);
            color: white;
            font-size: 0.8rem;
            font-weight: 700;
            flex-shrink: 0;
        }

        .step-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #0f172a;
            margin: 0;
        }

        /* Primary button */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em;
            box-shadow: 0 8px 24px -6px rgba(37, 99, 235, 0.5);
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px -6px rgba(37, 99, 235, 0.55) !important;
        }

        div.stButton > button[kind="primary"]:active {
            transform: translateY(0);
        }

        /* Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] textarea {
            border-radius: 10px !important;
            border-color: #cbd5e1 !important;
        }

        div[data-baseweb="radio"] label {
            background: #f8fafc;
            border-radius: 10px;
            padding: 0.35rem 0.5rem;
            border: 1px solid #e2e8f0;
            transition: border-color 0.2s, background 0.2s;
        }

        /* Answer panel */
        .answer-panel {
            animation: slideUp 0.6s ease-out both;
        }

        .answer-panel div[data-testid="stVerticalBlockBorderWrapper"] {
            border-left: 4px solid #2563eb !important;
            background: rgba(255, 255, 255, 0.95) !important;
        }

        /* Expander */
        div[data-testid="stExpander"] {
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            background: rgba(255,255,255,0.7);
        }

        /* Divider subtle */
        hr {
            margin: 1.75rem 0 !important;
            border-color: rgba(148, 163, 184, 0.35) !important;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        @keyframes heroFade {
            from { opacity: 0; transform: translateY(-12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 640px) {
            .cloud-hero h1 { font-size: 1.65rem !important; }
            .block-container { padding-top: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_app_styles()


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
# RETRIEVAL
# =========================================================

def retrieve_context(question, subject, top_k=TOP_K):

    index, metadata = load_subject_index(subject)

    question_words = [
        word.lower().strip(".,?!:;()[]{}")
        for word in question.split()
        if len(word.strip(".,?!:;()[]{}")) > 3
    ]

    # -----------------------------------------------------
    # KEYWORD SEARCH
    # -----------------------------------------------------

    keyword_results = []

    for idx, item in enumerate(metadata):

        text = item.get("text", "").lower()

        matches = sum(
            1
            for word in question_words
            if word in text
        )

        if matches > 0:

            result = item.copy()

            result["score"] = float(matches)
            result["keyword_matches"] = matches
            result["_index"] = idx

            keyword_results.append(result)

    # Best keyword matches first
    keyword_results.sort(
        key=lambda x: x["keyword_matches"],
        reverse=True
    )

    # -----------------------------------------------------
    # USE KEYWORD RESULTS WHEN AVAILABLE
    # -----------------------------------------------------

    if keyword_results:

        return keyword_results[:top_k]

    # -----------------------------------------------------
    # FAISS SEMANTIC SEARCH FALLBACK
    # -----------------------------------------------------

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        result = metadata[idx].copy()

        result["score"] = float(score)
        result["_index"] = int(idx)

        results.append(result)

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

        page = get_page_number(result)

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
# UI
# =========================================================

st.markdown(
    """
    <div class="cloud-hero">
        <span class="cloud-hero-badge">Class 11 · Cloud Learning</span>
        <h1>📚 AI Study Assistant</h1>
        <p>Ask questions directly from your Class 11 textbooks — grounded answers with source pages.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def step_heading(number, title):
    st.markdown(
        f"""
        <div class="step-header">
            <span class="step-num">{number}</span>
            <p class="step-title">{title}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# STEP 1 — CLASS
# =========================================================

st.markdown('<div class="step-card step-card-delay-1">', unsafe_allow_html=True)
with st.container(border=True):
    step_heading(1, "Class")
    st.selectbox(
        "Select Class",
        ["Class 11"],
        label_visibility="collapsed",
    )
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# STEP 2 — SUBJECT
# =========================================================

st.markdown('<div class="step-card step-card-delay-2">', unsafe_allow_html=True)
with st.container(border=True):
    step_heading(2, "Subject")
    selected_subject_name = st.selectbox(
        "Select Subject",
        SUPPORTED_SUBJECTS,
        label_visibility="collapsed",
    )
st.markdown("</div>", unsafe_allow_html=True)

selected_subject = SUBJECT_KEYS[
    selected_subject_name
]


# =========================================================
# STEP 3 — QUESTION
# =========================================================

st.markdown('<div class="step-card step-card-delay-3">', unsafe_allow_html=True)
with st.container(border=True):
    step_heading(3, "Ask your question")
    question = st.text_area(
        "Enter your question:",
        placeholder="Example: Explain photosynthesis",
        height=120,
        label_visibility="collapsed",
    )
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# STEP 4 — ANSWER FORMAT
# =========================================================

st.markdown('<div class="step-card step-card-delay-4">', unsafe_allow_html=True)
with st.container(border=True):
    step_heading(4, "Answer format")
    answer_mode = st.radio(
        "Choose answer format:",
        [
            "Explanation",
            "Summary",
            "Quiz"
        ],
        horizontal=True,
        label_visibility="collapsed",
    )
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)


# =========================================================
# ASK BUTTON
# =========================================================

ask_button = st.button(
    "🤖 Ask AI",
    type="primary",
    use_container_width=True,
)


# =========================================================
# PROCESS
# =========================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            f"Searching {selected_subject_name} textbook..."
        ):

            try:

                answer, sources = generate_answer(
                    question.strip(),
                    selected_subject,
                    answer_mode
                )

                st.markdown('<div class="answer-panel">', unsafe_allow_html=True)
                st.divider()

                with st.container(border=True):
                    st.markdown("#### Answer")
                    st.markdown(answer)

                # -------------------------------------------------
                # TEXTBOOK SOURCES
                # -------------------------------------------------

                if sources:

                    with st.expander(
                        "📖 Textbook Sources",
                        expanded=False,
                    ):

                        for source in sources:

                            page = get_page_number(
                                source
                            )

                            st.markdown(
                                f"**Page {page}**"
                            )

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )
