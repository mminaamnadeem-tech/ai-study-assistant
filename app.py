import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# Must be the first Streamlit command in this file.
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered",
)


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


def ui_container(bordered=False):
    """Streamlit container; border= needs Streamlit 1.29+."""
    if bordered:
        try:
            return st.container(border=True)
        except TypeError:
            return st.container()
    return st.container()


def inject_app_styles():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        :root {
            --bg: #fafafa;
            --surface: #ffffff;
            --border: #e4e4e7;
            --text: #18181b;
            --muted: #71717a;
            --accent: #18181b;
            --accent-soft: #f4f4f5;
            --radius: 14px;
            --shadow: 0 1px 2px rgba(24, 24, 27, 0.06), 0 8px 24px rgba(24, 24, 27, 0.06);
        }

        html, body, [class*="css"] {
            font-family: "Plus Jakarta Sans", system-ui, sans-serif;
        }

        .stApp {
            background: var(--bg);
            background-image:
                radial-gradient(circle at 1px 1px, rgba(24, 24, 27, 0.04) 1px, transparent 0);
            background-size: 24px 24px;
        }

        .block-container {
            max-width: 760px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }

        .app-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.85rem 1rem;
            margin: -1rem -1rem 1.5rem -1rem;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            min-width: 0;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--accent);
            color: #fff;
            font-size: 1rem;
            flex-shrink: 0;
        }

        .brand-title {
            margin: 0;
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text);
            line-height: 1.2;
        }

        .brand-sub {
            margin: 0;
            font-size: 0.78rem;
            color: var(--muted);
            line-height: 1.2;
        }

        .env-badge {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text);
            background: var(--accent-soft);
            border: 1px solid var(--border);
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            white-space: nowrap;
        }

        .page-head h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--text);
        }

        .page-head p {
            margin: 0.35rem 0 0 0;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0 0 1.25rem 0;
        }

        .pill {
            font-size: 0.75rem;
            font-weight: 600;
            color: #3f3f46;
            background: #fff;
            border: 1px solid var(--border);
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
        }

        .panel-title {
            margin: 0 0 0.15rem 0;
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text);
        }

        .panel-desc {
            margin: 0 0 1rem 0;
            font-size: 0.82rem;
            color: var(--muted);
        }

        .field-label {
            margin: 0.25rem 0 0.5rem 0;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #52525b;
        }

        .result-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin: 1.5rem 0 0.75rem 0;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid var(--border);
        }

        .result-meta-left {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            min-width: 0;
        }

        .live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.15);
            flex-shrink: 0;
        }

        .result-title {
            margin: 0;
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text);
        }

        .result-tag {
            font-size: 0.72rem;
            font-weight: 600;
            color: #3f3f46;
            background: #fff;
            border: 1px solid var(--border);
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            white-space: nowrap;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
            background: var(--surface) !important;
            box-shadow: var(--shadow);
            padding-top: 0.75rem;
            padding-bottom: 0.75rem;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] textarea {
            border-radius: 10px !important;
            border-color: var(--border) !important;
            background: #fff !important;
        }

        div[data-baseweb="radio"] > div {
            gap: 0.45rem !important;
        }

        div[data-baseweb="radio"] label {
            background: #fff !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 0.45rem 0.75rem !important;
        }

        div.stButton > button[kind="primary"] {
            background: var(--accent) !important;
            color: #fff !important;
            border: 1px solid var(--accent) !important;
            border-radius: 12px !important;
            min-height: 2.75rem;
            font-weight: 700 !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: #fff;
            box-shadow: var(--shadow);
        }

        hr {
            margin: 1.25rem 0 !important;
            border-color: var(--border) !important;
        }

        footer, #MainMenu { visibility: hidden; height: 0; }

        @media (max-width: 640px) {
            .app-topbar { flex-direction: column; align-items: flex-start; }
            .page-head h1 { font-size: 1.45rem; }
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

    if "page" in item:
        return item["page"]

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

    keyword_results.sort(
        key=lambda x: x["keyword_matches"],
        reverse=True
    )

    if keyword_results:

        return keyword_results[:top_k]

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
    <div class="app-topbar">
        <div class="brand">
            <span class="brand-icon">📚</span>
            <div>
                <p class="brand-title">Study Assistant</p>
                <p class="brand-sub">Textbook-grounded workspace</p>
            </div>
        </div>
        <span class="env-badge">Class 11 · Cloud</span>
    </div>

    <div class="page-head">
        <h1>Ask from your textbook</h1>
        <p>Choose a subject, ask a question, and get an answer with page references.</p>
    </div>

    <div class="pill-row">
        <span class="pill">Textbook-only answers</span>
        <span class="pill">Source pages included</span>
        <span class="pill">Explanation · Summary · Quiz</span>
    </div>
    """,
    unsafe_allow_html=True,
)


with ui_container(bordered=True):

    st.markdown(
        """
        <p class="panel-title">New question</p>
        <p class="panel-desc">Fill in the details below, then run the assistant.</p>
        """,
        unsafe_allow_html=True,
    )

    col_class, col_subject = st.columns(2)

    with col_class:

        st.markdown('<p class="field-label">Step 1 — Class</p>', unsafe_allow_html=True)

        st.selectbox(
            "Select Class",
            ["Class 11"],
            label_visibility="collapsed",
        )

    with col_subject:

        st.markdown('<p class="field-label">Step 2 — Subject</p>', unsafe_allow_html=True)

        selected_subject_name = st.selectbox(
            "Select Subject",
            SUPPORTED_SUBJECTS,
            label_visibility="collapsed",
        )

    selected_subject = SUBJECT_KEYS[
        selected_subject_name
    ]

    st.divider()

    st.markdown('<p class="field-label">Step 3 — Question</p>', unsafe_allow_html=True)

    question = st.text_area(
        "Enter your question:",
        placeholder="Example: Explain photosynthesis",
        height=140,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<p class="field-label">Step 4 — Answer format</p>', unsafe_allow_html=True)

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

    st.divider()

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

                st.markdown(
                    f"""
                    <div class="result-meta">
                        <div class="result-meta-left">
                            <span class="live-dot"></span>
                            <p class="result-title">Answer</p>
                        </div>
                        <span class="result-tag">{selected_subject_name} · {answer_mode}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with ui_container(bordered=True):
                    st.markdown(answer)

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

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )
