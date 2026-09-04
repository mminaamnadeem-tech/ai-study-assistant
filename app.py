import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# =========================================================
# APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered",
)

SUPPORTED_SUBJECTS = {
    "Physics": "physics",
    "Chemistry": "chemistry",
    "Biology": "biology",
}

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.30

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# =========================================================
# LOAD MODELS / API
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it in Streamlit Cloud → Settings → Secrets."
        )

    return Groq(api_key=api_key)


# =========================================================
# LOAD SUBJECT-SPECIFIC FAISS INDEX
# =========================================================

@st.cache_resource
def load_subject_index(subject_key):
    subject_dir = VECTORSTORE_DIR / subject_key

    index_path = subject_dir / "index.faiss"
    metadata_path = subject_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing FAISS index: {index_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}"
        )

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Index/metadata mismatch for {subject_key}: "
            f"{index.ntotal} vectors vs {len(metadata)} metadata entries."
        )

    return index, metadata


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(question, subject_key, top_k=TOP_K):
    model = load_embedding_model()
    index, metadata = load_subject_index(subject_key)

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        score = float(score)

        if score < SIMILARITY_THRESHOLD:
            continue

        result = metadata[idx].copy()
        result["score"] = score
        results.append(result)

    return results


# =========================================================
# GROQ GENERATION
# =========================================================

def generate_answer(question, subject_name, subject_key, answer_mode):
    results = retrieve_context(
        question=question,
        subject_key=subject_key,
        top_k=TOP_K,
    )

    # No sufficiently relevant textbook evidence.
    if not results:
        return (
            "I couldn't find this information in the selected textbook.",
            [],
        )

    context_parts = []

    for result in results:
        page = result.get("metadata", {}).get("page", "Unknown")
        text = result.get("text", "").strip()

        if text:
            context_parts.append(
                f"SOURCE PAGE: {page}\n{text}"
            )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """
You are a textbook-grounded AI Study Assistant for Class 11 students.

STRICT RULES:

1. Answer ONLY from the supplied textbook context.
2. Do NOT use outside knowledge.
3. Do NOT use internet knowledge.
4. Do NOT invent facts, definitions, formulas, examples,
   explanations, or conclusions.
5. Do NOT use information from another subject.
6. Do NOT assume information that is not supported by the
   supplied context.
7. If the supplied textbook context is insufficient to answer
   the question, respond exactly:
   "I couldn't find this information in the selected textbook."
8. Stay faithful to the textbook's terminology and content.
9. Do not claim a page/source contains information unless it
   is actually present in the supplied context.
10. Answer at a Class 11 student level.
"""

    if answer_mode == "Explanation":
        mode_instruction = """
Explain the answer clearly and step-by-step.
Include definitions, formulas, concepts, and textbook examples
only when they are supported by the supplied context.
"""

    elif answer_mode == "Summary":
        mode_instruction = """
Give a concise revision-oriented summary.
Focus on key points, definitions, concepts, formulas, and facts
supported by the supplied textbook context.
"""

    else:
        mode_instruction = """
Create a quiz using ONLY the supplied textbook context.

Include:
- MCQs
- Short questions
- Conceptual questions

Provide the answers after the questions.
Do not introduce information outside the textbook.
"""

    user_prompt = f"""
SELECTED SUBJECT:
{subject_name}

ANSWER MODE:
{answer_mode}

STUDENT QUESTION:
{question}

TEXTBOOK CONTEXT:
{context}

MODE INSTRUCTIONS:
{mode_instruction}
"""

    client = load_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content.strip()

    return answer, results


# =========================================================
# USER INTERFACE
# =========================================================

st.title("📚 AI Study Assistant")

st.caption(
    "Class 11 textbook-grounded study assistant"
)

st.divider()

st.subheader("Step 1 — Class")
st.selectbox(
    "Select Class",
    ["Class 11"],
    disabled=True,
)

st.subheader("Step 2 — Subject")

selected_subject_name = st.selectbox(
    "Select Subject",
    list(SUPPORTED_SUBJECTS.keys()),
)

selected_subject_key = SUPPORTED_SUBJECTS[
    selected_subject_name
]

st.subheader("Step 3 — Ask your Question")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: What is momentum?",
    height=120,
)

st.subheader("Step 4 — Answer Format")

answer_mode = st.radio(
    "Choose answer format:",
    ["Explanation", "Summary", "Quiz"],
    horizontal=True,
)

ask_button = st.button(
    "🤖 Ask AI",
    type="primary",
    use_container_width=True,
)

if ask_button:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner(
            f"Searching the Class 11 {selected_subject_name} textbook..."
        ):
            try:
                answer, sources = generate_answer(
                    question=question.strip(),
                    subject_name=selected_subject_name,
                    subject_key=selected_subject_key,
                    answer_mode=answer_mode,
                )

                st.divider()
                st.subheader("Answer")
                st.write(answer)

                if sources:
                    st.divider()
                    st.subheader("📖 Textbook Source")

                    shown_pages = set()

                    for source in sources:
                        metadata = source.get("metadata", {})
                        page = metadata.get("page", "Unknown")

                        if page not in shown_pages:
                            st.write(
                                f"Class 11 {selected_subject_name} "
                                f"— Page {page}"
                            )
                            shown_pages.add(page)

            except Exception as error:
                st.error(f"Application error: {error}")
