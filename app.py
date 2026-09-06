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

SUPPORTED_SUBJECTS = ["Physics", "Chemistry", "Biology"]

SUBJECT_KEYS = {
    "Physics": "phy",
    "Chemistry": "chem",
    "Biology": "bio",
}

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

# Your FAISS indexes appear to use L2 distance.
# Lower distance = more relevant.

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered",
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
# RETRIEVAL
# =========================================================

def retrieve_context(question, subject, top_k=TOP_K):
    index, metadata = load_subject_index(subject)

    question_words = [
        word.lower().strip(".,?!:;()[]{}")
        for word in question.split()
        if len(word.strip(".,?!:;()[]{}")) > 3
    ]

    # Find textbook chunks containing question keywords
    keyword_results = []

    for idx, item in enumerate(metadata):
        text = item.get("text", "").lower()

        matches = sum(
            1 for word in question_words
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

    # If keyword search finds enough relevant textbook chunks
    if keyword_results:
        return keyword_results[:top_k]

    # Fallback to semantic FAISS search
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

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        result = metadata[idx].copy()
        result["score"] = float(score)
        results.append(result)

    return results   


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question, subject, answer_mode):

    results = retrieve_context(
        question,
        subject,
        TOP_K
    )
    st.write("DEBUG RESULTS COUNT:", len(results))

    if not results:
        return (
            "I couldn't find this information in the selected textbook."
        ), []

    context_parts = []

    for result in results:

        page = result.get("metadata", {}).get(
            "page",
            "Unknown"
        )

        text = result.get("text", "")

        context_parts.append(
            f"SOURCE PAGE: {page}\n{text}"
        )

    context = "\n\n".join(context_parts)
    st.write("DEBUG CONTEXT:", context[:2000])

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

9. If the question asks for something not supported
by the supplied context, do not guess.
"""

    if answer_mode == "Explanation":

        mode_instruction = """
Explain the answer clearly and step-by-step.
Use definitions, concepts, formulas and examples
only when supported by the textbook context.
"""

    elif answer_mode == "Summary":

        mode_instruction = """
Give a concise revision-oriented answer.
Focus on important definitions, concepts,
formulas and facts supported by the textbook.
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

    answer = response.choices[0].message.content

    return answer, results


# =========================================================
# UI
# =========================================================

st.title("📚 AI Study Assistant")

st.write(
    "Ask questions directly from your Class 11 textbooks."
)

st.divider()


st.subheader("Step 1 — Class")

st.selectbox(
    "Select Class",
    ["Class 11"]
)


st.subheader("Step 2 — Subject")

selected_subject_name = st.selectbox(
    "Select Subject",
    SUPPORTED_SUBJECTS
)

selected_subject = SUBJECT_KEYS[
    selected_subject_name
]


st.subheader("Step 3 — Ask your Question")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: Explain photosynthesis",
    height=120
)


st.subheader("Step 4 — Answer Format")

answer_mode = st.radio(
    "Choose answer format:",
    [
        "Explanation",
        "Summary",
        "Quiz"
    ],
    horizontal=True
)


ask_button = st.button(
    "🤖 Ask AI",
    type="primary",
    use_container_width=True
)


# =========================================================
# PROCESS
# =========================================================

if ask_button:

    if not question.strip():

        st.warning("Please enter a question first.")

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

                st.divider()

                st.subheader("Answer")

                st.write(answer)

                if sources:

                    with st.expander("📖 Textbook Sources"):

                        for source in sources:

                            page = source.get(
                                "metadata", {}
                            ).get(
                                "page",
                                "Unknown"
                            )

                            st.write(
                                f"**Page {page}**"
                            )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )
