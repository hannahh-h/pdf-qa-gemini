import os

import fitz
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

# Streamlit Cloud uses st.secrets instead of .env
if not API_KEY:
    try:
        API_KEY = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        API_KEY = None

MODEL = "google/gemini-2.5-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 8
SIMILARITY_THRESHOLD = 0.40
MAX_TOKENS = 1000


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="PDF Q&A RAG",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# API Client
# ============================================================

if not API_KEY:
    st.error(
        "OPENROUTER_API_KEY is not configured. "
        "Add it to your local .env file or Streamlit Secrets."
    )
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# Load Embedding Model
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# PDF Loading
# ============================================================

def load_pdf(uploaded_file):
    document = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    pages = []
    total_characters = 0

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

            total_characters += len(text)

    document.close()

    return pages, total_characters


# ============================================================
# Text Chunking
# ============================================================

def create_chunks(pages, chunk_size=800, overlap=150):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


# ============================================================
# Embeddings
# ============================================================

def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings


# ============================================================
# Retrieval
# ============================================================

def retrieve_relevant_chunks(
    question,
    chunks,
    embeddings,
    model
):
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        embeddings,
        question_embedding
    )

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for index in top_indices:
        if scores[index] >= SIMILARITY_THRESHOLD:
            results.append({
                "text": chunks[index]["text"],
                "page": chunks[index]["page"],
                "score": float(scores[index])
            })

    return results


# ============================================================
# Answer Generation
# ============================================================

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return "I could not find the answer in the provided PDF."

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):
        context_parts.append(
            f"[Source {i} | Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a precise PDF question-answering assistant.

Answer the user's question using ONLY the information contained
in the provided PDF context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided PDF."

Do not use outside knowledge.
Do not invent information.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

Provide a concise and accurate answer.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using retrieved "
                        "PDF context only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error while generating answer: {e}")
        return None


# ============================================================
# Main Application
# ============================================================

st.title("📄 PDF Q&A — RAG Assistant")

st.markdown(
    """
Upload a PDF and ask questions about its contents.

The system uses **Retrieval-Augmented Generation (RAG)** to
retrieve relevant sections of the document before generating
an answer.
"""
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        **PDF Q&A RAG System**

        Built using:

        - Python
        - PyMuPDF
        - Sentence Transformers
        - NumPy
        - OpenRouter
        - Gemini 2.5 Flash
        """
    )

    st.divider()

    st.write("**Embedding model**")
    st.code(EMBEDDING_MODEL)

    st.write("**LLM**")
    st.code(MODEL)

    st.write("**Top-K retrieval**")
    st.write(TOP_K)

    st.write("**Similarity threshold**")
    st.write(SIMILARITY_THRESHOLD)


# ============================================================
# PDF Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # Process PDF
    # --------------------------------------------------------

    if (
        "processed_file" not in st.session_state
        or st.session_state.processed_file
        != uploaded_file.name
    ):

        with st.spinner(
            "Loading embedding model..."
        ):
            embedding_model = load_embedding_model()

        with st.spinner(
            "Reading PDF..."
        ):
            pages, total_characters = load_pdf(
                uploaded_file
            )

        if not pages:
            st.error(
                "No readable text was found in this PDF."
            )
            st.stop()

        with st.spinner(
            "Creating text chunks..."
        ):
            chunks = create_chunks(pages)

        with st.spinner(
            "Creating embeddings..."
        ):
            embeddings = create_embeddings(
                chunks,
                embedding_model
            )

        st.session_state.processed_file = (
            uploaded_file.name
        )

        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

        st.session_state.total_characters = (
            total_characters
        )


    # --------------------------------------------------------
    # Document Information
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages",
            len(st.session_state.pages)
        )

    with col2:
        st.metric(
            "Characters",
            f"{st.session_state.total_characters:,}"
        )

    with col3:
        st.metric(
            "Chunks",
            len(st.session_state.chunks)
        )

    st.divider()


    # --------------------------------------------------------
    # Question Input
    # --------------------------------------------------------

    st.subheader("Ask a question")

    question = st.text_input(
        "Question",
        placeholder=(
            "Example: What are the main components "
            "of a battery?"
        )
    )

    if st.button(
        "Ask",
        type="primary"
    ):

        if not question.strip():
            st.warning(
                "Please enter a question."
            )
            st.stop()

        embedding_model = load_embedding_model()

        with st.spinner(
            "Retrieving relevant information..."
        ):

            retrieved_chunks = retrieve_relevant_chunks(
                question,
                st.session_state.chunks,
                st.session_state.embeddings,
                embedding_model
            )

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                retrieved_chunks
            )


        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        if answer:

            st.subheader("Answer")

            st.markdown(answer)


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if retrieved_chunks:

            st.subheader("Sources")

            displayed_pages = set()

            for chunk in retrieved_chunks:

                page = chunk["page"]

                if page not in displayed_pages:

                    st.write(
                        f"Page {page}"
                    )

                    displayed_pages.add(page)


            # ------------------------------------------------
            # Retrieved Context
            # ------------------------------------------------

            with st.expander(
                "View retrieved context"
            ):

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    st.markdown(
                        f"**Source {i} — "
                        f"Page {chunk['page']} — "
                        f"Score: "
                        f"{chunk['score']:.4f}**"
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.divider()

else:

    st.info(
        "Upload a PDF above to get started."
    )