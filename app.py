import os
import time
import random
import hashlib
from io import BytesIO

import streamlit as st
import pandas as pd
import docx
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic, APIStatusError
from pdf2image import convert_from_bytes
import easyocr
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# ---------------------------
# API Key
# ---------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    try:
        ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        st.error("❌ No Anthropic API key found. Set it in environment variables or secrets.toml")
        st.stop()

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------------
# Embeddings + OCR
# ---------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ocr_reader = easyocr.Reader(["en"], gpu=False)


# ---------------------------
# Helpers
# ---------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def ocr_pdf(file_bytes):
    """OCR fallback for PDFs with no extractable text."""
    images = convert_from_bytes(file_bytes)
    text = ""
    for img in images:
        text += " ".join([line[1] for line in ocr_reader.readtext(np.array(img))]) + "\n"
    return text


@st.cache_data(show_spinner=False)
def extract_and_cache_text(file_bytes: bytes, file_name: str):
    """Extract text from file and cache result by file content hash."""
    if file_name.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(file_bytes))
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            if text.strip():
                return text
        except Exception:
            pass
        return ocr_pdf(file_bytes)

    elif file_name.endswith(".docx"):
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(file_bytes))
        return df.to_string()

    elif file_name.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
        return df.to_string()

    elif file_name.endswith(".pptx"):
        prs = Presentation(BytesIO(file_bytes))
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs)

    elif file_name.endswith(".txt"):
        return file_bytes.decode("utf-8")

    return ""


@st.cache_resource(show_spinner=False)
def build_faiss_index(text: str):
    """Build FAISS index & cache it for the text hash."""
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return chunks, index


def search_chunks(query, chunks, index, top_k=3):
    """Retrieve most relevant chunks."""
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0]]


# ---------------------------
# Claude API Helper
# ---------------------------
def query_claude(client, model, messages, max_tokens=1000, retries=3, system=None):
    """Query Claude with retries for overload errors."""
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return response.content[0].text
        except APIStatusError as e:
            if e.status_code == 529:  # overloaded
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"⚠️ Claude API overloaded. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            else:
                st.error(f"❌ API Error {e.status_code}: {e.message}")
                return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return None
    st.error("🚨 Claude API is still overloaded after multiple retries. Try again later.")
    return None


# ---------------------------
# Summarization (Batched)
# ---------------------------
def batch_summarize(text, batch_size=5):
    chunks = chunk_text(text)
    summaries = []

    for i in range(0, len(chunks), batch_size):
        batch = "\n\n".join(chunks[i:i + batch_size])
        with st.spinner(f"Summarizing batch {i//batch_size+1}/{(len(chunks)+batch_size-1)//batch_size}..."):
            response = query_claude(
                client,
                "claude-opus-4-1-20250805",
                messages=[{"role": "user", "content": f"Summarize this section:\n\n{batch}"}],
                system="You are an AI summarizer."
            )
            if response:
                summaries.append(response)
            time.sleep(0.5)  # backoff

    combined = "\n".join(summaries)
    final_summary = query_claude(
        client,
        "claude-opus-4-1-20250805",
        messages=[{"role": "user", "content": f"Combine into a single coherent summary:\n\n{combined}"}],
        system="You are an AI that merges summaries."
    )
    return final_summary or "⚠️ Summarization failed."


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Document Summarizer", layout="wide")
st.title("📄 Document Summarizer with Claude + RAG")

uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, XLSX, CSV, PPTX, TXT)",
    type=["pdf", "docx", "xlsx", "csv", "pptx", "txt"],
    accept_multiple_files=False
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    text = extract_and_cache_text(file_bytes, uploaded_file.name)

    if text.strip():
        chunks, index = build_faiss_index(text)
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.full_text = text
        st.success("✅ Document processed and indexed successfully!")
    else:
        st.error("No text could be extracted from the uploaded file.")

if "chunks" in st.session_state and st.session_state.chunks:
    mode = st.radio("Choose mode:", ["Q&A", "Summarization"])

    if mode == "Q&A":
        user_query = st.text_input("🔍 Ask a question about your document:")
        if st.button("Ask") and user_query:
            with st.spinner("Thinking..."):
                context_chunks = search_chunks(user_query, st.session_state.chunks, st.session_state.index)
                context_text = "\n".join(context_chunks)
                response = query_claude(
                    client,
                    "claude-opus-4-1-20250805",
                    messages=[{"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}],
                    system="Answer based only on the provided context."
                )
                if response:
                    st.subheader("🤖 Answer")
                    st.write(response)

    elif mode == "Summarization":
        if st.button("Summarize Document"):
            with st.spinner("Summarizing document..."):
                summary = batch_summarize(st.session_state.full_text)
                st.subheader("📌 Document Summary")
                st.write(summary)

