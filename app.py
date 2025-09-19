import os
import time
import random
from io import BytesIO

import streamlit as st
import pandas as pd
import docx
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import faiss
from anthropic import Anthropic, APIStatusError
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

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
# Embeddings
# ---------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ---------------------------
# Helpers
# ---------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

@st.cache_data(show_spinner=False)
def extract_and_cache_text(file_bytes: bytes, file_name: str):
    ext = file_name.split(".")[-1].lower()

    if ext == "pdf":
        try:
            reader = PdfReader(BytesIO(file_bytes))
            text = "\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )
            return text if text.strip() else None
        except Exception:
            return None

    elif ext == "docx":
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == "xlsx":
        df = pd.read_excel(BytesIO(file_bytes))
        return df.to_string()

    elif ext == "csv":
        df = pd.read_csv(BytesIO(file_bytes))
        return df.to_string()

    elif ext == "pptx":
        prs = Presentation(BytesIO(file_bytes))
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs)

    elif ext == "txt":
        return file_bytes.decode("utf-8")

    return None

@st.cache_resource(show_spinner=False)
def build_faiss_index(text: str):
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return chunks, index

def search_chunks(query, chunks, index, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0]]

def query_claude(client, model, messages, max_tokens=1000, retries=3, system=None):
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
            if e.status_code == 529:
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

def batch_summarize(text, batch_size=5):
    chunks = chunk_text(text)
    summaries = []
    for i in range(0, len(chunks), batch_size):
        batch = "\n\n".join(chunks[i:i + batch_size])
        with st.spinner(f"📄 Summarizing batch {i//batch_size+1}/{(len(chunks)+batch_size-1)//batch_size}..."):
            response = query_claude(
                client,
                "claude-opus-4-1-20250805",
                messages=[{"role": "user", "content": f"Summarize this section:\n\n{batch}"}],
                system="You are an AI summarizer."
            )
            if response:
                summaries.append(response)
            time.sleep(0.5)
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
st.set_page_config(page_title="AI Document Summarizer", layout="wide")
st.title("📄 AI-Powered Document Summarizer (Claude + RAG)")
st.caption("Upload your docs and get concise summaries or answers using AI with context-aware search.")

uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, XLSX, CSV, PPTX, TXT)",
    type=["pdf", "docx", "xlsx", "csv", "pptx", "txt"],
    accept_multiple_files=False
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    text = extract_and_cache_text(file_bytes, uploaded_file.name)
    ext = uploaded_file.name.split(".")[-1].lower()

    if text is None:
        st.warning(f"⚠️ OCR is disabled on Streamlit Cloud. Please upload documents with extractable text.")
    else:
        messages = {
            "pdf": "✅ PDF processed successfully!",
            "docx": "✅ Word document processed successfully!",
            "xlsx": "✅ Spreadsheet processed successfully!",
            "csv": "✅ Spreadsheet processed successfully!",
            "pptx": "✅ Presentation processed successfully!",
            "txt": "✅ Text file processed successfully!"
        }
        st.success(messages.get(ext, "✅ Document processed successfully!"))

        # Build FAISS index
        chunks, index = build_faiss_index(text)
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.full_text = text

        # ---------------------------
        # Mode Selection UI
        # ---------------------------
        mode_options = ["💬 Ask a Question", "📝 Get Summary"]
        selected_mode = st.selectbox("What would you like to do?", mode_options)

        # Display selected mode
        st.markdown(f"**Selected Mode:** {selected_mode}")

        # ---------------------------
        # Q&A Mode
        # ---------------------------
        if selected_mode == "💬 Ask a Question":
            user_query = st.text_input("🔍 Type your question here:")
            if st.button("Ask Now"):
                if user_query.strip():
                    ext_name = uploaded_file.name.split(".")[-1].upper()
                    with st.spinner(f"🧠 Searching for answers in your {ext_name}..."):
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

        # ---------------------------
        # Summarization Mode
        # ---------------------------
        elif selected_mode == "📝 Get Summary":
            if st.button("Generate Summary"):
                ext_name = uploaded_file.name.split(".")[-1].upper()
                with st.spinner(f"📄 Creating a concise summary of your {ext_name}..."):
                    summary = batch_summarize(st.session_state.full_text)
                    st.subheader("📌 Document Summary")
                    st.write(summary)
