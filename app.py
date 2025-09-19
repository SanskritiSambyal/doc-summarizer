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

# ---------- Clean Styling ----------
st.markdown("""
<style>
body, .stApp, .stTextInput>div>div>input {
    font-size: 17px;
}
.compact-label {
    font-weight: 500;
    margin-top: 0px;
    margin-bottom: 4px;
}
.output-box {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 10px;
    margin-top: 16px;
    font-size: 16px;
    line-height: 1.5;
}
.output-box h1, .output-box h2, .output-box h3 {
    font-size: 18px !important;
    font-weight: 600;
    margin-top: 8px;
    margin-bottom: 6px;
}
.output-box p {
    margin: 4px 0;
    font-size: 16px;
}
.stSelectbox>div>div>div, .stTextInput>div {
    margin-top: 0px !important;
    margin-bottom: 4px !important;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
h1 {
    margin-top: -70px !important;
    margin-bottom: 12px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 AI-Powered Document Summarizer & Chatbot (Claude + RAG)")

# ---------- Layout Columns ----------
col_sidebar, col_main = st.columns([1, 3])

# ---------- Sidebar ----------
with col_sidebar:
    st.markdown('<div class="compact-label">Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "file_uploader",
        type=["pdf","docx","xlsx","csv","pptx","txt"],
        label_visibility="collapsed"
    )

    text_extracted = False
    if uploaded_file:
        file_bytes = uploaded_file.read()
        text = extract_and_cache_text(file_bytes, uploaded_file.name)
        if text:
            text_extracted = True
            chunks, index = build_faiss_index(text)
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.full_text = text

            ext = uploaded_file.name.split(".")[-1].lower()
            messages = {
                "pdf": "✅ PDF processed successfully!",
                "docx": "✅ Word document processed successfully!",
                "xlsx": "✅ Spreadsheet processed successfully!",
                "csv": "✅ Spreadsheet processed successfully!",
                "pptx": "✅ Presentation processed successfully!",
                "txt": "✅ Text file processed successfully!"
            }
            st.success(messages.get(ext, "✅ Document processed successfully!"))
        else:
            st.warning("⚠️ OCR is disabled on Streamlit Cloud. Please upload documents with extractable text.")

    # Show mode selector only if text was extracted
    if text_extracted:
        st.markdown('<div class="compact-label">Select Mode</div>', unsafe_allow_html=True)
        selected_mode = st.selectbox(
            "mode_selector",
            ["💬 Ask a Question", "📝 Get Summary"],
            key="selected_mode",
            label_visibility="collapsed"
        )

        # ---------- Collapsible History ----------
        with st.expander("🕘 History"):
            if "qa_history" not in st.session_state:
                st.session_state.qa_history = []
            if "summary_history" not in st.session_state:
                st.session_state.summary_history = []

            # Q&A History
            with st.expander("🗨️ Q&A History"):
                if st.session_state.qa_history:
                    for entry in reversed(st.session_state.qa_history[-5:]):
                        st.markdown(
                            f'<div class="output-box"><b>Q:</b> {entry["query"]}<br><b>A:</b> {entry["answer"]}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No Q&A history yet.")

            # Summary History
            with st.expander("📝 Summary History"):
                if st.session_state.summary_history:
                    for i, s in enumerate(reversed(st.session_state.summary_history[-5:]), 1):
                        st.markdown(
                            f'<div class="output-box"><b>Summary {i}:</b><br>{s}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No summaries yet.")
# ---------- Main Column ----------
with col_main:
    if text_extracted:
        # ---------- Q&A ----------
        if selected_mode == "💬 Ask a Question":
            st.markdown('<div class="compact-label">Your Question</div>', unsafe_allow_html=True)
            user_query = st.text_input(
                "question_input",
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )

            ask_disabled = user_query.strip() == ""
            col1, col2 = st.columns([0.25, 0.75])
            with col1:
                ask_clicked = st.button("Ask Now", disabled=ask_disabled)

            if ask_clicked:
                with st.spinner(f"🧠 Searching for answers in your {ext}..."):
                    context_chunks = search_chunks(user_query, st.session_state.chunks, st.session_state.index)
                    context_text = "\n".join(context_chunks)
                    response = query_claude(
                        client,
                        "claude-opus-4-1-20250805",
                        messages=[{"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}],
                        system="Answer based only on the provided context."
                    )
                if response:
                    st.session_state.qa_history.append({"query": user_query, "answer": response})
                    st.markdown(f'<div class="output-box"><b>🤖 Answer:</b><br>{response}</div>', unsafe_allow_html=True)

        # ---------- Summary ----------
        elif selected_mode == "📝 Get Summary":
            col1, col2 = st.columns([0.25, 0.75])
            with col1:
                summary_clicked = st.button("Generate Summary")

            if summary_clicked:
                with st.spinner(f"📄 Creating a concise summary of your {ext}..."):
                    summary = batch_summarize(st.session_state.full_text)
                    st.session_state.summary_history.append(summary)
                    st.markdown(f'<div class="output-box"><b>📌 Document Summary:</b><br>{summary}</div>', unsafe_allow_html=True)
