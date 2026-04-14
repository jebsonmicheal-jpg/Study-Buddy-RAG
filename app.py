import base64
import csv
import html
import hashlib
import importlib
import io
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from collections import Counter

import fitz
import ollama
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

APP_TITLE = "Study Buddy RAG"
DEFAULT_PDF = Path("Artifacts") / "Articles.pdf"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
FLASHCARD_MODEL = "llama3.2:3b"
EXAM_MODEL = "llama3.2:3b"
FAISS_INDEX_DIR = Path("faiss_index")
FAISS_META_FILE = FAISS_INDEX_DIR / "meta.json"
PDF_LIBRARY_FILE = Path("pdf_library.json")
KB_STATS_FILE = Path("kb_stats.json")
UPLOADS_DIR = Path("uploaded_pdfs")
PROGRESS_FILE = Path("progress.json")
VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "25"))
VISION_TEXT_THRESHOLD = int(os.getenv("VISION_TEXT_THRESHOLD", "100"))


st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #000000;
            --surface: #0b0b0b;
            --surface-soft: #121212;
            --text: #ffffff;
            --muted: #c9c9c9;
            --line: #2a2a2a;
            --brand: #4f8cff;
            --brand-2: #8b5cf6;
            --ok: #34d399;
            --warn: #f59e0b;
            --side-bg: #000000;
            --side-surface: #0b0b0b;
            --side-line: #242424;
            --side-text: #ffffff;
            --side-muted: #d1d5db;
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        .stSidebar {
            background: var(--side-bg);
            border-right: 1px solid var(--side-line);
        }
        .stSidebar > div {
            color: var(--side-text) !important;
        }
        .stSidebar .stMarkdown {
            color: var(--side-text) !important;
        }

        .hero-wrap {
            background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
            border-radius: 18px;
            padding: 2rem;
            color: white;
            box-shadow: 0 12px 36px rgba(37, 99, 235, 0.25);
            margin-bottom: 2rem;
        }
        .hero-wrap h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 900;
        }
        .hero-wrap p {
            margin: 0.5rem 0 0 0;
            font-size: 0.95rem;
            opacity: 0.95;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        .metric-card .k {
            color: var(--muted) !important;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .metric-card .v {
            color: var(--text) !important;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }

        .panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, div {
            color: var(--text) !important;
        }

        [data-testid="stChatMessageContent"] {
            color: var(--text) !important;
        }

        .stChatInputContainer,
        .stTextInput > div > div {
            background: var(--surface) !important;
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--muted) !important;
            border-radius: 12px;
            padding: 0.45rem 0.9rem;
            border: 1px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            color: var(--brand) !important;
            background: var(--surface-soft);
            border: 1px solid #2c4b8a;
            font-weight: 700;
        }

        .stAlert {
            border-radius: 12px !important;
            border: 1px solid var(--line) !important;
        }
        .stExpander {
            border-radius: 12px !important;
            border: 1px solid var(--line) !important;
            background: var(--surface) !important;
        }

        .chat-container {
            display: flex;
            flex-direction: column;
            height: 70vh;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 1rem;
            padding-right: 0.5rem;
        }
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        .chat-messages::-webkit-scrollbar-track {
            background: transparent;
        }
        .chat-messages::-webkit-scrollbar-thumb {
            background: var(--line);
            border-radius: 3px;
        }
        .chat-input-sticky {
            position: sticky;
            bottom: 0;
            background: var(--surface);
            padding: 1rem 0;
            border-top: 1px solid var(--line);
            z-index: 100;
        }

        .mcq-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        }
        .mcq-question {
            color: var(--text) !important;
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.8rem;
        }
        .mcq-option {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.6rem;
            cursor: pointer;
            color: var(--text) !important;
            font-size: 0.95rem;
        }
        .mcq-option:hover {
            background: #1a1a1a;
            border-color: var(--brand);
        }
        .mcq-answer {
            background: #102018;
            border-left: 4px solid #34d399;
            padding: 0.8rem;
            border-radius: 6px;
            margin-top: 0.8rem;
            color: #dcfce7 !important;
            font-size: 0.95rem;
        }
        .sb-brand {
            color: var(--brand) !important;
            font-weight: 700;
        }
        .chip {
            display: inline-block;
            background: #101828;
            border: 1px solid #1e3a8a;
            color: #93c5fd;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-right: 0.4rem;
        }
        .flashcards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .flip-card {
            background: transparent;
            height: 220px;
            perspective: 1000px;
        }
        .flip-card-inner {
            position: relative;
            width: 100%;
            height: 100%;
            text-align: left;
            transition: transform 0.6s;
            transform-style: preserve-3d;
        }
        .flip-card:hover .flip-card-inner {
            transform: rotateY(180deg);
        }
        .flip-card-front, .flip-card-back {
            position: absolute;
            width: 100%;
            height: 100%;
            -webkit-backface-visibility: hidden;
            backface-visibility: hidden;
            border-radius: 14px;
            border: 1px solid var(--line);
            padding: 1rem;
            overflow-y: auto;
        }
        .flip-card-front {
            background: var(--surface);
            color: var(--text);
        }
        .flip-card-back {
            background: #101828;
            color: #dbeafe;
            transform: rotateY(180deg);
            border-color: #1e3a8a;
        }
        .flashcard-label {
            font-size: 0.78rem;
            color: var(--muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@500;700;800&display=swap');

    * {
        transition: all 0.3s ease;
    }

    .stApp,
    .stSidebar,
    .stMarkdown,
    .stText,
    p,
    li,
    label,
    h4,
    h5,
    h6,
    input,
    textarea,
    button {
        font-family: 'Inter', sans-serif !important;
    }

    .material-icons,
    .material-icons-round,
    .material-icons-outlined,
    .material-symbols-outlined,
    .material-symbols-rounded,
    .material-symbols-sharp,
    [class*="codicon"] {
        font-family: "Material Icons", "Material Symbols Outlined", "Material Symbols Rounded", "Material Symbols Sharp", "codicon" !important;
    }

    .stApp, .stSidebar, .stMarkdown, .stText, p, li, label, div {
        line-height: 1.45 !important;
        letter-spacing: 0.01em;
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    h1, h2, h3, .hero-title, .orbitron {
        font-family: 'Orbitron', sans-serif !important;
    }

    html {
        scroll-behavior: smooth;
    }

    body, .stApp {
        background: #0a0a0f !important;
    }

    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #11111a;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #9b59b6, #4f8ef7);
        border-radius: 10px;
        border: 2px solid #11111a;
    }

    .premium-hero {
        min-height: 72vh;
        border-radius: 22px;
        background: linear-gradient(120deg, #0a0a0f, #1a1a2e, #16213e);
        background-size: 300% 300%;
        animation: heroGradient 12s ease infinite;
        border: 1px solid rgba(79, 142, 247, 0.25);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.55);
        padding: 2rem;
        margin-bottom: 1.25rem;
    }

    @keyframes heroGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .typewriter {
        overflow: hidden;
        white-space: nowrap;
        border-right: 2px solid #4f8ef7;
        width: 0;
        animation: typing 2.3s steps(24, end) forwards, blink 0.75s step-end infinite;
        font-size: clamp(1.8rem, 4vw, 3rem);
        font-weight: 800;
        margin: 0;
    }
    @keyframes typing { from { width: 0 } to { width: 100% } }
    @keyframes blink { from, to { border-color: transparent } 50% { border-color: #4f8ef7 } }

    .hero-subtitle {
        opacity: 0;
        animation: fadeInUp 0.9s ease 1s forwards;
        color: #d6def7;
        margin-top: 0.6rem;
    }

    .feature-grid {
        margin-top: 1.2rem;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 1rem;
        transform: translateY(20px);
        opacity: 0;
        animation: fadeInUp 0.8s ease forwards;
    }
    .feature-card:nth-child(2) { animation-delay: 0.12s; }
    .feature-card:nth-child(3) { animation-delay: 0.24s; }
    .feature-card:hover {
        box-shadow: 0 0 0 1px rgba(79, 142, 247, 0.6), 0 0 28px rgba(79, 142, 247, 0.35);
        transform: translateY(-4px);
    }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .hero-cta {
        display: inline-block;
        margin-top: 1.1rem;
        background: linear-gradient(90deg, #4f8ef7, #9b59b6);
        color: #fff !important;
        font-weight: 700;
        padding: 0.7rem 1.1rem;
        border-radius: 999px;
        box-shadow: 0 0 20px rgba(79, 142, 247, 0.35);
        animation: pulseBtn 2s ease infinite;
    }
    @keyframes pulseBtn {
        0% { transform: scale(1); }
        50% { transform: scale(1.04); }
        100% { transform: scale(1); }
    }

    .model-pill-wrap {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin: 0.45rem 0 0.7rem 0;
    }
    .model-pill {
        border: 1px solid rgba(255, 255, 255, 0.2);
        background: rgba(255, 255, 255, 0.05);
        border-radius: 999px;
        padding: 0.28rem 0.6rem;
        font-size: 0.78rem;
    }
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        animation: pulseDot 1.4s ease infinite;
    }
    .dot.ok { background: #1abc9c; box-shadow: 0 0 8px #1abc9c; }
    .dot.bad { background: #ff4d4d; box-shadow: 0 0 8px #ff4d4d; }
    @keyframes pulseDot {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }

    .upload-zone {
        border: 1px dashed rgba(79, 142, 247, 0.55);
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: inset 0 0 20px rgba(79, 142, 247, 0.08);
    }
    .upload-zone:hover {
        border-style: solid;
        border-color: #1abc9c;
    }

    [data-testid="stFileUploader"] section {
        padding: 0.5rem !important;
    }
    [data-testid="stFileUploader"] button {
        white-space: nowrap !important;
        min-width: 90px;
        gap: 0.35rem !important;
    }
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p {
        line-height: 1.4 !important;
    }

    .typing {
        display: inline-flex;
        gap: 4px;
        align-items: center;
        padding: 0.5rem 0.65rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.04);
    }
    .typing span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #4f8ef7;
        animation: bounce 1s infinite;
    }
    .typing span:nth-child(2) { animation-delay: 0.15s; }
    .typing span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
        40% { transform: translateY(-6px); opacity: 1; }
    }

    .chat-row { display: flex; margin: 0.45rem 0; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .chat-bubble {
        max-width: 82%;
        border-radius: 14px;
        padding: 0.65rem 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.16);
    }
    .chat-bubble.user {
        background: linear-gradient(135deg, rgba(79, 142, 247, 0.95), rgba(155, 89, 182, 0.95));
        color: #fff;
    }
    .chat-bubble.assistant {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(8px);
        animation: fadeInUp 0.35s ease;
    }

    .skeleton {
        height: 12px;
        border-radius: 6px;
        margin: 0.45rem 0;
        background: linear-gradient(90deg, #1b1b24 25%, #2a2a35 50%, #1b1b24 75%);
        background-size: 200% 100%;
        animation: shimmer 1.4s linear infinite;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    .stButton > button:hover {
        transform: scale(1.05);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(155, 89, 182, 0.35) !important;
        box-shadow: inset -1px 0 0 rgba(79, 142, 247, 0.35);
    }

    div[role="radiogroup"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 0.4rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        margin-bottom: 1rem;
    }
    div[role="radiogroup"] label {
        border-radius: 999px !important;
        border: 1px solid transparent;
        padding: 0.42rem 0.7rem !important;
    }
    div[role="radiogroup"] label:hover {
        border-color: rgba(79, 142, 247, 0.45);
        box-shadow: 0 0 14px rgba(79, 142, 247, 0.2);
    }

    .copy-tip {
        font-size: 0.8rem;
        color: #9aa6ce;
    }

    @media (max-width: 900px) {
        .feature-grid { grid-template-columns: 1fr; }
        .premium-hero { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_chat_model() -> ChatOllama:
    return ChatOllama(
        model=st.session_state.get("chat_model", DEFAULT_CHAT_MODEL),
        base_url=OLLAMA_HOST,
        temperature=0.2,
    )


def get_flashcard_model() -> ChatOllama:
    return ChatOllama(
        model=FLASHCARD_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.2,
    )


def get_exam_model() -> ChatOllama:
    return ChatOllama(
        model=EXAM_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.2,
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        base_url=OLLAMA_HOST,
    )


def get_ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


@st.cache_data(ttl=30)
def get_model_status_map() -> dict[str, bool]:
    targets = ["llama3.2:3b", "nomic-embed-text", "llava:7b"]
    status = {name: False for name in targets}
    try:
        response = get_ollama_client().list()
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])
        names = set()
        for model in models or []:
            model_name = getattr(model, "model", None)
            if model_name is None and isinstance(model, dict):
                model_name = model.get("model") or model.get("name")
            if model_name:
                names.add(str(model_name))
        for name in targets:
            status[name] = name in names
    except Exception:
        pass
    return status


def render_model_status_pills() -> None:
    status_map = get_model_status_map()
    pills = []
    for model_name in ["llama3.2:3b", "nomic-embed-text", "llava:7b"]:
        dot_class = "ok" if status_map.get(model_name) else "bad"
        state = "online" if status_map.get(model_name) else "offline"
        pills.append(
            f'<span class="model-pill"><span class="dot {dot_class}"></span>{model_name} · {state}</span>'
        )
    st.markdown(f"<div class='model-pill-wrap'>{''.join(pills)}</div>", unsafe_allow_html=True)


def render_premium_hero() -> None:
    st.markdown(
        """
        <section class="premium-hero">
            <h1 class="typewriter hero-title">Study Buddy 📚</h1>
            <p class="hero-subtitle">Local AI study cockpit with premium interactions, fast retrieval, and exam-first workflows.</p>
            <div class="feature-grid">
                <div class="feature-card"><strong>Doubt Solver</strong><br/>Context-grounded answers with source citations.</div>
                <div class="feature-card"><strong>Quiz Mode</strong><br/>Practice MCQs with score tracking and insights.</div>
                <div class="feature-card"><strong>Study Notes</strong><br/>Quick topic summaries built from your PDFs.</div>
            </div>
            <span class="hero-cta">Get Started →</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_chat_bubble(role: str, content: str) -> None:
    safe = html.escape(content).replace("\n", "<br>")
    role_class = "user" if role == "user" else "assistant"
    st.markdown(
        f"<div class='chat-row {role_class}'><div class='chat-bubble {role_class}'>{safe}</div></div>",
        unsafe_allow_html=True,
    )


def render_skeleton_loader(lines: int = 4) -> None:
    for _ in range(lines):
        st.markdown("<div class='skeleton'></div>", unsafe_allow_html=True)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def ensure_storage_dirs() -> None:
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._-") or "pdf"
    suffix = Path(name).suffix.lower() or ".pdf"
    return f"{stem}{suffix}"


def default_pdf_library() -> dict:
    return {"uploaded_pdfs": []}


def default_kb_stats() -> dict:
    return {
        "signature": "",
        "built_at": "",
        "total_pages": 0,
        "total_chunks": 0,
        "source_chunk_counts": {},
    }


def load_pdf_library() -> dict:
    defaults = default_pdf_library()
    if not PDF_LIBRARY_FILE.exists():
        return defaults
    try:
        with open(PDF_LIBRARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return defaults
        data.setdefault("uploaded_pdfs", [])
        if not isinstance(data["uploaded_pdfs"], list):
            data["uploaded_pdfs"] = []
        return data
    except Exception:
        return defaults


def save_pdf_library(library: dict) -> None:
    ensure_storage_dirs()
    with open(PDF_LIBRARY_FILE, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)


def load_kb_stats() -> dict:
    defaults = default_kb_stats()
    if not KB_STATS_FILE.exists():
        return defaults
    try:
        with open(KB_STATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return defaults
        for key, value in defaults.items():
            data.setdefault(key, value)
        if not isinstance(data.get("source_chunk_counts"), dict):
            data["source_chunk_counts"] = {}
        return data
    except Exception:
        return defaults


def save_kb_stats(stats: dict) -> None:
    ensure_storage_dirs()
    with open(KB_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def create_default_pdf_record() -> dict | None:
    if not DEFAULT_PDF.exists():
        return None
    return {
        "id": "default_articles",
        "kind": "default",
        "original_name": DEFAULT_PDF.name,
        "display_name": DEFAULT_PDF.name,
        "stored_name": DEFAULT_PDF.name,
        "path": str(DEFAULT_PDF),
        "file_hash": compute_file_hash(DEFAULT_PDF),
    }


def register_uploaded_pdfs(uploaded_files) -> tuple[int, int]:
    if not uploaded_files:
        return 0, 0

    ensure_storage_dirs()
    library = load_pdf_library()
    existing_hashes = {item.get("file_hash") for item in library.get("uploaded_pdfs", [])}
    added = 0
    skipped = 0

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getbuffer()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash in existing_hashes:
            skipped += 1
            continue

        stored_name = f"{file_hash[:12]}__{safe_filename(uploaded_file.name)}"
        stored_path = UPLOADS_DIR / stored_name
        with open(stored_path, "wb") as f:
            f.write(file_bytes)

        library["uploaded_pdfs"].append(
            {
                "id": file_hash[:12],
                "kind": "uploaded",
                "original_name": uploaded_file.name,
                "display_name": uploaded_file.name,
                "stored_name": stored_name,
                "path": str(stored_path),
                "file_hash": file_hash,
                "size": len(file_bytes),
                "uploaded_at": int(time.time()),
            }
        )
        existing_hashes.add(file_hash)
        added += 1

    save_pdf_library(library)
    return added, skipped


def remove_uploaded_pdf(record_id: str) -> bool:
    library = load_pdf_library()
    records = library.get("uploaded_pdfs", [])
    target = next((record for record in records if record.get("id") == record_id), None)
    if not target:
        return False

    target_path = Path(target.get("path", ""))
    if target_path.exists():
        try:
            target_path.unlink()
        except Exception:
            pass

    library["uploaded_pdfs"] = [record for record in records if record.get("id") != record_id]
    save_pdf_library(library)
    return True


def get_selected_pdf_records(include_default: bool = True) -> list[dict]:
    records = list(load_pdf_library().get("uploaded_pdfs", []))
    if include_default:
        default_record = create_default_pdf_record()
        if default_record:
            records.append(default_record)
    return records


def build_documents_from_sources(source_records: list[dict]) -> tuple[list[Document], dict[str, int], int]:
    all_documents: list[Document] = []
    page_counts: dict[str, int] = {}
    total_pages = 0

    for record in source_records:
        path = Path(record.get("path", ""))
        if not path.exists():
            continue
        source_name = record.get("display_name") or record.get("original_name") or path.name
        source_id = record.get("id") or source_name
        documents, page_count = build_page_documents(path, source_name=source_name, source_id=source_id)
        all_documents.extend(documents)
        page_counts[source_id] = page_count
        total_pages += page_count

    return all_documents, page_counts, total_pages


def build_source_items(docs: list[Document], limit: int = 8) -> list[dict]:
    items = []
    for doc in docs[:limit]:
        snippet = re.sub(r"\s+", " ", doc.page_content[:240]).strip()
        items.append(
            {
                "source": doc.metadata.get("source", "PDF"),
                "page": doc.metadata.get("page", "?"),
                "snippet": snippet,
            }
        )
    return items


def render_sources_expander(sources: list[dict], label: str = "Sources") -> None:
    if not sources:
        return
    with st.expander(label):
        for idx, source in enumerate(sources, start=1):
            st.markdown(
                f"**{idx}. {source.get('source', 'PDF')} · page {source.get('page', '?')}**"
            )
            st.caption(source.get("snippet", ""))


def save_uploaded_pdf(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None
    temp_pdf = Path(tempfile.gettempdir()) / uploaded_file.name
    with open(temp_pdf, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_pdf


def render_page_to_png(page: fitz.Page) -> bytes | None:
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None


def summarize_page_image(image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode()
    try:
        response = get_ollama_client().chat(
            model=st.session_state.get("vision_model", DEFAULT_VISION_MODEL),
            messages=[
                {
                    "role": "user",
                    "content": "Describe this page concisely focusing on diagrams, tables, and key visuals. Be brief.",
                    "images": [base64_image],
                }
            ],
        )
        return response.message.content
    except Exception as exc:
        return f"[Vision analysis failed: {str(exc)[:50]}]"


def build_page_documents(pdf_path: Path, source_name: str | None = None, source_id: str | None = None) -> tuple[list[Document], int]:
    documents = []
    try:
        vision_enabled = st.session_state.get("use_vision", False)
        source_name = source_name or pdf_path.name
        source_id = source_id or source_name
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc[page_num]
                text = normalize_text(page.get_text())
                if text.strip():
                    doc_obj = Document(
                        page_content=text,
                        metadata={"source": source_name, "source_id": source_id, "page": page_num + 1},
                    )
                    documents.append(doc_obj)

                should_use_vision = False
                if vision_enabled:
                    text_len = len(text)
                    should_use_vision = text_len < VISION_TEXT_THRESHOLD

                if should_use_vision:
                    image_bytes = render_page_to_png(page)
                    if image_bytes:
                        image_summary = summarize_page_image(image_bytes)
                        if image_summary:
                            doc_obj = Document(
                                page_content=f"[VISUAL: {image_summary}]",
                                metadata={
                                    "source": source_name,
                                    "source_id": source_id,
                                    "page": page_num + 1,
                                    "type": "visual",
                                },
                            )
                            documents.append(doc_obj)
        return documents, page_count
    except Exception as exc:
        st.error(f"PDF parsing error: {exc}")
        return [], 0


def build_vectorstore(documents: list[Document]) -> tuple[FAISS, int, dict[str, int], float]:
    build_start_time = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No chunks available for embedding")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    batch_size = 20
    worker_count = 4
    text_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    total_batches = len(text_batches)
    embedding_model_name = st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

    progress = st.progress(0.0, text=f"Embedding batch 0 of {total_batches}")
    batch_results: list[list[list[float]] | None] = [None] * total_batches

    def embed_batch(batch_texts: list[str]) -> list[list[float]]:
        embedder = OllamaEmbeddings(model=embedding_model_name, base_url=OLLAMA_HOST)
        return embedder.embed_documents(batch_texts)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(embed_batch, batch_text): batch_idx
            for batch_idx, batch_text in enumerate(text_batches)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            batch_idx = future_to_idx[future]
            batch_results[batch_idx] = future.result()
            completed += 1
            progress.progress(completed / total_batches, text=f"Embedding batch {completed} of {total_batches}")

    text_embeddings: list[tuple[str, list[float]]] = []
    for batch_idx, batch_text in enumerate(text_batches):
        vectors = batch_results[batch_idx] or []
        text_embeddings.extend(zip(batch_text, vectors))

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=get_embeddings(),
        metadatas=metadatas,
    )

    progress.empty()
    chunk_counts = Counter(str(doc.metadata.get("source_id", doc.metadata.get("source", "PDF"))) for doc in chunks)
    build_time_seconds = time.time() - build_start_time
    return vectorstore, len(chunks), dict(chunk_counts), build_time_seconds


def faiss_index_exists(index_dir: Path = FAISS_INDEX_DIR) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def compute_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_index_signature(source_records: list[dict]) -> str:
    sources = []
    for record in source_records:
        path = Path(record.get("path", ""))
        sources.append(
            {
                "id": record.get("id", record.get("source_name", "")),
                "name": record.get("display_name", record.get("original_name", path.name)),
                "hash": record.get("file_hash") or (compute_file_hash(path) if path.exists() else ""),
                "kind": record.get("kind", "uploaded"),
            }
        )
    payload = {
        "embedding_model": st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "use_vision": st.session_state.get("use_vision", False),
        "vision_max_pages": VISION_MAX_PAGES,
        "vision_text_threshold": VISION_TEXT_THRESHOLD,
        "sources": sorted(sources, key=lambda item: item["id"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_index_metadata(index_meta_file: Path = FAISS_META_FILE) -> dict | None:
    if not index_meta_file.exists():
        return None
    try:
        with open(index_meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_vectorstore(vectorstore: FAISS, metadata: dict, index_dir: Path = FAISS_INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    with open(FAISS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f)


def load_vectorstore(index_dir: Path = FAISS_INDEX_DIR) -> FAISS | None:
    if not faiss_index_exists(index_dir):
        return None
    try:
        return FAISS.load_local(
            str(index_dir),
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def get_vectorstore_documents(vectorstore: FAISS) -> list[Document]:
    store = getattr(vectorstore, "docstore", None)
    store_dict = getattr(store, "_dict", {})
    return [doc for doc in store_dict.values() if isinstance(doc, Document)]


def get_vectorstore_stats(vectorstore: FAISS) -> tuple[int, int, list[Document]]:
    docs = get_vectorstore_documents(vectorstore)
    pages = {doc.metadata.get("page") for doc in docs if doc.metadata.get("page") is not None}
    chunk_count = len(getattr(vectorstore, "index_to_docstore_id", {}))
    return len(pages), chunk_count, docs


def should_load_saved_index(source_records: list[dict]) -> tuple[bool, dict | None]:
    if not faiss_index_exists():
        return False, None
    metadata = load_index_metadata()
    if not source_records:
        return True, metadata
    if not metadata:
        return False, None
    expected_signature = build_index_signature(source_records)
    return metadata.get("signature") == expected_signature, metadata


def format_sources(docs: list[Document]) -> list[dict]:
    return build_source_items(docs)


def answer_question(question: str) -> tuple[str, list[Document], float]:
    start_time = time.time()
    if not st.session_state.get("vectorstore"):
        raise ValueError("Knowledge base not initialized")

    try:
        similar_docs = st.session_state.vectorstore.max_marginal_relevance_search(question, k=8, fetch_k=20)
    except Exception:
        similar_docs = st.session_state.vectorstore.similarity_search(question, k=8)

    context = "\n\n".join(
        [
            f"[Page {doc.metadata.get('page', '?')}] {doc.page_content[:700]}"
            for doc in similar_docs
        ]
    )

    history = st.session_state.get("conversation_history", [])[-6:]
    history_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a study assistant. Use the retrieved PDF context first. "
                "If relevant context exists, answer directly and clearly instead of saying irrelevant. "
                "Only say information is unavailable when context truly lacks it. "
                "When possible, mention page numbers from the context.",
            ),
            (
                "human",
                "Conversation so far:\n{history}\n\nContext from PDF:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    chain = prompt | get_chat_model()
    response = chain.invoke({"history": history_text, "context": context, "question": question})
    answer = response.content if hasattr(response, "content") else str(response)
    elapsed = time.time() - start_time
    return answer, similar_docs, elapsed


def generate_mcqs(num_questions: int = 5) -> list[dict]:
    """Generate MCQs from indexed PDF context."""
    if not st.session_state.get("vectorstore"):
        return []

    sample_docs = st.session_state.vectorstore.similarity_search("key concepts important", k=6)
    context = "\n\n".join([doc.page_content[:300] for doc in sample_docs])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert MCQ generator for study material. "
                "Generate exactly {num} multiple-choice questions with clear structure. "
                "For each question: provide question text, 4 options (a, b, c, d), correct answer letter, and brief explanation. "
                "Use this format:\nMCQ X) Question text?\na) Option\nb) Option\nc) Option\nd) Option\nAnswer: X\nExplanation: Brief reason.",
            ),
            (
                "human",
                "Generate {num} MCQs based on this PDF content:\n\n{context}",
            ),
        ]
    )

    chain = prompt | get_chat_model()
    try:
        response = chain.invoke({"num": num_questions, "context": context})
        text = response.content if hasattr(response, "content") else str(response)
        mcqs = parse_mcq_response(text)
        return mcqs[:num_questions]
    except Exception as exc:
        st.error(f"MCQ generation failed: {exc}")
        return []


def normalize_exam_answer(value) -> str:
    if isinstance(value, int):
        if 0 <= value <= 3:
            return "ABCD"[value]
        if 1 <= value <= 4:
            return "ABCD"[value - 1]
    text = str(value).strip().upper()
    text = re.sub(r"[^A-D]", "", text)
    return text[:1] if text[:1] in "ABCD" else ""


def parse_exam_questions_response(text: str) -> list[dict]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    questions: list[dict] = []
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    question = str(item.get("question", "")).strip()
                    options = item.get("options", [])
                    answer = normalize_exam_answer(item.get("answer", item.get("correct_answer", "")))
                    explanation = str(item.get("explanation", "")).strip()
                    concept = str(item.get("concept", item.get("topic", ""))).strip()
                    source_page = str(item.get("source_page", item.get("page", "?"))).strip() or "?"
                    if isinstance(options, str):
                        options = [opt.strip() for opt in re.split(r"\n+|\|", options) if opt.strip()]
                    if question and len(options) >= 4 and answer in "ABCD":
                        questions.append(
                            {
                                "question": question,
                                "options": [str(opt).strip() for opt in options[:4]],
                                "answer": answer,
                                "explanation": explanation,
                                "concept": concept,
                                "source_page": source_page,
                            }
                        )
        except Exception:
            questions = []

    if questions:
        return questions

    fallback = parse_mcq_response(text)
    for mcq in fallback:
        options = []
        for option in mcq.get("options", [])[:4]:
            cleaned_option = re.sub(r"^[abcdABCD][\)\.\-\:]\s*", "", option).strip()
            options.append(cleaned_option)
        while len(options) < 4:
            options.append(f"Option {len(options) + 1}")
        questions.append(
            {
                "question": str(mcq.get("question", "")).strip(),
                "options": options,
                "answer": normalize_exam_answer(mcq.get("answer", "")) or "A",
                "explanation": str(mcq.get("explanation", "")).strip(),
                "concept": "",
                "source_page": "?",
            }
        )
    return questions


def generate_exam_questions(topic: str, num_questions: int, difficulty: str) -> list[dict]:
    if not st.session_state.get("vectorstore"):
        return []

    retrieved = st.session_state.vectorstore.similarity_search(topic, k=max(10, num_questions * 2))
    context = "\n\n".join(
        [f"[Page {d.metadata.get('page', '?')}] {d.page_content[:800]}" for d in retrieved]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an exam question generator for study material. "
                "Generate exactly {num_questions} MCQs in valid JSON only. "
                "Each item must have: question, options (list of 4 strings), answer (A/B/C/D), explanation, concept, source_page. "
                "Difficulty is {difficulty}. "
                "Make questions based only on the provided PDF context. "
                "Do not add markdown or extra text.",
            ),
            (
                "human",
                "Topic: {topic}\nDifficulty: {difficulty}\n\nContext:\n{context}",
            ),
        ]
    )

    chain = prompt | get_exam_model()
    try:
        response = chain.invoke(
            {"topic": topic, "difficulty": difficulty, "context": context, "num_questions": num_questions}
        )
        text = response.content if hasattr(response, "content") else str(response)
        questions = parse_exam_questions_response(text)
        return questions[:num_questions]
    except Exception as exc:
        st.error(f"Exam question generation failed: {exc}")
        return []


def build_exam_report(exam: dict) -> dict:
    questions = exam.get("questions", [])
    total = len(questions)
    score = 0
    question_reports = []
    weak_topics: list[str] = []
    weak_seen = set()

    for idx, question in enumerate(questions, start=1):
        selected = exam.get("answers", {}).get(str(idx), "")
        correct = str(question.get("answer", "")).strip().upper()
        is_correct = selected == correct
        if is_correct:
            score += 1
        else:
            concept = question.get("concept") or question.get("source_page") or exam.get("topic", "General")
            concept = str(concept).strip() or exam.get("topic", "General")
            if concept.lower() not in weak_seen:
                weak_seen.add(concept.lower())
                weak_topics.append(concept)
        question_reports.append(
            {
                "number": idx,
                "question": question.get("question", ""),
                "selected": selected or "Not answered",
                "correct": correct,
                "options": question.get("options", []),
                "explanation": question.get("explanation", ""),
                "concept": question.get("concept", ""),
                "source_page": question.get("source_page", "?"),
                "is_correct": is_correct,
            }
        )

    score_pct = round((score / total) * 100, 1) if total else 0.0
    started_at = float(exam.get("started_at", time.time()))
    ended_at = float(exam.get("ended_at", time.time()))
    time_taken = max(0.0, ended_at - started_at)

    return {
        "topic": exam.get("topic", "Exam"),
        "difficulty": exam.get("difficulty", "medium"),
        "total_questions": total,
        "correct_answers": score,
        "score_pct": score_pct,
        "time_limit_minutes": int(exam.get("time_limit_minutes", 0)),
        "time_taken_seconds": time_taken,
        "submitted_by_timeout": bool(exam.get("submitted_by_timeout", False)),
        "weak_topics": weak_topics,
        "question_reports": question_reports,
    }


def finalize_exam(timeout: bool = False) -> None:
    if st.session_state.get("exam_submitted"):
        return

    exam = {
        "topic": st.session_state.get("exam_topic", "Exam"),
        "difficulty": st.session_state.get("exam_difficulty", "medium"),
        "time_limit_minutes": st.session_state.get("exam_time_limit_minutes", 0),
        "started_at": st.session_state.get("exam_started_at", time.time()),
        "ended_at": time.time(),
        "submitted_by_timeout": timeout,
        "questions": st.session_state.get("exam_questions", []),
        "answers": {},
    }

    for idx, _question in enumerate(exam["questions"], start=1):
        answer_key = f"exam_answer_{st.session_state.get('exam_run_id', 0)}_{idx}"
        selected = st.session_state.get(answer_key, "")
        exam["answers"][str(idx)] = str(selected).strip().upper()

    report = build_exam_report(exam)
    st.session_state.exam_report = report
    st.session_state.exam_submitted = True
    st.session_state.exam_running = False
    add_quiz_score(int(report["score_pct"]))
    add_topic_studied(str(report["topic"]))
    for weak_topic in report.get("weak_topics", []):
        add_topic_studied(str(weak_topic))


def render_exam_report(report: dict) -> None:
    st.markdown("### 📋 Exam Report")
    st.metric("Total Score", f"{report.get('score_pct', 0):.1f}%")
    st.metric("Correct Answers", f"{report.get('correct_answers', 0)}/{report.get('total_questions', 0)}")
    st.metric("Time Taken", f"{report.get('time_taken_seconds', 0):.0f} seconds")

    st.markdown("**Weak Topics**")
    weak_topics = report.get("weak_topics", [])
    if weak_topics:
        for topic in weak_topics:
            st.write(f"- {topic}")
    else:
        st.write("- None")

    st.markdown("---")
    for item in report.get("question_reports", []):
        with st.container(border=True):
            st.write(f"**Question {item['number']}**")
            st.write(item.get("question", ""))
            st.write(f"**Your answer:** {item.get('selected', 'Not answered')}")
            st.write(f"**Correct answer:** {item.get('correct', '')}")
            st.write(f"**Explanation:** {item.get('explanation', '')}")
            st.caption(f"Concept: {item.get('concept', '')} | Page: {item.get('source_page', '?')}")


def build_exam_report_pdf(report: dict) -> bytes:
    try:
        canvas_module = importlib.import_module("reportlab.pdfgen.canvas")
        letter = importlib.import_module("reportlab.lib.pagesizes").letter
        colors = importlib.import_module("reportlab.lib.colors")
    except Exception as exc:
        raise RuntimeError(f"reportlab not available: {exc}") from exc

    buffer = io.BytesIO()
    canvas = canvas_module.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    def write_line(text: str, font: str = "Helvetica", size: int = 10, color=None, gap: int = 14):
        nonlocal y
        if y < 60:
            canvas.showPage()
            y = height - 50
        canvas.setFont(font, size)
        if color is not None:
            canvas.setFillColor(color)
        canvas.drawString(40, y, str(text)[:120])
        canvas.setFillColor(colors.black)
        y -= gap

    write_line("Exam Report", font="Helvetica-Bold", size=16, gap=22)
    write_line(f"Topic: {report.get('topic', '')}")
    write_line(f"Difficulty: {report.get('difficulty', '')}")
    write_line(f"Score: {report.get('score_pct', 0):.1f}%")
    write_line(f"Correct: {report.get('correct_answers', 0)}/{report.get('total_questions', 0)}")
    write_line(f"Time taken: {report.get('time_taken_seconds', 0):.0f} seconds")
    write_line("Weak Topics:", font="Helvetica-Bold", size=11, gap=16)
    weak_topics = report.get("weak_topics", [])
    if weak_topics:
        for topic in weak_topics:
            write_line(f"- {topic}")
    else:
        write_line("- None")

    write_line("Questions:", font="Helvetica-Bold", size=11, gap=16)
    for item in report.get("question_reports", []):
        write_line(f"Q{item.get('number')}: {item.get('question', '')}", font="Helvetica-Bold", size=9, gap=12)
        write_line(f"Your answer: {item.get('selected', 'Not answered')}")
        write_line(f"Correct answer: {item.get('correct', '')}")
        write_line(f"Explanation: {item.get('explanation', '')}")
        write_line(f"Concept: {item.get('concept', '')} | Page: {item.get('source_page', '?')}", size=8, gap=12)
        y -= 6

    canvas.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def reset_exam_state() -> None:
    st.session_state.exam_questions = []
    st.session_state.exam_topic = ""
    st.session_state.exam_difficulty = "medium"
    st.session_state.exam_num_questions = 5
    st.session_state.exam_time_limit_minutes = 10
    st.session_state.exam_started_at = 0.0
    st.session_state.exam_end_at = 0.0
    st.session_state.exam_run_id = 0
    st.session_state.exam_running = False
    st.session_state.exam_submitted = False
    st.session_state.exam_report = {}
    st.session_state.exam_timeout = False


def parse_mcq_response(text: str) -> list[dict]:
    """Parse MCQ response into structured format."""
    mcqs = []
    current_mcq = None
    lines = text.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a question line (MCQ X))
        if line.lower().startswith("mcq ") or (")" in line and any(c.isdigit() for c in line[:5])):
            if current_mcq and "question" in current_mcq:
                mcqs.append(current_mcq)
            current_mcq = {"question": line, "options": [], "answer": "", "explanation": ""}
        
        # Check for options (a), b), c), d))
        elif current_mcq and line and len(line) > 0 and line[0] in "abcd" and ")" in line:
            current_mcq["options"].append(line)
        
        # Check for answer
        elif current_mcq and "answer:" in line.lower():
            current_mcq["answer"] = line.split(":", 1)[1].strip() if ":" in line else ""
        
        # Check for explanation
        elif current_mcq and "explanation:" in line.lower():
            current_mcq["explanation"] = line.split(":", 1)[1].strip() if ":" in line else ""
    
    # Add the last MCQ if it exists
    if current_mcq and "question" in current_mcq:
        mcqs.append(current_mcq)
    
    return mcqs


def generate_study_notes(topic: str, max_points: int = 8) -> str:
    """Generate concise study notes from indexed PDF context."""
    if not st.session_state.get("vectorstore"):
        return "Knowledge base not initialized."

    retrieved = st.session_state.vectorstore.similarity_search(topic, k=12)
    text_first = [d for d in retrieved if d.metadata.get("type", "text") != "visual"]
    selected_docs = text_first[:8] if text_first else retrieved[:8]

    context = "\n\n".join(
        [f"[Page {d.metadata.get('page', '?')}] {d.page_content[:700]}" for d in selected_docs]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a study notes assistant. Create clean, exam-ready notes from the provided PDF context. "
                "Use short headings and bullet points. Keep it factual and concise.",
            ),
            (
                "human",
                "Topic: {topic}\nMax points: {max_points}\n\nContext:\n{context}\n\n"
                "Return:\n"
                "1) Quick summary (2-3 lines)\n"
                "2) Key points as bullets\n"
                "3) Important terms/formulas (if any)\n"
                "4) Mention page numbers when possible.",
            ),
        ]
    )

    chain = prompt | get_chat_model()
    response = chain.invoke({"topic": topic, "max_points": max_points, "context": context})
    return response.content if hasattr(response, "content") else str(response)


def parse_flashcards_response(text: str) -> list[dict]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    cards: list[dict] = []

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    question = str(item.get("question", item.get("q", ""))).strip()
                    answer = str(item.get("answer", item.get("a", ""))).strip()
                    source_page = str(item.get("source_page", item.get("page", "?"))).strip() or "?"
                    if question and answer:
                        cards.append(
                            {
                                "question": question,
                                "answer": answer,
                                "source_page": source_page,
                            }
                        )
        except Exception:
            cards = []

    if cards:
        return cards

    # Fallback parser for non-JSON outputs: captures Q:/A:/Page: style blocks.
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Card\s*\d+[:\-)\.]\s*)?Q(?:uestion)?\s*[:\-)\.]+\s*(.*?)\s*"
        r"\n\s*A(?:nswer)?\s*[:\-)\.]+\s*(.*?)\s*"
        r"(?:\n\s*(?:Source\s*Page|Page)\s*[:\-)\.]+\s*([^\n]+))?",
        flags=re.IGNORECASE | re.DOTALL,
    )
    fallback_cards = []
    for match in pattern.finditer(cleaned):
        question = re.sub(r"\s+", " ", match.group(1)).strip()
        answer = re.sub(r"\s+", " ", match.group(2)).strip()
        source_page = (match.group(3) or "?").strip()
        if question and answer:
            fallback_cards.append(
                {
                    "question": question,
                    "answer": answer,
                    "source_page": source_page,
                }
            )
    return fallback_cards


def build_flashcards_from_chunks(retrieved_docs: list[Document], num_cards: int) -> list[dict]:
    cards: list[dict] = []
    for doc in retrieved_docs:
        page = str(doc.metadata.get("page", "?"))
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        for sentence in sentences:
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if len(cleaned) < 40:
                continue
            cards.append(
                {
                    "question": f"What does the document say about: {cleaned[:90]}...?",
                    "answer": cleaned,
                    "source_page": page,
                }
            )
            if len(cards) >= num_cards:
                return cards
    return cards


def generate_flashcards(topic: str, num_cards: int = 10) -> list[dict]:
    if not st.session_state.get("vectorstore"):
        return []

    retrieved = st.session_state.vectorstore.similarity_search(topic, k=14)
    context = "\n\n".join(
        [f"[Page {d.metadata.get('page', '?')}] {d.page_content[:700]}" for d in retrieved]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You create concise study flashcards from provided context. "
                "Return exactly {num_cards} cards and nothing else. "
                "Use this exact plain-text format for each card:\n"
                "CARD\nQ: <question>\nA: <answer>\nPAGE: <page_number>\nENDCARD\n"
                "Rules: do not output JSON, do not use markdown code fences, do not skip ENDCARD. "
                "PAGE must be a page number that exists in the provided context.",
            ),
            (
                "human",
                "Topic: {topic}\n\nContext:\n{context}",
            ),
        ]
    )

    chain = prompt | get_flashcard_model()
    response = chain.invoke({"topic": topic, "context": context, "num_cards": num_cards})
    text = response.content if hasattr(response, "content") else str(response)
    cards = parse_flashcards_response(text)[:num_cards]

    if len(cards) < num_cards:
        card_blocks = re.findall(r"CARD\s*(.*?)\s*ENDCARD", text, flags=re.DOTALL | re.IGNORECASE)
        parsed_from_blocks: list[dict] = []
        for block in card_blocks:
            q_match = re.search(r"Q\s*:\s*(.*)", block, flags=re.IGNORECASE)
            a_match = re.search(r"A\s*:\s*(.*)", block, flags=re.IGNORECASE)
            p_match = re.search(r"PAGE\s*:\s*([^\n]+)", block, flags=re.IGNORECASE)
            if not q_match or not a_match:
                continue
            parsed_from_blocks.append(
                {
                    "question": q_match.group(1).strip(),
                    "answer": a_match.group(1).strip(),
                    "source_page": (p_match.group(1).strip() if p_match else "?"),
                }
            )
        cards = (cards + parsed_from_blocks)[:num_cards]

    if len(cards) < num_cards:
        fallback_cards = build_flashcards_from_chunks(retrieved, num_cards)
        cards = (cards + fallback_cards)[:num_cards]

    return cards


def flashcards_to_csv(cards: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["question", "answer", "source_page"])
    writer.writeheader()
    writer.writerows(cards)
    return output.getvalue()


def render_flashcards(cards: list[dict]) -> None:
    if not cards:
        return

    total = len(cards)
    current_idx = int(st.session_state.get("flashcard_index", 0))
    current_idx = max(0, min(current_idx, total - 1))
    flipped = bool(st.session_state.get("flashcard_flipped", False))
    card = cards[current_idx]

    question = html.escape(str(card.get("question", ""))).replace("\n", "<br>")
    answer = html.escape(str(card.get("answer", ""))).replace("\n", "<br>")
    source_page = html.escape(str(card.get("source_page", "?")))
    flip_class = " style='transform: rotateY(180deg);'" if flipped else ""

    left, center, right = st.columns([1, 6, 1])
    with left:
        if st.button("⬅", key="flash_prev", use_container_width=True, help="Previous card"):
            st.session_state.flashcard_index = (current_idx - 1) % total
            st.session_state.flashcard_flipped = False
            st.rerun()

    with center:
        st.markdown(
            (
                f'<div class="flip-card" style="height:280px;">'
                f'<div class="flip-card-inner"{flip_class}>'
                f'<div class="flip-card-front">'
                f'<div class="flashcard-label orbitron">Question</div>'
                f'<p style="font-size:1.15rem;"><strong>{question}</strong></p>'
                f'</div>'
                f'<div class="flip-card-back" style="background:#1abc9c22; border-color:#1abc9c;">'
                f'<div class="flashcard-label orbitron">Answer</div>'
                f'<p>{answer}</p>'
                f'<p style="margin-top:0.8rem;"><strong>Source page:</strong> {source_page}</p>'
                f'</div>'
                f'</div>'
                f'</div>'
            ),
            unsafe_allow_html=True,
        )
        if st.button("Flip Card", key="flash_flip", use_container_width=True, help="Flip card"):
            st.session_state.flashcard_flipped = not flipped
            st.rerun()
        st.caption(f"Card {current_idx + 1} of {total}")

    with right:
        if st.button("➡", key="flash_next", use_container_width=True, help="Next card"):
            st.session_state.flashcard_index = (current_idx + 1) % total
            st.session_state.flashcard_flipped = False
            st.rerun()


def default_progress_data() -> dict:
    return {
        "last_active_date": "",
        "study_streak": 0,
        "total_questions_asked": 0,
        "quiz_scores": [],
        "topics_studied": [],
    }


def load_progress_data() -> dict:
    defaults = default_progress_data()
    if not PROGRESS_FILE.exists():
        return defaults
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return defaults
        for key, value in defaults.items():
            data.setdefault(key, value)
        if not isinstance(data.get("quiz_scores"), list):
            data["quiz_scores"] = []
        if not isinstance(data.get("topics_studied"), list):
            data["topics_studied"] = []
        return data
    except Exception:
        return defaults


def save_progress_data() -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.progress, f, indent=2)


def update_daily_streak() -> None:
    progress = st.session_state.progress
    today = date.today()
    today_str = today.isoformat()
    last_active = progress.get("last_active_date", "")

    if not last_active:
        progress["study_streak"] = 1
        progress["last_active_date"] = today_str
        save_progress_data()
        return

    if last_active == today_str:
        return

    try:
        last_date = date.fromisoformat(last_active)
    except ValueError:
        progress["study_streak"] = 1
        progress["last_active_date"] = today_str
        save_progress_data()
        return

    if last_date == today - timedelta(days=1):
        progress["study_streak"] = int(progress.get("study_streak", 0)) + 1
    else:
        progress["study_streak"] = 1

    progress["last_active_date"] = today_str
    save_progress_data()


def add_topic_studied(topic: str) -> None:
    cleaned = re.sub(r"\s+", " ", topic).strip()
    if not cleaned:
        return
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip() + "..."
    topics = st.session_state.progress.setdefault("topics_studied", [])
    lowered = {t.lower() for t in topics}
    if cleaned.lower() not in lowered:
        topics.append(cleaned)
        save_progress_data()


def increment_total_questions() -> None:
    value = int(st.session_state.progress.get("total_questions_asked", 0))
    st.session_state.progress["total_questions_asked"] = value + 1
    save_progress_data()


def add_quiz_score(score: int) -> None:
    st.session_state.progress.setdefault("quiz_scores", []).append(
        {"date": date.today().isoformat(), "score": int(score)}
    )
    save_progress_data()


def render_progress_dashboard() -> None:
    progress = st.session_state.progress
    quiz_scores = progress.get("quiz_scores", [])
    quizzes_taken = len(quiz_scores)
    avg_score = (
        sum(float(item.get("score", 0)) for item in quiz_scores) / quizzes_taken
        if quizzes_taken
        else 0.0
    )

    st.markdown("---")
    st.markdown("### 🔥 Progress Tracker")
    st.metric("Daily Study Streak", f"{int(progress.get('study_streak', 0))} days")
    st.metric("Total Questions Asked", int(progress.get("total_questions_asked", 0)))
    st.metric("Quizzes Taken", quizzes_taken)
    st.metric("Average Quiz Score", f"{avg_score:.1f}%")

    q_count = int(progress.get("total_questions_asked", 0))
    t_count = len(progress.get("topics_studied", []))
    qz_count = quizzes_taken
    ring_html = f"""
    <div style='display:flex; gap:8px; justify-content:space-between; margin:8px 0 10px 0;'>
        <div style='text-align:center;'>
            <svg width='72' height='72' viewBox='0 0 120 120'>
                <circle cx='60' cy='60' r='48' stroke='#2b2b35' stroke-width='10' fill='none'/>
                <circle cx='60' cy='60' r='48' stroke='#4f8ef7' stroke-width='10' fill='none' stroke-dasharray='{min(300, q_count*8)} 302' transform='rotate(-90 60 60)'/>
                <text x='60' y='66' text-anchor='middle' fill='#fff' font-size='18'>{q_count}</text>
            </svg>
            <div class='copy-tip'>Questions</div>
        </div>
        <div style='text-align:center;'>
            <svg width='72' height='72' viewBox='0 0 120 120'>
                <circle cx='60' cy='60' r='48' stroke='#2b2b35' stroke-width='10' fill='none'/>
                <circle cx='60' cy='60' r='48' stroke='#9b59b6' stroke-width='10' fill='none' stroke-dasharray='{min(300, qz_count*30)} 302' transform='rotate(-90 60 60)'/>
                <text x='60' y='66' text-anchor='middle' fill='#fff' font-size='18'>{qz_count}</text>
            </svg>
            <div class='copy-tip'>Quizzes</div>
        </div>
        <div style='text-align:center;'>
            <svg width='72' height='72' viewBox='0 0 120 120'>
                <circle cx='60' cy='60' r='48' stroke='#2b2b35' stroke-width='10' fill='none'/>
                <circle cx='60' cy='60' r='48' stroke='#1abc9c' stroke-width='10' fill='none' stroke-dasharray='{min(300, t_count*18)} 302' transform='rotate(-90 60 60)'/>
                <text x='60' y='66' text-anchor='middle' fill='#fff' font-size='18'>{t_count}</text>
            </svg>
            <div class='copy-tip'>Topics</div>
        </div>
    </div>
    """
    st.markdown(ring_html, unsafe_allow_html=True)

    if quiz_scores:
        try:
            plt = importlib.import_module("matplotlib.pyplot")
        except Exception:
            st.warning("matplotlib is not installed. Install dependencies to see quiz score charts.")
            return
        labels = [item.get("date", "") for item in quiz_scores]
        values = [float(item.get("score", 0)) for item in quiz_scores]
        fig, ax = plt.subplots(figsize=(4.2, 2.2))
        ax.bar(range(len(values)), values, color="#4f8cff")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score")
        ax.set_title("Quiz Scores Over Time")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.caption("No quiz scores yet.")

    st.markdown("**Covered Topics**")
    topics = progress.get("topics_studied", [])
    if topics:
        for idx, topic in enumerate(topics, start=1):
            st.checkbox(topic, value=True, disabled=True, key=f"topic_done_{idx}_{hash(topic)}")
    else:
        st.caption("No topics tracked yet.")


def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "conversation_history": [],
        "generated_notes": "",
        "vectorstore": None,
        "source_documents": [],
        "knowledge_ready": False,
        "chat_model": DEFAULT_CHAT_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "vision_model": DEFAULT_VISION_MODEL,
        "use_vision": False,
        "page_count": 0,
        "chunk_count": 0,
        "mcqs": [],
        "quiz_submitted": False,
        "quiz_score_pct": 0.0,
        "flashcards": [],
        "flashcard_index": 0,
        "flashcard_flipped": False,
        "progress": load_progress_data(),
        "pdf_library": load_pdf_library(),
        "kb_stats": load_kb_stats(),
        "exam_questions": [],
        "exam_topic": "important concepts",
        "exam_difficulty": "medium",
        "exam_num_questions": 5,
        "exam_time_limit_minutes": 10,
        "exam_started_at": 0.0,
        "exam_end_at": 0.0,
        "exam_run_id": 0,
        "exam_running": False,
        "exam_submitted": False,
        "exam_report": {},
        "exam_timeout": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


initialize_session_state()
update_daily_streak()


def render_kb_manager() -> None:
    st.markdown("**Knowledge Base Manager**")
    st.markdown("<div class='upload-zone'>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploads",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_add, col_refresh = st.columns(2)
    with col_add:
        if st.button("Add PDFs to Library", use_container_width=True):
            if uploaded_files:
                added, skipped = register_uploaded_pdfs(uploaded_files)
                if added or skipped:
                    st.session_state.pdf_library = load_pdf_library()
                    st.session_state.vectorstore = None
                    st.session_state.knowledge_ready = False
                    st.success(f"Added {added} PDF(s); skipped {skipped} duplicate(s).")
                    st.rerun()
            else:
                st.info("Select one or more PDFs first.")
    with col_refresh:
        if st.button("Refresh Library", use_container_width=True):
            st.session_state.pdf_library = load_pdf_library()
            st.session_state.kb_stats = load_kb_stats()
            st.rerun()

    st.caption("Uploaded PDFs are stored locally and reused on rebuild.")

    records = st.session_state.pdf_library.get("uploaded_pdfs", [])
    source_chunk_counts = st.session_state.kb_stats.get("source_chunk_counts", {})
    if records:
        for record in records:
            source_id = record.get("id", record.get("display_name", ""))
            chunk_count = int(source_chunk_counts.get(source_id, 0))
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**{record.get('original_name', record.get('display_name', 'PDF'))}**")
                st.caption(f"{chunk_count} chunks")
            with right:
                if st.button("Remove", key=f"remove_pdf_{source_id}", use_container_width=True):
                    if remove_uploaded_pdf(source_id):
                        st.session_state.pdf_library = load_pdf_library()
                        st.session_state.vectorstore = None
                        st.session_state.knowledge_ready = False
                        st.session_state.kb_stats = load_kb_stats()
                        st.warning("PDF removed. Rebuild the knowledge base to refresh the index.")
                        st.rerun()
    else:
        st.caption("No uploaded PDFs in the library yet.")

with st.sidebar:
    st.markdown(
        """
        <div class="hero-wrap">
            <h1>📚 Study Buddy</h1>
            <p>Local RAG powered by Ollama · Zero API keys · 100% private</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Build Knowledge Base**")
    render_model_status_pills()
    use_default = st.checkbox("Use default PDF", value=True, help="Use the default PDF if available")
    col1, col2 = st.columns(2)
    with col1:
        use_vision_toggle = st.checkbox(
            "Enable Visual Page Analysis (slower)",
            value=st.session_state.use_vision,
        )
        st.session_state.use_vision = use_vision_toggle
        if st.session_state.use_vision:
            st.warning("Vision mode will significantly slow KB build time.")

    with col2:
        if st.button("Build KB 🔨", use_container_width=True, type="primary"):
            selected_sources = get_selected_pdf_records(include_default=use_default)
            if not selected_sources:
                st.error("Please upload at least one PDF or use the default PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    step_placeholder = st.empty()
                    build_step_bar = st.progress(0.0, text="Starting build...")
                    can_load_saved_index, saved_meta = should_load_saved_index(selected_sources)
                    loaded_vectorstore = load_vectorstore() if can_load_saved_index else None

                    if loaded_vectorstore is not None:
                        step_placeholder.info("Loading existing FAISS index...")
                        build_step_bar.progress(1.0, text="Loaded from disk")
                        vectorstore = loaded_vectorstore
                        documents = get_vectorstore_documents(loaded_vectorstore)
                        if saved_meta and "page_count" in saved_meta and "chunk_count" in saved_meta:
                            page_count = int(saved_meta["page_count"])
                            chunk_count = int(saved_meta["chunk_count"])
                            source_chunk_counts = saved_meta.get("source_chunk_counts", {})
                        else:
                            page_count, chunk_count, _ = get_vectorstore_stats(loaded_vectorstore)
                            source_chunk_counts = {}
                        loaded_from_disk = True
                    else:
                        step_placeholder.info("Extracting text...")
                        build_step_bar.progress(0.2, text="Extracting text...")
                        documents, source_page_counts, page_count = build_documents_from_sources(selected_sources)
                        if not documents:
                            st.error("No content extracted from the selected PDFs")
                            vectorstore = None
                        else:
                            step_placeholder.info("Chunking...")
                            build_step_bar.progress(0.45, text="Chunking...")
                            step_placeholder.info("Embedding...")
                            build_step_bar.progress(0.6, text="Embedding...")
                            vectorstore, chunk_count, source_chunk_counts, build_time_seconds = build_vectorstore(documents)
                            step_placeholder.info("Storing in FAISS...")
                            build_step_bar.progress(0.9, text="Storing in FAISS...")
                            metadata = {
                                "signature": build_index_signature(selected_sources),
                                "page_count": page_count,
                                "chunk_count": chunk_count,
                                "build_time_seconds": round(build_time_seconds, 2),
                                "source_chunk_counts": source_chunk_counts,
                                "source_page_counts": source_page_counts,
                                "sources": [
                                    {
                                        "id": record.get("id"),
                                        "name": record.get("display_name", record.get("original_name", "PDF")),
                                        "kind": record.get("kind", "uploaded"),
                                    }
                                    for record in selected_sources
                                ],
                                "saved_at": int(time.time()),
                            }
                            save_vectorstore(vectorstore, metadata)
                            save_kb_stats(
                                {
                                    "signature": metadata["signature"],
                                    "built_at": metadata["saved_at"],
                                    "total_pages": page_count,
                                    "total_chunks": chunk_count,
                                    "build_time_seconds": metadata["build_time_seconds"],
                                    "source_chunk_counts": source_chunk_counts,
                                }
                            )
                            st.session_state.kb_stats = load_kb_stats()
                            build_step_bar.progress(1.0, text="Done")
                            step_placeholder.empty()
                            loaded_from_disk = False

                    if vectorstore is not None:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.source_documents = documents
                        st.session_state.generated_notes = ""
                        st.session_state.knowledge_ready = True
                        st.session_state.page_count = page_count
                        st.session_state.chunk_count = chunk_count
                        st.session_state.kb_stats = load_kb_stats() if loaded_from_disk else st.session_state.kb_stats
                        if loaded_from_disk:
                            st.success(f"KB loaded from disk: {page_count} pages, {chunk_count} chunks")
                            st.toast("KB loaded successfully")
                        else:
                            st.success(
                                f"KB built and saved: {page_count} pages, {chunk_count} chunks in {build_time_seconds:.2f} seconds"
                            )
                            st.toast("KB built successfully")

    st.markdown("---")
    render_kb_manager()

    st.markdown("---")
    st.markdown("**Model Configuration**")

    model = st.selectbox(
        "Chat Model",
        ["llama3.2:3b", "llama2", "neural-chat"],
        index=0,
        key="model_select",
    )
    st.session_state.chat_model = model

    embed_model = st.selectbox(
        "Embedding Model",
        ["nomic-embed-text", "all-minilm"],
        index=0,
        key="embed_select",
    )
    st.session_state.embedding_model = embed_model

    vision_model = st.selectbox(
        "Vision Model",
        ["llava:7b", "llava:13b"],
        index=0,
        key="vision_select",
    )
    st.session_state.vision_model = vision_model

    st.markdown("---")
    if st.session_state.knowledge_ready:
        st.markdown("✅ **Knowledge Base Ready**")
    else:
        st.markdown("⏳ **Build knowledge base to start**")

    render_progress_dashboard()

col1, col2, col3, col4 = st.columns(4)

status_text = "Ready" if st.session_state.knowledge_ready else "Idle"
status_color = "#0f766e" if st.session_state.knowledge_ready else "#b45309"

with col1:
    st.markdown(
        f"<div class='metric-card'><div class='k'>Status</div><div class='v' style='color: {status_color}'>{status_text}</div></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div class='metric-card'><div class='k'>PDF Pages</div><div class='v'>{st.session_state.get('page_count', 0)}</div></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div class='metric-card'><div class='k'>Context Chunks</div><div class='v'>{st.session_state.get('chunk_count', 0)}</div></div>",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"<div class='metric-card'><div class='k'>Chat Memory</div><div class='v'>{len(st.session_state.conversation_history)}</div></div>",
        unsafe_allow_html=True,
    )

selected_tab = st.radio(
    "Navigation",
    ["💬 Doubt Solver", "📝 Quiz Mode", "🧪 Exam Mode", "🃏 Flashcards", "📖 Study Notes", "🧠 How It Works"],
    horizontal=True,
    label_visibility="collapsed",
    key="custom_tab_nav",
)

if selected_tab == "💬 Doubt Solver":
    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        head_left, head_right = st.columns([8, 1])
        with head_left:
            st.subheader("Ask anything from your study material")
        with head_right:
            if st.button("🗑", key="clear_chat_icon", help="Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.toast("Chat history cleared")
                st.rerun()
        st.caption("Examples: Explain architecture diagram · Summarize chapter 2 · List key workflows")

        if not st.session_state.messages:
            st.info("Build the knowledge base first, then start chatting.")
        else:
            for message in st.session_state.messages:
                render_chat_bubble(message["role"], message["content"])
                if message.get("role") == "assistant" and message.get("sources"):
                    render_sources_expander(message["sources"])

        user_question = st.chat_input("Type your doubt here...")

        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.conversation_history.append({"role": "user", "content": user_question})
            increment_total_questions()
            add_topic_studied(user_question)

            if not st.session_state.knowledge_ready:
                error_msg = "Please build the knowledge base first."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
            else:
                typing_box = st.empty()
                typing_box.markdown(
                    "<div class='typing'><span></span><span></span><span></span></div>",
                    unsafe_allow_html=True,
                )
                try:
                    answer, source_docs, elapsed = answer_question(user_question)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": build_source_items(source_docs),
                        }
                    )
                    st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    error_msg = f"I could not answer that yet: {exc}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
                typing_box.empty()
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Study Assistant")
        st.markdown(
            """
            - Ask short, direct doubts for better retrieval.
            - Mention page/topic names if available.
            - Keep image analysis enabled for architecture diagrams.
            - Rebuild the knowledge base after changing PDFs.
            """
        )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Conversation Memory**")
        st.metric("Messages in session", len(st.session_state.conversation_history))
        if st.button("Clear chat history", key="clear_chat", use_container_width=True, help="Clear all chat entries"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.toast("Chat history cleared")
            st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Current Setup**")
        st.markdown(f"- Answer model: `{st.session_state.chat_model}`")
        st.markdown(f"- Embedding model: `{st.session_state.embedding_model}`")
        st.markdown(f"- Vision model: `{st.session_state.vision_model}`")
        st.markdown(f"- Vision mode: {'✅ On' if st.session_state.use_vision else '❌ Off'}")
        st.markdown("</div>", unsafe_allow_html=True)

if selected_tab == "📝 Quiz Mode":
    st.subheader("📝 Quiz Mode - MCQ Generator")
    st.caption("Generate multiple-choice questions from your PDF for quick revision.")

    if not st.session_state.knowledge_ready:
        st.error("⚠️ Build the knowledge base first to generate MCQs.")
    else:
        col_num, col_gen = st.columns([1, 3])
        with col_num:
            num_mcqs = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5)
        with col_gen:
            if st.button("Generate MCQs 🎯", type="primary", use_container_width=True):
                render_skeleton_loader(3)
                st.session_state.mcqs = generate_mcqs(num_mcqs)
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score_pct = 0.0
                add_topic_studied("Quiz practice")
                if st.session_state.mcqs:
                    st.toast("Quiz generated")

        if st.session_state.mcqs and len(st.session_state.mcqs) > 0:
            st.markdown("---")
            for idx, mcq in enumerate(st.session_state.mcqs, start=1):
                with st.container(border=True):
                    st.write(f"**MCQ {idx}: {mcq.get('question', 'Question')}**")
                    options = mcq.get("options", [])
                    user_choice = st.radio(
                        "Choose option",
                        options,
                        key=f"quiz_option_{idx}",
                        label_visibility="collapsed",
                    )
                    if st.session_state.quiz_submitted:
                        correct_letter = normalize_exam_answer(mcq.get("answer", ""))
                        selected_letter = normalize_exam_answer(user_choice)
                        if selected_letter == correct_letter:
                            st.success("Correct")
                        else:
                            st.error(f"Incorrect · Correct answer: {correct_letter}")
                    
                    with st.expander("📋 Show Answer & Explanation"):
                        answer_text = mcq.get("answer", "N/A")
                        explanation_text = mcq.get("explanation", "N/A")
                        st.success(f"**✅ Answer:** {answer_text}")
                        st.info(f"**💡 Explanation:** {explanation_text}")

            st.markdown("---")
            col_exp, col_clear = st.columns(2)
            with col_exp:
                if st.button("📥 Export MCQs as Text", use_container_width=True):
                    export_text = "\n\n".join(
                        [
                            f"MCQ {idx}: {mcq.get('question', '')}\n"
                            f"{chr(10).join(mcq.get('options', []))}\n"
                            f"Answer: {mcq.get('answer', '')}\n"
                            f"Explanation: {mcq.get('explanation', '')}"
                            for idx, mcq in enumerate(st.session_state.mcqs, start=1)
                        ]
                    )
                    st.download_button(
                        label="⬇️ Download MCQs",
                        data=export_text,
                        file_name="mcqs.txt",
                        mime="text/plain",
                    )
            with col_clear:
                if st.button("🔄 Clear MCQs", use_container_width=True):
                    st.session_state.mcqs = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score_pct = 0.0
                    st.rerun()

            if st.button("Submit Quiz", type="primary", use_container_width=True, key="submit_quiz_btn"):
                total = len(st.session_state.mcqs)
                correct = 0
                for idx, mcq in enumerate(st.session_state.mcqs, start=1):
                    selected = st.session_state.get(f"quiz_option_{idx}", "")
                    if normalize_exam_answer(selected) == normalize_exam_answer(mcq.get("answer", "")):
                        correct += 1
                pct = (correct / total) * 100 if total else 0.0
                st.session_state.quiz_submitted = True
                st.session_state.quiz_score_pct = pct
                add_quiz_score(int(pct))
                if pct >= 80:
                    st.balloons()
                    st.toast("Excellent score")
                st.rerun()

            if st.session_state.quiz_submitted:
                pct = st.session_state.quiz_score_pct
                ring_color = "#1abc9c" if pct >= 80 else "#4f8ef7"
                st.markdown(
                    (
                        "<svg width='180' height='180' viewBox='0 0 120 120'>"
                        "<circle cx='60' cy='60' r='46' stroke='#2d2d3a' stroke-width='10' fill='none'/>"
                        f"<circle cx='60' cy='60' r='46' stroke='{ring_color}' stroke-width='10' fill='none' "
                        f"stroke-dasharray='{2.89 * pct} 289' transform='rotate(-90 60 60)'/>"
                        f"<text x='60' y='66' text-anchor='middle' fill='#fff' font-size='16'>{pct:.1f}%</text>"
                        "</svg>"
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            quiz_score = st.slider("Your quiz score (%)", min_value=0, max_value=100, value=70, key="quiz_score_input")
            if st.button("Save Quiz Score", use_container_width=True, key="save_quiz_score"):
                add_quiz_score(quiz_score)
                st.success("Quiz score saved to progress tracker.")

if selected_tab == "🧪 Exam Mode":
        st.subheader("🧪 Exam Mode")
        st.caption("Choose your exam settings, answer the MCQs, and submit before time runs out.")

        if not st.session_state.knowledge_ready:
            st.error("⚠️ Build the knowledge base first to start an exam.")
        else:
            if not st.session_state.exam_running and not st.session_state.exam_submitted:
                col_topic, col_num, col_diff, col_time = st.columns([2, 1, 1, 1])
                with col_topic:
                    exam_topic = st.text_input("Exam topic", value=st.session_state.exam_topic, key="exam_topic_input")
                with col_num:
                    exam_num_questions = st.selectbox("Number of questions", [5, 10, 20], index=[5, 10, 20].index(st.session_state.exam_num_questions), key="exam_num_questions_input")
                with col_diff:
                    exam_difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=["easy", "medium", "hard"].index(st.session_state.exam_difficulty), key="exam_difficulty_input")
                with col_time:
                    exam_time_limit_minutes = st.number_input("Time limit (minutes)", min_value=1, max_value=120, value=int(st.session_state.exam_time_limit_minutes), step=1, key="exam_time_limit_input")

                if st.button("Start Exam", type="primary", use_container_width=True):
                    with st.spinner("Generating exam questions..."):
                        exam_questions = generate_exam_questions(exam_topic, int(exam_num_questions), exam_difficulty)
                        if not exam_questions:
                            st.error("No exam questions were generated. Try a different topic or rebuild the KB.")
                        else:
                            st.session_state.exam_topic = exam_topic
                            st.session_state.exam_num_questions = int(exam_num_questions)
                            st.session_state.exam_difficulty = exam_difficulty
                            st.session_state.exam_time_limit_minutes = int(exam_time_limit_minutes)
                            st.session_state.exam_questions = exam_questions
                            st.session_state.exam_started_at = time.time()
                            st.session_state.exam_end_at = st.session_state.exam_started_at + (int(exam_time_limit_minutes) * 60)
                            st.session_state.exam_run_id = int(time.time())
                            st.session_state.exam_running = True
                            st.session_state.exam_submitted = False
                            st.session_state.exam_report = {}
                            st.session_state.exam_timeout = False
                            st.rerun()

            if st.session_state.exam_running and not st.session_state.exam_submitted:
                remaining_seconds = int(st.session_state.exam_end_at - time.time())
                timer_placeholder = st.empty()
                if remaining_seconds <= 0:
                    timer_placeholder.error("⏰ Time is up. Submitting exam automatically...")
                    finalize_exam(timeout=True)
                    st.rerun()
                else:
                    minutes, seconds = divmod(max(0, remaining_seconds), 60)
                    timer_placeholder.markdown(f"### ⏳ Time Remaining: {minutes:02d}:{seconds:02d}")

                st.markdown("---")
                for idx, question in enumerate(st.session_state.exam_questions, start=1):
                    with st.container(border=True):
                        st.write(f"**Question {idx}**")
                        st.write(question.get("question", ""))
                        for option_letter, option_text in zip("ABCD", question.get("options", [])):
                            st.write(f"{option_letter}. {option_text}")
                        st.radio(
                            "Choose your answer",
                            ["A", "B", "C", "D"],
                            horizontal=True,
                            key=f"exam_answer_{st.session_state.exam_run_id}_{idx}",
                        )

                col_submit, col_reset = st.columns(2)
                with col_submit:
                    if st.button("Submit Exam", type="primary", use_container_width=True):
                        finalize_exam(timeout=False)
                        st.rerun()
                with col_reset:
                    if st.button("Reset Exam", use_container_width=True):
                        reset_exam_state()
                        st.rerun()

                time.sleep(1)
                st.rerun()

            if st.session_state.exam_submitted and st.session_state.exam_report:
                render_exam_report(st.session_state.exam_report)

                try:
                    pdf_bytes = build_exam_report_pdf(st.session_state.exam_report)
                    st.download_button(
                        label="⬇️ Download Exam Report PDF",
                        data=pdf_bytes,
                        file_name="exam_report.pdf",
                        mime="application/pdf",
                    )
                except Exception as exc:
                    st.warning(f"PDF export is unavailable right now: {exc}")

                col_again, col_clear = st.columns(2)
                with col_again:
                    if st.button("Start New Exam", use_container_width=True):
                        reset_exam_state()
                        st.rerun()
                with col_clear:
                    if st.button("Clear Exam Report", use_container_width=True):
                        reset_exam_state()
                        st.rerun()

if selected_tab == "🃏 Flashcards":
    st.subheader("🃏 Flashcard Generator")
    st.caption("Generate Q&A flashcards from retrieved chunks for a topic.")

    if not st.session_state.knowledge_ready:
        st.error("⚠️ Build the knowledge base first to generate flashcards.")
    else:
        col_topic, col_count, col_btn = st.columns([2, 1, 1])
        with col_topic:
            flashcard_topic = st.text_input("Flashcard topic", value="important concepts", key="flashcard_topic")
        with col_count:
            flashcard_count = st.number_input("Count", min_value=1, max_value=30, value=10, step=1, key="flashcard_count")
        with col_btn:
            if st.button("Generate Flashcards 🃏", type="primary", use_container_width=True):
                render_skeleton_loader(3)
                try:
                    st.session_state.flashcards = generate_flashcards(flashcard_topic, num_cards=int(flashcard_count))
                    st.session_state.flashcard_index = 0
                    st.session_state.flashcard_flipped = False
                    add_topic_studied(flashcard_topic)
                    if st.session_state.flashcards:
                        st.toast("Flashcards generated")
                    else:
                        st.warning("Could not parse flashcards from model output. Try again with a narrower topic.")
                except Exception as exc:
                    st.error(f"Flashcard generation failed: {exc}")

        cards = st.session_state.get("flashcards", [])
        if cards:
            render_flashcards(cards)
            csv_data = flashcards_to_csv(cards)
            st.download_button(
                label="⬇️ Export Flashcards to CSV",
                data=csv_data,
                file_name="flashcards.csv",
                mime="text/csv",
            )
            if st.button("🔄 Clear Flashcards", use_container_width=True):
                st.session_state.flashcards = []
                st.session_state.flashcard_index = 0
                st.session_state.flashcard_flipped = False
                st.toast("Flashcards cleared")
                st.rerun()

if selected_tab == "📖 Study Notes":
    st.subheader("📖 Indexed source snippets")
    if st.session_state.knowledge_ready and st.session_state.vectorstore is not None:
        query_preview = st.text_input("Preview snippets for topic", value="architecture overview")

        c1, c2 = st.columns([1, 1])
        with c1:
            points = st.slider("Study note points", min_value=4, max_value=15, value=8)
        with c2:
            if st.button("Generate Study Notes", type="primary", use_container_width=True):
                render_skeleton_loader(4)
                try:
                    st.session_state.generated_notes = generate_study_notes(query_preview, points)
                    add_topic_studied(query_preview)
                    st.toast("Study notes generated")
                except Exception as exc:
                    st.session_state.generated_notes = f"Could not generate notes: {exc}"

        if st.session_state.generated_notes:
            head_l, head_r = st.columns([6, 1])
            with head_l:
                st.markdown("### 🧠 Generated Study Notes")
            with head_r:
                if st.button("📋", key="copy_notes_btn", help="Copy notes text", use_container_width=True):
                    st.toast("Notes copied")
            st.markdown(
                f"<div style='border-left:4px solid #9b59b6; padding-left:0.8rem;'>{st.session_state.generated_notes}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

        raw_docs = st.session_state.vectorstore.similarity_search(query_preview, k=12)
        text_docs = [d for d in raw_docs if d.metadata.get("type", "text") != "visual"]
        sample_docs = text_docs[:4] if text_docs else raw_docs[:4]

        for index, doc in enumerate(sample_docs, start=1):
            doc_type = doc.metadata.get("type", "text")
            with st.expander(
                f"Snippet {index} · {doc.metadata.get('source', 'PDF')} page {doc.metadata.get('page', '?')} ({doc_type})"
            ):
                st.write(doc.page_content)
    else:
        st.info("Build the knowledge base first to preview snippets.")

if selected_tab == "🧠 How It Works":
    st.subheader("🧠 How It Works")
    st.markdown(
        """
        ### Your Study Workflow

        **1. Upload PDF**
        - Upload any PDF or use the default sample
        - Enable vision mode to analyze diagrams and architecture

        **2. Build Knowledge Base**
        - Extract text from all pages
        - Split into overlapping chunks (1200 chars, 200 char overlap)
        - Generate embeddings using local Ollama model
        - Create FAISS vector index for fast retrieval

        **3. Ask Doubts**
        - Type your question in the chat
        - System finds 5 most relevant chunks from your PDF
        - Ollama LLM generates context-aware answer
        - All conversation history stored in session memory (no database needed)

        **4. Generate MCQs**
        - Create practice questions automatically
        - Each MCQ includes answer and explanation
        - Export to text for offline study

        ### Session Memory
        - **Chat history**: All messages kept in session memory (no database)
        - **Conversation context**: Enables follow-up questions
        - **Vector store**: Rebuilt each session for accuracy
        - **Clear history**: Manually clear anytime via Study Assistant panel

        ### Model Stack
        - **Chat**: `llama3.2:3b` (3 billion parameters)
        - **Embeddings**: `nomic-embed-text` (semantic search)
        - **Vision**: `llava:7b` (diagram understanding)
        - **Vector DB**: FAISS (CPU-based, in-memory)

        ### Privacy
        ✅ Fully local - no API calls  
        ✅ No internet required  
        ✅ 100% data privacy  
        ✅ Works offline  
        """
    )


