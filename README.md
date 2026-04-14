# Study Buddy RAG (Local PDF Doubt Solver)

Study Buddy is a complete local RAG app built with Streamlit + Ollama.
It can parse uploaded PDFs, analyze visual pages, answer doubts from context, generate quizzes, and produce study notes.

No cloud API key is required.

---

## 1) Current capabilities

- Doubt Solver chat using retrieved PDF context
- Quiz Mode (MCQ generation + export)
- Study Notes generation from indexed chunks
- Visual-page analysis (architecture diagrams / image-heavy pages)
- Source snippet preview by topic
- Session chat memory (no external database)
- Fully local inference (Ollama models)

---

## 2) Tech stack

### Runtime
- Python 3.12 (tested)
- Streamlit
- Ollama

### Retrieval + LLM
- LangChain Core / Community
- LangChain Ollama
- FAISS (in-memory vector store)
- Recursive text splitter

### PDF and utilities
- PyMuPDF (`fitz`)
- pypdf
- python-dotenv

### Models used
- `llama3.2:3b` (chat / answers)
- `nomic-embed-text` (embeddings)
- `llava:7b` (vision summaries)

---

## 3) Project structure

```text
Question-Answering-System-using-RAG/
  app.py
  requirements.txt
  README.md
  LICENSE
  Artifacts/
    Articles.pdf
```

---

## 4) Prerequisites

1. Install Ollama: https://ollama.com/
2. Install Python 3.10+ (3.12 recommended)
3. Pull models (one-time):

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama pull llava:7b
```

---

## 5) Installation

### Option A: Standard venv setup

```powershell
git clone https://github.com/KalyanM45/Question-Answering-System-using-RAG.git
cd Question-Answering-System-using-RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Exact interpreter command used in this environment

```powershell
Set-Location "C:\Users\Micheal Jebson M\OneDrive\Desktop\genai\Question-Answering-System-using-RAG"
& "C:/Users/Micheal Jebson M/AppData/Local/Programs/Python/Python312/python.exe" -m pip install -r requirements.txt
```

---

## 6) Run the app completely (Windows)

### Step 1: Start Ollama service

```powershell
ollama serve
```

### Step 2: Run Streamlit

```powershell
Set-Location "C:\Users\Micheal Jebson M\OneDrive\Desktop\genai\Question-Answering-System-using-RAG"
& "C:/Users/Micheal Jebson M/AppData/Local/Programs/Python/Python312/python.exe" -m streamlit run app.py --server.port 8504
```

Then open:
- http://localhost:8504

If port is busy, change to another port (8505/8506, etc.).

### Optional clean restart command

```powershell
Stop-Process -Name "python" -Force 2>$null
Set-Location "C:\Users\Micheal Jebson M\OneDrive\Desktop\genai\Question-Answering-System-using-RAG"
& "C:/Users/Micheal Jebson M/AppData/Local/Programs/Python/Python312/python.exe" -m streamlit run app.py --server.port 8504
```

---

## 7) How to use

1. Open the app URL from terminal.
2. In sidebar, verify models:
   - Chat: `llama3.2:3b`
   - Embedding: `nomic-embed-text`
   - Vision: `llava:7b`
3. Upload PDF (uploaded file is prioritized).
4. Click **Build KB 🔨**.
5. Use tabs:
   - **Doubt Solver**: ask questions from the document
   - **Quiz Mode**: generate MCQs and export
   - **Study Notes**: generate topic-wise notes
   - **How It Works**: app workflow summary

---

## 8) Environment configuration (optional)

Create `.env`:

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_VISION_MODEL=llava:7b
```

Defaults are already set in `app.py`, so `.env` is optional.

---

## 9) Features currently in app UI

- Dark theme (black background / white text)
- Knowledge base status cards (pages, chunks, memory)
- Chat memory per session
- Study-note generation with topic input and point count
- Snippet preview with page metadata
- MCQ generation with answer/explanation reveal

---

## 10) Troubleshooting

### `ERR_CONNECTION_REFUSED` / `/_stcore/health`
- Streamlit is not running.
- Restart app from terminal and open the latest URL.

### Port not available
- Change port:

```powershell
& "C:/Users/Micheal Jebson M/AppData/Local/Programs/Python/Python312/python.exe" -m streamlit run app.py --server.port 8505
```

### `No module named ...`
- Install dependencies with the same interpreter used to run Streamlit.

### Slow KB build
- Disable vision for faster indexing.
- Large PDFs with image summaries will take longer.

### Answers feel irrelevant
- Rebuild KB after changing/uploading PDF.
- Ask specific topic/page-based questions.
- Keep model names exactly matching pulled Ollama models.

---

## 11) Notes on architecture

- PDF text + visual summaries are converted into `Document` chunks.
- Embeddings are generated locally and stored in FAISS in-memory.
- Retrieval uses context-first answering in chat.
- Study notes and MCQs are generated from retrieved context.

---

## 12) Privacy

- Fully local inference and processing
- No external LLM API calls
- Documents stay on local machine

---

## 13) License

See [LICENSE](LICENSE).

---

## 14) Credits

Original repository adapted and modernized into a local Ollama-first RAG system with improved UI, study notes, and quiz workflow.
