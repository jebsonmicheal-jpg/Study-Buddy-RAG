# Copilot Instructions

## Project shape
- This is a single-entrypoint Streamlit app in `app.py` for a fully local RAG workflow.
- The app is intentionally privacy-first: PDFs are processed locally, embeddings and answers come from Ollama, and no external API key is used.
- Keep changes focused unless a task explicitly asks for a refactor; the current architecture is a compact one-file pipeline.

## Core data flow
- PDF input comes from either the uploaded file or the fallback sample at `Artifacts/Articles.pdf`.
- The knowledge-base build path is: `build_page_documents()` → `build_vectorstore()` → `answer_question()` / `generate_mcqs()` / `generate_study_notes()`.
- Page text is normalized before chunking, and optional vision summaries are added as separate `Document` entries with `metadata.type = "visual"`.
- Retrieval is context-first: answers should cite page numbers when possible and prefer the indexed PDF context over general model knowledge.

## State and UI conventions
- `st.session_state` is the source of truth for chat history, KB readiness, selected models, generated notes, MCQs, and the vector store.
- Session keys used by the app include `messages`, `conversation_history`, `vectorstore`, `source_documents`, `knowledge_ready`, `mcqs`, `generated_notes`, `page_count`, and `chunk_count`.
- The UI is organized into four tabs: `Doubt Solver`, `Quiz Mode`, `Study Notes`, and `How It Works`.
- The right-hand chat panel is for workflow guidance and session controls, not for retrieval logic.

## Model and integration points
- Ollama is the only LLM/embedding backend. Defaults come from `OLLAMA_HOST`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_VISION_MODEL`.
- `ChatOllama` is used for generation, `OllamaEmbeddings` for FAISS indexing, and `ollama.Client()` is used for page-image vision summaries.
- Keep model names aligned with the sidebar selectors and the README defaults (`llama3.2:3b`, `nomic-embed-text`, `llava:7b`).

## Project-specific patterns
- The app uses custom inline CSS for the dark theme, metric cards, sticky chat input, and MCQ cards; preserve those class names if you extend the UI.
- `build_page_documents()` should keep adding per-page metadata (`source`, `page`, and optional `type`) so downstream previews and citations stay accurate.
- `generate_mcqs()` expects the LLM to output a strict format, and `parse_mcq_response()` is brittle by design; keep the question / options / answer / explanation structure consistent.
- `generate_study_notes()` deliberately filters out visual chunks first so study notes favor text passages over image summaries.

## Workflow knowledge
- To run locally on Windows, start Ollama first (`ollama serve`), then launch Streamlit with the same Python interpreter used for dependency installs.
- The README documents a working command pattern using Python 3.12 and `streamlit run app.py --server.port 8504`.
- If the app is slow or the KB feels noisy, the intended control is to disable vision before rebuilding the KB.

## When editing
- Prefer small, targeted edits to `app.py` and keep behavior aligned with the current session-state-driven design.
- If you add new document metadata or retrieval behavior, update the snippets, citations, and helper text in the UI together.
- Avoid introducing a database, remote API dependency, or multi-file architecture unless the task explicitly requires it.