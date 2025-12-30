# Young RAG (Air‑Gapped Template)

Young RAG is a Streamlit-based Retrieval-Augmented Generation (RAG) template that:
- Ingests text from a Postgres database.
- Chunks text by paragraphs with token-aware sizing and overlap.
- Stores embeddings in a local Chroma vector database.
- Provides a multi-turn chat UI with conversational retrieval and app-rendered citations.

This project was designed to run locally on an air‑gapped, high‑security environment**. It ships with OpenAI API integrations by default, but you should **replace those calls with your locally hosted model endpoints** (e.g., vLLM or another internal inference gateway) before deploying in a restricted environment.) Use of Postgres was a stakeholder requirement but can be switched for any database client. 

## How it works

1. **Ingestion:**
   - The app reads rows from Postgres (`path`, `text`) and chunks each document by paragraphs.
   - Chunks target **>=300 tokens** (except the last chunk), and oversized chunks are split down to **<=1000 tokens**.
   - Chunks are stored in Chroma with metadata containing the original `path` and `chunk_index`.

2. **Retrieval + Chat:**
   - The chat flow uses LangChain’s `ConversationalRetrievalChain` with `ConversationBufferMemory` for multi-turn context.
   - Top documents are retrieved from Chroma, and citations are rendered **by the app** below the assistant response.

## Local development

### Prerequisites
- Python 3.10
- Postgres database accessible on the host configured in `app.py`
- OpenAI API key (for default configuration)

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

## Air‑gapped / high‑security deployment notes

This repository is intended as a **template** for secure environments. If running on an air‑gapped machine:

1. **Replace OpenAI API calls**
   - The app currently uses `ChatOpenAI` and `OpenAIEmbeddings` (see `app.py`).
   - Swap these with locally hosted model endpoints such as **vLLM** or another internal API that provides:
     - **Chat/completions** for the LLM
     - **Embedding generation** for document indexing

2. **Keep data local**
   - Run Postgres, Chroma, and the Streamlit app on the same secured network.
   - Ensure no outbound connectivity is required by your LLM or embedding services.

3. **Environment configuration**
   - Set credentials in `app.py` or via environment variables as appropriate for your environment.

## Project structure

- `app.py` — Streamlit app, ingestion, chunking, retrieval, and chat UI
- `requirements.txt` — Python dependencies

## Notes

- The default Postgres table expected by the app is `documents` with `path` and `text` columns.
- The “Check for new data” button updates Chroma if new documents are detected.
