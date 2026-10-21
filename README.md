# NexusRAG — Neural Knowledge Hub & AI Assistant

An end-to-end, privacy-preserving, production-grade **Retrieval Augmented Generation (RAG)** platform featuring an immersive, glassmorphic interactive web UI, local vector embeddings, and local/cloud LLM synthesis.

---

## ⚡ Quick Start (1-Click Launch)

Simply double-click:
```bat
start_all.bat
```
This automatically boots:
1. **FastAPI Backend** on `http://127.0.0.1:8000`
2. **React + Vite Frontend** on `http://localhost:5173`
3. Launches your default browser directly into **NexusRAG**.

*Alternatively, run separately:*
- Backend: Double-click `start_backend.bat`
- Frontend: Double-click `start_frontend.bat`

---

## 🌟 Key Features

### 1. 💬 AI Neural Chat with Real-Time Grounding
- Talk directly with your private documents using **Ollama Llama 3** (local, free) or **OpenAI**.
- Rich markdown responses (tables, code blocks, lists, bullet points).
- **Interactive Source Citations**: Every answer displays expandable citation pills showing the source document name, similarity match percentage, and exact retrieved chunk text.
- 1-click clipboard copy, response latency badge, and curated prompt starters.

### 2. 🔍 Vector Similarity Inspector
- Dedicated laboratory view to test semantic queries directly against **ChromaDB**.
- Real-time similarity progress bars, cosine distance scores, chunk metadata, and text previews without invoking the LLM.

### 3. 📁 Knowledge Base & Document Hub
- Drag-and-drop document uploader supporting **PDF, DOCX, and TXT**.
- Real-time ingestion: automatic text cleaning, recursive character splitting (1000 char size, 200 char overlap), local vector embeddings with `all-MiniLM-L6-v2` (384 dimensions), and storage in ChromaDB.
- Document explorer displaying file sizes, modification dates, chunk counts, and database reset tool.

### 4. 🎛️ Interactive RAG Tuning Controls (Sidebar)
- **Top-K Chunks Slider**: Dynamically adjust how many chunks are retrieved per query (1 to 10).
- **LLM Temperature Slider**: Fine-tune factuality vs creativity (0.0 to 1.0).
- Live system status pills showing Ollama model status, ChromaDB chunk count, and backend health.

---

## 🏗️ Architecture

```
my-rag-project/
├── frontend/                     # React 19 + Vite 8 Modern UI
│   ├── src/
│   │   ├── App.jsx               # Immersive Chat, Inspector, & Knowledge Base UI
│   │   ├── App.css               # Futuristic glassmorphism & neon styling
│   │   ├── index.css             # Global tokens & typography
│   │   └── main.jsx
│   ├── vite.config.js            # Reverse proxy (/api -> :8000)
│   └── package.json
│
├── rag_pipeline/                 # Python RAG Engine
│   ├── api.py                    # FastAPI server & endpoints
│   ├── config.py                 # Central configuration
│   ├── generator.py              # LLM synthesis with citations & error handling
│   ├── injector.py               # Document injection orchestrator
│   ├── document_loader.py        # PDF, DOCX, TXT loaders
│   ├── text_processor.py         # Recursive character text chunker
│   ├── embedding_manager.py      # SentenceTransformers & OpenAI embedders
│   ├── vector_store.py           # ChromaDB & FAISS persistence
│   ├── documents/                # Ingested documents (e.g. report.pdf.pdf)
│   └── vector_db/chroma_store/   # Local ChromaDB persistent vector database
│
├── start_all.bat                 # 1-Click launcher
├── start_backend.bat             # Backend launcher
├── start_frontend.bat            # Frontend launcher
└── .env                          # Environment variables & provider settings
```

---

## ⚙️ Configuration (`.env`)

```ini
# LLM Provider: "ollama" (free local) or "openai" (cloud)
LLM_PROVIDER=ollama
LLM_MODEL_NAME=llama3
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1

# Embeddings: "sentence-transformers" (free local) or "openai"
EMBEDDING_PROVIDER=sentence-transformers

# Vector Store
CHROMA_PERSIST_DIR=./vector_db/chroma_store
COLLECTION_NAME=rag_documents
```

