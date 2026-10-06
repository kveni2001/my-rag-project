# config.py

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Literal

load_dotenv()


@dataclass
class Config:
    """Central configuration for the RAG pipeline."""

    # ── Document Loading ──
    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".doc", ".txt")

    # ── Chunking Strategy ──
    CHUNK_SIZE: int = 1000            # characters per chunk
    CHUNK_OVERLAP: int = 200          # overlap between chunks
    SEPARATOR: str = "\n"

    # ── Embedding Model ──
    # Options: "sentence-transformers", "openai"
    EMBEDDING_PROVIDER: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    
    # For sentence-transformers (FREE, runs locally)
    ST_MODEL_NAME: str = "all-MiniLM-L6-v2"    # 384 dimensions, fast
    # Other options:
    # "all-mpnet-base-v2"          → 768 dims, better quality
    # "multi-qa-MiniLM-L6-cos-v1"  → optimized for Q&A

    # For OpenAI (PAID, API-based)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_NAME: str = "text-embedding-3-small"

    # ── Vector Store ──
    # Options: "chroma", "faiss"
    VECTOR_STORE_TYPE: Literal["chroma", "faiss"] = "chroma"
    CHROMA_PERSIST_DIR: str = "./vector_db/chroma_store"
    FAISS_INDEX_PATH: str = "./vector_db/faiss_store"   
    COLLECTION_NAME: str = "rag_documents"              

    # ── Processing ──
        # ── LLM Generation ──
    # Options: "openai" (paid, API) or "ollama" (free, local)
    LLM_PROVIDER: Literal["openai", "ollama"] = "ollama"
    
    # If using OpenAI: "gpt-3.5-turbo", "gpt-4o", etc.
    # If using Ollama: "llama3", "mistral", "phi3", etc.
    LLM_MODEL_NAME: str = "llama3" 
    
    # Ollama local API URL (default)
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    BATCH_SIZE: int = 64         
    
                                                                                                                                 # documents per batch for embedding


config = Config()