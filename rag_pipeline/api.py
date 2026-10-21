# api.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import shutil
import os
import time
from pathlib import Path

from generator import RAGGenerator
from config import config

app = FastAPI(title="Interactive RAG API", description="Production-grade API for Retrieval Augmented Generation")

# Allow React to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components globally (reuse generator's injector instance)
generator = RAGGenerator()
injector = generator.injector

# --- Request Models ---
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=1.0)

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

# --- Endpoints ---

@app.get("/api/health")
async def get_health():
    """Health check for backend services."""
    stats = injector.get_stats()
    return {
        "status": "healthy",
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL_NAME,
        "embedding_provider": stats.get("embedding_provider"),
        "total_documents": stats.get("total_documents", 0),
        "timestamp": time.time(),
    }

@app.get("/api/stats")
async def get_stats():
    """Get Vector DB stats and configuration."""
    stats = injector.get_stats()
    stats["llm_provider"] = config.LLM_PROVIDER
    stats["llm_model"] = config.LLM_MODEL_NAME
    stats["supported_extensions"] = list(config.SUPPORTED_EXTENSIONS)
    return stats

@app.get("/api/documents")
async def list_documents():
    """List documents in the storage folder with metadata."""
    docs_dir = config.BASE_DIR / "documents"
    docs_dir.mkdir(exist_ok=True)
    
    files = []
    for f in docs_dir.iterdir():
        if f.is_file() and not f.name.startswith("."):
            stat = f.stat()
            files.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": stat.st_size,
                "size_formatted": f"{stat.st_size / (1024 * 1024):.2f} MB" if stat.st_size >= 1024 * 1024 else f"{stat.st_size / 1024:.1f} KB",
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "extension": f.suffix.lower(),
            })

    total_chunks = injector.vector_store.get_count()
    return {
        "files": files,
        "total_files": len(files),
        "total_chunks_in_db": total_chunks,
    }

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a file and inject it into the Vector DB."""
    docs_dir = config.BASE_DIR / "documents"
    docs_dir.mkdir(exist_ok=True)

    # Sanitize filename
    safe_name = Path(file.filename).name
    file_path = str(docs_dir / safe_name)
    
    # Save file locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
        
    # Inject into DB
    result = injector.inject_file(file_path)
    return result

@app.post("/api/chat")
async def chat_with_docs(request: ChatRequest):
    """Ask a question to the LLM about the documents, returning full answer and citations."""
    start_time = time.time()
    res = generator.ask_with_sources(
        query=request.query,
        top_k=request.top_k or 5,
        temperature=request.temperature or 0.2
    )
    elapsed = round(time.time() - start_time, 2)
    res["latency_seconds"] = elapsed
    return res

@app.post("/api/search")
async def search_vectors(request: SearchRequest):
    """Semantic similarity search directly against vector store."""
    start_time = time.time()
    results = injector.search(request.query, top_k=request.top_k or 5)
    elapsed = round(time.time() - start_time, 3)
    return {
        "query": request.query,
        "results": results,
        "total_found": len(results),
        "search_time_seconds": elapsed,
    }

@app.post("/api/clear")
async def clear_store():
    """Clear all vector embeddings from the vector store."""
    try:
        injector.vector_store.delete_collection()
        return {"success": True, "message": "Vector store cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {e}")