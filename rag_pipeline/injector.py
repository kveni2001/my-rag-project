# injector.py

import time
import logging
from typing import List, Optional
from pathlib import Path

from config import config
from document_loader import DocumentLoader, LoadedDocument
from text_processor import TextProcessor, TextChunk
from embedding_manager import EmbeddingManager
from vector_store import get_vector_store, VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RAGInjector:
    """
    End-to-end pipeline: File → Load → Chunk → Embed → Store
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_provider: str = None,
        vector_store_type: str = None,
    ):
        logger.info("=" * 60)
        logger.info("Initializing RAG Injector Pipeline")
        logger.info("=" * 60)

        # Initialize components
        self.loader = DocumentLoader()

        self.processor = TextProcessor(
            chunk_size=chunk_size or config.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        )

        self.embedder = EmbeddingManager(
            provider=embedding_provider or config.EMBEDDING_PROVIDER
        )

        self.vector_store = get_vector_store(
            store_type=vector_store_type or config.VECTOR_STORE_TYPE,
            dimension=self.embedder.dimension,
        )

        logger.info("Pipeline ready!\n")

    def inject_file(self, file_path: str) -> dict:
        """Inject a single file into the vector store."""
        start_time = time.time()

        # Step 1: Load
        document = self.loader.load_file(file_path)
        if not document:
            return {"success": False, "error": "Failed to load file"}

        # Step 2: Chunk
        chunks = self.processor.process_document(document)
        if not chunks:
            return {"success": False, "error": "No chunks generated"}

        # Step 3: Embed
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Step 4: Store
        self.vector_store.add_chunks(chunks, embeddings)

        elapsed = time.time() - start_time

        result = {
            "success": True,
            "file": file_path,
            "chunks_created": len(chunks),
            "total_in_store": self.vector_store.get_count(),
            "time_seconds": round(elapsed, 2),
        }

        logger.info(f"✅ Injected {Path(file_path).name}: {len(chunks)} chunks in {elapsed:.2f}s")
        return result

    def inject_directory(self, dir_path: str, recursive: bool = True) -> dict:
        """Inject all supported files from a directory."""
        start_time = time.time()

        # Step 1: Load all documents
        documents = self.loader.load_directory(dir_path, recursive=recursive)
        if not documents:
            return {"success": False, "error": "No documents found"}

        # Step 2: Chunk all documents
        all_chunks = self.processor.process_documents(documents)
        if not all_chunks:
            return {"success": False, "error": "No chunks generated"}

        # Step 3: Embed all chunks
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Step 4: Store all
        self.vector_store.add_chunks(all_chunks, embeddings)

        elapsed = time.time() - start_time

        result = {
            "success": True,
            "directory": dir_path,
            "files_processed": len(documents),
            "total_chunks": len(all_chunks),
            "total_in_store": self.vector_store.get_count(),
            "time_seconds": round(elapsed, 2),
        }

        logger.info(
            f"\n✅ Injection complete: {len(documents)} files → "
            f"{len(all_chunks)} chunks in {elapsed:.2f}s"
        )
        return result

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search the vector store with a text query."""
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results

    def get_stats(self) -> dict:
        """Get current pipeline statistics."""
        return {
            "embedding_provider": self.embedder.provider,
            "embedding_dimension": self.embedder.dimension,
            "vector_store_type": config.VECTOR_STORE_TYPE,
            "total_documents": self.vector_store.get_count(),
            "chunk_size": self.processor.chunk_size,
            "chunk_overlap": self.processor.chunk_overlap,
        }