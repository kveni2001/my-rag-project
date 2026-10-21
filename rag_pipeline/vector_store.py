# vector_store.py

import os
import pickle
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

from config import config
from text_processor import TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Abstract interface for vector stores."""

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        raise NotImplementedError

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        raise NotImplementedError

    def get_count(self) -> int:
        raise NotImplementedError

    def delete_collection(self):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
#  ChromaDB Implementation
# ─────────────────────────────────────────────────────────────

class ChromaVectorStore(VectorStore):
    """Vector store using ChromaDB (persistent, easy to use)."""

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None,
    ):
        import chromadb

        raw_persist = persist_dir or config.CHROMA_PERSIST_DIR
        persist_path = Path(raw_persist)
        if not persist_path.is_absolute():
            persist_path = (config.BASE_DIR / persist_path).resolve()

        self.persist_dir = str(persist_path)
        self.collection_name = collection_name or config.COLLECTION_NAME

        # Ensure storage directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            f"ChromaDB initialized at '{self.persist_dir}' | "
            f"Collection: '{self.collection_name}' | "
            f"Existing documents: {self.collection.count()}"
        )

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        batch_size: int = 500,
    ):
        """Add chunks with their embeddings to ChromaDB."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")

        if not chunks:
            logger.info("No chunks provided to add.")
            return

        # Check existing IDs to prevent duplicates
        existing_ids = set()
        try:
            result = self.collection.get(ids=[c.chunk_id for c in chunks])
            if result and "ids" in result:
                existing_ids = set(result["ids"])
        except Exception:
            pass

        new_chunks = []
        new_embeddings = []
        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_embeddings.append(emb)

        if not new_chunks:
            logger.info("No new chunks to add (all chunks already exist).")
            return

        skipped = len(chunks) - len(new_chunks)
        if skipped > 0:
            logger.info(f"Skipped {skipped} duplicate chunks.")

        logger.info(f"Inserting {len(new_chunks)} new chunks into ChromaDB...")

        # Batch insert
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i: i + batch_size]
            batch_embeddings = new_embeddings[i: i + batch_size]

            self.collection.add(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c.content for c in batch_chunks],
                metadatas=[c.metadata for c in batch_chunks],
            )

        logger.info(f"✅ Total items in ChromaDB: {self.collection.count()}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search for similar chunks in ChromaDB."""
        if self.collection.count() == 0:
            logger.warning("Vector store is empty.")
            return []

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self.collection.query(**kwargs)

        search_results = []
        if results and "ids" in results and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                search_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "similarity_score": round(1.0 - distance, 4),
                })

        return search_results

    def get_count(self) -> int:
        return self.collection.count()

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
        # Recreate an empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared collection '{self.collection_name}'")


# ─────────────────────────────────────────────────────────────
#  FAISS Implementation
# ─────────────────────────────────────────────────────────────

class FAISSVectorStore(VectorStore):
    """Vector store using FAISS (in-memory with disk persistence)."""

    def __init__(
        self,
        index_path: str = None,
        dimension: int = 384,
    ):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss package is not installed. Install via `pip install faiss-cpu`."
            )

        raw_index = index_path or config.FAISS_INDEX_PATH
        idx_path = Path(raw_index)
        if not idx_path.is_absolute():
            idx_path = (config.BASE_DIR / idx_path).resolve()
        self.index_path = str(idx_path)
        self.dimension = dimension

        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.chunk_ids: List[str] = []

        # Load from disk if index exists, otherwise create new
        if self._load_index():
            logger.info(f"FAISS index loaded from '{self.index_path}' | Docs: {len(self.documents)}")
        else:
            self.index = self.faiss.IndexFlatIP(dimension)
            logger.info(f"New FAISS index created | Dimension: {dimension}")

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]):
        """Add chunks with their embeddings to FAISS."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length.")

        if not chunks:
            return

        existing_ids = set(self.chunk_ids)
        new_indices = [i for i, c in enumerate(chunks) if c.chunk_id not in existing_ids]

        if not new_indices:
            logger.info("No new chunks to add (all chunks already exist).")
            return

        new_chunks = [chunks[i] for i in new_indices]
        new_embeddings = [embeddings[i] for i in new_indices]

        logger.info(f"Adding {len(new_chunks)} new chunks to FAISS...")

        vectors = np.array(new_embeddings, dtype=np.float32)
        self.faiss.normalize_L2(vectors)

        self.index.add(vectors)

        for chunk in new_chunks:
            self.documents.append(chunk.content)
            self.metadatas.append(chunk.metadata)
            self.chunk_ids.append(chunk.chunk_id)

        self._save_index()
        logger.info(f"✅ Total items in FAISS: {self.index.ntotal}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search for similar chunks in FAISS."""
        if self.index.ntotal == 0:
            logger.warning("FAISS vector store is empty.")
            return []

        query_vector = np.array([query_embedding], dtype=np.float32)
        self.faiss.normalize_L2(query_vector)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            result = {
                "id": self.chunk_ids[idx],
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "similarity_score": round(float(score), 4),
                "distance": round(float(1.0 - score), 4),
            }

            if filter_metadata:
                match = all(
                    result["metadata"].get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            results.append(result)

        return results

    def get_count(self) -> int:
        return self.index.ntotal

    def delete_collection(self):
        """Reset the index."""
        self.index = self.faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadatas = []
        self.chunk_ids = []
        self._save_index()
        logger.info("FAISS index cleared.")

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        self.faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))

        meta = {
            "documents": self.documents,
            "metadatas": self.metadatas,
            "chunk_ids": self.chunk_ids,
        }
        with open(os.path.join(self.index_path, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)

    def _load_index(self) -> bool:
        """Load FAISS index and metadata from disk."""
        index_file = os.path.join(self.index_path, "index.faiss")
        meta_file = os.path.join(self.index_path, "metadata.pkl")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            return False

        try:
            self.index = self.faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)

            self.documents = meta["documents"]
            self.metadatas = meta["metadatas"]
            self.chunk_ids = meta["chunk_ids"]
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False


# ─────────────────────────────────────────────────────────────
#  Factory Function
# ─────────────────────────────────────────────────────────────

def get_vector_store(store_type: str = None, dimension: int = 384) -> VectorStore:
    """Factory to create and return the appropriate vector store."""
    store_type = store_type or config.VECTOR_STORE_TYPE

    if store_type == "chroma":
        return ChromaVectorStore()
    elif store_type == "faiss":
        return FAISSVectorStore(dimension=dimension)
    else:
        raise ValueError(f"Unknown vector store type: '{store_type}'. Supported: 'chroma', 'faiss'")