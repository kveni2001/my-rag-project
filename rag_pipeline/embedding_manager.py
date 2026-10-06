# embedding_manager.py

import os
import logging
from typing import List
from tqdm import tqdm

from config import config

# Silence noisy HTTP and HuggingFace logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages text embeddings using either:
    - sentence-transformers (free, local)
    - OpenAI API (paid, cloud)
    """

    def __init__(self, provider: str = None):
        self.provider = provider or config.EMBEDDING_PROVIDER
        self.model = None
        self.dimension = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model based on provider."""
        if self.provider == "sentence-transformers":
            self._load_sentence_transformer()
        elif self.provider == "openai":
            self._load_openai()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _load_sentence_transformer(self):
        """Load local sentence-transformer model quickly from local cache."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model '{config.ST_MODEL_NAME}'...")

        # Try loading strictly from local cache first (instant load, no internet hang)
        try:
            self.model = SentenceTransformer(
                config.ST_MODEL_NAME,
                model_kwargs={"local_files_only": True}
            )
        except Exception:
            # Fallback to standard download if not cached yet
            self.model = SentenceTransformer(config.ST_MODEL_NAME)

        # Get embedding vector dimension
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding model ready. Dimension: {self.dimension}")

    def _load_openai(self):
        """Initialize OpenAI embedding client."""
        from openai import OpenAI

        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")

        self.model = OpenAI(api_key=config.OPENAI_API_KEY)
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.dimension = dim_map.get(config.OPENAI_MODEL_NAME, 1536)

    def embed_texts(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Embed a list of text strings into vector representations."""
        batch_size = batch_size or config.BATCH_SIZE

        if self.provider == "sentence-transformers":
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings.tolist()

        elif self.provider == "openai":
            all_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                batch = texts[i: i + batch_size]
                response = self.model.embeddings.create(
                    input=batch,
                    model=config.OPENAI_MODEL_NAME,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        results = self.embed_texts([query])
        return results[0]