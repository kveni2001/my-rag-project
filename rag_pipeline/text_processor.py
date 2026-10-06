# text_processor.py

import re
import hashlib
from typing import List, Dict
from dataclasses import dataclass
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import LoadedDocument

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata and unique ID."""
    chunk_id: str
    content: str
    metadata: Dict
    char_count: int
    word_count: int


class TextProcessor:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",     # Double newline (paragraph break)
                "\n",       # Single newline
                ". ",       # Sentence end
                "? ",       # Question end
                "! ",       # Exclamation end
                "; ",       # Semicolon
                ", ",       # Comma
                " ",        # Space
                "",         # Character level (last resort)
            ],
            is_separator_regex=False,
        )

    def process_document(self, document: LoadedDocument) -> List[TextChunk]:
        """Process a single document into chunks."""
        # Clean the text first
        cleaned_text = self._clean_text(document.content)

        if not cleaned_text.strip():
            logger.warning(f"Empty document after cleaning: {document.source}")
            return []

        # Split into chunks
        raw_chunks = self.splitter.split_text(cleaned_text)

        # Create TextChunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(document.source, i, chunk_text)

            chunk = TextChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata={
                    **document.metadata,          # inherit document metadata
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "chunk_size": len(chunk_text),
                },
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
            )
            chunks.append(chunk)

        logger.info(
            f"  → {document.metadata.get('filename', 'unknown')}: "
            f"{len(raw_chunks)} chunks "
            f"(avg {sum(c.char_count for c in chunks) // max(len(chunks), 1)} chars)"
        )
        return chunks

    def process_documents(self, documents: List[LoadedDocument]) -> List[TextChunk]:
        """Process multiple documents into chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    # ── Private Methods ──

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple whitespace/newlines
        text = re.sub(r'\n{3,}', '\n\n', text)         # max 2 consecutive newlines
        text = re.sub(r'[ \t]{2,}', ' ', text)          # collapse spaces/tabs
        text = re.sub(r'(\n\s*){3,}', '\n\n', text)     # clean empty lines

        # Remove special characters that break embeddings
        text = text.replace('\x00', '')                  # null bytes
        text = re.sub(r'[^\S\n]+', ' ', text)            # normalize whitespace (keep \n)

        return text.strip()

    def _generate_chunk_id(self, source: str, index: int, content: str) -> str:
        """Generate a unique, deterministic ID for each chunk."""
        raw = f"{source}::chunk_{index}::{content[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()