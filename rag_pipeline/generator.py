# generator.py

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config import config
from injector import RAGInjector

logger = logging.getLogger(__name__)

class RAGGenerator:
    def __init__(self):
        self.injector = RAGInjector()
        self._init_client()

    def _init_client(self):
        """Setup or refresh LLM Client."""
        if config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing in .env")
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            
        elif config.LLM_PROVIDER == "ollama":
            # Ollama provides an OpenAI-compatible API
            self.client = OpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key="ollama" # required by client, ignored by Ollama
            )
        else:
            raise ValueError(f"Unknown LLM Provider: {config.LLM_PROVIDER}")

        self.model_name = config.LLM_MODEL_NAME

    def ask_with_sources(self, query: str, top_k: int = 5, temperature: float = 0.2) -> Dict[str, Any]:
        """Ask a question and return both the generated answer and retrieved source chunks."""
        logger.info(f"Searching documents for: '{query}' (top_k={top_k})")
        results = self.injector.search(query, top_k=top_k)

        sources = []
        for r in results:
            meta = r.get("metadata", {})
            sources.append({
                "id": r.get("id", ""),
                "filename": meta.get("filename", "Unknown Source"),
                "file_type": meta.get("file_type", ""),
                "page_count": meta.get("page_count", 0),
                "similarity_score": round(float(r.get("similarity_score", 0.0)), 4),
                "content": r.get("content", ""),
                "metadata": meta,
            })

        if not results:
            return {
                "answer": "I couldn't find any relevant documents in the database to answer this question. Please upload relevant files or try a different search phrase.",
                "sources": [],
                "model": self.model_name,
                "provider": config.LLM_PROVIDER
            }

        # Build context from retrieved chunks
        context_text = ""
        for i, res in enumerate(results, 1):
            source = res.get("metadata", {}).get("filename", "Unknown Source")
            score = round(float(res.get("similarity_score", 0.0)) * 100, 1)
            context_text += f"\n--- Source {i}: {source} (Relevance: {score}%) ---\n{res['content']}\n"

        system_prompt = (
            "You are a helpful, professional, and knowledgeable AI assistant. "
            "You are provided with context retrieved from uploaded internal documents. "
            "Your job is to answer the user's question clearly, accurately, and thoroughly based on this context. "
            "Always cite document sources or relevant sections when appropriate. "
            "If the context does not contain the answer, politely state: 'Based on the provided documents, I could not find information on this topic.' "
            "Do not fabricate facts."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context Information:\n{context_text}\n\nUser Question: {query}"}
        ]

        logger.info(f"Generating answer using {self.model_name} (provider: {config.LLM_PROVIDER})...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=False
            )
            full_answer = response.choices[0].message.content or ""

            return {
                "answer": full_answer.strip(),
                "sources": sources,
                "model": self.model_name,
                "provider": config.LLM_PROVIDER
            }

        except Exception as e:
            err_str = str(e)
            logger.error(f"LLM Generation Error: {err_str}")
            hint = ""
            if "Connection" in err_str or "11434" in err_str:
                hint = " (Note: Please ensure Ollama is running locally on port 11434 with 'ollama run llama3')"
            return {
                "answer": f"Unable to generate response from AI model: {err_str}{hint}",
                "sources": sources,
                "model": self.model_name,
                "provider": config.LLM_PROVIDER,
                "error": err_str
            }

    def ask(self, query: str, top_k: int = 5, stream: bool = False) -> str:
        """Backward-compatible helper returning the answer string."""
        res = self.ask_with_sources(query, top_k=top_k)
        return res["answer"]