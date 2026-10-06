# generator.py

import logging
from openai import OpenAI
from config import config
from injector import RAGInjector

logger = logging.getLogger(__name__)

class RAGGenerator:
    def __init__(self):
        self.injector = RAGInjector()
        
        # Setup LLM Client
        if config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing in .env")
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            
        elif config.LLM_PROVIDER == "ollama":
            # Ollama has an OpenAI-compatible API!
            self.client = OpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key="ollama" # required but ignored by Ollama
            )
        else:
            raise ValueError(f"Unknown LLM Provider: {config.LLM_PROVIDER}")

        self.model_name = config.LLM_MODEL_NAME

    def ask(self, query: str, top_k: int = 5, stream: bool = True):
        """Ask a question and get an answer based on ingested documents."""
        
        # 1. Retrieve relevant chunks from the Vector DB
        print(f"\n🔍 Searching documents for: '{query}'...")
        results = self.injector.search(query, top_k=top_k)
        
        if not results:
            return "I couldn't find any relevant documents in the database to answer this."

        # 2. Format the context
        context_text = ""
        for i, res in enumerate(results, 1):
            source = res['metadata'].get('filename', 'Unknown Source')
            context_text += f"\n--- Document {i} (Source: {source}) ---\n{res['content']}\n"

        # 3. Build the Prompt
        system_prompt = (
            "You are a helpful, professional AI assistant. "
            "You will be provided with context from internal documents. "
            "Your job is to answer the user's question based ONLY on this context. "
            "If the answer is not contained in the context, say 'I don't know based on the provided documents.' "
            "Do not make up information."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context Information:\n{context_text}\n\nQuestion: {query}"}
        ]

        print(f"🤖 Generating answer using {self.model_name}...\n")
        print("="*60)

        # 4. Call the LLM
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            temperature=0.3 # Low temperature for more factual answers
        )

        # 5. Handle output
        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full_response += text
            print("\n" + "="*60)
            return full_response
        else:
            answer = response.choices[0].message.content
            print(answer)
            print("\n" + "="*60)
            return answer