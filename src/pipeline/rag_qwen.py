from __future__ import annotations

from dataclasses import asdict

from src.embedding.sentence_embedder import SentenceTransformerEmbedder
from src.generation.prompts import build_rag_prompt
from src.generation.qwen_client import QwenApiClient
from src.retrieval.qdrant_store import LocalQdrantStore


class RagQwenPipeline:
    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        store: LocalQdrantStore,
        llm: QwenApiClient,
        top_k: int = 5,
        max_tokens: int = 128,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_tokens = max_tokens

    def answer_question(self, question: str) -> dict:
        query_vector = self.embedder.encode_one(question)
        chunks = self.store.search(query_vector=query_vector, top_k=self.top_k)
        prompt = build_rag_prompt(question, chunks)
        answer = self.llm.generate(prompt, max_tokens=self.max_tokens)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [asdict(chunk) for chunk in chunks],
            "prompt": prompt,
        }

    def answer_questions(self, questions: list[str]) -> list[dict]:
        return [self.answer_question(question) for question in questions]
