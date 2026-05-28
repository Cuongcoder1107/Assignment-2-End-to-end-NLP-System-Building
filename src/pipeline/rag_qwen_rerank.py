from __future__ import annotations

from dataclasses import asdict

from src.embedding.sentence_embedder import SentenceTransformerEmbedder
from src.generation.prompts import build_rag_prompt
from src.generation.qwen_client import QwenApiClient
from src.retrieval.qdrant_store import LocalQdrantStore
from src.retrieval.reranker import CrossEncoderReranker


class RagQwenRerankPipeline:
    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        store: LocalQdrantStore,
        reranker: CrossEncoderReranker,
        llm: QwenApiClient,
        retrieve_top_k: int = 20,
        rerank_top_k: int = 5,
        max_tokens: int = 64,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.llm = llm
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.max_tokens = max_tokens

    def answer_question(self, question: str) -> dict:
        query_vector = self.embedder.encode_one(question)
        retrieved_chunks = self.store.search(query_vector=query_vector, top_k=self.retrieve_top_k)
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_k=self.rerank_top_k)
        prompt = build_rag_prompt(question, reranked_chunks)
        answer = self.llm.generate(prompt, max_tokens=self.max_tokens)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [asdict(chunk) for chunk in retrieved_chunks],
            "reranked_chunks": [asdict(chunk) for chunk in reranked_chunks],
            "prompt": prompt,
        }

    def answer_questions(self, questions: list[str]) -> list[dict]:
        return [self.answer_question(question) for question in questions]
