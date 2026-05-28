from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.retrieval.qdrant_store import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_path: str, batch_size: int = 16) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = CrossEncoder(model_path)

    def rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(question, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        reranked = []
        for chunk, score in ranked[:top_k]:
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    title=chunk.title,
                    url=chunk.url,
                    topic=chunk.topic,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return reranked
