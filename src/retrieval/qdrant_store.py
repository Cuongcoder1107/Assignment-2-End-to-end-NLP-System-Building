from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    title: str
    url: str
    topic: str
    text: str
    score: float


class LocalQdrantStore:
    def __init__(self, storage_path: str | Path, collection_name: str) -> None:
        self.storage_path = str(storage_path)
        self.collection_name = collection_name
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=self.storage_path)

    def recreate_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors,
        batch_size: int = 128,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = []
            for offset, (chunk, vector) in enumerate(zip(batch_chunks, batch_vectors)):
                point_id = start + offset
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "source_id": chunk["source_id"],
                            "title": chunk.get("title", ""),
                            "url": chunk.get("url", ""),
                            "topic": chunk.get("topic", ""),
                            "document_type": chunk.get("document_type", ""),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "text": chunk["text"],
                            "word_count": chunk.get("word_count", 0),
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    source_id=str(payload.get("source_id", "")),
                    title=str(payload.get("title", "")),
                    url=str(payload.get("url", "")),
                    topic=str(payload.get("topic", "")),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                )
            )
        return results

    def close(self) -> None:
        self.client.close()
