from __future__ import annotations

from src.retrieval.qdrant_store import RetrievedChunk


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{index}] Source: {chunk.title}\n"
            f"URL: {chunk.url}\n"
            f"Text: {chunk.text}"
        )
    context = "\n\n".join(context_blocks)

    return (
        "You are a factual question-answering system for questions about Pittsburgh and Carnegie Mellon University.\n"
        "Use only the provided context.\n"
        "Return the shortest factual answer possible.\n"
        "Do not explain your reasoning.\n"
        "Do not write a full sentence unless the answer itself requires it.\n"
        "If the answer is not in the context, return exactly: unknown\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_closed_book_prompt(question: str) -> str:
    return (
        "Answer the question with the shortest factual answer possible.\n"
        "Do not explain your reasoning.\n"
        "If you do not know the answer, return exactly: unknown\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
