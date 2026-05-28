import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding.sentence_embedder import SentenceTransformerEmbedder
from src.generation.qwen_client import QwenApiClient
from src.pipeline.rag_qwen_rerank import RagQwenRerankPipeline
from src.retrieval.qdrant_store import LocalQdrantStore
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.env import get_env_value
from src.utils.io import read_lines, write_jsonl, write_lines


def write_trace_csv(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "question",
                "answer",
                "top_retrieved_chunk_id",
                "top_retrieved_score",
                "top_reranked_chunk_id",
                "top_reranked_score",
                "top_reranked_title",
                "top_reranked_url",
            ],
        )
        writer.writeheader()
        for row_id, row in enumerate(rows, start=1):
            top_retrieved = row["retrieved_chunks"][0] if row["retrieved_chunks"] else {}
            top_reranked = row["reranked_chunks"][0] if row["reranked_chunks"] else {}
            writer.writerow(
                {
                    "row_id": row_id,
                    "question": row["question"],
                    "answer": row["answer"],
                    "top_retrieved_chunk_id": top_retrieved.get("chunk_id", ""),
                    "top_retrieved_score": f"{top_retrieved.get('score', 0.0):.6f}" if top_retrieved else "",
                    "top_reranked_chunk_id": top_reranked.get("chunk_id", ""),
                    "top_reranked_score": f"{top_reranked.get('score', 0.0):.6f}" if top_reranked else "",
                    "top_reranked_title": top_reranked.get("title", ""),
                    "top_reranked_url": top_reranked.get("url", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qdrant + cross-encoder reranker + Qwen RAG.")
    parser.add_argument("--questions", default="data/test/questions.txt")
    parser.add_argument("--output", default="system_outputs/system_output_2_rerank.txt")
    parser.add_argument("--trace-jsonl", default="artifacts/runs/rag_qwen_rerank_trace.jsonl")
    parser.add_argument("--trace-csv", default="artifacts/runs/rag_qwen_rerank_trace.csv")
    parser.add_argument("--embedding-model-path", default="models/sentence-transformers--all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model-path", default="models/cross-encoder--ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--qdrant-path", default="qdrant_storage")
    parser.add_argument("--collection", default="pittsburgh_cmu_chunks")
    parser.add_argument("--qwen-api-url", default=os.environ.get("QWEN_API_URL") or get_env_value("QWEN_API_URL"))
    parser.add_argument("--retrieve-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    questions = read_lines(args.questions)
    embedder = SentenceTransformerEmbedder(args.embedding_model_path)
    store = LocalQdrantStore(args.qdrant_path, args.collection)
    reranker = CrossEncoderReranker(args.reranker_model_path)
    llm = QwenApiClient(args.qwen_api_url, default_max_tokens=args.max_tokens)
    pipeline = RagQwenRerankPipeline(
        embedder=embedder,
        store=store,
        reranker=reranker,
        llm=llm,
        retrieve_top_k=args.retrieve_top_k,
        rerank_top_k=args.rerank_top_k,
        max_tokens=args.max_tokens,
    )
    rows = pipeline.answer_questions(questions)

    write_lines(args.output, [row["answer"] for row in rows])
    write_jsonl(args.trace_jsonl, rows)
    write_trace_csv(args.trace_csv, rows)
    store.close()

    print(f"Questions answered: {len(rows)}")
    print(f"Retrieve top-k: {args.retrieve_top_k}")
    print(f"Rerank top-k: {args.rerank_top_k}")
    print(f"Output written to: {args.output}")
    print(f"Trace written to: {args.trace_csv}")


if __name__ == "__main__":
    main()
