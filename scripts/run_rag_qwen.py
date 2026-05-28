import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding.sentence_embedder import SentenceTransformerEmbedder
from src.generation.qwen_client import QwenApiClient
from src.pipeline.rag_qwen import RagQwenPipeline
from src.retrieval.qdrant_store import LocalQdrantStore
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
                "top_chunk_id",
                "top_score",
                "top_title",
                "top_url",
            ],
        )
        writer.writeheader()
        for row_id, row in enumerate(rows, start=1):
            top = row["retrieved_chunks"][0] if row["retrieved_chunks"] else {}
            writer.writerow(
                {
                    "row_id": row_id,
                    "question": row["question"],
                    "answer": row["answer"],
                    "top_chunk_id": top.get("chunk_id", ""),
                    "top_score": f"{top.get('score', 0.0):.6f}" if top else "",
                    "top_title": top.get("title", ""),
                    "top_url": top.get("url", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qdrant + Qwen RAG over a questions.txt file.")
    parser.add_argument("--questions", default="data/test/questions.txt")
    parser.add_argument("--output", default="system_outputs/system_output_1.txt")
    parser.add_argument("--trace-jsonl", default="artifacts/runs/rag_qwen_trace.jsonl")
    parser.add_argument("--trace-csv", default="artifacts/runs/rag_qwen_trace.csv")
    parser.add_argument("--model-path", default="models/sentence-transformers--all-MiniLM-L6-v2")
    parser.add_argument("--qdrant-path", default="qdrant_storage")
    parser.add_argument("--collection", default="pittsburgh_cmu_chunks")
    parser.add_argument("--qwen-api-url", default=os.environ.get("QWEN_API_URL") or get_env_value("QWEN_API_URL"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    questions = read_lines(args.questions)
    embedder = SentenceTransformerEmbedder(args.model_path)
    store = LocalQdrantStore(args.qdrant_path, args.collection)
    llm = QwenApiClient(args.qwen_api_url, default_max_tokens=args.max_tokens)
    pipeline = RagQwenPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    rows = pipeline.answer_questions(questions)

    write_lines(args.output, [row["answer"] for row in rows])
    write_jsonl(args.trace_jsonl, rows)
    write_trace_csv(args.trace_csv, rows)

    print(f"Questions answered: {len(rows)}")
    print(f"Output written to: {args.output}")
    print(f"Trace written to: {args.trace_csv}")
    store.close()


if __name__ == "__main__":
    main()
