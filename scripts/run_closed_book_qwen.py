import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.qwen_client import QwenApiClient
from src.pipeline.closed_book_qwen import ClosedBookQwenPipeline
from src.utils.env import get_env_value
from src.utils.io import read_lines, write_jsonl, write_lines


def write_trace_csv(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "question", "answer"])
        writer.writeheader()
        for row_id, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "row_id": row_id,
                    "question": row["question"],
                    "answer": row["answer"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-book Qwen baseline over a questions.txt file.")
    parser.add_argument("--questions", default="data/test/questions.txt")
    parser.add_argument("--output", default="system_outputs/system_output_closed_book.txt")
    parser.add_argument("--trace-jsonl", default="artifacts/runs/closed_book_qwen_trace.jsonl")
    parser.add_argument("--trace-csv", default="artifacts/runs/closed_book_qwen_trace.csv")
    parser.add_argument("--qwen-api-url", default=os.environ.get("QWEN_API_URL") or get_env_value("QWEN_API_URL"))
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    questions = read_lines(args.questions)
    llm = QwenApiClient(args.qwen_api_url, default_max_tokens=args.max_tokens)
    pipeline = ClosedBookQwenPipeline(llm=llm, max_tokens=args.max_tokens)
    rows = pipeline.answer_questions(questions)

    write_lines(args.output, [row["answer"] for row in rows])
    write_jsonl(args.trace_jsonl, rows)
    write_trace_csv(args.trace_csv, rows)

    print(f"Questions answered: {len(rows)}")
    print(f"Output written to: {args.output}")
    print(f"Trace written to: {args.trace_csv}")


if __name__ == "__main__":
    main()
