import argparse
import csv
from pathlib import Path


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8", newline="\n")


def write_annotation_template(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "split", "topic", "source_hint", "question", "reference_answer", "review_notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "row_id": row_id,
                    "split": row["split"].strip(),
                    "topic": row.get("topic", "").strip(),
                    "source_hint": row.get("source_hint", "").strip(),
                    "question": row["question"].strip(),
                    "reference_answer": row["reference_answer"].strip(),
                    "review_notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export QA master TSV into assignment train/test text files.")
    parser.add_argument("--input", default="data/annotations/qa_master.tsv")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    required = {"split", "question", "reference_answer"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    by_split = {"train": [], "test": []}
    for row in rows:
        split = row["split"].strip().lower()
        if split not in by_split:
            raise ValueError(f"Unknown split: {split}")
        question = row["question"].strip()
        answer = row["reference_answer"].strip()
        if not question or not answer:
            raise ValueError(f"Empty question/answer in row: {row}")
        by_split[split].append(row)

    data_dir = Path(args.data_dir)
    for split, split_rows in by_split.items():
        write_lines(data_dir / split / "questions.txt", [row["question"].strip() for row in split_rows])
        write_lines(data_dir / split / "reference_answers.txt", [row["reference_answer"].strip() for row in split_rows])
        write_annotation_template(data_dir / split / "annotation_template.csv", split_rows)

    summary = [
        "QA annotation export summary",
        f"Input: {args.input}",
        f"Train examples: {len(by_split['train'])}",
        f"Test examples: {len(by_split['test'])}",
    ]
    (data_dir / "annotations" / "qa_export_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
