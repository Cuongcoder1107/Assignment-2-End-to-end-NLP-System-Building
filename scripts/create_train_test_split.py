import argparse
import csv
import random
from pathlib import Path


def read_questions(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Question" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'Question' column.")

        rows = []
        for index, row in enumerate(reader, start=1):
            question = (row.get("Question") or "").strip()
            if not question:
                continue
            answer = (
                row.get("Answer")
                or row.get("Reference Answer")
                or row.get("reference_answer")
                or row.get("reference_answers")
                or ""
            ).strip()
            rows.append(
                {
                    "original_index": index,
                    "question": question,
                    "reference_answer": answer,
                }
            )
    return rows


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8", newline="\n")


def write_annotation_template(path: Path, split_name: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "row_id",
                "original_index",
                "question",
                "reference_answer",
                "notes",
            ],
        )
        writer.writeheader()
        for row_id, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "split": split_name,
                    "row_id": row_id,
                    "original_index": row["original_index"],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/test QA split files from a question CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7111)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    rows = read_questions(Path(args.input_csv))
    if not rows:
        raise ValueError("No non-empty questions found in the input CSV.")

    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    train_size = round(len(shuffled) * args.train_ratio)
    train_rows = shuffled[:train_size]
    test_rows = shuffled[train_size:]

    data_dir = Path(args.data_dir)
    write_lines(data_dir / "train" / "questions.txt", [row["question"] for row in train_rows])
    write_lines(data_dir / "train" / "reference_answers.txt", [row["reference_answer"] for row in train_rows])
    write_lines(data_dir / "test" / "questions.txt", [row["question"] for row in test_rows])
    write_lines(data_dir / "test" / "reference_answers.txt", [row["reference_answer"] for row in test_rows])

    write_annotation_template(data_dir / "train" / "annotation_template.csv", "train", train_rows)
    write_annotation_template(data_dir / "test" / "annotation_template.csv", "test", test_rows)

    summary = [
        "Train/test split summary",
        f"Input CSV: {Path(args.input_csv)}",
        f"Total questions: {len(rows)}",
        f"Train ratio: {args.train_ratio}",
        f"Seed: {args.seed}",
        f"Train questions: {len(train_rows)}",
        f"Test questions: {len(test_rows)}",
        f"Input had reference answers: {any(row['reference_answer'] for row in rows)}",
    ]
    (data_dir / "split_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
