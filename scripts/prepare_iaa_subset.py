import argparse
import csv
import random
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an IAA subset and independent annotation templates.")
    parser.add_argument("--questions", default="data/test/questions.txt")
    parser.add_argument("--answers", default="data/test/reference_answers.txt")
    parser.add_argument("--output-dir", default="data/annotations/iaa")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7111)
    args = parser.parse_args()

    questions = read_lines(Path(args.questions))
    answers = read_lines(Path(args.answers))
    if len(questions) != len(answers):
        raise ValueError("Questions and reference answers must have the same number of lines.")

    indices = list(range(len(questions)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    selected = sorted(indices[: min(args.sample_size, len(indices))])

    subset_rows = []
    annotator_rows = []
    for row_id, index in enumerate(selected, start=1):
        subset_rows.append(
            {
                "iaa_id": row_id,
                "test_index": index + 1,
                "question": questions[index],
                "reference_answer": answers[index],
            }
        )
        annotator_rows.append(
            {
                "iaa_id": row_id,
                "test_index": index + 1,
                "question": questions[index],
                "draft_reference_answer": answers[index],
                "annotated_answer": "",
                "notes": "",
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "iaa_subset.csv",
        subset_rows,
        ["iaa_id", "test_index", "question", "reference_answer"],
    )
    write_csv(
        output_dir / "annotator_a.csv",
        annotator_rows,
        ["iaa_id", "test_index", "question", "draft_reference_answer", "annotated_answer", "notes"],
    )
    write_csv(
        output_dir / "annotator_b.csv",
        annotator_rows,
        ["iaa_id", "test_index", "question", "draft_reference_answer", "annotated_answer", "notes"],
    )

    summary = [
        "IAA subset preparation summary",
        f"Source questions: {args.questions}",
        f"Source answers: {args.answers}",
        f"Test examples available: {len(questions)}",
        f"IAA sample size: {len(selected)}",
        f"Seed: {args.seed}",
        "Annotator files contain blank annotated_answer columns for independent review.",
    ]
    (output_dir / "iaa_subset_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
