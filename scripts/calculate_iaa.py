import argparse
import csv
import re
import string
from collections import Counter
from pathlib import Path


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def token_f1(a: str, b: str) -> float:
    a_tokens = normalize_answer(a).split()
    b_tokens = normalize_answer(b).split()
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0

    common = Counter(a_tokens) & Counter(b_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def read_annotations(path: Path, allow_draft: bool) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        answer = (row.get("annotated_answer") or "").strip()
        if not answer and allow_draft:
            answer = (row.get("draft_reference_answer") or "").strip()
        if not answer:
            raise ValueError(f"Missing annotated_answer for iaa_id={row.get('iaa_id')} in {path}")
        result[row["iaa_id"]] = {
            "question": row.get("question", ""),
            "answer": answer,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate exact agreement and token F1 agreement for two annotators.")
    parser.add_argument("--annotator-a", default="data/annotations/iaa/annotator_a.csv")
    parser.add_argument("--annotator-b", default="data/annotations/iaa/annotator_b.csv")
    parser.add_argument("--output", default="data/annotations/iaa/iaa_results.csv")
    parser.add_argument("--allow-draft", action="store_true", help="Use draft_reference_answer when annotated_answer is blank.")
    args = parser.parse_args()

    annotations_a = read_annotations(Path(args.annotator_a), args.allow_draft)
    annotations_b = read_annotations(Path(args.annotator_b), args.allow_draft)
    common_ids = sorted(set(annotations_a).intersection(annotations_b), key=lambda value: int(value))
    if not common_ids:
        raise ValueError("No shared iaa_id values found.")

    rows = []
    exact_matches = 0
    f1_scores = []
    for iaa_id in common_ids:
        answer_a = annotations_a[iaa_id]["answer"]
        answer_b = annotations_b[iaa_id]["answer"]
        exact = int(normalize_answer(answer_a) == normalize_answer(answer_b))
        f1 = token_f1(answer_a, answer_b)
        exact_matches += exact
        f1_scores.append(f1)
        rows.append(
            {
                "iaa_id": iaa_id,
                "question": annotations_a[iaa_id]["question"],
                "annotator_a_answer": answer_a,
                "annotator_b_answer": answer_b,
                "exact_match": exact,
                "token_f1": f"{f1:.4f}",
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iaa_id",
                "question",
                "annotator_a_answer",
                "annotator_b_answer",
                "exact_match",
                "token_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    exact_agreement = exact_matches / len(common_ids)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    summary = [
        "IAA results summary",
        f"Compared examples: {len(common_ids)}",
        f"Exact agreement: {exact_agreement:.4f}",
        f"Average token F1 agreement: {avg_f1:.4f}",
        f"Detailed results: {output_path}",
    ]
    (output_path.parent / "iaa_results_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
