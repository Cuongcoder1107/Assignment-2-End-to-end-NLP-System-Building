import argparse
import csv
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import read_lines


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, references: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(reference) for reference in references))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_recall(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    return overlap / len(ref_tokens)


def best_overlap_metric(prediction: str, references: list[str], metric_fn) -> float:
    return max(metric_fn(prediction, reference) for reference in references)


def read_test_topics(qa_master_path: Path) -> list[str]:
    with qa_master_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("split") == "test"]
    return [row.get("topic", "unknown") or "unknown" for row in rows]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA outputs with EM, token F1, and answer recall.")
    parser.add_argument("--questions", default="data/test/questions.txt")
    parser.add_argument("--references", default="data/test/reference_answers.txt")
    parser.add_argument("--qa-master", default="data/annotations/qa_master.tsv")
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument("--error-dir", default="analysis/error_analysis")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=[
            "closed_book=system_outputs/system_output_closed_book.txt",
            "rag_top5=system_outputs/system_output_1.txt",
            "rag_rerank=system_outputs/system_output_2_rerank.txt",
        ],
        help="System specs in name=path format.",
    )
    args = parser.parse_args()

    questions = read_lines(args.questions)
    reference_lines = read_lines(args.references)
    topics = read_test_topics(Path(args.qa_master))
    if not (len(questions) == len(reference_lines) == len(topics)):
        raise ValueError("Questions, references, and test topics must have the same length.")

    systems = []
    for spec in args.systems:
        if "=" not in spec:
            raise ValueError(f"Invalid system spec: {spec}")
        name, path = spec.split("=", 1)
        predictions = read_lines(path)
        if len(predictions) != len(questions):
            raise ValueError(f"{name} has {len(predictions)} lines, expected {len(questions)}.")
        systems.append((name, path, predictions))

    summary_rows = []
    by_topic_values = defaultdict(lambda: defaultdict(list))
    detailed_rows = []
    error_rows = []

    for system_name, system_path, predictions in systems:
        em_values = []
        f1_values = []
        recall_values = []

        for index, (question, reference_line, prediction, topic) in enumerate(
            zip(questions, reference_lines, predictions, topics),
            start=1,
        ):
            references = [part.strip() for part in reference_line.split(";") if part.strip()]
            em = exact_match(prediction, references)
            f1 = best_overlap_metric(prediction, references, token_f1)
            recall = best_overlap_metric(prediction, references, answer_recall)

            em_values.append(em)
            f1_values.append(f1)
            recall_values.append(recall)
            by_topic_values[(system_name, topic)]["em"].append(em)
            by_topic_values[(system_name, topic)]["f1"].append(f1)
            by_topic_values[(system_name, topic)]["recall"].append(recall)

            row = {
                "row_id": index,
                "topic": topic,
                "system": system_name,
                "question": question,
                "reference_answer": reference_line,
                "prediction": prediction,
                "exact_match": f"{em:.4f}",
                "token_f1": f"{f1:.4f}",
                "answer_recall": f"{recall:.4f}",
            }
            detailed_rows.append(row)
            if em < 1.0:
                error_rows.append(row)

        summary_rows.append(
            {
                "system": system_name,
                "output_path": system_path,
                "num_questions": len(questions),
                "exact_match": f"{average(em_values):.4f}",
                "token_f1": f"{average(f1_values):.4f}",
                "answer_recall": f"{average(recall_values):.4f}",
            }
        )

    topic_rows = []
    for (system_name, topic), values in sorted(by_topic_values.items()):
        topic_rows.append(
            {
                "system": system_name,
                "topic": topic,
                "num_questions": len(values["em"]),
                "exact_match": f"{average(values['em']):.4f}",
                "token_f1": f"{average(values['f1']):.4f}",
                "answer_recall": f"{average(values['recall']):.4f}",
            }
        )

    output_dir = Path(args.output_dir)
    error_dir = Path(args.error_dir)
    write_csv(
        output_dir / "evaluation_summary.csv",
        summary_rows,
        ["system", "output_path", "num_questions", "exact_match", "token_f1", "answer_recall"],
    )
    write_csv(
        output_dir / "evaluation_by_topic.csv",
        topic_rows,
        ["system", "topic", "num_questions", "exact_match", "token_f1", "answer_recall"],
    )
    write_csv(
        output_dir / "evaluation_detailed.csv",
        detailed_rows,
        [
            "row_id",
            "topic",
            "system",
            "question",
            "reference_answer",
            "prediction",
            "exact_match",
            "token_f1",
            "answer_recall",
        ],
    )
    write_csv(
        error_dir / "system_errors.csv",
        error_rows,
        [
            "row_id",
            "topic",
            "system",
            "question",
            "reference_answer",
            "prediction",
            "exact_match",
            "token_f1",
            "answer_recall",
        ],
    )

    for row in summary_rows:
        print(
            f"{row['system']}: EM={row['exact_match']} "
            f"F1={row['token_f1']} Recall={row['answer_recall']}"
        )


if __name__ == "__main__":
    main()
