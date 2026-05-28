import argparse
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup
from pypdf import PdfReader


BOILERPLATE_PATTERNS = [
    r"^skip to (main )?content$",
    r"^skip navigation$",
    r"^main navigation$",
    r"^primary navigation$",
    r"^secondary navigation$",
    r"^open navigation$",
    r"^close navigation$",
    r"^toggle navigation$",
    r"^menu$",
    r"^search$",
    r"^site search$",
    r"^subscribe$",
    r"^sign up$",
    r"^log in$",
    r"^privacy policy$",
    r"^terms of use$",
    r"^cookie policy$",
    r"^accept cookies$",
    r"^all rights reserved\.?$",
    r"^copyright\b",
    r"^share this\b",
    r"^follow us\b",
    r"^back to top$",
    r"^loading\.{0,3}$",
    r"^advertisement$",
    r"^related articles$",
    r"^read more$",
    r"^view all$",
]

BOILERPLATE_RE = re.compile("|".join(f"(?:{p})" for p in BOILERPLATE_PATTERNS), re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"\S+")


def normalize_space(value: str) -> str:
    value = html.unescape(value or "")
    value = value.replace("\u00a0", " ")
    value = value.replace("\u200b", "")
    value = SPACE_RE.sub(" ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = MULTI_NEWLINE_RE.sub("\n\n", value)
    return value.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    if BOILERPLATE_RE.search(stripped):
        return True
    if re.fullmatch(r"[\W_]+", stripped):
        return True
    if re.fullmatch(r"(facebook|twitter|instagram|youtube|linkedin|tiktok|x)", stripped, re.IGNORECASE):
        return True
    return False


def clean_lines(text: str) -> str:
    lines = []
    seen_consecutive = None

    for raw_line in normalize_space(text).splitlines():
        line = normalize_space(raw_line)
        if is_noise_line(line):
            continue
        if line == seen_consecutive:
            continue
        seen_consecutive = line
        lines.append(line)

    return "\n".join(lines).strip()


def prune_html(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "form", "button", "input", "select"]):
        tag.decompose()

    noisy_re = re.compile(
        r"(nav|menu|footer|cookie|consent|subscribe|newsletter|social|share|breadcrumb|advert|promo|modal|popup|search|skip|accessibility)",
        re.IGNORECASE,
    )
    for tag in list(soup.find_all(True)):
        if tag.name in {"html", "body", "main", "article"}:
            continue
        attrs_dict = getattr(tag, "attrs", None)
        if not isinstance(attrs_dict, dict):
            continue
        attrs = " ".join(
            str(value)
            for key, value in attrs_dict.items()
            if key in {"class", "id", "role", "aria-label"}
        )
        if attrs and noisy_re.search(attrs):
            tag.decompose()


def extract_html_text(path: Path, source_id: str) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    title = normalize_space(soup.title.get_text(" ")) if soup.title else ""
    prune_html(soup)

    if source_id.startswith("wiki_"):
        main = soup.select_one("#mw-content-text") or soup.select_one("main") or soup.body or soup
    else:
        main = soup.select_one("main") or soup.select_one("article") or soup.body or soup

    text = main.get_text("\n", strip=True)
    return title, clean_lines(text)


def extract_pdf_text_with_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    return "\n\n".join(pages)


def extract_pdf_text_with_pdfplumber(path: Path) -> str:
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
    return "\n\n".join(pages)


def extract_pdf_text(path: Path) -> str:
    try:
        text = extract_pdf_text_with_pypdf(path)
        if word_count(text) >= 25:
            return clean_lines(text)
    except Exception:
        pass
    return clean_lines(extract_pdf_text_with_pdfplumber(path))


def split_long_paragraph(words: list[str], max_words: int, overlap_words: int) -> list[str]:
    parts = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return parts


def chunk_text(text: str, max_words: int, overlap_words: int, min_words: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    chunks = []
    current = []
    current_count = 0

    def flush_current() -> None:
        nonlocal current, current_count
        if current and (current_count >= min_words or not chunks):
            chunks.append("\n".join(current).strip())
        current = []
        current_count = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

        if len(words) > max_words:
            flush_current()
            for part in split_long_paragraph(words, max_words, overlap_words):
                if word_count(part) >= min_words:
                    chunks.append(part)
            continue

        if current_count + len(words) > max_words:
            overlap_text = ""
            if overlap_words > 0 and current:
                previous_words = " ".join(current).split()
                overlap_text = " ".join(previous_words[-overlap_words:])
            flush_current()
            if overlap_text and word_count(overlap_text) + len(words) <= max_words:
                current.append(overlap_text)
                current_count = word_count(overlap_text)

        current.append(paragraph)
        current_count += len(words)

    flush_current()
    return [chunk for chunk in chunks if chunk.strip()]


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Pittsburgh/CMU documents and create retrieval chunks.")
    parser.add_argument("--metadata", default="data/raw/metadata/metadata.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=35)
    parser.add_argument("--min-words", type=int, default=35)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    processed_dir = Path(args.processed_dir)
    chunks_dir = Path(args.chunks_dir)
    documents_dir = processed_dir / "documents"
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    processed_at = datetime.now(timezone.utc).isoformat()
    documents = []
    chunks = []
    errors = []

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        local_path = row.get("local_path", "").strip()
        if not local_path:
            continue

        path = Path(local_path)
        if not path.exists():
            errors.append({"source_id": row["source_id"], "error": f"Missing local file: {local_path}"})
            continue

        try:
            source_id = row["source_id"]
            document_type = row.get("document_type", "")
            if path.suffix.lower() == ".pdf" or document_type == "pdf":
                extracted_title = ""
                text = extract_pdf_text(path)
            else:
                extracted_title, text = extract_html_text(path, source_id)

            if word_count(text) < args.min_words:
                errors.append({"source_id": source_id, "error": "Too little text after cleaning"})
                continue

            title = row.get("title") or extracted_title
            document_record = {
                "source_id": source_id,
                "title": title,
                "url": row.get("url", ""),
                "topic": row.get("topic", ""),
                "document_type": document_type,
                "local_path": local_path,
                "text_path": str(documents_dir / f"{source_id}.txt"),
                "word_count": word_count(text),
                "char_count": len(text),
                "processed_at": processed_at,
                "content_sha1": stable_hash(text),
            }
            documents.append(document_record)
            (documents_dir / f"{source_id}.txt").write_text(text, encoding="utf-8", newline="\n")

            for index, chunk in enumerate(chunk_text(text, args.max_words, args.overlap_words, args.min_words)):
                chunk_id = f"{source_id}__chunk_{index:04d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source_id": source_id,
                        "title": title,
                        "url": row.get("url", ""),
                        "topic": row.get("topic", ""),
                        "document_type": document_type,
                        "local_path": local_path,
                        "chunk_index": index,
                        "text": chunk,
                        "word_count": word_count(chunk),
                        "char_count": len(chunk),
                        "content_sha1": stable_hash(chunk),
                        "processed_at": processed_at,
                    }
                )
        except Exception as exc:
            errors.append({"source_id": row.get("source_id", ""), "error": str(exc)})

    write_jsonl(processed_dir / "corpus.jsonl", documents)
    write_jsonl(chunks_dir / "chunks.jsonl", chunks)

    write_csv(
        processed_dir / "corpus_metadata.csv",
        documents,
        [
            "source_id",
            "title",
            "url",
            "topic",
            "document_type",
            "local_path",
            "text_path",
            "word_count",
            "char_count",
            "processed_at",
            "content_sha1",
        ],
    )
    write_csv(
        chunks_dir / "chunks.csv",
        chunks,
        [
            "chunk_id",
            "source_id",
            "title",
            "url",
            "topic",
            "document_type",
            "local_path",
            "chunk_index",
            "text",
            "word_count",
            "char_count",
            "content_sha1",
            "processed_at",
        ],
    )
    write_csv(processed_dir / "preprocess_errors.csv", errors, ["source_id", "error"])

    topic_counts = {}
    for document in documents:
        topic_counts[document["topic"]] = topic_counts.get(document["topic"], 0) + 1

    chunk_word_counts = [chunk["word_count"] for chunk in chunks]
    avg_chunk_words = sum(chunk_word_counts) / len(chunk_word_counts) if chunk_word_counts else 0
    stats = [
        "Preprocessing summary",
        f"Processed at: {processed_at}",
        f"Documents processed: {len(documents)}",
        f"Documents skipped/errors: {len(errors)}",
        f"Chunks created: {len(chunks)}",
        f"Average chunk words: {avg_chunk_words:.1f}",
        f"Min chunk words: {min(chunk_word_counts) if chunk_word_counts else 0}",
        f"Max chunk words: {max(chunk_word_counts) if chunk_word_counts else 0}",
        "",
        "Documents by topic:",
    ]
    for topic, count in sorted(topic_counts.items()):
        stats.append(f"- {topic}: {count}")
    (chunks_dir / "chunk_stats.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")

    print("\n".join(stats))


if __name__ == "__main__":
    main()
