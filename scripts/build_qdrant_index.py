import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding.sentence_embedder import SentenceTransformerEmbedder
from src.retrieval.qdrant_store import LocalQdrantStore
from src.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Index cleaned chunks into embedded local Qdrant.")
    parser.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    parser.add_argument("--model-path", default="models/sentence-transformers--all-MiniLM-L6-v2")
    parser.add_argument("--qdrant-path", default="qdrant_storage")
    parser.add_argument("--collection", default="pittsburgh_cmu_chunks")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks)
    embedder = SentenceTransformerEmbedder(args.model_path, batch_size=args.batch_size, show_progress=True)
    vectors = embedder.encode([chunk["text"] for chunk in chunks])

    store = LocalQdrantStore(args.qdrant_path, args.collection)
    store.recreate_collection(vector_size=embedder.dimension)
    store.upsert_chunks(chunks, vectors, batch_size=args.batch_size)

    print(f"Chunks indexed: {len(chunks)}")
    print(f"Embedding model: {args.model_path}")
    print(f"Vector dimension: {embedder.dimension}")
    print(f"Qdrant storage: {args.qdrant_path}")
    print(f"Collection: {args.collection}")
    store.close()


if __name__ == "__main__":
    main()
