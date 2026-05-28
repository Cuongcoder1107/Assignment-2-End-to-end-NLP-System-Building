import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.qwen_client import QwenApiClient
from src.utils.env import get_env_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Qwen ngrok API.")
    parser.add_argument("--qwen-api-url", default=os.environ.get("QWEN_API_URL") or get_env_value("QWEN_API_URL"))
    parser.add_argument("--prompt", default="Answer with one word: What city is Carnegie Mellon University in?")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    client = QwenApiClient(args.qwen_api_url, default_max_tokens=args.max_tokens)
    print(client.generate(args.prompt, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
