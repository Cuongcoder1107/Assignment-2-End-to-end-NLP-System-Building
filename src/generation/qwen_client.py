from __future__ import annotations

import re

import requests


class QwenApiClient:
    def __init__(self, api_url: str, timeout: int = 120, default_max_tokens: int = 128) -> None:
        if not api_url:
            raise ValueError("Qwen API URL is required.")
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/generate"):
            self.api_url = f"{self.api_url}/generate"
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        response = requests.post(
            self.api_url,
            json={
                "prompt": prompt,
                "max_tokens": max_tokens or self.default_max_tokens,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if "response" not in payload:
            raise ValueError(f"Qwen API response missing 'response': {payload}")
        return clean_answer(str(payload["response"]))


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^(answer\s*:\s*)", "", answer, flags=re.IGNORECASE)
    answer = answer.splitlines()[0].strip() if answer else "unknown"
    answer = answer.strip(" .,:;\"'")
    return answer or "unknown"
