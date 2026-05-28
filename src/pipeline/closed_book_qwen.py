from __future__ import annotations

from src.generation.prompts import build_closed_book_prompt
from src.generation.qwen_client import QwenApiClient


class ClosedBookQwenPipeline:
    def __init__(self, llm: QwenApiClient, max_tokens: int = 128) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def answer_question(self, question: str) -> dict:
        prompt = build_closed_book_prompt(question)
        answer = self.llm.generate(prompt, max_tokens=self.max_tokens)
        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
        }

    def answer_questions(self, questions: list[str]) -> list[dict]:
        return [self.answer_question(question) for question in questions]
