# Example Outputs for Analysis

## RAG + Reranker Fixes Retrieval/Answering

Question: Which two rivers meet to form the Ohio River in Pittsburgh?

Reference: Allegheny River and Monongahela River

Closed-book: unknown

RAG top-5: Allegheny River

RAG + reranker: Allegheny and Monongahela

Observation: Reranking improved the answer by surfacing context that mentions both rivers.

## RAG + Reranker Corrects Wording

Question: What nickname connects Pittsburgh to steelmaking?

Reference: Steel City

Closed-book: Pittsburgh is known as the "Steel City

RAG top-5: City of Steel

RAG + reranker: Steel City

Observation: Reranking led to the expected concise entity-style answer.

## Remaining Metric Mismatch from Equivalent Formatting

Question: How many Super Bowl championships are associated with the Steelers on Visit Pittsburgh?

Reference: six

RAG + reranker: 6

Observation: The answer is semantically correct but fails exact match because the reference uses a word rather than a digit.

## Remaining Error

Question: When was the Pittsburgh Pirates franchise founded?

Reference: 1881

RAG + reranker: 1882

Observation: The reranked context or generated answer still produced the wrong year, so this remains a true factual error.
