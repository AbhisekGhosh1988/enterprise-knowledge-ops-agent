# Evaluation and Guardrail Documentation

## 1. Purpose

The Enterprise Knowledge Operations Agent includes evaluation and guardrail mechanisms to identify:

- insufficient retrieval
- irrelevant retrieval
- low grounding confidence
- ungrounded responses
- configured conflicting outputs

The evaluation information is produced in a structured format and displayed alongside the final response.

The evaluation layer supports the assignment requirements for grounding checks, retrieval relevance signals, agent decision traces, failure detection, and structured evaluation output.

---

## 2. Retrieval Evaluation

The Retriever Agent uses the Chroma vector store to retrieve relevant document chunks.

For each retrieved document, a retrieval distance score is captured.

The evaluation layer calculates the average retrieval score.

The current implementation uses the following empirical threshold:

```text
RETRIEVAL_DISTANCE_THRESHOLD = 0.65