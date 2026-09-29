from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_ollama import ChatOllama


class VerifierAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0
        )

    def verify(
        self,
        query: str,
        answer: str,
        documents: List[Document]
    ) -> Dict[str, Any]:

        sources = list(
            dict.fromkeys(
                doc.metadata.get("source", "unknown")
                for doc in documents
            )
        )

        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'unknown')}\n"
                f"{doc.page_content}"
                for doc in documents
            ]
        )

        prompt = f"""
You are the Verifier Agent in an enterprise knowledge operations system.

Verify whether the proposed answer is supported by the retrieved source documents.

User Query:
{query}

Proposed Answer:
{answer}

Retrieved Source Documents:
{context}

Instructions:
- Check whether the answer is grounded in the retrieved documents.
- Do not use outside knowledge.
- Identify unsupported claims.
- Check whether the answer is consistent with the retrieved evidence.
- Provide a concise verification explanation.

Return your response in the following format:

GROUNDED: YES or NO
VERIFICATION: <brief explanation>
"""

        response = self.llm.invoke(prompt)

        content = response.content

        if isinstance(content, str):
            verification_text = content.strip()

        elif isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        text_parts.append(text)

                elif isinstance(item, str):
                    text_parts.append(item)

            verification_text = "\n".join(text_parts).strip()

        else:
            verification_text = str(content).strip()

        grounded = "GROUNDED: YES" in verification_text.upper()

        return {
            "grounded": grounded,
            "verification": verification_text,
            "sources": sources
        }