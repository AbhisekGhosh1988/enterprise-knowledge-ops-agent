from typing import Dict, List

from langchain_core.documents import Document
from langchain_ollama import ChatOllama


class AnalystAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0
        )

    def analyze(
        self,
        query: str,
        documents: List[Document],
        memory: List[Dict[str, str]]
    ) -> str:

        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'unknown')}\n"
                f"{doc.page_content}"
                for doc in documents
            ]
        )

        memory_context = "\n".join(
            [
                f"Previous Query: {item['query']}\n"
                f"Previous Response: {item['response']}"
                for item in memory
            ]
        )

        prompt = f"""
You are the Analyst Agent in an enterprise knowledge operations system.

Answer the user's question using ONLY the retrieved source documents.

User Query:
{query}

Retrieved Source Documents:
{context}

Previous Memory:
{memory_context}

Instructions:
- Reason over the retrieved documents.
- Use information from multiple documents when required.
- Do not invent facts.
- If the retrieved documents do not contain enough information, say so.
- Produce a concise answer that can be checked by a verifier.
"""

        response = self.llm.invoke(prompt)

        content = response.content

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        text_parts.append(text)

                elif isinstance(item, str):
                    text_parts.append(item)

            return "\n".join(text_parts).strip()

        return str(content).strip()