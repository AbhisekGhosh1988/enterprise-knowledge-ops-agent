import re
from typing import Dict, List


class OrchestratorAgent:

    def plan(self, query: str) -> Dict[str, object]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        cleaned_query = query.strip()

        sub_queries = self._decompose_query(cleaned_query)

        plan: List[str] = [
            "retrieve",
            "analyze",
            "verify"
        ]

        return {
            "query": cleaned_query,
            "plan": plan,
            "sub_queries": sub_queries
        }

    def _decompose_query(self, query: str) -> List[str]:
        parts = re.split(
            r"\s+(?:and|also)\s+",
            query,
            flags=re.IGNORECASE
        )

        sub_queries = [
            part.strip(" ?.")
            for part in parts
            if part.strip(" ?.")
        ]

        return sub_queries