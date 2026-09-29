from typing import Dict, List


class MemoryAgent:

    def __init__(self):
        self._memory: List[Dict[str, str]] = []

    def add_interaction(
        self,
        query: str,
        response: str
    ) -> None:
        self._memory.append(
            {
                "query": query,
                "response": response
            }
        )

    def get_memory(self) -> List[Dict[str, str]]:
        return self._memory.copy()

    def clear_memory(self) -> None:
        self._memory.clear()