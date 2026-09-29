from typing import List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.retrieval.vector_store import search_documents


class RetrieverAgent:

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        return search_documents(
            self.vector_store,
            query,
            k
        )