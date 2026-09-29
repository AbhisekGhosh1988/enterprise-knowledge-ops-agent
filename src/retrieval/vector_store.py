from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


CHROMA_PATH = Path("data/chroma")

EMBEDDING_MODEL = "nomic-embed-text"


def create_vector_store(documents: List[Document]) -> Chroma:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_name="enterprise_documents"
    )

    return vector_store


def search_documents(
    vector_store: Chroma,
    query: str,
    k: int = 3
):
    return vector_store.similarity_search_with_score(
        query,
        k=k
    )