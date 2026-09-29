from typing import List, Dict, Any, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from src.agents.orchestrator import OrchestratorAgent
from src.agents.retriever import RetrieverAgent
from src.agents.analyst import AnalystAgent
from src.agents.verifier import VerifierAgent
from src.agents.memory import MemoryAgent
from src.ingestion.document_loader import (
    load_documents,
    split_documents
)
from src.retrieval.vector_store import create_vector_store


class AgentState(TypedDict, total=False):
    query: str
    plan: List[str]
    sub_queries: List[str]
    documents: List[Document]
    retrieval_scores: List[float]
    memory: List[Dict[str, str]]
    answer: str
    verification: Dict[str, Any]
    final_answer: str
    trace: List[str]


class KnowledgeOperationsWorkflow:

    def __init__(self):
        documents = load_documents()
        chunks = split_documents(documents)

        vector_store = create_vector_store(chunks)

        self.orchestrator = OrchestratorAgent()
        self.retriever = RetrieverAgent(vector_store)
        self.analyst = AnalystAgent()
        self.verifier = VerifierAgent()
        self.memory = MemoryAgent()

        self.graph = self._build_graph()

    def _build_graph(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("memory", self._memory_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("verifier", self._verifier_node)

        workflow.add_edge(START, "orchestrator")
        workflow.add_edge("orchestrator", "memory")
        workflow.add_edge("memory", "retriever")
        workflow.add_edge("retriever", "analyst")

        workflow.add_edge("analyst","verifier")

        workflow.add_edge("verifier",END)

        return workflow.compile()

    def _orchestrator_node(self, state: AgentState) -> AgentState:
        result = self.orchestrator.plan(state["query"])

        return {
            "plan": result["plan"],
            "sub_queries": result["sub_queries"],
            "trace": ["orchestrator"]
        }

    def _retriever_node(self, state: AgentState) -> AgentState:
        sub_queries = state.get(
            "sub_queries",
            [state["query"]]
        )

        all_results = []

        for sub_query in sub_queries:
            results = self.retriever.retrieve(sub_query)
            all_results.extend(results)

        documents = [
            document
            for document, score in all_results
        ]

        retrieval_scores = [
            score
            for document, score in all_results
        ]

        return {
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "trace": state.get("trace", []) + ["retriever"]
        }

    def _analyst_node(
        self,
        state: AgentState
    ) -> AgentState:

        answer = self.analyst.analyze(
            query=state["query"],
            documents=state["documents"],
            memory=state["memory"]
        )

        return {
            "answer": answer,
            "trace": state.get("trace", []) + ["analyst"]
        }

    def _verifier_node(
        self,
        state: AgentState
    ) -> AgentState:

        verification = self.verifier.verify(
            query=state["query"],
            answer=state["answer"],
            documents=state["documents"]
        )

        self.memory.add_interaction(
            state["query"],
            state["answer"]
        )

        return {
            "verification": verification,
            "final_answer": state["answer"],
            "trace": state.get("trace", []) + ["verifier"]
        }

    def run(self, query: str) -> AgentState:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.graph.invoke(
            {
                "query": query
            }
        )

    def _memory_node(self, state: AgentState) -> AgentState:
        return {
            "memory": self.memory.get_memory(),
            "trace": state.get("trace", []) + ["memory"]
        }