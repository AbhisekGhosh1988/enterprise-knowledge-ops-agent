from src.ingestion.document_loader import (
    load_documents,
    split_documents
)

from src.retrieval.vector_store import (
    create_vector_store,
    search_documents
)

from src.agents.retriever import RetrieverAgent
from src.agents.memory import MemoryAgent
from src.agents.analyst import AnalystAgent
from src.agents.verifier import VerifierAgent
from src.agents.orchestrator import OrchestratorAgent
from src.graph.workflow import KnowledgeOperationsWorkflow

from langchain_core.documents import Document
from unittest.mock import MagicMock


class MockLLM:
    def __init__(self, response_text):
        self.response_text = response_text

    def invoke(self, prompt):
        response = MagicMock()
        response.content = self.response_text
        return response


def test_load_documents():
    documents = load_documents()

    assert len(documents) == 3

    sources = {
        document.metadata["source"]
        for document in documents
    }

    assert sources == {
        "company_policy.txt",
        "leave_policy.txt",
        "employee_handbook.txt",
    }


def test_split_documents():
    documents = load_documents()
    chunks = split_documents(documents)

    assert len(chunks) >= len(documents)

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert "source" in chunk.metadata


def test_create_vector_store():
    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    assert vector_store is not None


def test_search_documents():
    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    results = search_documents(
        vector_store,
        "Who approves remote work?"
    )

    assert len(results) > 0

    document, score = results[0]

    assert isinstance(document, Document)
    assert isinstance(score, float)

    assert "source" in document.metadata


def test_retriever_agent():
    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    retriever = RetrieverAgent(vector_store)

    results = retriever.retrieve(
        "Who approves remote work?"
    )

    assert len(results) > 0

    document, score = results[0]

    assert isinstance(document, Document)
    assert isinstance(score, float)

    assert "source" in document.metadata


def test_memory_agent():
    memory = MemoryAgent()

    memory.add_interaction(
        "Who approves remote work?",
        "Manager approval is required."
    )

    stored_memory = memory.get_memory()

    assert len(stored_memory) == 1
    assert stored_memory[0]["query"] == "Who approves remote work?"
    assert stored_memory[0]["response"] == "Manager approval is required."

    memory.clear_memory()

    assert memory.get_memory() == []


def test_analyst_agent():
    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    retriever = RetrieverAgent(vector_store)

    results = retriever.retrieve(
        "Who approves remote work?"
    )

    memory = MemoryAgent()

    analyst = AnalystAgent()

    # Mock Gemini so this unit test makes no API request.
    analyst.llm = MockLLM(
        "Manager approval is required for remote work."
    )

    answer = analyst.analyze(
        query="Who approves remote work?",
        documents=[
            document for document, score in results
        ],
        memory=memory.get_memory()
    )

    assert answer
    assert len(answer.strip()) > 0


def test_verifier_agent():
    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    retriever = RetrieverAgent(vector_store)

    results = retriever.retrieve(
        "Who approves remote work?"
    )

    memory = MemoryAgent()

    analyst = AnalystAgent()

    # Mock Gemini for Analyst.
    analyst.llm = MockLLM(
        "Manager approval is required for remote work."
    )

    answer = analyst.analyze(
        query="Who approves remote work?",
        documents=[
            document for document, score in results
        ],
        memory=memory.get_memory()
    )

    verifier = VerifierAgent()

    # Mock Gemini for Verifier.
    verifier.llm = MockLLM(
        """GROUNDED: YES
EXPLANATION: The answer is supported by the provided source documents.
SOURCES: company_policy.txt"""
    )

    verification = verifier.verify(
        query="Who approves remote work?",
        answer=answer,
        documents=[
            document for document, score in results
        ]
    )

    assert isinstance(verification, dict)
    assert "grounded" in verification
    assert "verification" in verification
    assert "sources" in verification
    assert len(verification["sources"]) > 0


def test_orchestrator_agent():
    orchestrator = OrchestratorAgent()

    result = orchestrator.plan(
        "Who approves remote work?"
    )

    assert result["query"] == "Who approves remote work?"

    assert result["plan"] == [
        "retrieve",
        "analyze",
        "verify"
    ]


def test_orchestrator_rejects_empty_query():
    orchestrator = OrchestratorAgent()

    try:
        orchestrator.plan("")
        assert False
    except ValueError as error:
        assert str(error) == "Query cannot be empty."


def test_langgraph_workflow():
    workflow = KnowledgeOperationsWorkflow()

    # Replace the live Gemini calls created inside the workflow
    # with deterministic mock responses.
    workflow.analyst.llm = MockLLM(
        "Manager approval is required for remote work."
    )

    workflow.verifier.llm = MockLLM(
        """GROUNDED: YES
EXPLANATION: The answer is supported by the provided source documents.
SOURCES: company_policy.txt"""
    )

    result = workflow.run(
        "Who approves remote work?"
    )

    assert result["plan"] == [
        "retrieve",
        "analyze",
        "verify"
    ]

    assert result["documents"]

    assert result["answer"]

    assert result["verification"]

    assert result["final_answer"]

    assert result["trace"] == [
        "orchestrator",
        "memory",
        "retriever",
        "analyst",
        "verifier"
    ]


def test_multi_document_query():
    workflow = KnowledgeOperationsWorkflow()

    # Replace the live Gemini calls created inside the workflow
    # with deterministic mock responses.
    workflow.analyst.llm = MockLLM(
        "The leave policy defines leave rules, while the employee handbook describes remote work."
    )

    workflow.verifier.llm = MockLLM(
        """GROUNDED: YES
EXPLANATION: The answer is supported by the provided source documents.
SOURCES: leave_policy.txt, employee_handbook.txt"""
    )

    result = workflow.run(
        "What are the leave rules and what does the employee handbook say about remote work?"
    )

    assert result["plan"] == [
        "retrieve",
        "analyze",
        "verify"
    ]

    assert len(result["sub_queries"]) >= 2

    assert result["documents"]

    sources = set(
        document.metadata.get("source")
        for document in result["documents"]
    )

    assert len(sources) > 1

    assert result["answer"]

    assert result["verification"]

    assert result["final_answer"]

    assert result["trace"] == [
        "orchestrator",
        "memory",
        "retriever",
        "analyst",
        "verifier"
    ]


def test_evaluation_detects_insufficient_retrieval():
    from src.evaluation.evaluator import EvaluationResult

    result = {
        "retrieval_scores": [],
        "verification": {},
        "trace": [],
        "final_answer": ""
    }

    evaluation = EvaluationResult(
        query="Unknown question",
        result=result
    )

    evaluation_result = evaluation.to_dict()

    assert evaluation_result["retrieval"]["status"] == \
        "INSUFFICIENT_RETRIEVAL"

    assert evaluation_result["retrieval"]["average_score"] is None


def test_guardrail_allows_grounded_response():
    from src.evaluation.evaluator import EvaluationResult

    result = {
        "retrieval_scores": [0.6, 0.7],
        "verification": {
            "grounded": True,
            "verification": "GROUNDED: YES",
            "sources": ["company_policy.txt"]
        },
        "trace": [],
        "final_answer": "The manager approves the exception."
    }

    evaluation = EvaluationResult(
        query="Who approves the exception?",
        result=result
    )

    evaluation_result = evaluation.to_dict()

    assert evaluation_result["grounding_status"] == "GROUNDED"
    assert evaluation_result["guardrail"]["action"] == "ALLOW_RESPONSE"


def test_guardrail_limits_ungrounded_response():
    from src.evaluation.evaluator import EvaluationResult

    result = {
        "retrieval_scores": [0.6, 0.7],
        "verification": {
            "grounded": False,
            "verification": "GROUNDED: NO",
            "sources": ["company_policy.txt"]
        },
        "trace": [],
        "final_answer": "This answer is not supported by the documents."
    }

    evaluation = EvaluationResult(
        query="Unknown question",
        result=result
    )

    evaluation_result = evaluation.to_dict()

    assert evaluation_result["grounding_status"] == "UNGROUNDED"
    assert evaluation_result["guardrail"]["action"] == "LIMIT_RESPONSE"


def test_conflict_detection():
    from src.evaluation.evaluator import EvaluationResult

    result = {
        "retrieval_scores": [0.6],
        "verification": {
            "grounded": True,
            "verification": "GROUNDED: YES",
            "sources": ["company_policy.txt"]
        },
        "conflicting_outputs": True,
        "trace": [],
        "final_answer": "Conflicting answer."
    }

    evaluation = EvaluationResult(
        query="Conflicting question",
        result=result
    )

    evaluation_result = evaluation.to_dict()

    assert evaluation_result["conflict_detection"]["status"] == \
        "CONFLICT_DETECTED"

    assert evaluation_result["conflict_detection"]["action"] == \
        "LIMIT_RESPONSE"


def test_no_conflict_detection():
    from src.evaluation.evaluator import EvaluationResult

    result = {
        "retrieval_scores": [0.6],
        "verification": {
            "grounded": True,
            "verification": "GROUNDED: YES",
            "sources": ["company_policy.txt"]
        },
        "conflicting_outputs": False,
        "trace": [],
        "final_answer": "Supported answer."
    }

    evaluation = EvaluationResult(
        query="Normal question",
        result=result
    )

    evaluation_result = evaluation.to_dict()

    assert evaluation_result["conflict_detection"]["status"] == \
        "NO_CONFLICT"

    assert evaluation_result["conflict_detection"]["action"] == \
        "ALLOW_RESPONSE"


def test_grounding_confidence_supported():
    from src.evaluation.evaluator import EvaluationResult

    result = {
        "retrieval_scores": [0.45, 0.50],
        "verification": {
            "grounded": True,
            "verification": "GROUNDED: YES",
            "sources": ["company_policy.txt"]
        },
        "trace": [],
        "final_answer": "The manager approves the exception."
    }

    evaluation = EvaluationResult(
        query="Who approves the exception?",
        result=result
    )

    evaluation_result = evaluation.to_dict()

    assert evaluation_result["grounding_confidence"]["status"] == \
        "SUPPORTED"


def test_grounding_confidence_low():
    from src.evaluation.evaluator import EvaluationResult

    result = {
        "retrieval_scores": [0.80, 0.82],
        "verification": {
            "grounded": True,
            "verification": "GROUNDED: YES",
            "sources": ["company_policy.txt"]
        },
        "trace": [],
        "final_answer": "Unsupported question."
    }

    evaluation = EvaluationResult(
        query="Unrelated question",
        result=result
    )

    evaluation_result = evaluation.to_dict()

    assert evaluation_result["grounding_confidence"]["status"] == \
        "LOW"