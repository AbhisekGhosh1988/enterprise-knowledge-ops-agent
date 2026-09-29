# Unit Test Case Documentation

## 1. Purpose

The test suite validates the main components of the Enterprise Knowledge Operations Agent.

The tests cover:

- document ingestion
- document splitting
- vector store creation
- document retrieval
- Retriever Agent
- Memory Agent
- Analyst Agent
- Verifier Agent
- Orchestrator Agent
- query validation
- LangGraph workflow
- multi-document queries
- retrieval evaluation
- grounding evaluation
- guardrails
- conflict detection
- grounding confidence

The test suite uses mocked LLM responses for agent-level and workflow tests so that the tests do not depend on live LLM inference.

---

## 2. Test Execution

The complete test suite is executed using:

```bash
pytest