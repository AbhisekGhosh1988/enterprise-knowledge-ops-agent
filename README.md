# Enterprise Knowledge Operations Agent

A fully local, multi-agent enterprise knowledge operations system built
with **Python, LangChain, LangGraph, Chroma, Ollama, and Streamlit**.

The project implements the Agentic AI Developer case study requirements
around enterprise document ingestion, vector retrieval, multi-agent
orchestration, reasoning and synthesis, grounding/validation,
guardrails, evaluation, observability, explainability, and unit testing.

> **Important:** The enterprise documents included with this project are
> synthetic test documents created only to demonstrate the system. They
> are not real company policies.

------------------------------------------------------------------------

## 1. Project Overview

The goal of this project is to build an enterprise knowledge operations
agent that can answer questions requiring information from enterprise
documents while making the answer process observable and verifiable.

The system is designed around five distinct agent roles:

1.  **Orchestrator Agent** --- plans the workflow and decomposes complex
    queries.
2.  **Memory Agent** --- provides conversation/session context.
3.  **Retriever Agent** --- retrieves relevant document chunks from
    Chroma.
4.  **Analyst Agent** --- reasons over the retrieved evidence and
    synthesizes an answer.
5.  **Verifier Agent** --- checks whether the proposed answer is
    grounded in the retrieved evidence.

An evaluation layer then assesses retrieval quality, grounding,
confidence, guardrails, conflict status, and the execution trace.

The application can be used from:

-   a command-line interface, and
-   a simple Streamlit UI.

The runtime is fully local and uses Ollama for both embeddings and the
local language model.

------------------------------------------------------------------------

## 2. Assignment Alignment

The implementation was created against the supplied Agentic AI Developer
case study.

The case study expects an application that can:

-   ingest enterprise documents,
-   use vector retrieval,
-   support multi-agent roles for planning, retrieval, reasoning,
    validation, and memory,
-   handle complex multi-step and cross-document questions,
-   orchestrate and route work,
-   validate and ground responses,
-   provide guardrails,
-   expose evaluation and observability signals,
-   provide explainability,
-   and run locally.

The assignment also identifies the following expected deliverables:

-   Application code base
-   Fully functional application
-   Architecture and agent-flow diagram
-   Evaluation and guardrail documentation
-   Unit-test case documentation

This repository contains those deliverables.

------------------------------------------------------------------------

## 3. Scope

### Included

-   Local enterprise-document ingestion
-   Text document chunking
-   Vector embeddings
-   Chroma vector database
-   Query decomposition
-   Multi-agent LangGraph workflow
-   Retrieval with similarity/distance scores
-   Multi-document reasoning
-   Source attribution
-   Grounding verification
-   Retrieval relevance evaluation
-   Grounding confidence evaluation
-   Response guardrails
-   Agent execution traces
-   Structured evaluation output
-   Streamlit evaluation UI
-   Automated unit tests
-   Evaluation and guardrail documentation

### Explicitly out of scope

The case study identifies the following areas as out of scope:

-   Cloud/production deployment
-   Enterprise authentication/access control
-   Model training or fine-tuning
-   Advanced UI

This implementation therefore focuses on the local agentic workflow
rather than production infrastructure.

------------------------------------------------------------------------

# 4. What Was Built

The final system follows this execution path:

``` text
User Query
    |
    v
Orchestrator Agent
    |
    v
Memory Agent
    |
    v
Retriever Agent
    |
    v
Analyst Agent
    |
    v
Verifier Agent
    |
    v
Evaluation Layer
    |
    v
Final Response
```

The workflow is implemented using LangGraph.

The important design principle is that the Analyst does not
independently search the internet or use outside knowledge. It receives
the retrieved enterprise document context and is instructed to answer
from that evidence.

The Verifier then checks the proposed answer against the same retrieved
evidence.

------------------------------------------------------------------------

# 5. Architecture

## 5.1 High-level architecture

``` text
                         +------------------+
                         |    User Query    |
                         +--------+---------+
                                  |
                                  v
                       +----------+----------+
                       | Orchestrator Agent  |
                       | Planning /          |
                       | Decomposition       |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |    Memory Agent     |
                       | Session Context     |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |   Retriever Agent   |
                       | Chroma + Embeddings |
                       +----------+----------+
                                  |
                         Retrieved Documents
                         + Retrieval Scores
                                  |
                                  v
                       +----------+----------+
                       |    Analyst Agent    |
                       | Reasoning /          |
                       | Synthesis            |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |   Verifier Agent    |
                       | Grounding /         |
                       | Source Validation   |
                       +----------+----------+
                                  |
                                  v
                       +----------+----------+
                       |  Evaluation Layer   |
                       | Relevance           |
                       | Grounding           |
                       | Confidence          |
                       | Guardrails          |
                       | Conflict Status     |
                       +----------+----------+
                                  |
                                  v
                           Final Response
```

The architecture diagram is also maintained separately in:

``` text
docs/architecture.md
```

------------------------------------------------------------------------

# 6. Agent Responsibilities

## 6.1 Orchestrator Agent

File:

``` text
src/agents/orchestrator.py
```

Responsibilities:

-   validate the incoming query,
-   clean the query,
-   decompose a complex query into sub-queries,
-   establish the execution plan.

The current execution plan is:

``` text
retrieve -> analyze -> verify
```

Query decomposition is intentionally rule-based. The current
implementation splits queries around connectors such as:

``` text
and
also
```

### Important assumption

The Orchestrator is **not an LLM-based planner** in the current
implementation.

This was intentionally kept simple because the assignment requires query
planning/decomposition and orchestration but does not explicitly require
an LLM to perform the planning step.

------------------------------------------------------------------------

## 6.2 Memory Agent

File:

``` text
src/agents/memory.py
```

Responsibilities:

-   retain previous query/response interactions,
-   expose previous interactions to later Analyst calls,
-   provide conversational context during the current application
    session.

### Important limitation

The current Memory Agent uses **in-process session memory**.

It is not a persistent enterprise memory database.

------------------------------------------------------------------------

## 6.3 Retriever Agent

File:

``` text
src/agents/retriever.py
```

Responsibilities:

-   receive retrieval queries,
-   call the Chroma vector store,
-   return relevant document chunks and their retrieval distances.

The Retriever delegates the actual vector search to:

``` text
src/retrieval/vector_store.py
```

------------------------------------------------------------------------

## 6.4 Analyst Agent

File:

``` text
src/agents/analyst.py
```

Responsibilities:

-   reason over retrieved documents,
-   synthesize information from multiple documents,
-   use memory context when available,
-   avoid unsupported claims,
-   explicitly indicate when the retrieved evidence is insufficient.

The Analyst uses the local Ollama model:

``` text
llama3.2:3b
```

The prompt explicitly instructs the model to use only retrieved source
documents.

------------------------------------------------------------------------

## 6.5 Verifier Agent

File:

``` text
src/agents/verifier.py
```

Responsibilities:

-   inspect the proposed answer,
-   compare it against retrieved documents,
-   determine whether it is grounded,
-   provide a verification explanation,
-   expose source documents used for the response.

The verifier returns a structured result containing:

``` text
grounded
verification
sources
```

The expected verifier output contains:

``` text
GROUNDED: YES
```

or:

``` text
GROUNDED: NO
```

------------------------------------------------------------------------

# 7. Document Ingestion and RAG Pipeline

Source documents are stored under:

``` text
data/documents/
```

The current synthetic test set contains:

``` text
company_policy.txt
leave_policy.txt
employee_handbook.txt
```

Documents are loaded by:

``` text
src/ingestion/document_loader.py
```

The ingestion process is:

``` text
Text Documents
      |
      v
TextLoader
      |
      v
Document Objects
      |
      v
RecursiveCharacterTextSplitter
      |
      v
Document Chunks
      |
      v
Ollama Embeddings
      |
      v
Chroma
```

### Chunking configuration

The current splitter uses:

``` text
chunk_size = 500
chunk_overlap = 50
```

### Embedding model

``` text
nomic-embed-text
```

### Vector database

``` text
Chroma
```

The Chroma data is persisted under:

``` text
data/chroma/
```

------------------------------------------------------------------------

# 8. Retrieval

The Retriever uses:

``` python
similarity_search_with_score()
```

The returned value is a **distance**, not a normalized relevance
percentage.

For the current Chroma setup:

``` text
lower distance = more similar
higher distance = less similar
```

Retrieval returns the top:

``` text
k = 3
```

results by default.

Retrieval scores are preserved in the LangGraph state and passed to the
evaluation layer.

------------------------------------------------------------------------

# 9. Retrieval Relevance Evaluation

The evaluation layer calculates the average retrieval distance.

Current threshold:

``` text
RETRIEVAL_DISTANCE_THRESHOLD = 0.65
```

Interpretation:

``` text
average distance <= 0.65
    -> RETRIEVAL_AVAILABLE

average distance > 0.65
    -> IRRELEVANT_RETRIEVAL
```

### Important assumption

The value `0.65` is an **empirical calibration for the supplied
synthetic test documents**.

It is not:

-   a universal Chroma threshold,
-   a mathematical guarantee of relevance,
-   or a threshold specified by the assignment.

It is documented as an application-level evaluation threshold.

------------------------------------------------------------------------

# 10. Reasoning and Multi-Document Synthesis

A complex query can be decomposed into multiple sub-queries.

For example:

``` text
What are the leave rules and what does the employee handbook say
about remote work?
```

can be decomposed into separate retrieval concerns.

The Retriever executes retrieval for each sub-query and combines the
resulting documents before sending them to the Analyst.

The Analyst can therefore reason across multiple retrieved documents.

This supports the case-study requirement for cross-document reasoning.

------------------------------------------------------------------------

# 11. Grounding and Verification

After the Analyst generates an answer, the Verifier receives:

-   original query,
-   proposed answer,
-   retrieved documents.

The Verifier checks whether the answer is supported by the retrieved
evidence.

Possible result:

``` text
GROUNDED: YES
```

or:

``` text
GROUNDED: NO
```

The verification explanation is displayed in the UI and included in the
structured evaluation result.

------------------------------------------------------------------------

# 12. Guardrails

The evaluation layer combines retrieval quality and verifier results.

## 12.1 Allow response

When:

-   retrieval is considered relevant, and
-   the verifier establishes grounding,

the system returns:

``` text
ALLOW_RESPONSE
```

## 12.2 Limit response

When:

-   retrieval is insufficient,
-   retrieval is irrelevant,
-   or grounding cannot be established,

the system returns:

``` text
LIMIT_RESPONSE
```

The CLI/UI then avoids presenting the generated answer as a reliable
answer and instead explains that the available evidence is insufficient
or irrelevant.

------------------------------------------------------------------------

# 13. Grounding Confidence

Grounding confidence is currently qualitative.

Possible status:

``` text
SUPPORTED
```

or:

``` text
LOW
```

`LOW` is produced when:

-   retrieval is insufficient,
-   retrieval is irrelevant,
-   or the verifier cannot establish grounding.

`SUPPORTED` is produced when relevant retrieval is available and the
verifier establishes grounding.

This is a decision signal, not a probabilistic confidence score.

------------------------------------------------------------------------

# 14. Conflict Detection

The evaluation result includes a structured conflict-detection field:

``` text
NO_CONFLICT
```

or:

``` text
CONFLICT_DETECTED
```

### Important limitation

The current workflow does **not** independently generate multiple
competing agent answers and perform autonomous contradiction analysis.

Instead, the evaluator supports a structured `conflicting_outputs` flag
so that conflict information can be incorporated into the evaluation
result.

This limitation is intentionally documented rather than claiming a
capability that the current implementation does not provide.

------------------------------------------------------------------------

# 15. Agent Execution Trace

Every workflow execution records the agents involved.

Typical trace:

``` text
[
    "orchestrator",
    "memory",
    "retriever",
    "analyst",
    "verifier"
]
```

This trace is included in the evaluation output and displayed by the
Streamlit UI.

It provides a simple, inspectable record of the execution path.

------------------------------------------------------------------------

# 16. Structured Evaluation Output

The evaluation layer produces structured information including:

``` text
query
retrieval
retrieval_scores
sources
grounding_status
grounding_confidence
guardrail
conflict_detection
grounded
verification
trace
answer
```

This makes the evaluation result inspectable instead of returning only a
final natural-language answer.

------------------------------------------------------------------------

# 17. Synthetic Test Documents

The project currently uses three synthetic documents:

``` text
data/documents/company_policy.txt
data/documents/leave_policy.txt
data/documents/employee_handbook.txt
```

These were created because no real enterprise documents were supplied
for the demonstration.

Every document is explicitly marked as synthetic test data.

They are intended to exercise:

-   document ingestion,
-   chunking,
-   vector retrieval,
-   multi-document reasoning,
-   source attribution,
-   grounding,
-   irrelevant retrieval detection,
-   and guardrail behavior.

They should not be treated as actual organizational policies.

------------------------------------------------------------------------

# 18. Technology Stack

  Component           Technology
  ------------------- -----------------------------
  Language            Python
  Agent framework     LangGraph
  LLM orchestration   LangChain
  Local LLM           Ollama / `llama3.2:3b`
  Embeddings          Ollama / `nomic-embed-text`
  Vector database     Chroma
  Document loading    LangChain Community
  Text splitting      LangChain Text Splitters
  Workflow            LangGraph StateGraph
  UI                  Streamlit
  Testing             Pytest
  Configuration       python-dotenv
  Runtime             Local machine

------------------------------------------------------------------------

# 19. Project Structure

``` text
enterprise-knowledge-ops-agent/
│
├── data/
│   └── documents/
│       ├── company_policy.txt
│       ├── leave_policy.txt
│       └── employee_handbook.txt
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── document_loader.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── retriever.py
│   │   ├── analyst.py
│   │   ├── verifier.py
│   │   └── memory.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py
│   │
│   └── main.py
│
├── tests/
│   ├── __init__.py
│   └── test_agents.py
│
├── docs/
│   ├── architecture.md
│   ├── evaluation-and-guardrails.md
│   └── unit-test-cases.md
│
├── data/chroma/
│   └── generated vector-store data
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

`data/chroma/` is generated runtime data and does not represent source
enterprise documents.

------------------------------------------------------------------------

# 20. Prerequisites for a New Machine

The following assumes a Windows machine.

Install:

1.  Python
2.  Git
3.  Ollama

Verify Python:

``` powershell
python --version
```

Verify Git:

``` powershell
git --version
```

Verify Ollama:

``` powershell
ollama --version
```

------------------------------------------------------------------------

# 21. Install and Prepare Ollama

The project uses two local Ollama models.

### Language model

``` text
llama3.2:3b
```

Pull it:

``` powershell
ollama pull llama3.2:3b
```

### Embedding model

``` text
nomic-embed-text
```

Pull it:

``` powershell
ollama pull nomic-embed-text
```

Verify installed models:

``` powershell
ollama list
```

Both models should be present.

The application uses Ollama locally, so an internet connection is not
required for model inference after the models have been downloaded.

------------------------------------------------------------------------

# 22. Clone the Project

Clone the repository:

``` powershell
git clone <repository-url>
```

Then enter the project:

``` powershell
cd enterprise-knowledge-ops-agent
```

If the project was supplied as a ZIP instead, extract it and open a
terminal in the extracted project directory.

------------------------------------------------------------------------

# 23. Create a Python Virtual Environment

From the project root:

``` powershell
python -m venv .venv
```

Activate it:

``` powershell
.venv\Scripts\activate
```

After activation, the terminal should show the virtual environment name.

------------------------------------------------------------------------

# 24. Install Python Dependencies

Upgrade pip:

``` powershell
python -m pip install --upgrade pip
```

Install project dependencies:

``` powershell
pip install -r requirements.txt
```

The main runtime dependencies include:

``` text
langchain
langgraph
langchain-ollama
langchain-chroma
chromadb
langchain-community
langchain-text-splitters
pypdf
python-dotenv
pytest
streamlit
```

------------------------------------------------------------------------

# 25. Environment Configuration

The current implementation does not require a cloud API key.

An `.env` file may therefore remain empty.

Do not add cloud API keys unnecessarily.

The runtime uses:

``` text
Ollama
```

for local LLM inference and embeddings.

The `.gitignore` should continue to exclude:

``` text
.env
.venv/
__pycache__/
.pytest_cache/
```

and generated local artifacts as appropriate.

------------------------------------------------------------------------

# 26. Verify the Test Documents

Before running the application, confirm:

``` text
data/documents/company_policy.txt
data/documents/leave_policy.txt
data/documents/employee_handbook.txt
```

exist.

If you replace these documents with your own `.txt` enterprise
documents, the ingestion pipeline will load `.txt` files from the same
directory.

The documents should contain meaningful enterprise knowledge relevant to
the questions you intend to ask.

------------------------------------------------------------------------

# 27. Build / Use the Vector Store

The current workflow creates the Chroma vector store when
`KnowledgeOperationsWorkflow` is initialized.

The relevant flow is:

``` text
load_documents()
        ↓
split_documents()
        ↓
create_vector_store()
```

The vector store is persisted under:

``` text
data/chroma/
```

Therefore, no separate database server is required.

If you intentionally replace the source documents and need a clean
rebuild, remove the generated Chroma directory and start the application
again.

On Windows PowerShell:

``` powershell
Remove-Item -Recurse -Force data\chroma
```

Then rerun the application.

Only remove this directory when you intentionally want to rebuild the
vector store.

------------------------------------------------------------------------

# 28. Run the Command-Line Application

From the project root with the virtual environment activated:

``` powershell
python -m src.main
```

You will be prompted:

``` text
Enter your question:
```

Enter a question based on the supplied documents.

The CLI displays:

``` text
=== FINAL ANSWER ===

=== SOURCES ===

=== VERIFICATION ===

=== EVALUATION ===
```

The evaluation section includes retrieval, grounding, guardrail,
conflict, and trace information.

------------------------------------------------------------------------

# 29. Run the Streamlit Application

The recommended evaluator interface is the Streamlit application.

Run:

``` powershell
streamlit run src/ui/app.py
```

Streamlit will display a local URL in the terminal.

Open that URL in a browser.

The UI provides:

-   question input,
-   final answer,
-   sources,
-   verification,
-   evaluation metrics,
-   retrieval details,
-   guardrail details,
-   grounding details,
-   conflict detection,
-   agent execution trace.

------------------------------------------------------------------------

# 30. How to Test the Application

The Streamlit interface includes a **How to test** section containing
two recommended scenarios.

## Scenario A --- Supported multi-document query

Use:

``` text
What are the leave rules and what does the employee handbook say about remote work?
```

This is intended to demonstrate:

-   query decomposition,
-   retrieval,
-   multiple source documents,
-   reasoning/synthesis,
-   source attribution,
-   grounding,
-   evaluation,
-   and agent tracing.

Expected source documents include relevant synthetic enterprise
documents such as:

``` text
employee_handbook.txt
company_policy.txt
```

The exact retrieval set can depend on the vector retrieval results.

------------------------------------------------------------------------

## Scenario B --- Unsupported/irrelevant query

Use:

``` text
What is the company's policy for underwater mining operations?
```

This is intentionally outside the knowledge contained in the synthetic
test documents.

The expected behavior is that the evaluation layer identifies
weak/irrelevant retrieval and limits the response rather than presenting
an unsupported answer as reliable.

A representative result is:

``` text
retrieval = IRRELEVANT_RETRIEVAL
grounding_status = UNGROUNDED
grounding_confidence = LOW
guardrail = LIMIT_RESPONSE
```

This scenario demonstrates failure detection and guardrail behavior.

------------------------------------------------------------------------

# 31. Run the Automated Tests

From the project root:

``` powershell
pytest -q
```

The current test suite contains **19 tests**.

The tests use mocked LLM responses where appropriate so that the
unit-test suite does not depend on live model inference.

The tests cover:

-   document ingestion,
-   document splitting,
-   vector-store creation,
-   retrieval,
-   Retriever Agent,
-   Memory Agent,
-   Analyst Agent,
-   Verifier Agent,
-   Orchestrator Agent,
-   empty-query validation,
-   LangGraph workflow,
-   multi-document query handling,
-   retrieval evaluation,
-   irrelevant retrieval,
-   guardrails,
-   grounding confidence,
-   conflict evaluation hook,
-   structured evaluation output.

Expected current result:

``` text
19 passed
```

------------------------------------------------------------------------

# 32. Testing Strategy

The unit tests deliberately avoid requiring an LLM response for every
test.

A lightweight mock object is used for Analyst/Verifier testing.

This makes the tests:

-   deterministic,
-   faster,
-   less dependent on Ollama availability,
-   easier to run in development and CI.

The live Ollama application is separately tested through the functional
demonstration scenarios.

Detailed test documentation is available in:

``` text
docs/unit-test-cases.md
```

------------------------------------------------------------------------

# 33. Evaluation and Guardrail Documentation

Detailed evaluation documentation is available in:

``` text
docs/evaluation-and-guardrails.md
```

It documents:

-   retrieval evaluation,
-   retrieval threshold,
-   grounding,
-   grounding confidence,
-   guardrails,
-   source attribution,
-   execution traces,
-   failure detection,
-   conflict detection limitation,
-   local runtime configuration,
-   functional validation,
-   and assignment alignment.

------------------------------------------------------------------------

# 34. Architecture Documentation

Detailed architecture and agent flow are available in:

``` text
docs/architecture.md
```

This includes the Mermaid architecture diagram and the relationship
between:

``` text
Orchestrator
Memory
Retriever
Analyst
Verifier
Evaluation
```

------------------------------------------------------------------------

# 35. Unit Test Documentation

Detailed unit-test documentation is available in:

``` text
docs/unit-test-cases.md
```

It describes the 19 current test cases and the testing approach.

------------------------------------------------------------------------

# 36. Design Assumptions

This section records decisions made during implementation so that
reviewers do not have to infer them.

### Assumption A --- Fully local model execution

The implementation uses Ollama rather than a cloud LLM.

Reason:

-   the case study requires a fully local application,
-   it avoids external API dependency,
-   it makes functional demonstration reproducible on the local machine.

### Assumption B --- Synthetic documents are acceptable test data

No real enterprise knowledge base was provided.

Therefore synthetic documents were created specifically for functional
testing.

They are explicitly marked as synthetic.

### Assumption C --- Rule-based planning is sufficient

The Orchestrator performs lightweight deterministic decomposition rather
than using an LLM to create the plan.

This keeps the planning layer deterministic and inspectable while still
providing query decomposition and routing.

### Assumption D --- Retrieval distance is an evaluation signal

Chroma returns distance values through:

``` text
similarity_search_with_score()
```

These are treated as retrieval signals.

They are not presented as percentages.

### Assumption E --- Retrieval threshold is empirical

The threshold:

``` text
0.65
```

was calibrated against the supplied synthetic test documents.

It should be recalibrated if a materially different document collection,
embedding model, or retrieval configuration is introduced.

### Assumption F --- Session memory is sufficient

The Memory Agent stores interactions in memory for the current
workflow/application lifetime.

Persistent enterprise memory was not implemented because it is not
explicitly required by the assignment.

------------------------------------------------------------------------

# 37. Known Limitations

The following limitations are intentional and documented.

## 37.1 Rule-based query decomposition

The Orchestrator currently uses deterministic rules rather than an LLM
planner.

## 37.2 Session memory

Memory is not persisted to an external database.

## 37.3 Conflict detection

Conflict detection is currently a structured evaluation hook and does
not independently compare multiple competing generated answers.

## 37.4 Retrieval threshold

The `0.65` retrieval-distance threshold is specific to the current test
setup and should not be treated as universally applicable.

## 37.5 Local model capability

The application uses:

``` text
llama3.2:3b
```

which is intentionally lightweight for local execution. Its reasoning
quality can differ from larger hosted models.

## 37.6 Text documents

The current ingestion implementation scans:

``` text
data/documents/*.txt
```

The project includes `pypdf` in the dependency set, but the current
document loader is implemented around text files.

------------------------------------------------------------------------

# 38. Troubleshooting

## Problem: `ModuleNotFoundError`

Make sure the command is executed from the project root:

``` text
enterprise-knowledge-ops-agent/
```

For Streamlit, use:

``` powershell
streamlit run src/ui/app.py
```

The UI contains project-root path handling so that the `src` package can
be imported correctly.

------------------------------------------------------------------------

## Problem: Ollama model not found

Run:

``` powershell
ollama list
```

Make sure these models exist:

``` text
llama3.2:3b
nomic-embed-text
```

If missing:

``` powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

------------------------------------------------------------------------

## Problem: Ollama is not running

Start the Ollama application/service and verify:

``` powershell
ollama list
```

Then rerun the application.

------------------------------------------------------------------------

## Problem: Retrieval results look wrong after changing documents

Delete the generated Chroma store:

``` powershell
Remove-Item -Recurse -Force data\chroma
```

Then restart the application so the documents are embedded again.

------------------------------------------------------------------------

## Problem: Tests fail because of environment setup

Confirm the virtual environment is active:

``` powershell
.venv\Scripts\activate
```

Then reinstall dependencies:

``` powershell
pip install -r requirements.txt
```

Run:

``` powershell
pytest -q
```

------------------------------------------------------------------------

# 39. Security and Governance Notes

The project includes application-level controls for:

-   grounding,
-   retrieval relevance,
-   response limiting,
-   source attribution,
-   confidence signaling,
-   and evaluation traceability.

However, this is a local case-study implementation and does not provide
enterprise authentication or authorization.

Do not treat the application as a production enterprise access-control
system.

Do not place confidential production documents into the supplied
synthetic test setup without reviewing the required security and
data-handling controls for the target environment.

------------------------------------------------------------------------

# 40. Why There Is No Cloud Deployment

The case study explicitly places cloud/production deployment outside the
core scope.

The final implementation therefore does not require:

-   AWS services,
-   Azure OpenAI,
-   hosted vector databases,
-   cloud authentication,
-   or cloud deployment.

Everything required for the demonstrated workflow runs locally.

------------------------------------------------------------------------

# 41. Why Ollama Was Used

The final runtime uses Ollama because the application needs a local LLM
and embedding provider.

The runtime therefore consists of:

``` text
Ollama
    |
    +-- llama3.2:3b
    |
    +-- nomic-embed-text
```

This also avoids depending on external API keys during normal execution.

------------------------------------------------------------------------

# 42. Runtime Flow in Detail

For every query:

``` text
1. User enters query
        |
2. Orchestrator validates query
        |
3. Orchestrator creates plan
        |
4. Orchestrator decomposes query if applicable
        |
5. Memory Agent provides previous session context
        |
6. Retriever executes vector searches
        |
7. Retrieval distances are captured
        |
8. Analyst receives retrieved documents
        |
9. Analyst generates evidence-based answer
        |
10. Verifier checks answer against documents
        |
11. Evaluation layer evaluates retrieval
        |
12. Evaluation layer evaluates grounding
        |
13. Evaluation layer determines confidence
        |
14. Guardrail decides ALLOW_RESPONSE or LIMIT_RESPONSE
        |
15. Sources and evaluation information are exposed
        |
16. Agent execution trace is exposed
```

------------------------------------------------------------------------

# 43. Example Successful Evaluation

For a supported multi-document query, a representative evaluation
contains:

``` json
{
  "retrieval": {
    "status": "RETRIEVAL_AVAILABLE"
  },
  "grounding_status": "GROUNDED",
  "grounding_confidence": {
    "status": "SUPPORTED"
  },
  "guardrail": {
    "action": "ALLOW_RESPONSE"
  },
  "conflict_detection": {
    "status": "NO_CONFLICT"
  },
  "trace": [
    "orchestrator",
    "memory",
    "retriever",
    "analyst",
    "verifier"
  ]
}
```

The exact retrieval scores can vary with the vector store and documents.

------------------------------------------------------------------------

# 44. Example Failure Evaluation

For the intentionally unsupported underwater-mining query, the tested
system produced the following type of result:

``` json
{
  "retrieval": {
    "status": "IRRELEVANT_RETRIEVAL"
  },
  "grounding_status": "UNGROUNDED",
  "grounding_confidence": {
    "status": "LOW"
  },
  "guardrail": {
    "action": "LIMIT_RESPONSE"
  },
  "conflict_detection": {
    "status": "NO_CONFLICT"
  },
  "trace": [
    "orchestrator",
    "memory",
    "retriever",
    "analyst",
    "verifier"
  ]
}
```

This demonstrates that the system can identify when the available
evidence does not support a reliable answer.

------------------------------------------------------------------------

# 45. Assignment Deliverables Checklist

The expected deliverables are covered as follows:

  -------------------------------------------------------------------------
  Assignment Deliverable              Location / Evidence
  ----------------------------------- -------------------------------------
  Application code base               `src/`

  Fully functional application        `src/main.py`, `src/ui/app.py`

  Architecture and agent flow         `docs/architecture.md`

  Evaluation and guardrail            `docs/evaluation-and-guardrails.md`
  documentation                       

  Unit test case documentation        `docs/unit-test-cases.md`

  Automated tests                     `tests/test_agents.py`

  Synthetic test knowledge base       `data/documents/`
  -------------------------------------------------------------------------

------------------------------------------------------------------------

# 46. Final Project Status

The current implementation provides:

``` text
[✓] Multi-agent architecture
[✓] Orchestrator
[✓] Retriever
[✓] Analyst
[✓] Verifier
[✓] Memory
[✓] LangGraph workflow
[✓] Document ingestion
[✓] Chunking
[✓] Local embeddings
[✓] Chroma vector store
[✓] Multi-document retrieval
[✓] Query decomposition
[✓] Source attribution
[✓] Grounding verification
[✓] Retrieval relevance evaluation
[✓] Grounding confidence
[✓] Guardrails
[✓] Evaluation output
[✓] Agent execution trace
[✓] Failure detection
[✓] Streamlit UI
[✓] CLI
[✓] Unit tests
[✓] Evaluation documentation
[✓] Architecture documentation
[✓] Unit-test documentation
```

The implementation intentionally documents its limitations rather than
presenting unsupported capabilities as implemented features.

------------------------------------------------------------------------

# 47. Quick Start

For a developer who already has Python, Git, and Ollama installed:

``` powershell
git clone <repository-url>
cd enterprise-knowledge-ops-agent

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull nomic-embed-text

pytest -q

streamlit run src/ui/app.py
```

Then try:

``` text
What are the leave rules and what does the employee handbook say about remote work?
```

and:

``` text
What is the company's policy for underwater mining operations?
```

The first demonstrates the grounded multi-document path.

The second demonstrates irrelevant retrieval, low grounding confidence,
and response limiting.

------------------------------------------------------------------------

# 48. Related Documentation

``` text
docs/architecture.md
docs/evaluation-and-guardrails.md
docs/unit-test-cases.md
```

These documents provide more focused detail while this README serves as
the overall single-source project guide.

------------------------------------------------------------------------

# 49. Final Note

This README describes the implementation as it exists, including its
design decisions, assumptions, tested behavior, limitations, and setup
process.

It should be updated whenever a material implementation change is made
to the project so that the README remains the single source of truth for
the application.
