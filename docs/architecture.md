# Enterprise Knowledge Operations Agent
## Architecture and Agent Flow

```mermaid
flowchart TD
    U[User Query]

    O[Orchestrator Agent<br/>Planning and Query Decomposition]

    M[Memory Agent<br/>Conversation Context]

    R[Retriever Agent<br/>Chroma Vector Store<br/>Ollama Embeddings]

    A[Analyst Agent<br/>Reasoning and Synthesis<br/>Ollama LLM]

    V[Verifier Agent<br/>Grounding and Source Validation<br/>Ollama LLM]

    E[Evaluation Layer<br/>Retrieval Relevance<br/>Grounding Confidence<br/>Guardrails<br/>Conflict Detection]

    F[Final Response]

    U --> O
    O --> M
    M --> R
    R --> A
    A --> V
    V --> E
    E --> F

    R -.->|Retrieved Documents<br/>+ Retrieval Scores| A
    V -.->|Verification Result<br/>+ Sources| E
    O -.->|Plan + Sub-Queries| E