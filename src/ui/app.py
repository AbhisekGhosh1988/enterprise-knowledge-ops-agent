import streamlit as st

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import Evaluator


st.set_page_config(
    page_title="Enterprise Knowledge Operations Agent",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def get_evaluator():
    return Evaluator()


def main():
    st.title("Enterprise Knowledge Operations Agent")

    st.write(
        "Ask questions about the enterprise knowledge documents. "
        "The system retrieves evidence, generates an answer, "
        "verifies grounding, and applies evaluation guardrails."
    )
    with st.expander("How to test"):
        st.markdown(
            """
    ### Supported query

    Try:

    > What are the leave rules and what does the employee handbook say about remote work?

    This demonstrates:
    - Multi-document retrieval
    - Query decomposition
    - Cross-document synthesis
    - Source attribution
    - Grounding verification

    ### Unsupported query

    Try:

    > What is the company's policy for underwater mining operations?

    This demonstrates:
    - Irrelevant retrieval detection
    - Low grounding confidence
    - Guardrail response limitation

    The application uses the enterprise knowledge documents in the `data/documents` directory.
    """
        )
    query = st.text_area(
        "Enter your question",
        height=100,
        placeholder=(
            "Example: What are the leave rules and what does "
            "the employee handbook say about remote work?"
        )
    )

    if st.button("Ask Knowledge Agent", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Processing your question..."):
            evaluator = get_evaluator()
            evaluation = evaluator.evaluate(query.strip())

        st.divider()

        # --------------------------------------------------
        # Final Answer
        # --------------------------------------------------

        st.subheader("Final Answer")

        if evaluation["guardrail"]["action"] == "LIMIT_RESPONSE":
            st.warning(
                "The system could not provide a reliable answer "
                "because the retrieved source evidence was "
                "insufficient or irrelevant."
            )
        else:
            st.write(evaluation["answer"])

        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        st.subheader("Sources")

        sources = evaluation.get("sources", [])

        if sources:
            for source in sources:
                st.write(f"- {source}")
        else:
            st.write("No sources identified.")

        # --------------------------------------------------
        # Verification
        # --------------------------------------------------

        st.subheader("Verification")

        verification = evaluation.get("verification", "")

        if verification:
            st.code(verification)
        else:
            st.write("No verification information available.")

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        st.subheader("Evaluation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Retrieval",
                evaluation["retrieval"]["status"]
            )

        with col2:
            st.metric(
                "Grounding",
                evaluation["grounding_confidence"]["status"]
            )

        with col3:
            st.metric(
                "Guardrail",
                evaluation["guardrail"]["action"]
            )

        # --------------------------------------------------
        # Retrieval Details
        # --------------------------------------------------

        with st.expander("Retrieval Details"):
            st.write(
                "Average retrieval distance:",
                evaluation["retrieval"]["average_score"]
            )

            st.write(
                "Retrieval scores:",
                evaluation["retrieval_scores"]
            )

        # --------------------------------------------------
        # Guardrail Details
        # --------------------------------------------------

        with st.expander("Guardrail Details"):
            st.write(
                "Action:",
                evaluation["guardrail"]["action"]
            )

            st.write(
                "Reason:",
                evaluation["guardrail"]["reason"]
            )

        # --------------------------------------------------
        # Grounding Details
        # --------------------------------------------------

        with st.expander("Grounding Details"):
            st.write(
                "Status:",
                evaluation["grounding_status"]
            )

            st.write(
                "Confidence:",
                evaluation["grounding_confidence"]["status"]
            )

            st.write(
                "Reason:",
                evaluation["grounding_confidence"]["reason"]
            )

        # --------------------------------------------------
        # Conflict Detection
        # --------------------------------------------------

        with st.expander("Conflict Detection"):
            st.write(
                "Status:",
                evaluation["conflict_detection"]["status"]
            )

            st.write(
                "Action:",
                evaluation["conflict_detection"]["action"]
            )

        # --------------------------------------------------
        # Agent Trace
        # --------------------------------------------------

        with st.expander("Agent Execution Trace"):
            trace = evaluation.get("trace", [])

            if trace:
                st.write(" → ".join(trace))
            else:
                st.write("No trace available.")


if __name__ == "__main__":
    main()