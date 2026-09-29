import json

from src.evaluation.evaluator import Evaluator


def main():
    evaluator = Evaluator()

    query = input("Enter your question: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    evaluation = evaluator.evaluate(query)

    print("\n=== FINAL ANSWER ===")

    if evaluation["guardrail"]["action"] == "LIMIT_RESPONSE":
        print(
            "I cannot provide a reliable answer because "
            "the retrieved source evidence is insufficient or "
            "irrelevant to the question."
        )
    else:
        print(evaluation["answer"])

    print("\n=== SOURCES ===")
    for source in evaluation["sources"]:
        print(f"- {source}")

    print("\n=== VERIFICATION ===")
    print(evaluation["verification"])

    print("\n=== EVALUATION ===")
    print(
        json.dumps(
            {
                "retrieval": evaluation["retrieval"],
                "retrieval_scores": evaluation["retrieval_scores"],
                "grounding_status": evaluation["grounding_status"],
                "grounding_confidence": evaluation[
                    "grounding_confidence"
                ],
                "guardrail": evaluation["guardrail"],
                "conflict_detection": evaluation[
                    "conflict_detection"
                ],
                "trace": evaluation["trace"]
            },
            indent=2
        )
    )


if __name__ == "__main__":
    main()