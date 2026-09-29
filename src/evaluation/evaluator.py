from typing import Any, Dict

from src.graph.workflow import KnowledgeOperationsWorkflow


RETRIEVAL_DISTANCE_THRESHOLD = 0.65


class EvaluationResult:

    def __init__(
        self,
        query: str,
        result: Dict[str, Any]
    ):
        self.query = query
        self.result = result

    def _evaluate_retrieval(self) -> Dict[str, Any]:
        scores = self.result.get("retrieval_scores", [])

        if not scores:
            return {
                "status": "INSUFFICIENT_RETRIEVAL",
                "average_score": None
            }

        average_score = sum(scores) / len(scores)

        if average_score > RETRIEVAL_DISTANCE_THRESHOLD:
            return {
                "status": "IRRELEVANT_RETRIEVAL",
                "average_score": average_score
            }

        return {
            "status": "RETRIEVAL_AVAILABLE",
            "average_score": average_score
        }

    def _evaluate_guardrail(
        self,
        verification: Dict[str, Any]
    ) -> Dict[str, Any]:

        grounded = verification.get(
            "grounded",
            False
        )

        retrieval_evaluation = self._evaluate_retrieval()

        if retrieval_evaluation["status"] in {
            "INSUFFICIENT_RETRIEVAL",
            "IRRELEVANT_RETRIEVAL"
        }:
            return {
                "action": "LIMIT_RESPONSE",
                "reason": (
                    "The retrieved evidence was insufficient or "
                    "irrelevant to the user query."
                )
            }

        if not grounded:
            return {
                "action": "LIMIT_RESPONSE",
                "reason": (
                    "The answer could not be sufficiently grounded "
                    "in the retrieved source documents."
                )
            }

        return {
            "action": "ALLOW_RESPONSE",
            "reason": (
                "The answer was grounded in relevant retrieved "
                "source documents."
            )
        }

    def _evaluate_conflicts(self) -> Dict[str, Any]:
        conflicting_outputs = self.result.get(
            "conflicting_outputs",
            False
        )

        if conflicting_outputs:
            return {
                "status": "CONFLICT_DETECTED",
                "action": "LIMIT_RESPONSE"
            }

        return {
            "status": "NO_CONFLICT",
            "action": "ALLOW_RESPONSE"
        }

    def _evaluate_grounding_confidence(
        self,
        verification: Dict[str, Any]
    ) -> Dict[str, Any]:

        grounded = verification.get(
            "grounded",
            False
        )

        retrieval = self._evaluate_retrieval()

        if retrieval["status"] in {
            "INSUFFICIENT_RETRIEVAL",
            "IRRELEVANT_RETRIEVAL"
        }:
            return {
                "status": "LOW",
                "reason": (
                    "Retrieved evidence is insufficient or irrelevant."
                )
            }

        if not grounded:
            return {
                "status": "LOW",
                "reason": (
                    "The verifier could not establish grounding."
                )
            }

        return {
            "status": "SUPPORTED",
            "reason": (
                "The response is grounded in relevant "
                "retrieved evidence."
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        verification = self.result.get(
            "verification",
            {}
        )

        retrieval = self._evaluate_retrieval()

        guardrail = self._evaluate_guardrail(
            verification
        )

        conflict = self._evaluate_conflicts()

        grounding_confidence = (
            self._evaluate_grounding_confidence(
                verification
            )
        )

        return {
            "query": self.query,

            "retrieval": retrieval,

            "retrieval_scores": self.result.get(
                "retrieval_scores",
                []
            ),

            "sources": verification.get(
                "sources",
                []
            ),

            "grounding_status": (
                "GROUNDED"
                if verification.get("grounded", False)
                else "UNGROUNDED"
            ),

            "grounding_confidence": grounding_confidence,

            "guardrail": guardrail,

            "conflict_detection": conflict,

            "grounded": verification.get(
                "grounded",
                False
            ),

            "verification": verification.get(
                "verification",
                ""
            ),

            "trace": self.result.get(
                "trace",
                []
            ),

            "answer": self.result.get(
                "final_answer",
                ""
            )
        }


class Evaluator:

    def __init__(self):
        self.workflow = KnowledgeOperationsWorkflow()

    def evaluate(
        self,
        query: str
    ) -> Dict[str, Any]:

        result = self.workflow.run(query)

        evaluation = EvaluationResult(
            query=query,
            result=result
        )

        return evaluation.to_dict()