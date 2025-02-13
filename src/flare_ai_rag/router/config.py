from dataclasses import dataclass

from flare_ai_rag.openrouter.model import Model

BASE_PROMPT = """You are a query router. Analyze the query below and classify it by returning exactly one of these values—without any additional text:\n\n

    - **{answer_option}**: Use this if the query is clear, specific, and can be answered with factual information.\n
    - **{clarify_option}**: Use this if the query is ambiguous, vague, or needs additional context.\n
    - **{reject_option}**: Use this if the query is inappropriate, harmful, or completely out of scope.\n\n

    Query: {query}\n\n

    Return exactly one word: ANSWER, CLARIFY, or REJECT.
"""


@dataclass(frozen=True)
class RouterConfig:
    base_prompt: str
    model: Model
    answer_option: str
    clarify_option: str
    reject_option: str

    @staticmethod
    def load(router_model: Model) -> "RouterConfig":
        """Loads the router Model."""
        # Define the classification options.
        answer_option = "ANSWER"
        clarify_option = "CLARIFY"
        reject_option = "REJECT"

        base_prompt = BASE_PROMPT.format(
            answer_option=answer_option,
            clarify_option=clarify_option,
            reject_option=reject_option,
            query="{query}",
        )

        return RouterConfig(
            base_prompt=base_prompt,
            model=router_model,
            answer_option=answer_option,
            clarify_option=clarify_option,
            reject_option=reject_option,
        )
