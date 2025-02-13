from dataclasses import dataclass

from flare_ai_rag.openrouter.model import Model
from flare_ai_rag.router.prompts import BASE_PROMPT


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
