from dataclasses import dataclass

from flare_ai_rag.ai import Model
from flare_ai_rag.responder.prompts import RESPONDER_INSTRUCTION, RESPONDER_PROMPT


@dataclass(frozen=True)
class ResponderConfig:
    model: Model | None
    system_prompt: str
    query_prompt: str

    @staticmethod
    def load(model_config: dict | None = None) -> "ResponderConfig":
        """Loads the Responder config."""
        if not model_config:
            # When using Gemini
            model = None
        else:
            # When using OpenRouter
            model = Model(
                model_id=model_config["id"],
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
            )

        return ResponderConfig(
            model=model,
            system_prompt=RESPONDER_INSTRUCTION,
            query_prompt=RESPONDER_PROMPT,
        )
