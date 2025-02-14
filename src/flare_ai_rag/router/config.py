from dataclasses import dataclass

from flare_ai_rag.config import config
from flare_ai_rag.openrouter.model import Model
from flare_ai_rag.utils import loader

# Load base prompt
BASE_PROMPT = loader.load_txt(config.input_path / "router" / "prompts.txt")


@dataclass(frozen=True)
class RouterConfig:
    base_prompt: str
    model: Model
    answer_option: str
    clarify_option: str
    reject_option: str

    @staticmethod
    def load(model_config: dict) -> "RouterConfig":
        """Loads the router Model."""
        model = Model(
            model_id=model_config["id"],
            max_tokens=model_config["max_tokens"],
            temperature=model_config["temperature"],
        )

        return RouterConfig(
            base_prompt=BASE_PROMPT,
            model=model,
            answer_option="ANSWER",
            clarify_option="CLARIFY",
            reject_option="REJECT",
        )
