import structlog

from flare_ai_rag.config import config
from flare_ai_rag.openrouter.client import OpenRouterClient
from flare_ai_rag.router.config import RouterConfig
from flare_ai_rag.router.router import QueryRouter
from flare_ai_rag.utils import loader

logger = structlog.get_logger(__name__)


def main() -> None:
    # Initialize OpenRouter client
    client = OpenRouterClient(
        api_key=config.open_router_api_key, base_url=config.open_router_base_url
    )

    # Set up responder config
    model_config = loader.load_json(config.input_path / "input_parameters.json")[
        "router_model"
    ]
    router_config = RouterConfig.load(model_config)

    # Initialize the QueryRouter.
    router = QueryRouter(client=client, config=router_config)

    # List of sample queries to classify.
    queries = [
        "What is the capital of France?",
        "Is Flare an EVM chain?",
        "What is the FTSO?",
    ]

    # Process each query and print its classification.
    for query in queries:
        classification = router.route_query(query)
        logger.info("Query processed.", classification=classification)


if __name__ == "__main__":
    main()
