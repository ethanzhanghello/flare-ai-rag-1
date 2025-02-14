import pandas as pd
import structlog
from qdrant_client import QdrantClient

from flare_ai_rag.config import config
from flare_ai_rag.openrouter.client import OpenRouterClient
from flare_ai_rag.responder.config import ResponderConfig
from flare_ai_rag.responder.responder import OpenRouterResponder
from flare_ai_rag.retriever.config import QdrantConfig
from flare_ai_rag.retriever.qdrant_collection import generate_collection
from flare_ai_rag.retriever.qdrant_retriever import QdrantRetriever
from flare_ai_rag.router.config import RouterConfig
from flare_ai_rag.router.router import QueryRouter
from flare_ai_rag.utils import loader

logger = structlog.get_logger(__name__)


def setup_clients(input_config: dict) -> tuple[OpenRouterClient, QdrantClient]:
    """Initialize OpenRouter and Qdrant clients."""
    # Setup OpenRouter client.
    openrouter_client = OpenRouterClient(
        api_key=config.open_router_api_key, base_url=config.open_router_base_url
    )

    # Setup Qdrant client.
    qdrant_config = QdrantConfig.load(input_config["qdrant_config"])
    qdrant_client = QdrantClient(host=qdrant_config.host, port=qdrant_config.port)

    return openrouter_client, qdrant_client


def setup_router(
    openrouter_client: OpenRouterClient, input_config: dict
) -> QueryRouter:
    """Initialize the query router."""
    router_model_config = input_config["router_model"]
    router_config = RouterConfig.load(router_model_config)
    return QueryRouter(client=openrouter_client, config=router_config)


def setup_responder(
    openrouter_client: OpenRouterClient, input_config: dict
) -> OpenRouterResponder:
    """Initialize the responder."""
    responder_config = input_config["responder_model"]
    responder_config = ResponderConfig.load(responder_config)
    return OpenRouterResponder(
        client=openrouter_client, responder_config=responder_config
    )


def setup_retriever(
    qdrant_client: QdrantClient,
    input_config: dict,
    df_docs: pd.DataFrame,
    collection: str | None = None,
) -> QdrantRetriever:
    """Initialize the Qdrant retriever."""
    qdrant_config = QdrantConfig.load(input_config["qdrant_config"])

    # (Re)generate qdrant collection
    if collection:
        generate_collection(
            df_docs, qdrant_client, qdrant_config, collection_name=collection
        )
    # Return retriever
    return QdrantRetriever(client=qdrant_client, qdrant_config=qdrant_config)


def main() -> None:
    # Load input configuration.
    input_config = loader.load_json(config.input_path / "input_parameters.json")

    # Setup clients.
    openrouter_client, qdrant_client = setup_clients(input_config)

    # Setup the router.
    router = setup_router(openrouter_client, input_config)

    # Process user query.
    query = loader.load_txt(config.input_path / "query.txt")
    classification = router.route_query(query)
    logger.info("Queried classified.", classification=classification)

    if classification == "ANSWER":
        df_docs = pd.read_csv(config.data_path / "docs.csv", delimiter=",")
        logger.info("Loaded CSV Data.", num_rows=len(df_docs))

        # Retrieve docs
        retriever = setup_retriever(
            qdrant_client, input_config, df_docs, collection="docs_collection"
        )
        retrieved_docs = retriever.semantic_search(query, top_k=5)

        # Prepare answer
        responder = setup_responder(openrouter_client, input_config)
        answer = responder.generate_response(query, retrieved_docs)
        logger.info("Answer retrieved.", answer=answer)
    elif classification == "CLARIFY":
        logger.info("Your query needs clarification. Please provide more details.")
    elif classification == "REJECT":
        logger.info("Your query has been rejected as it is out of scope.")
    else:
        logger.info("Unexpected classification.", classification=classification)


if __name__ == "__main__":
    main()
