import pandas as pd
import structlog
from qdrant_client import QdrantClient

from flare_ai_rag.ai import GeminiEmbedding, GeminiProvider
from flare_ai_rag.responder import GeminiResponder, ResponderConfig
from flare_ai_rag.retriever import QdrantConfig, QdrantRetriever, generate_collection
from flare_ai_rag.router import GeminiRouter, RouterConfig
from flare_ai_rag.settings import settings
from flare_ai_rag.utils import load_json, load_txt, save_json

logger = structlog.get_logger(__name__)


def setup_qdrant(input_config: dict) -> QdrantClient:
    """Initialize Qdrant client."""
    logger.info("Setting up Qdrant client...")
    qdrant_config = QdrantConfig.load(input_config["qdrant_config"])
    qdrant_client = QdrantClient(host=qdrant_config.host, port=qdrant_config.port)
    logger.info("Qdrant client has been set up.")

    return qdrant_client


def setup_router(input_config: dict) -> GeminiRouter:
    """Initialize the Gemini Provider and the Gemini Router."""
    # Setup router config
    router_model_config = input_config["router_model"]
    router_config = RouterConfig.load(router_model_config)

    # Setup Gemini client based on Router config
    gemini_provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        system_instruction=router_config.system_prompt,
    )

    return GeminiRouter(client=gemini_provider, config=router_config)


def setup_retriever(
    qdrant_client: QdrantClient,
    input_config: dict,
    df_docs: pd.DataFrame,
    collection_name: str | None = None,
) -> QdrantRetriever:
    """Initialize the Qdrant retriever."""
    # Set up Qdrant config
    qdrant_config = QdrantConfig.load(input_config["qdrant_config"])

    # Set up Gemini Embedding client
    embedding_client = GeminiEmbedding(settings.gemini_api_key)
    # (Re)generate qdrant collection
    if collection_name:
        generate_collection(
            df_docs,
            qdrant_client,
            qdrant_config,
            collection_name=collection_name,
            embedding_client=embedding_client,
        )
        logger.info(
            "The Qdrant collection has been generated.", collection_name=collection_name
        )
    # Return retriever
    return QdrantRetriever(
        client=qdrant_client,
        qdrant_config=qdrant_config,
        embedding_client=embedding_client,
    )


def setup_responder(input_config: dict) -> GeminiResponder:
    """Initialize the responder."""
    # Set up Responder Config.
    responder_config = input_config["responder_model"]
    responder_config = ResponderConfig.load(responder_config)

    # Set up a new Gemini Provider based on Responder Config.
    gemini_provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        system_instruction=responder_config.system_prompt,
    )
    return GeminiResponder(client=gemini_provider, responder_config=responder_config)


def main() -> None:
    # Load input configuration.
    input_config = load_json(settings.input_path / "input_parameters.json")

    # Set up the Gemini Router
    router = setup_router(input_config)

    # Load data
    df_docs = pd.read_csv(settings.data_path / "docs.csv", delimiter=",")
    logger.info("Loaded CSV Data.", num_rows=len(df_docs))

    # Set up qdrant client.
    qdrant_client = setup_qdrant(input_config)

    # Set up retriever. (Use Gemini Embedding.)
    retriever = setup_retriever(
        qdrant_client, input_config, df_docs, collection_name="docs_collection"
    )

    # Set up responder. (Use Gemini Provider.)
    responder = setup_responder(input_config)

    # Process user query.
    query = load_txt(settings.input_path / "query.txt")
    classification = router.route_query(query)
    logger.info(
        "Queried has been classified by the Router.", classification=classification
    )

    if classification == "ANSWER":
        retrieved_docs = retriever.semantic_search(query, top_k=5)
        logger.info("Docs have been retrieved.")

        # Prepare answer
        answer = responder.generate_response(query, retrieved_docs)
        logger.info("Response has been generated.", answer=answer)

        # Save answer
        output_file = settings.data_path / "rag_answer.json"
        save_json(
            {
                "query": query,
                "answer": answer,
            },
            output_file,
        )

    elif classification == "CLARIFY":
        logger.info("Your query needs clarification. Please provide more details.")
    elif classification == "REJECT":
        logger.info("Your query has been rejected as it is out of scope.")
    else:
        logger.info("Unexpected classification.", classification=classification)


if __name__ == "__main__":
    main()
