import structlog
from qdrant_client import QdrantClient

from flare_ai_rag.config import config
from flare_ai_rag.retriever.config import QdrantConfig
from flare_ai_rag.retriever.qdrant_retriever import QdrantRetriever
from flare_ai_rag.utils import loader

logger = structlog.get_logger(__name__)


def main() -> None:
    # Load Qdrant config
    config_json = loader.load_json(config.input_path / "input_parameters.json")
    qdrant_config = QdrantConfig.load(config_json["qdrant_config"])

    # Initialize Qdrant client
    client = QdrantClient(host=qdrant_config.host, port=qdrant_config.port)

    # Initialize the retriever.
    retriever = QdrantRetriever(client=client, qdrant_config=qdrant_config)

    # Define a sample query.
    query = "What is Flare?"

    # Perform semantic search.
    results = retriever.semantic_search(query, top_k=5)

    # Print out the search results.
    for result in results:
        logger.info("Search Results:", result=result)


if __name__ == "__main__":
    main()
