import pandas as pd
import structlog
from qdrant_client import QdrantClient

from flare_ai_rag.ai import GeminiEmbedding
from flare_ai_rag.retriever.config import QdrantConfig
from flare_ai_rag.retriever.qdrant_collection import generate_collection
from flare_ai_rag.settings import settings
from flare_ai_rag.utils import load_json

logger = structlog.get_logger(__name__)


def main() -> None:
    # Load Qdrant config
    config_json = load_json(settings.input_path / "input_parameters.json")
    qdrant_config = QdrantConfig.load(config_json["qdrant_config"])

    # Load the CSV file.
    df_docs = pd.read_csv(settings.data_path / "docs.csv", delimiter=",")
    logger.info("Loaded CSV Data.", num_rows=len(df_docs))

    # Initialize Qdrant client.
    client = QdrantClient(host=qdrant_config.host, port=qdrant_config.port)

    # Initialize Gemini client
    embedding_client = GeminiEmbedding(api_key=settings.gemini_api_key)

    generate_collection(
        df_docs,
        client,
        qdrant_config,
        collection_name="docs_collection",
        embedding_client=embedding_client,
    )


if __name__ == "__main__":
    main()
