import pandas as pd
import structlog
from qdrant_client import QdrantClient

from flare_ai_rag.config import config
from flare_ai_rag.retriever.config import QdrantConfig
from flare_ai_rag.retriever.qdrant_collection import generate_collection
from flare_ai_rag.utils import loader

logger = structlog.get_logger(__name__)


def main() -> None:
    # Load Qdrant config
    config_json = loader.load_json(config.input_path / "input_parameters.json")
    qdrant_config = QdrantConfig.load(config_json["qdrant_config"])

    # Load the CSV file.
    df_docs = pd.read_csv(config.data_path / "docs.csv", delimiter=",")
    logger.info("Loaded CSV Data.", num_rows=len(df_docs))

    # Initialize Qdrant client.
    client = QdrantClient(host=qdrant_config.host, port=qdrant_config.port)

    generate_collection(
        df_docs, client, qdrant_config, collection_name="docs_collection"
    )


if __name__ == "__main__":
    main()
