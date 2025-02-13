
import pandas as pd
import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

from flare_ai_rag.config import config
from flare_ai_rag.retriever.config import QdrantConfig
from flare_ai_rag.retriever.qdrant_collection import create_collection
from flare_ai_rag.utils import loader

logger = structlog.get_logger(__name__)


def generate_collection(
    df_docs: pd.DataFrame,
    client: QdrantClient,
    qdrant_config: QdrantConfig,
    collection_name: str,
):
    """Routine for generating a Qdrant collection for a specific CSV file type."""
    # Create the collection.
    create_collection(client, collection_name, qdrant_config.vector_size)
    logger.info("Created the collection.", collection_name=collection_name)

    # Load the embedding model.
    embedding_model = SentenceTransformer(qdrant_config.embedding_model)

    # For each document in the CSV, compute its embedding and prepare a Qdrant point.
    points = []
    for i, row in df_docs.iterrows():
        doc_id = i
        content = row["Contents"]

        # Check if content is missing or not a string.
        if pd.isna(content) or not isinstance(content, str):
            logger.warning(
                "Skipping document due to missing or invalid content.",
                filename=row["Filename"],
            )
            continue

        try:
            # Compute the embedding for the document content.
            embedding = embedding_model.encode(content).tolist()
        except Exception as e:
            logger.error(
                "Error encoding document.", filename=row["Filename"], error=str(e)
            )
            continue

        # Prepare the payload.
        payload = {
            "filename": row["Filename"],
            "metadata": row["Metadata"],
            "text": content,
        }

        # Create a Qdrant point.
        point = PointStruct(id=doc_id, vector=embedding, payload=payload)
        points.append(point)

    if points:
        # Upload the points into the Qdrant collection.
        client.upsert(collection_name=collection_name, points=points)
        logger.info(
            "Collection generated and documents inserted into Qdrant successfully.",
            collection_name=collection_name,
            num_points=len(points),
        )
    else:
        logger.warning("No valid documents found to insert.")


def main():
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
