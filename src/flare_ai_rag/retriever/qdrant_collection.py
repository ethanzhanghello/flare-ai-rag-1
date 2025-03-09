import google.api_core.exceptions
import pandas as pd
import structlog
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from flare_ai_rag.ai import EmbeddingTaskType, GeminiEmbedding
from flare_ai_rag.retriever.config import RetrieverConfig

# ✅ Ensure Structlog is Configured
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer(),  # Outputs structured logs
    ]
)
logger = structlog.get_logger(__name__)

PROCESSED_DIR = "processed_data/"  # Folder where preprocessed & external data is stored

def _create_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """
    Creates a Qdrant collection with the given parameters.
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def generate_collection(
    df_docs: pd.DataFrame,
    qdrant_client: QdrantClient,
    retriever_config: RetrieverConfig,
    embedding_client: GeminiEmbedding,
) -> None:
    """
    Routine for generating a Qdrant collection with support for Flare FTSO data.
    """
    _create_collection(qdrant_client, retriever_config.collection_name, retriever_config.vector_size)

    # Load preprocessed metadata
    preprocessed_metadata_path = os.path.join(PROCESSED_DIR, "metadata.json")
    if os.path.exists(preprocessed_metadata_path):
        with open(preprocessed_metadata_path, "r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
        metadata_dict = {doc["original"]: doc for doc in metadata}
    else:
        logger.warning("No preprocessed metadata found. Using raw CSV data.")
        metadata_dict = {}

    points = []
    
    # ✅ Process standard documents
    for idx, (_, row) in enumerate(df_docs.iterrows(), start=1):
        file_name = row["file_name"]
        content = row["content"]

        # Use preprocessed text if available
        if file_name in metadata_dict:
            preprocessed_doc = metadata_dict[file_name]
            chunk_path = os.path.join(PROCESSED_DIR, preprocessed_doc["filename"])
            try:
                with open(chunk_path, "r", encoding="utf-8") as file:
                    content = file.read()
            except Exception as e:
                logger.warning(f"Could not read preprocessed chunk: {chunk_path}. Error: {e}")
                continue

        if not isinstance(content, str) or len(content) < 10:
            logger.warning("Skipping document due to missing or invalid content.", filename=file_name)
            continue

        try:
            embedding = embedding_client.embed_content(
                embedding_model=retriever_config.embedding_model,
                task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
                contents=content,
                title=file_name,
            )
        except google.api_core.exceptions.InvalidArgument as e:
            if "400 Request payload size exceeds the limit" in str(e):
                logger.warning("Skipping document due to size limit.", filename=file_name)
                continue
            logger.exception("Error encoding document (InvalidArgument).", filename=file_name)
            continue
        except Exception:
            logger.exception("Error encoding document (general).", filename=file_name)
            continue

        payload = {
            "filename": file_name,
            "metadata": row["meta_data"],
            "text": content,
        }

        # Add metadata
        if file_name in metadata_dict:
            payload.update(metadata_dict[file_name])

        points.append(PointStruct(id=idx, vector=embedding, payload=payload))

    # ✅ Process external Flare FTSO data
    flare_data_path = os.path.join(PROCESSED_DIR, "flare_data.json")
    if os.path.exists(flare_data_path):
        with open(flare_data_path, "r", encoding="utf-8") as meta_file:
            flare_content = json.load(meta_file)

        for idx, entry in enumerate(flare_content, start=len(points) + 1):
            text_content = f"Flare FTSO Data: {entry}"  # Convert JSON to text format

            try:
                embedding = embedding_client.embed_content(
                    embedding_model=retriever_config.embedding_model,
                    task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
                    contents=text_content,
                    title="Flare FTSO Update"
                )
            except Exception as e:
                logger.warning(f"Failed to embed Flare data: {e}")
                continue  # Skip this entry if embedding fails

            points.append(PointStruct(id=idx, vector=embedding, payload={"dataset": "flare_data", "text": text_content}))

    if points:
        qdrant_client.upsert(collection_name=retriever_config.collection_name, points=points)
        logger.info(f"✅ Stored {len(points)} documents in Qdrant.")
    else:
        logger.warning("No valid documents found to insert.")
