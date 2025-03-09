import structlog  # Ensure logger is available
from typing import override, Any
from qdrant_client import QdrantClient
from flare_ai_rag.ai import EmbeddingTaskType, GeminiEmbedding
from flare_ai_rag.retriever.base import BaseRetriever
from flare_ai_rag.retriever.config import RetrieverConfig
import os
import json

logger = structlog.get_logger(__name__)  # Define logger

PROCESSED_DIR = "processed_data/"  # Folder where preprocessed & external data is stored

class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        client: QdrantClient,
        retriever_config: RetrieverConfig,
        embedding_client: GeminiEmbedding,
    ) -> None:
        """Initialize the QdrantRetriever."""
        self.client = client
        self.retriever_config = retriever_config
        self.embedding_client = embedding_client

    @override
    def semantic_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Perform semantic search using preprocessed document chunks and Flare data.
        Returns a **single list of documents** instead of a dictionary.
        """
        query_vector = self.embedding_client.embed_content(
            embedding_model=self.retriever_config.embedding_model,
            contents=query,
            task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
        )

        results = self.client.search(
            collection_name=self.retriever_config.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        retrieved_docs = []  # ✅ A single list instead of a dictionary

        for hit in results:
            if hit.payload:
                dataset = hit.payload.get("dataset", "RAG")
                text = hit.payload.get("text", "")
                metadata = hit.payload.get("metadata", {})

                doc_entry = {
                    "text": text,
                    "score": hit.score,
                    "source": metadata.get("original", dataset),
                }
                retrieved_docs.append(doc_entry)  # ✅ Append everything to a single list
            else:
                logger.warning(f"⚠️ Missing payload for search result: {hit}")

        return retrieved_docs  # ✅ Now it returns List[Dict[str, Any]]

def search_relevant_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Wrapper function to call `semantic_search` from `QdrantRetriever`.
    Ensures correct parameter types before retrieving documents.
    """
    try:
        # ✅ Initialize Qdrant Client
        qdrant_client = QdrantClient(host="localhost", port=6333)

        # ✅ Correct RetrieverConfig Initialization
        retriever_config = RetrieverConfig(
            embedding_model="models/text-embedding-004",
            collection_name="documents",
            vector_size=768,
            host="localhost",
            port=6333
        )

        # ✅ Initialize Gemini Embedding
        embedding_client = GeminiEmbedding(api_key="YOUR_GEMINI_API_KEY")

        retriever = QdrantRetriever(
            client=qdrant_client,
            retriever_config=retriever_config,
            embedding_client=embedding_client,
        )

        retrieved_data = retriever.semantic_search(query=query, top_k=top_k)
        

        return retrieved_data
    
    except Exception as e:
        logger.error(f"Error in search_relevant_documents: {e}")
        return []  # ✅ Return an empty list instead of an incorrect type
