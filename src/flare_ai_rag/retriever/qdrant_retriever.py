import structlog  # Ensure logger is available
from typing import override
from qdrant_client import QdrantClient
from flare_ai_rag.ai import EmbeddingTaskType, GeminiEmbedding
from flare_ai_rag.retriever.base import BaseRetriever
from flare_ai_rag.retriever.config import RetrieverConfig

logger = structlog.get_logger(__name__)  # Define logger

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
    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Perform semantic search using **preprocessed document chunks** and external datasets.
        """
        query_vector = self.embedding_client.embed_content(
            embedding_model="models/text-embedding-004",
            contents=query,
            task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
        )

        results = self.client.search(
            collection_name=self.retriever_config.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        output = {"standard_docs": [], "extra_data": []}

        for hit in results:
            if hit.payload:
                dataset = hit.payload.get("dataset", "RAG")
                text = hit.payload.get("text", "")
                metadata = hit.payload.get("metadata", {})

                # Structure the data properly
                doc_entry = {
                    "text": text,
                    "score": hit.score,
                    "source": metadata.get("original", dataset),
                }

                if dataset in ["github_data.json", "google_trends.json", "flare_data.json"]:
                    output["extra_data"].append(doc_entry)
                else:
                    output["standard_docs"].append(doc_entry)
            else:
                logger.warning(f"⚠️ Missing payload for search result: {hit}")
    
        return output
