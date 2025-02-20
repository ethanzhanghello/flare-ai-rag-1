from typing import override

from qdrant_client import QdrantClient

from flare_ai_rag.ai import GeminiClient
from flare_ai_rag.retriever.base_retriever import BaseRetriever
from flare_ai_rag.retriever.config import QdrantConfig


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        client: QdrantClient,
        qdrant_config: QdrantConfig,
        gemini_client: GeminiClient,
    ) -> None:
        """Initialize the QdrantRetriever."""
        self.client = client
        self.qdrant_config = qdrant_config
        # Instantiate the embedding model using the model name from config.
        self.gemini_client = gemini_client

    @override
    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Perform semantic search by converting the query into a vector
        and searching in Qdrant.

        :param query: The input query.
        :param top_k: Number of top results to return.
        :return: A list of dictionaries, each representing a retrieved document.
        """
        # Convert the query into a vector embedding using the
        # SentenceTransformer instance.
        query_vector = self.gemini_client.embed_content(
            model="text-embedding-004", contents=query
        )

        # Search Qdrant for similar vectors.
        results = self.client.search(
            collection_name=self.qdrant_config.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        # Process and return results.
        output = []
        for hit in results:
            if hit.payload:
                text = hit.payload.get("text", "")
                metadata = {k: v for k, v in hit.payload.items() if k != "text"}
            else:
                text = ""
                metadata = ""
            output.append({"text": text, "score": hit.score, "metadata": metadata})
        return output
