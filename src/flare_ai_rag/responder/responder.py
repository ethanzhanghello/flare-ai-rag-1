from typing import Any, override

from flare_ai_rag.ai import GeminiProvider, OpenRouterClient
from flare_ai_rag.responder import BaseResponder, ResponderConfig
from flare_ai_rag.utils import parse_chat_response
from flare_ai_rag.retriever.qdrant_retriever import search_relevant_documents  # Import retrieval function


class GeminiResponder(BaseResponder):
    def __init__(
        self, client: GeminiProvider, responder_config: ResponderConfig
    ) -> None:
        """
        Initialize the responder with a GeminiProvider.

        :param client: An instance of GeminiProvider.
        :param responder_config: Configuration settings for AI responses.
        """
        self.client = client
        self.responder_config = responder_config

    @override
    def generate_response(self, query: str, retrieved_documents: list[dict]) -> str:
        """
        Generate a final answer using the query, retrieved context, and real-world data.
        Dynamically adjusts retrieval size, includes citations, and handles unclear responses.

        :param query: The input query.
        :param retrieved_documents: A list of dictionaries containing retrieved docs.
        :return: The generated answer as a string.
        """
        # Retrieve additional context from BigQuery & Flare
        external_data = search_relevant_documents(query, top_k=5)["extra_data"]

        # Dynamically adjust the number of retrieved documents
        num_docs = min(len(retrieved_documents), 8) if len(query) > 100 else min(len(retrieved_documents), 5)

        context = "📚 List of retrieved documents:\n"
        citations = []

        for idx, doc in enumerate(retrieved_documents[:num_docs], start=1):
            title = doc.get("title", f"Document {idx}")
            author = doc.get("author", "Unknown Author")
            date = doc.get("date", "Unknown Date")
            text_snippet = doc.get("text", "")[:200]  # Limit snippet length for readability

            context += f"📌 {title} (by {author}, {date}):\n{text_snippet}...\n\n"
            citations.append(f"[{idx}] {title}")

        # Add external data (GitHub repos, Google Trends, Flare blockchain)
        if external_data:
            context += "\n🌍 Additional Data from BigQuery & Flare:\n"
            for idx, entry in enumerate(external_data, start=len(citations) + 1):
                context += f"🔹 {entry['source']}: {entry['text'][:200]}...\n"
                citations.append(f"[{idx}] {entry['source']}")

        # Compose the structured prompt for Gemini
        prompt = (
            f"{context}User query: {query}\n"
            f"{self.responder_config.query_prompt}"
        )

        # Generate response using Gemini
        response = self.client.generate(
            prompt,
            response_mime_type=None,
            response_schema=None,
        )

        # Detect unclear responses and refine using external data
        unclear_responses = ["I'm not sure", "I don't know", "Sorry", "I cannot find"]
        if any(phrase.lower() in response.text.lower() for phrase in unclear_responses) or len(response.text) < 30:
            print("🔄 Low-confidence response detected. Refining answer with more retrieval...")
            refined_query = f"Provide more details about: {query}"
            return self.generate_response(refined_query, retrieved_documents)

        # Append citations to response
        return response.text + "\n\n📚 Sources: " + ", ".join(citations)


class OpenRouterResponder(BaseResponder):
    def __init__(
        self, client: OpenRouterClient, responder_config: ResponderConfig
    ) -> None:
        """
        Initialize the responder with an OpenRouter client and the model to use.

        :param client: An instance of OpenRouterClient.
        :param responder_config: Configuration settings for AI responses.
        """
        self.client = client
        self.responder_config = responder_config

    @override
    def generate_response(self, query: str, retrieved_documents: list[dict]) -> str:
        """
        Generate a final answer using the query, retrieved documents, and additional knowledge.
        Dynamically adjusts retrieval and citation inclusion.

        :param query: The input query.
        :param retrieved_documents: A list of dictionaries containing retrieved docs.
        :return: The generated answer as a string.
        """
        # Retrieve external data (BigQuery & Flare)
        external_data = search_relevant_documents(query, top_k=5)["extra_data"]

        context = "📚 List of retrieved preprocessed documents:\n"
        citations = []

        for idx, doc in enumerate(retrieved_documents, start=1):
            title = doc.get("title", f"Document {idx}")
            author = doc.get("author", "Unknown Author")
            date = doc.get("date", "Unknown Date")
            text_snippet = doc.get("text", "")[:200]

            context += f"📌 {title} (by {author}, {date}):\n{text_snippet}...\n\n"
            citations.append(f"[{idx}] {title}")

        # Add external knowledge from real-world datasets
        if external_data:
            context += "\n🌍 Additional Data from BigQuery & Flare:\n"
            for idx, entry in enumerate(external_data, start=len(citations) + 1):
                context += f"🔹 {entry['source']}: {entry['text'][:200]}...\n"
                citations.append(f"[{idx}] {entry['source']}")

        # Compose the structured prompt for OpenRouter
        prompt = (
            f"{context}User query: {query}\n"
            f"{self.responder_config.query_prompt}"
        )

        # Prepare the payload for OpenRouter API.
        payload: dict[str, Any] = {
            "model": self.responder_config.model.model_id,
            "messages": [
                {"role": "system", "content": self.responder_config.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        if self.responder_config.model.max_tokens is not None:
            payload["max_tokens"] = self.responder_config.model.max_tokens
        if self.responder_config.model.temperature is not None:
            payload["temperature"] = self.responder_config.model.temperature

        # Send the prompt to the OpenRouter API.
        response = self.client.send_chat_completion(payload)

        return parse_chat_response(response) + "\n\n📚 Sources: " + ", ".join(citations)
