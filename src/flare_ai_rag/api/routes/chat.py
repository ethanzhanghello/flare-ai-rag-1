import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from flare_ai_rag.responder import GeminiResponder
from flare_ai_rag.retriever import QdrantRetriever
from flare_ai_rag.router import GeminiRouter

logger = structlog.get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """
    Pydantic model for chat message validation.

    Attributes:
        message (str): The chat message content, must not be empty
    """

    message: str = Field(..., min_length=1)


class ChatRouter:
    """
    A simple chat router that processes incoming messages using the RAG pipeline.

    It wraps the existing query classification, document retrieval, and response
    generation components to handle a conversation in a single endpoint.
    """

    def __init__(
        self,
        router: APIRouter,
        query_router: GeminiRouter,
        retriever: QdrantRetriever,
        responder: GeminiResponder,
    ) -> None:
        """
        Initialize the ChatRouter.

        Args:
            router (APIRouter): FastAPI router to attach endpoints.
            query_router: Component that classifies the query.
            retriever: Component that retrieves relevant documents.
            responder: Component that generates a response.
        """
        self._router = router
        self.query_router = query_router
        self.retriever = retriever
        self.responder = responder
        self.logger = logger.bind(router="chat")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """
        Set up FastAPI routes for the chat endpoint.
        """

        @self._router.post("/")
        async def chat(message: ChatMessage) -> dict[str, str] | None:  # pyright: ignore [reportUnusedFunction]
            """
            Process a chat message through the RAG pipeline.
            Returns a response containing the query classification and the answer.
            """
            try:
                self.logger.debug("Received chat message", message=message.message)
                # Classify the query.
                classification = self.query_router.route_query(message.message)
                self.logger.info("Query classified", classification=classification)

                if classification == "ANSWER":
                    # Retrieve relevant documents.
                    retrieved_docs = self.retriever.semantic_search(
                        message.message, top_k=5
                    )
                    self.logger.info("Documents retrieved")

                    # Generate the final answer using retrieved context.
                    answer = self.responder.generate_response(
                        message.message, retrieved_docs
                    )
                    self.logger.info("Response generated", answer=answer)
                    return {"classification": classification, "response": answer}

                # Map static responses for CLARIFY and REJECT.
                static_responses = {
                    "CLARIFY": "Please provide additional context.",
                    "REJECT": "The query is out of scope.",
                }

                if classification in static_responses:
                    return {
                        "classification": classification,
                        "response": static_responses[classification],
                    }

            except Exception as e:
                self.logger.exception("Chat processing failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    @property
    def router(self) -> APIRouter:
        """Return the underlying FastAPI router with registered endpoints."""
        return self._router
