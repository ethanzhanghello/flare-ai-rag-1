from google import genai


class GeminiClient:
    def __init__(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)

    def embed_content(self, model: str, contents: str) -> list[float]:
        result = self.client.models.embed_content(model=model, contents=contents)
        embedding = result.embeddings

        if not embedding or embedding[0].values is None:
            msg = "No embedding was returned from the API."
            raise ValueError(msg)
        return embedding[0].values
