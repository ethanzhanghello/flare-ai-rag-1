[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Flare](https://img.shields.io/badge/flare-network-e62058.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNCIgaGVpZ2h0PSIzNCI+PHBhdGggZD0iTTkuNC0uMWEzMjAuMzUgMzIwLjM1IDAgMCAwIDIuOTkuMDJoMi4yOGExMTA2LjAxIDExMDYuMDEgMCAwIDEgOS4yMy4wNGMzLjM3IDAgNi43My4wMiAxMC4xLjA0di44N2wuMDEuNDljLS4wNSAyLTEuNDMgMy45LTIuOCA1LjI1YTkuNDMgOS40MyAwIDAgMS02IDIuMDdIMjAuOTJsLTIuMjItLjAxYTQxNjEuNTcgNDE2MS41NyAwIDAgMS04LjkyIDBMMCA4LjY0YTIzNy4zIDIzNy4zIDAgMCAxLS4wMS0xLjUxQy4wMyA1LjI2IDEuMTkgMy41NiAyLjQgMi4yIDQuNDcuMzcgNi43LS4xMiA5LjQxLS4wOXoiIGZpbGw9IiNFNTIwNTgiLz48cGF0aCBkPSJNNy42NSAxMi42NUg5LjJhNzU5LjQ4IDc1OS40OCAwIDAgMSA2LjM3LjAxaDMuMzdsNi42MS4wMWE4LjU0IDguNTQgMCAwIDEtMi40MSA2LjI0Yy0yLjY5IDIuNDktNS42NCAyLjUzLTkuMSAyLjVhNzA3LjQyIDcwNy40MiAwIDAgMC00LjQtLjAzbC0zLjI2LS4wMmMtMi4xMyAwLTQuMjUtLjAyLTYuMzgtLjAzdi0uOTdsLS4wMS0uNTVjLjA1LTIuMSAxLjQyLTMuNzcgMi44Ni01LjE2YTcuNTYgNy41NiAwIDAgMSA0LjgtMnoiIGZpbGw9IiNFNjIwNTciLz48cGF0aCBkPSJNNi4zMSAyNS42OGE0Ljk1IDQuOTUgMCAwIDEgMi4yNSAyLjgzYy4yNiAxLjMuMDcgMi41MS0uNiAzLjY1YTQuODQgNC44NCAwIDAgMS0zLjIgMS45MiA0Ljk4IDQuOTggMCAwIDEtMi45NS0uNjhjLS45NC0uODgtMS43Ni0xLjY3LTEuODUtMy0uMDItMS41OS4wNS0yLjUzIDEuMDgtMy43NyAxLjU1LTEuMyAzLjM0LTEuODIgNS4yNy0uOTV6IiBmaWxsPSIjRTUyMDU3Ii8+PC9zdmc+&colorA=FFFFFF)](https://dev.flare.network/)

# Flare AI RAG

Flare AI template for Retrieval-Augmented Generation (RAG).

## 📌 Prerequisites

Before getting started, ensure you have:

- A **Python 3.12** environment.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed for dependency management.
- An [OpenRouter API Key](https://openrouter.ai/settings/keys).

## Environment Setup

### Step 1: Install Dependencies

Install all required dependencies by running:

```bash
uv sync --all-extras
```

### Step 2: Configure Environment Variables

Set your OpenRouter API key in your environment:

```bash
export OPENROUTER_API_KEY=<your-openrouter-api-key>
```

## 🚀 Running the Flare-AI-RAG

- **Setup a Qdrant service**: make sure that Qdrant is up an running before running your script.
You can quickly start a Qdrant instance using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

The RAG consists of a router, a retriever, and a responder, all configurable within `src/input_parameters.json`.

Once configured, add your query to `src/query.txt` and run:

```bash
uv run start-rag
```
