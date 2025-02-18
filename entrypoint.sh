# Start Qdrant
qdrant --storage-type in_memory &

# Wait for Qdrant to initialize
sleep 3

# Start RAG
uv run start-rag