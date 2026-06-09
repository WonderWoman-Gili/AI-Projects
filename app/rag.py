from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import requests
import uuid

# -------------------------
# CONFIG (DOCKER SAFE)
# -------------------------
client = QdrantClient(host="qdrant", port=6333)
COLLECTION = "docs"

EMBED_URL = "http://ollama:11434/api/embeddings"
LLM_URL = "http://ollama:11434/api/generate"

VECTOR_SIZE = 768  # IMPORTANT: matches nomic-embed-text


# -------------------------
# INIT COLLECTION
# -------------------------
def init_collection():
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            "size": VECTOR_SIZE,
            "distance": "Cosine"
        }
    )


# -------------------------
# EMBEDDING
# -------------------------
def embed(text: str):
    try:
        response = requests.post(
            EMBED_URL,
            json={
                "model": "nomic-embed-text",
                "prompt": text
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]

    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")


# -------------------------
# INDEXING (ADD DATA)
# -------------------------
def add_document(text: str):
    vector = embed(text)

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text}
            )
        ]
    )


# -------------------------
# RETRIEVAL (SEARCH)
# -------------------------
def search(query: str):
    query_vector = embed(query)

    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=3,
        with_payload=True
    )

    return [point.payload["text"] for point in results]