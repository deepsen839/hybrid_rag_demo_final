import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-db")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "large_pdf_hybrid_rag",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# BGE-small uses a 512-token model context.
# 450 + 75 overlap gives a safe deterministic chunk.
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "75"))

VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "15"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "15"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

BM25_PATH = os.getenv(
    "BM25_PATH",
    "/workspace/data/bm25.pkl",
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)
