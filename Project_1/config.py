# config.py — central config for the entire RAG project
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DIR      = str(BASE_DIR / "chroma_db")
BM25_PICKLE     = str(BASE_DIR / "bm25_index.pkl")

# ── Models ─────────────────────────────────────────────────────────────────
EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL       = "llama3.2"

# ── Chroma ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "langchain_docs"

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80

# ── Retrieval ───────────────────────────────────────────────────────────────
VECTOR_K        = 10   # candidates from ChromaDB
BM25_K          = 10   # candidates from BM25
RERANK_TOP_K    = 5    # chunks kept after cross-encoder reranking

# ── LangChain doc URLs to index ─────────────────────────────────────────────
# These pages are fetched at ingestion time — add/remove as you like
LANGCHAIN_DOC_URLS = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/rag/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/lcel/",
    "https://python.langchain.com/docs/concepts/chat_models/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/embedding_models/",
    "https://python.langchain.com/docs/concepts/text_splitters/",
    "https://python.langchain.com/docs/concepts/document_loaders/",
    "https://python.langchain.com/docs/how_to/qa_sources/",
    "https://python.langchain.com/docs/how_to/streaming/",
    "https://python.langchain.com/docs/how_to/hybrid/",
]
