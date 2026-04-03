# config.py — central config for the entire RAG project
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DIR      = str(BASE_DIR / "chroma_db")
BM25_PICKLE     = str(BASE_DIR / "bm25_index.pkl")
PDF_DIR         = str(BASE_DIR / "papers")  # drop your PDF files here

# ── Models ─────────────────────────────────────────────────────────────────
EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL       = "llama3.2"

# ── Chroma ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "pdf_docs"

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80

# ── Retrieval ───────────────────────────────────────────────────────────────
VECTOR_K        = 10   # candidates from ChromaDB
BM25_K          = 10   # candidates from BM25
RERANK_TOP_K    = 5    # chunks kept after cross-encoder reranking
