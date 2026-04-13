# config.py — central config for the entire RAG project
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DIR      = str(BASE_DIR / "chroma_db")
BM25_PICKLE     = str(BASE_DIR / "bm25_index.pkl")
PDF_DIR         = str(BASE_DIR / "papers")  # drop your PDF files here

# ── Models ─────────────────────────────────────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL        = "llama3.2"
AVAILABLE_MODELS = ["llama3.2", "mistral", "gemma3:1b"]  # single source of truth

# ── Ollama host (overridden by OLLAMA_HOST env var in Docker) ───────────────
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── Chroma ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "pdf_docs"

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80

# ── Retrieval ───────────────────────────────────────────────────────────────
VECTOR_K        = 10   # candidates from ChromaDB
BM25_K          = 10   # candidates from BM25
RERANK_TOP_K    = 5    # chunks kept after cross-encoder reranking

# ── AGENTIC AI config ────────────────────────────────────────────────────────
# These values control the agent loop defined in agent.py.
# AGENT_MAX_ITERATIONS caps how many tool calls the agent can make in one run
# before LangGraph forces it to stop (prevents infinite loops).
AGENT_MAX_ITERATIONS = 6
