# ingest.py
# Run this ONCE to build your knowledge base.
# Usage: python ingest.py
#
# What it does:
#   1. Fetches LangChain doc pages from the web
#   2. Chunks them with RecursiveCharacterTextSplitter
#   3. Embeds chunks → stores in ChromaDB (vector store)
#   4. Builds a BM25 index → saves as a pickle file
#
# After this runs you'll see:
#   ./chroma_db/    ← vector store (persistent)
#   ./bm25_index.pkl ← keyword index

import pickle
import shutil
import sys
import time
import bs4
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from config import (
    CHROMA_DIR, BM25_PICKLE, EMBED_MODEL,
    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
    BM25_K, LANGCHAIN_DOC_URLS,
)


# ── Helper ──────────────────────────────────────────────────────────────────

def print_step(n: int, title: str):
    print(f"\n{'='*55}")
    print(f"  Step {n}: {title}")
    print(f"{'='*55}")


# ── Main ingestion pipeline ──────────────────────────────────────────────────

def run_ingestion(force: bool = False):
    """
    Full ingestion pipeline.
    Set force=True to wipe and rebuild an existing knowledge base.
    """

    # ── Guard: don't re-run unless forced ───────────────────────────────────
    chroma_path = Path(CHROMA_DIR)
    bm25_path   = Path(BM25_PICKLE)

    if chroma_path.exists() and bm25_path.exists() and not force:
        print("\n  Knowledge base already exists.")
        print("   Run with --force to rebuild it from scratch.\n")
        return

    if force and chroma_path.exists():
        print("\n🗑  Removing existing ChromaDB...")
        shutil.rmtree(chroma_path)

    # ── Step 1: Load pages ───────────────────────────────────────────────────
    print_step(1, "Fetching LangChain docs from the web")

    loader = WebBaseLoader(
        web_paths=LANGCHAIN_DOC_URLS,
        bs_kwargs={
            # Pull only the main article/content sections — skip nav bars
            "parse_only": bs4.SoupStrainer(
                ["article", "main", "div"],
                attrs={"class": lambda c: c and any(
                    kw in c for kw in
                    ["markdown", "docMainContainer", "theme-doc-markdown", "content"]
                )},
            )
        },
        requests_per_second=1,   # be polite to the server
    )

    t0 = time.time()
    raw_docs = loader.load()
    elapsed = time.time() - t0

    # Clean up empty docs (some pages may have no parseable content)
    raw_docs = [d for d in raw_docs if d.page_content.strip()]

    print(f"  Loaded {len(raw_docs)} pages in {elapsed:.1f}s")
    for doc in raw_docs:
        src   = doc.metadata.get("source", "?")
        chars = len(doc.page_content)
        print(f"  • {src}  ({chars:,} chars)")

    # ── Step 2: Chunk ────────────────────────────────────────────────────────
    print_step(2, f"Chunking  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on natural boundaries first
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)

    print(f"  Created {len(chunks)} chunks from {len(raw_docs)} pages")
    avg = sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
    print(f"  Average chunk length: {avg:.0f} chars")

    # ── Step 3: Embed → ChromaDB ─────────────────────────────────────────────
    print_step(3, f"Embedding with {EMBED_MODEL}")
    print("  (this takes a minute the first time — model downloads ~400 MB)")

    t0 = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    # Embed in one shot — Chroma handles batching internally
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    elapsed = time.time() - t0
    count = vectorstore._collection.count()
    print(f"  Stored {count:,} vectors in ChromaDB  ({elapsed:.1f}s)")

    # ── Step 4: Build BM25 index ─────────────────────────────────────────────
    print_step(4, "Building BM25 keyword index")

    bm25_retriever      = BM25Retriever.from_documents(chunks)
    bm25_retriever.k    = BM25_K

    with open(BM25_PICKLE, "wb") as f:
        pickle.dump(bm25_retriever, f)
    size_kb = Path(BM25_PICKLE).stat().st_size // 1024
    print(f"  BM25 index saved → {BM25_PICKLE}  ({size_kb} KB)")

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("    Ingestion complete!  Ready to query.")
    print(f"      ChromaDB : {CHROMA_DIR}")
    print(f"      BM25     : {BM25_PICKLE}")
    print(f"{'='*55}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--force" in sys.argv
    run_ingestion(force=force)
