# ingest.py
# Run this ONCE (or whenever you add new PDFs) to build your knowledge base.
# Usage: python ingest.py
#        python ingest.py --force   ← wipe and rebuild
#
# What it does:
#   1. Loads all PDF files from the ./papers/ folder
#   2. Chunks them with RecursiveCharacterTextSplitter
#   3. Embeds chunks → stores in ChromaDB (vector store)
#   4. Builds a BM25 index → saves as a pickle file
#
# After this runs you'll see:
#   ./chroma_db/     ← vector store (persistent)
#   ./bm25_index.pkl ← keyword index

import pickle
import shutil
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from config import (
    CHROMA_DIR, BM25_PICKLE, EMBED_MODEL, PDF_DIR,
    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, BM25_K,
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

    # ── Guard: check docs folder exists and has PDFs ─────────────────────────
    pdf_path    = Path(PDF_DIR)
    chroma_path = Path(CHROMA_DIR)
    bm25_path   = Path(BM25_PICKLE)

    if not pdf_path.exists() or not list(pdf_path.glob("*.pdf")):
        print(f"\n  No PDFs found in '{PDF_DIR}'.")
        print("  Create the folder and drop your PDF files in it, then re-run.\n")
        return

    if chroma_path.exists() and bm25_path.exists() and not force:
        print("\n  Knowledge base already exists.")
        print("  Run with --force to rebuild it from scratch.\n")
        return

    if force and chroma_path.exists():
        print("\n  Removing existing ChromaDB...")
        shutil.rmtree(chroma_path)

    # ── Step 1: Load PDFs ────────────────────────────────────────────────────
    print_step(1, f"Loading PDFs from {PDF_DIR}")

    loader = PyPDFDirectoryLoader(PDF_DIR)

    t0 = time.time()
    raw_docs = loader.load()
    elapsed = time.time() - t0

    # Remove empty pages
    raw_docs = [d for d in raw_docs if d.page_content.strip()]

    print(f"  Loaded {len(raw_docs)} pages in {elapsed:.1f}s")
    # Show per-file summary
    file_counts = {}
    for doc in raw_docs:
        fname = Path(doc.metadata.get("source", "?")).name
        file_counts[fname] = file_counts.get(fname, 0) + 1
    for fname, pages in file_counts.items():
        print(f"  • {fname}  ({pages} pages)")

    # ── Step 2: Chunk ────────────────────────────────────────────────────────
    print_step(2, f"Chunking  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

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

    bm25_retriever   = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_K

    with open(BM25_PICKLE, "wb") as f:
        pickle.dump(bm25_retriever, f)
    size_kb = Path(BM25_PICKLE).stat().st_size // 1024
    print(f"  BM25 index saved -> {BM25_PICKLE}  ({size_kb} KB)")

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
