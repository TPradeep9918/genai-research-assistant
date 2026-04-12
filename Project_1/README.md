# Research Paper Q&A — Production RAG System

A fully local, production-grade Retrieval-Augmented Generation (RAG) system that lets you query research papers in natural language. Built with hybrid retrieval, cross-encoder reranking, query rewriting, and a multi-LLM Streamlit UI — no cloud APIs required.

---

## What This Project Does

You drop a PDF research paper into the `papers/` folder. The system indexes it into a dual retrieval store (vector + keyword). When you ask a question — even in casual language — the pipeline rewrites your query into domain-specific terminology, retrieves the most relevant chunks, reranks them, and generates a cited answer using a locally running LLM.

---

## Architecture

```
User Question
     │
     ▼
Query Rewriter (LLM)          ← maps casual phrasing to paper terminology
     │
     ▼
Hybrid Retriever
 ├── BM25 (keyword)   ─── top 10 candidates
 └── ChromaDB (vector) ── top 10 candidates
     │
     ▼
RRF Fusion                    ← merges 20 candidates, removes duplicates
     │
     ▼
Cross-Encoder Reranker        ← scores all candidates against the query
     │
     ▼
Top 5 Chunks (with relevance scores)
     │
     ▼
Citation-Enforced Prompt      ← LLM must cite [1][2] for every claim
     │
     ▼
Local LLM via Ollama          ← llama3.2 / mistral / gemma3:1b
     │
     ▼
Answer + Source Explorer (Streamlit UI)
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Hybrid retrieval** | BM25 keyword search + ChromaDB vector search fused via Reciprocal Rank Fusion |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` rescores all candidates — picks top 5 by relevance |
| **Query rewriting** | LLM rewrites casual questions into paper-specific terminology before retrieval |
| **Citation enforcement** | Prompt forces the LLM to cite every claim with `[chunk_number]` — no hallucination |
| **Multi-LLM support** | Switch between `llama3.2`, `mistral`, `gemma3:1b` from the sidebar — no restart |
| **Fully local** | No OpenAI key, no cloud calls — runs entirely on your machine via Ollama |
| **Evaluation suite** | Measures context relevance, citation rate, and grounded response rate across all 3 models |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | [Ollama](https://ollama.com) — llama3.2, mistral, gemma3:1b |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (768-dim, normalised) |
| Vector store | ChromaDB (persistent on disk) |
| Keyword store | BM25 via `rank-bm25` (pickled index) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Orchestration | LangChain (LCEL pipeline) |
| UI | Streamlit |
| PDF parsing | PyPDF |

---

## Project Structure

```
Project_1/
├── papers/               ← drop your PDF files here
├── chroma_db/            ← auto-generated vector store (persistent)
├── bm25_index.pkl        ← auto-generated keyword index
│
├── config.py             ← all tunable parameters in one place
├── ingest.py             ← PDF → chunks → ChromaDB + BM25
├── retriever.py          ← HybridRerankedRetriever (BM25 + vector + cross-encoder)
├── chain.py              ← citation-enforced prompt + doc formatter
├── app.py                ← Streamlit chat UI with query rewriter
└── evaluate.py           ← model comparison: context relevance, citation%, grounded%
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull LLM models
```bash
ollama pull llama3.2
ollama pull mistral
ollama pull gemma3:1b
```

### 4. Add your PDFs
Drop any research paper PDFs into the `papers/` folder.

### 5. Build the knowledge base
```bash
python ingest.py
```
This loads your PDFs, chunks them (500 tokens, 80 overlap), embeds with `all-mpnet-base-v2`, stores in ChromaDB, and builds a BM25 index. Run with `--force` to rebuild from scratch.

### 6. Launch the app
```bash
python -m streamlit run app.py --server.fileWatcherType none
```
Open http://localhost:8501 in your browser.

### 7. (Optional) Run evaluation
```bash
python evaluate.py
```
Evaluates all 3 models across 7 benchmark questions and prints a comparison table.

---

## Configuration (`config.py`)

```python
EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL       = "llama3.2"

CHUNK_SIZE      = 500    # tokens per chunk
CHUNK_OVERLAP   = 80     # overlap between chunks

VECTOR_K        = 10     # candidates from ChromaDB
BM25_K          = 10     # candidates from BM25
RERANK_TOP_K    = 5      # chunks kept after cross-encoder reranking
```

---

## How Retrieval Works

1. **Ingestion** — PDFs are split into 500-token chunks with 80-token overlap. Each chunk is embedded into a 768-dimensional vector and stored in ChromaDB. A separate BM25 index is built for keyword matching.

2. **Query rewriting** — Before retrieval, the LLM rephrases the user's question using precise academic terminology (e.g. "scalar product" → "scaled dot-product attention mechanism").

3. **Hybrid retrieval** — The rewritten query runs against both BM25 (top 10) and ChromaDB (top 10), giving 20 candidate chunks.

4. **RRF fusion** — Reciprocal Rank Fusion merges the two ranked lists using the formula `score += 1 / (60 + rank)`, deduplicating by content prefix.

5. **Cross-encoder reranking** — All candidates are scored by the cross-encoder against the query. The top 5 by relevance score are kept.

6. **Cited answer generation** — The top 5 chunks are formatted as a numbered context block. The LLM is instructed to cite every claim using `[chunk_number]` notation and to refuse answering if the context is insufficient.

---

## Evaluation Metrics

The `evaluate.py` script runs 7 benchmark questions across all 3 models and measures:

| Metric | What it measures | Target |
|---|---|---|
| **Context Relevance** | Avg cross-encoder score of retrieved chunks (sigmoid-normalised to 0–1) | > 0.70 |
| **Citation Rate** | % of answers containing at least one `[N]` citation | 100% |
| **Grounded Rate** | % of answers that don't fall back to "docs don't cover this" | > 80% |

---

## Technologies 

- **RAG pipeline from scratch** — not a tutorial copy; every component is wired together manually using LangChain LCEL
- **Hybrid search** — most production RAG systems use hybrid retrieval; this implements both BM25 and dense vector search with RRF fusion
- **Reranking** — cross-encoder reranking is a production technique that significantly improves answer quality over naive top-k retrieval
- **Query understanding** — query rewriting with an LLM addresses the vocabulary mismatch problem between user language and document language
- **Hallucination mitigation** — citation-enforced prompting is a real production guardrail, not just a prompt suggestion
- **Multi-LLM architecture** — the system is model-agnostic; swapping models requires zero code changes
- **Evaluation** — quantitative metrics on retrieval quality and answer grounding, not just manual spot-checking
- **Fully local** — demonstrates understanding of privacy-first deployment and offline-capable systems
