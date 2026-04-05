# Research Paper Q&A — Production RAG System

> Ask questions about any research paper in plain English and get cited, accurate answers — powered entirely by local LLMs. No internet. No API keys. No hallucinations.

**Skills demonstrated:** Retrieval-Augmented Generation (RAG) · Large Language Models (LLMs) · Vector Databases · Hybrid Search · Cross-Encoder Reranking · Query Rewriting · LangChain · Streamlit · Ollama · Python

---

## The Problem This Solves

Reading research papers is slow. Finding a specific answer buried across 15 pages is frustrating. Generic chatbots hallucinate when asked about papers they haven't seen.

This project builds a Q&A system that:
- Reads your actual PDF papers
- Understands what you're asking even if you use informal language
- Returns answers grounded **only** in your documents, with citations for every claim
- Runs entirely on your local machine — no data leaves your computer

---

## How It Works — Plain English

Think of it as a smart research assistant with a 4-step process:

**Step 1 — Understand your question**
Your question goes to an LLM that rewrites it in precise academic language. So *"how does the scalar product work?"* becomes *"How does the scaled dot-product attention mechanism function?"* — much easier to search.

**Step 2 — Find the right passages**
The rewritten question searches your papers two ways simultaneously:
- **Keyword search (BM25)** — finds passages with exact matching words
- **Semantic search (ChromaDB)** — finds passages with similar meaning even if the words differ

Both searches return 10 candidates each (20 total), then a fusion algorithm ranks the combined list.

**Step 3 — Pick the best chunks**
A cross-encoder AI model reads every candidate passage and the question together, scoring how relevant each one actually is. Only the top 5 most relevant passages move forward.

**Step 4 — Generate a cited answer**
The top 5 passages are shown to the LLM with a strict instruction: *answer only from these passages and cite every claim with [1], [2], etc.* If the papers don't cover the question, it says so honestly instead of making things up.

---

## Architecture

```
Your Question
      │
      ▼
 Query Rewriter  ──────── LLM rephrases question into academic terminology
      │
      ▼
 Hybrid Retriever
  ├── BM25 Keyword Search  ──── top 10 by exact word match
  └── ChromaDB Vector Search ── top 10 by semantic similarity
      │
      ▼
 RRF Fusion  ──────────────── merges + deduplicates 20 candidates
      │
      ▼
 Cross-Encoder Reranker  ───── scores each passage against the question
      │
      ▼
 Top 5 Passages  (with relevance scores)
      │
      ▼
 Citation-Enforced LLM  ────── must cite [1][2] for every claim made
      │
      ▼
 Answer + Source Explorer (Streamlit UI)
```

---

## Key Features

| Feature | What it does | Why it matters |
|---|---|---|
| **Query rewriting** | Rephrases your question before search | Fixes vocabulary mismatch between how users ask vs. how papers write |
| **Hybrid retrieval** | BM25 + ChromaDB searched in parallel | Catches answers missed by either method alone |
| **RRF fusion** | Merges both ranked lists into one | No duplicates; best of both search strategies |
| **Cross-encoder reranking** | Re-scores all 20 candidates for true relevance | Eliminates irrelevant chunks that scored high on keyword overlap |
| **Citation enforcement** | Every claim must have a `[N]` source reference | Prevents hallucination; every answer is traceable |
| **Multi-LLM sidebar** | Switch models without restarting the app | Compare llama3.2, mistral, gemma3:1b side by side |
| **Fully local** | No OpenAI, no cloud, no API keys | Private, offline, cost-free |
| **Evaluation suite** | Quantitative metrics across all 3 models | Measures context relevance, citation rate, grounded response rate |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM inference | Ollama (llama3.2 · mistral · gemma3:1b) | Runs LLMs locally |
| Embeddings | `all-mpnet-base-v2` (768-dim) | Converts text to searchable vectors |
| Vector store | ChromaDB (persistent) | Stores and searches embeddings |
| Keyword search | BM25 via rank-bm25 | Exact/fuzzy keyword matching |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder relevance scoring |
| Orchestration | LangChain LCEL | Wires all components into a pipeline |
| UI | Streamlit | Interactive chat interface |
| PDF parsing | PyPDF | Extracts text from research papers |
| Language | Python 3.10+ | — |

---

## Project Structure

```
Project_1/
│
├── papers/               ← PUT YOUR PDF FILES HERE
├── chroma_db/            ← auto-created: stores vector embeddings
├── bm25_index.pkl        ← auto-created: keyword search index
│
├── config.py             ← all settings in one place (chunk size, model names, k values)
├── ingest.py             ← reads PDFs → creates chunks → builds ChromaDB + BM25
├── retriever.py          ← hybrid search + RRF fusion + cross-encoder reranking
├── chain.py              ← citation-enforced prompt template + document formatter
├── app.py                ← Streamlit UI with query rewriter and model switcher
└── evaluate.py           ← benchmarks all 3 models on 7 questions, prints score table
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running (`ollama serve`)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Pull the LLM models (one-time, ~7 GB total)
```bash
ollama pull llama3.2    # 2.0 GB
ollama pull mistral     # 4.4 GB
ollama pull gemma3:1b   # 815 MB
```

### Step 3 — Add your research papers
Drop any PDF files into the `papers/` folder. The project currently includes `NIPS-2017-attention-is-all-you-need-Paper.pdf` as an example.

### Step 4 — Build the knowledge base
```bash
python ingest.py
```
This runs once and takes ~1–2 minutes. It chunks your PDFs into 500-token pieces, embeds them into ChromaDB, and builds the BM25 keyword index. Run `python ingest.py --force` to rebuild after adding new PDFs.

### Step 5 — Start the app
```bash
python -m streamlit run app.py --server.fileWatcherType none
```
Open **http://localhost:8501** in your browser.

### Step 6 — (Optional) Run the evaluation
```bash
python evaluate.py
```
Benchmarks all 3 models across 7 questions about the Transformer paper and prints a comparison table.

---

## Configuration

All tunable parameters live in `config.py` — change them without touching any other file:

```python
EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"  # embedding model
CROSS_ENCODER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # reranker model
LLM_MODEL       = "llama3.2"                                 # default LLM

CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 80    # overlap between adjacent chunks

VECTOR_K        = 10    # how many candidates to fetch from ChromaDB
BM25_K          = 10    # how many candidates to fetch from BM25
RERANK_TOP_K    = 5     # how many chunks to keep after reranking
```

To add more papers: drop PDFs in `papers/` and run `python ingest.py --force`.
To add more models: add the model name to `AVAILABLE_MODELS` in `app.py` after running `ollama pull <model>`.

---

## Evaluation Metrics

`evaluate.py` runs 7 benchmark questions across all 3 models without needing any external APIs:

| Model | Context Relevance | Citation Rate | Grounded Rate | Avg Score |
|---|---|---|---|---|
| llama3.2 | 0.838 | 100% | 100% | **0.946** |
| mistral | 0.838 | 100% | 100% | **0.946** |
| gemma3:1b | 0.838 | 100% | 100% | **0.946** |

**Targets:** context relevance > 0.70 · citation rate = 100% · grounded rate > 80%

| Metric | Definition |
|---|---|
| **Context Relevance** | Average cross-encoder score of retrieved chunks (sigmoid-normalised 0–1) |
| **Citation Rate** | % of answers that include at least one `[N]` citation |
| **Grounded Rate** | % of answers that don't fall back to "docs don't cover this" |

---

## Example Questions to Try

These work well with the included "Attention Is All You Need" paper:

- *"How does scaled dot-product attention work?"*
- *"What is multi-head attention and why is it better than single-head?"*
- *"What BLEU score did the Transformer achieve on WMT 2014 English-to-German?"*
- *"Why did the authors use positional encoding instead of RNNs?"*
- *"How does the encoder-decoder structure work?"*

---

## Skills This Project Demonstrates

This project was built to reflect real-world GenAI engineering practices:

| Skill | Where it appears |
|---|---|
| RAG pipeline design | End-to-end: ingest → retrieve → rerank → generate |
| Hybrid search | BM25 + dense vector retrieval with RRF fusion (`retriever.py`) |
| Cross-encoder reranking | Two-stage retrieval for precision over recall (`retriever.py`) |
| Query understanding | LLM-based query rewriting to bridge vocabulary gaps (`app.py`) |
| Hallucination mitigation | Citation-enforced prompting — no unsourced claims (`chain.py`) |
| LLM orchestration | LangChain LCEL pipelines with composable runnables (`chain.py`, `app.py`) |
| Vector databases | ChromaDB with persistent storage and similarity search (`ingest.py`) |
| Model-agnostic design | 3 LLMs selectable at runtime — zero code changes needed (`app.py`) |
| Quantitative evaluation | Custom metrics without requiring OpenAI (`evaluate.py`) |
| Local-first deployment | Fully offline via Ollama — production-ready privacy model |

---

## Author

Built by **Pradeep** as part of a GenAI research assistant portfolio project.
Demonstrates production RAG techniques applicable to enterprise document Q&A, legal research tools, and internal knowledge base systems.
