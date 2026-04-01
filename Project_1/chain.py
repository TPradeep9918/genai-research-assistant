# chain.py
# Builds the LCEL RAG chain:
#
#   question (str)
#       │
#       ├──► retriever  →  top-5 chunks (with sources)
#       │
#       └──► CITATION_PROMPT  (context + question)
#               │
#               ▼
#           ChatOllama (llama3.2)
#               │
#               ▼
#           StrOutputParser  →  answer (str)
#
# Exported API:
#   get_chain() → (chain, retriever)
#       chain    : Runnable[str, str]   — invoke with a question string
#       retriever: Runnable[str, list]  — invoke for raw Document objects

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama

from retriever import build_retriever
from config import LLM_MODEL


# ── Prompt ───────────────────────────────────────────────────────────────────

CITATION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant on LangChain documentation.

Answer the question using ONLY the numbered context chunks provided below.
For every claim you make, cite the chunk number in square brackets — e.g. [1], [2].
If multiple chunks support the same claim, cite all of them — e.g. [1][3].
If the context does not contain enough information, say:
  "The provided docs don't cover this — please check the official LangChain docs."

Do NOT add information from outside the provided context.

--- Context ---
{context}
--- End of context ---

Question: {question}

Answer (with inline citations):"""
)


# ── Utilities ────────────────────────────────────────────────────────────────

def format_docs_with_citations(docs: list) -> str:
    """
    Converts a list of Document objects into a numbered context block.

    Example output:
        [1] Source: https://python.langchain.com/docs/concepts/rag/
        LCEL is the recommended way to compose LangChain components...

        ---

        [2] Source: https://python.langchain.com/docs/concepts/retrievers/
        A retriever is an interface that returns documents...
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ── Chain factory ─────────────────────────────────────────────────────────────

_chain     = None
_retriever = None


def get_chain():
    """
    Lazy-load the chain and retriever (models are heavy — load once, reuse).
    Returns (chain, retriever).

    chain    — call chain.invoke("your question") → str answer with citations
    retriever— call retriever.invoke("your question") → list[Document]
    """
    global _chain, _retriever

    if _chain is not None:
        return _chain, _retriever

    print("\n  Building RAG chain...")

    # Retriever: BM25 + ChromaDB + CrossEncoder reranker
    _retriever = build_retriever()

    # LLM: local Ollama, deterministic (temperature=0)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # LCEL chain — the pipe operator (|) wires everything together
    _chain = (
        {
            # Branch 1: retrieve docs → format as numbered context string
            "context": _retriever | RunnableLambda(format_docs_with_citations),
            # Branch 2: pass the question through unchanged
            "question": RunnablePassthrough(),
        }
        | CITATION_PROMPT
        | llm
        | StrOutputParser()
    )

    print("  Chain ready.\n")
    return _chain, _retriever
