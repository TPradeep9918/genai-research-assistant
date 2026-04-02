# chain.py
# Shared components for the RAG pipeline.
#
# Exported API:
#   CITATION_PROMPT          — ChatPromptTemplate used by app.py and evaluate.py
#   format_docs_with_citations — formats retrieved docs into a numbered context block

from langchain_core.prompts import ChatPromptTemplate


# ── Prompt ───────────────────────────────────────────────────────────────────

CITATION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant on research papers.

Answer the question using ONLY the numbered context chunks provided below.
For every claim you make, cite the chunk number in square brackets — e.g. [1], [2].
If multiple chunks support the same claim, cite all of them — e.g. [1][3].
If the context does not contain enough information, say:
  "The provided docs don't cover this — please check the original paper."

Do NOT add information from outside the provided context.

--- Context ---
{context}
--- End of context ---

Question: {question}

Answer (with inline citations):"""
)


# ── Utilities ────────────────────────────────────────────────────────────────

def format_docs_with_citations(docs: list) -> str:
    """Converts a list of Document objects into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)
