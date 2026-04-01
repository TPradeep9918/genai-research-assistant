import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from typing import List

from config import (
    CHROMA_DIR, BM25_PICKLE, EMBED_MODEL, CROSS_ENCODER,
    COLLECTION_NAME, VECTOR_K, BM25_K, RERANK_TOP_K,
)


class HybridRerankedRetriever(BaseRetriever):
    """BM25 + vector search fused via RRF, then cross-encoder reranked."""
    bm25: object = Field(default=None)
    vector: object = Field(default=None)
    cross_encoder: object = Field(default=None)
    top_n: int = Field(default=RERANK_TOP_K)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Step 1: retrieve from both sources
        bm25_docs  = self.bm25.invoke(query)
        vec_docs   = self.vector.invoke(query)

        # Step 2: RRF fusion (k=60)
        k = 60
        scores = {}
        all_docs = {}
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            scores[key]   = scores.get(key, 0) + 1 / (k + rank + 1)
            all_docs[key] = doc
        for rank, doc in enumerate(vec_docs):
            key = doc.page_content[:100]
            scores[key]   = scores.get(key, 0) + 1 / (k + rank + 1)
            all_docs[key] = doc

        # Step 3: sort by RRF score, take top candidates
        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        candidates  = [all_docs[k] for k in sorted_keys]

        # Step 4: cross-encoder rerank
        if not candidates:
            return []
        pairs  = [[query, doc.page_content] for doc in candidates]
        ce_scores = self.cross_encoder.score(pairs)
        ranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)

        return [doc for _, doc in ranked[: self.top_n]]


def build_retriever():
    print("  Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": VECTOR_K},
    )
    print(f"  ChromaDB loaded  ({vectorstore._collection.count():,} vectors)")

    print("  Loading BM25 index...")
    with open(BM25_PICKLE, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = BM25_K
    print("BM25 index loaded")

    print(f"  Loading cross-encoder ({CROSS_ENCODER})...")
    cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER)

    retriever = HybridRerankedRetriever(
        bm25=bm25_retriever,
        vector=vector_retriever,
        cross_encoder=cross_encoder,
        top_n=RERANK_TOP_K,
    )
    print(f"Hybrid + reranked retriever ready (top {RERANK_TOP_K} chunks)")
    return retriever