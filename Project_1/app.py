# app.py — Streamlit chat UI for the LangChain Docs RAG
# Run with:  python -m streamlit run app.py

import streamlit as st
from chain import get_chain

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LangChain Docs Q&A",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("LangChain Docs Q&A")
st.caption(
    "Production RAG — hybrid retrieval (BM25 + ChromaDB) · "
    "cross-encoder reranking · citation enforcement · Ollama llama3.2"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header(" Pipeline")
    st.markdown(
        """
        **Indexing**
        - Docs loaded from: LangChain website
        - Embeddings: `all-mpnet-base-v2`
        - Vector store: ChromaDB (persistent)
        - Keyword store: BM25

        **Querying**
        - Hybrid retrieval (RRF fusion)
        - Cross-encoder reranking
        - Top 5 chunks → Ollama llama3.2
        - Citation-enforced prompt
        """
    )
    st.divider()
    if st.button(" Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Load chain (cached across reruns) ────────────────────────────────────────

@st.cache_resource(show_spinner="  Loading models — Please wait")
def load_chain():
    return get_chain()

chain, retriever = load_chain()

# ── Chat state ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(" Retrieved sources"):
                for i, s in enumerate(msg["sources"], 1):
                    st.caption(f"**[{i}]** {s['url']}")
                    st.markdown(f"> {s['snippet']}...")
                    if i < len(msg["sources"]):
                        st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask anything about LangChain..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):

            # Run retriever separately so we can show sources
            retrieved_docs = retriever.invoke(question)

            # Run the full chain
            answer = chain.invoke(question)

        st.markdown(answer)

        # Show source chunks in an expander
        sources = []
        with st.expander(" Retrieved sources"):
            for i, doc in enumerate(retrieved_docs, 1):
                url     = doc.metadata.get("source", "unknown")
                snippet = doc.page_content.strip()[:250]
                sources.append({"url": url, "snippet": snippet})

                st.caption(f"**[{i}]** {url}")
                st.markdown(f"> {snippet}...")
                if i < len(retrieved_docs):
                    st.divider()

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
