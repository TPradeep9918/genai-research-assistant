# agent.py
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  AGENTIC AI — agent.py                                                      ║
# ║                                                                              ║
# ║  This file turns the static RAG pipeline into a dynamic reasoning agent.   ║
# ║                                                                              ║
# ║  OLD pipeline (app.py, Standard RAG mode) — fixed sequence, no decisions:  ║
# ║    User Question → Retrieve → Generate Answer → Done                       ║
# ║                                                                              ║
# ║  NEW agentic pipeline (this file) — reasoning loop with decisions:         ║
# ║    User Question                                                            ║
# ║        ↓                                                                    ║
# ║    [Agent thinks] → calls a Tool → sees result → [Agent thinks again] → … ║
# ║        ↓                                                                    ║
# ║    Final Answer  (when the agent decides it has enough information)         ║
# ║                                                                              ║
# ║  This "think → act → observe → think" pattern is called ReAct and is the  ║
# ║  defining behaviour of an agentic AI system.                               ║
# ║                                                                              ║
# ║  Read the numbered markers ❶–❻ in order to follow the flow:               ║
# ║    ❶ Tool definitions  (what the agent CAN do, built as closures)          ║
# ║    ❷ Agent State       (the agent's working memory)                        ║
# ║    ❸ Agent Node        (the LLM reasoning / decision step)                 ║
# ║    ❹ Routing           (loop or stop?)                                     ║
# ║    ❺ Graph             (wiring the loop together)                          ║
# ║    ❻ Public API        (called by app.py and evaluate.py)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from typing import TypedDict, Annotated

# LangGraph — the library that manages the agent loop
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# LangChain tool decorator and message types
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama

from chain import format_docs_with_citations
from config import OLLAMA_HOST, LLM_MODEL, AGENT_MAX_ITERATIONS


# System prompt that tells the LLM its role and strategy
_SYSTEM_PROMPT = """You are an expert research assistant with access to a local knowledge base of research papers.

You have three tools:
  • rewrite_query      — improve a vague question before searching
  • search_papers      — retrieve specific passages from the papers
  • get_paper_summary  — get a high-level overview of a topic

Strategy:
1. If the question uses casual or imprecise language, call rewrite_query first.
2. For specific facts or quotes, call search_papers.
3. For broad "what is X?" questions, call get_paper_summary.
4. If the first search results are insufficient, call search_papers again with a different query.
5. Once you have enough information, write your final answer with inline citations [N].
6. If the papers do not cover the topic, say so explicitly.

Never invent information. Every claim must be supported by retrieved context."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❷ — AGENT STATE (the agent's working memory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Every node in the graph reads from and writes to this shared state dict.
# `messages` uses LangGraph's add_messages reducer, which APPENDS new messages
# rather than replacing the whole list — so the full conversation is preserved
# across every iteration of the loop.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❶ — TOOL FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tools are created as closures inside build_agent_graph() so each invocation
# of run_agent() gets its own isolated set of tools bound to the caller's
# retriever and model_name. This eliminates the module-level global state that
# made the original design thread-unsafe under concurrent Streamlit requests.
#
# retrieved_docs is a mutable list (closure-captured) that tools write into,
# letting run_agent() return the last retrieved docs for source display in the UI.

def _make_tools(retriever, model_name: str, retrieved_docs: list):
    """
    Return (tools_list, tool_node) bound to the given retriever and model.

    retrieved_docs: caller-provided mutable list; search_papers writes the last
                    retrieved Document objects into it so the UI can show sources.
    """

    @tool
    def rewrite_query(question: str) -> str:
        """
        Rewrite a casual or vague question into precise academic terminology.

        Use this FIRST when the user's question uses everyday language that may
        not match the vocabulary used in the research papers.
        Example: "how does attention work?" → "scaled dot-product attention mechanism"
        """
        llm    = ChatOllama(model=model_name, temperature=0, base_url=OLLAMA_HOST)
        prompt = (
            "You are an expert in machine learning and NLP research. "
            "Rewrite the following question using precise academic terminology "
            "that would appear in research papers. "
            "Output ONLY the rewritten question, nothing else.\n\n"
            f"Original question: {question}"
        )
        response = llm.invoke(prompt)
        return response.content.strip()

    @tool
    def search_papers(query: str) -> str:
        """
        Search the local research paper knowledge base for information on a topic.

        Use this when you need specific facts, quotes, or technical details from
        the indexed papers. Returns numbered context chunks with source citations.
        You may call this multiple times with different queries if needed.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant passages found for this query. Try a different query."
        # Capture docs so run_agent() can surface them as sources in the UI
        retrieved_docs.clear()
        retrieved_docs.extend(docs)
        return format_docs_with_citations(docs)

    @tool
    def get_paper_summary(topic: str) -> str:
        """
        Retrieve and summarise what the indexed papers say about a broad topic.

        Use this for high-level overviews rather than specific facts.
        Best for 'what is X?' or 'explain X' style questions.
        """
        docs = retriever.invoke(topic)
        if not docs:
            return f"No content found about '{topic}' in the papers."
        retrieved_docs.clear()
        retrieved_docs.extend(docs)
        context = format_docs_with_citations(docs)
        llm    = ChatOllama(model=model_name, temperature=0, base_url=OLLAMA_HOST)
        prompt = (
            f"Summarise what the following research paper excerpts say about: {topic}\n\n"
            f"{context}\n\n"
            "Write a concise 2–3 sentence summary. "
            "Cite every claim with chunk numbers like [1] or [2][3]."
        )
        response = llm.invoke(prompt)
        return response.content.strip()

    agent_tools = [rewrite_query, search_papers, get_paper_summary]
    return agent_tools


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❺ — GRAPH CONSTRUCTION (wiring the loop together)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StateGraph defines the nodes and edges of the agent loop.
#
#   Nodes:
#     "agent" — runs the LLM  (agent_node)
#     "tools" — executes the tool the LLM requested  (LangGraph's ToolNode)
#
#   Edges:
#     START → "agent"                           (always start here)
#     "agent" → should_continue() → "tools"     (if LLM called a tool)
#     "agent" → should_continue() → END         (if LLM gave a final answer)
#     "tools" → "agent"                         (always loop back after a tool)
#
#   recursion_limit caps the total number of node visits so the loop cannot run
#   forever. Each question→tool round-trip visits 2 nodes (agent + tools), so
#   the limit is AGENT_MAX_ITERATIONS × 2.

def build_agent_graph(retriever, model_name: str, retrieved_docs: list):
    """
    AGENTIC AI ❺ — Graph builder.

    Constructs and compiles the LangGraph StateGraph that implements the
    reasoning loop. Tools are built as closures bound to retriever/model_name,
    making each call fully thread-safe.
    """
    agent_tools = _make_tools(retriever, model_name, retrieved_docs)

    # ── ❸ Agent Node ──────────────────────────────────────────────────────────
    # Runs the LLM on every iteration. The LLM decides whether to call a tool
    # (continue looping) or produce a final answer (stop).
    def agent_node(state: AgentState) -> AgentState:
        llm            = ChatOllama(model=model_name, temperature=0, base_url=OLLAMA_HOST)
        llm_with_tools = llm.bind_tools(agent_tools)
        messages       = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=_SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── ❹ Routing ─────────────────────────────────────────────────────────────
    # After every agent_node call, inspect the last message:
    #   has tool_calls → return "tools"  (keep looping)
    #   no  tool_calls → return "end"    (final answer)
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(agent_tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    # recursion_limit enforces AGENT_MAX_ITERATIONS (each round-trip = 2 nodes)
    return graph.compile(recursion_limit=AGENT_MAX_ITERATIONS * 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❻ — PUBLIC API (called by app.py and evaluate.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_agent(question: str, model_name: str, retriever) -> dict:
    """
    AGENTIC AI ❻ — Main entry point.

    Runs the agent reasoning loop for a single user question.

    Args:
        question:   The user's natural language question.
        model_name: Ollama model to use (e.g. "llama3.2").
        retriever:  HybridRerankedRetriever instance from retriever.py.

    Returns a dict:
        {
            "answer": str,         # the agent's final answer text
            "steps":  list[str],   # trace of tool calls made (shown in the UI)
            "docs":   list[Document],  # last set of retrieved docs (for source display)
        }
    """
    retrieved_docs: list = []
    graph = build_agent_graph(retriever, model_name, retrieved_docs)

    final_state = graph.invoke({"messages": [HumanMessage(content=question)]})

    answer = ""
    steps: list[str] = []

    for message in final_state["messages"]:
        if isinstance(message, AIMessage):
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_args = tc.get("args", {})
                    arg_str   = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
                    steps.append(f"`{tool_name}({arg_str})`")
            elif message.content:
                answer = message.content

    return {"answer": answer, "steps": steps, "docs": retrieved_docs}
