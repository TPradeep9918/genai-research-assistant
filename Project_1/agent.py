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
# ║  Read the numbered markers ❶–❼ in order to follow the flow:               ║
# ║    ❶ Global state shared with tools                                        ║
# ║    ❷ Tool definitions  (what the agent CAN do)                             ║
# ║    ❸ Agent State       (the agent's working memory)                        ║
# ║    ❹ Agent Node        (the LLM reasoning / decision step)                 ║
# ║    ❺ Routing           (loop or stop?)                                     ║
# ║    ❻ Graph             (wiring the loop together)                          ║
# ║    ❼ Public API        (called by app.py)                                  ║
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
from config import OLLAMA_HOST, LLM_MODEL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❶ — GLOBAL RESOURCES SHARED ACROSS TOOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tools are plain Python functions called by the LLM. The LLM can only pass
# the declared string/int parameters — it cannot pass Python objects like
# a retriever. So we store shared resources here at module level, then inject
# them once via init_agent() before the loop starts.

_retriever = None   # will be set to the HybridRerankedRetriever from retriever.py
_model_name = LLM_MODEL  # which Ollama model the agent uses


def init_agent(retriever, model_name: str = LLM_MODEL):
    """Inject shared resources before calling run_agent()."""
    global _retriever, _model_name
    _retriever = retriever
    _model_name = model_name


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❷ — TOOL DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A "tool" is a regular Python function decorated with @tool.
# The LLM reads the function name + docstring to know WHEN to call it.
# The LLM reads the parameter names to know HOW to call it.
# The agent can call zero, one, or many tools per reasoning turn — its choice.
#
# These three tools give the agent three different strategies:
#   • rewrite_query   — improve the query before searching
#   • search_papers   — retrieve specific passages (the same retriever as RAG)
#   • get_paper_summary — broad overview of a topic


@tool
def rewrite_query(question: str) -> str:
    """
    Rewrite a casual or vague question into precise academic terminology.

    Use this FIRST when the user's question uses everyday language that may
    not match the vocabulary used in the research papers.
    Example: "how does attention work?" → "scaled dot-product attention mechanism"
    """
    llm = ChatOllama(model=_model_name, temperature=0, base_url=OLLAMA_HOST)
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
    if _retriever is None:
        return "Error: retriever not initialised. Call init_agent() first."
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant passages found for this query. Try a different query."
    return format_docs_with_citations(docs)


@tool
def get_paper_summary(topic: str) -> str:
    """
    Retrieve and summarise what the indexed papers say about a broad topic.

    Use this for high-level overviews rather than specific facts.
    Best for 'what is X?' or 'explain X' style questions.
    """
    if _retriever is None:
        return "Error: retriever not initialised."
    docs = _retriever.invoke(topic)
    if not docs:
        return f"No content found about '{topic}' in the papers."

    context = format_docs_with_citations(docs)
    llm = ChatOllama(model=_model_name, temperature=0, base_url=OLLAMA_HOST)
    prompt = (
        f"Summarise what the following research paper excerpts say about: {topic}\n\n"
        f"{context}\n\n"
        "Write a concise 2–3 sentence summary. "
        "Cite every claim with chunk numbers like [1] or [2][3]."
    )
    response = llm.invoke(prompt)
    return response.content.strip()


# All tools in one list — passed to both the LLM (so it knows they exist)
# and to ToolNode (so it can execute them).
AGENT_TOOLS = [rewrite_query, search_papers, get_paper_summary]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❸ — AGENT STATE (the agent's working memory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Every node in the graph reads from and writes to this shared state dict.
# `messages` uses LangGraph's add_messages reducer, which APPENDS new messages
# rather than replacing the whole list — so the full conversation is preserved
# across every iteration of the loop.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❹ — AGENT NODE (the LLM reasoning / decision step)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This node runs the LLM on every iteration of the loop.
# The LLM sees the full message history (question + all previous tool results)
# and makes one of two decisions:
#
#   Decision A → Call a tool
#       Returns an AIMessage that contains `tool_calls` (structured JSON
#       saying which tool to run and with what arguments). The LLM does NOT
#       run the tool itself — it just declares its intent.
#
#   Decision B → Give a final answer
#       Returns an AIMessage with plain text content and no tool_calls.
#       The loop then ends.
#
# This decision-making is what separates an agent from a standard RAG chain.

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


def agent_node(state: AgentState) -> AgentState:
    """
    AGENTIC AI ❹ — Agent Node.

    Runs the LLM with tool-calling enabled. The LLM decides whether
    to call a tool (continue looping) or produce a final answer (stop).
    """
    llm = ChatOllama(model=_model_name, temperature=0, base_url=OLLAMA_HOST)

    # bind_tools() is the key call that gives the LLM awareness of the tools.
    # Without it, the LLM can only generate plain text.
    # With it, it can output structured tool_call instructions.
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    # Prepend system prompt on the first call (when no SystemMessage is present yet)
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❺ — ROUTING (loop or stop?)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# After every agent_node call, this function inspects the last message:
#   • Has tool_calls  → the LLM wants to use a tool → return "tools" (keep looping)
#   • No tool_calls   → the LLM wrote a final answer → return "end" (stop)
#
# This conditional routing IS the agent loop. Without it you'd have a
# straight-line pipeline just like the standard RAG chain.

def should_continue(state: AgentState) -> str:
    """
    AGENTIC AI ❺ — Routing function (conditional edge).

    Returns "tools" to continue the loop, or "end" to stop.
    """
    last_message = state["messages"][-1]

    # AIMessage.tool_calls is populated when the LLM wants to invoke a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"   # → loop: execute the requested tool

    return "end"         # → stop: LLM produced a final text answer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❻ — GRAPH CONSTRUCTION (wiring the loop together)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# StateGraph defines the nodes and edges of the agent loop.
#
#   Nodes:
#     "agent" — runs the LLM  (agent_node above)
#     "tools" — executes the tool the LLM requested  (LangGraph's ToolNode)
#
#   Edges:
#     START → "agent"                           (always start here)
#     "agent" → should_continue() → "tools"     (if LLM called a tool)
#     "agent" → should_continue() → END         (if LLM gave a final answer)
#     "tools" → "agent"                         (always loop back after a tool)
#
#   Visual flow:
#
#     START
#       ↓
#     [agent]  ←────────────────────────────┐
#       ↓                                   │
#     should_continue()                     │
#       ├── "tools" ──→ [tools] ────────────┘   (loop back for next reasoning step)
#       └── "end"   ──→ END                      (final answer reached)

def build_agent_graph():
    """
    AGENTIC AI ❻ — Graph builder.

    Constructs and compiles the LangGraph StateGraph that implements the
    reasoning loop. Returns a compiled graph ready to call .invoke() on.
    """
    graph = StateGraph(AgentState)

    # Register the two nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(AGENT_TOOLS))  # ToolNode auto-dispatches to the right tool

    # Entry point — always start at the agent (LLM reasoning) node
    graph.set_entry_point("agent")

    # Conditional edge after "agent": call should_continue() to decide next step
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",   # LLM requested a tool → execute it
            "end":   END,       # LLM wrote final answer → stop
        },
    )

    # Unconditional edge after "tools": always return to agent for next reasoning step
    graph.add_edge("tools", "agent")

    return graph.compile()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENTIC AI ❼ — PUBLIC API (called by app.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This is the only function app.py needs to call.
# It runs the full agent loop and returns the final answer plus a human-readable
# trace of every tool call the agent made along the way.

def run_agent(question: str, model_name: str, retriever) -> dict:
    """
    AGENTIC AI ❼ — Main entry point.

    Runs the agent reasoning loop for a single user question.

    Args:
        question:   The user's natural language question.
        model_name: Ollama model to use (e.g. "llama3.2").
        retriever:  HybridRerankedRetriever instance from retriever.py.

    Returns a dict:
        {
            "answer": str,       # the agent's final answer text
            "steps":  list[str], # trace of tool calls made (shown in the UI)
        }
    """
    # ❶ Inject shared resources so tools can access retriever + model
    init_agent(retriever, model_name)

    # ❻ Build the compiled graph (lightweight — just wires nodes together)
    graph = build_agent_graph()

    # Run the loop — LangGraph handles the iteration until should_continue() → "end"
    final_state = graph.invoke({"messages": [HumanMessage(content=question)]})

    # Extract final answer and step trace from the message history
    answer = ""
    steps = []

    for message in final_state["messages"]:
        if isinstance(message, AIMessage):
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Intermediate step — record which tool was called and with what
                for tc in message.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_args = tc.get("args", {})
                    arg_str = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
                    steps.append(f"`{tool_name}({arg_str})`")
            elif message.content:
                # Final answer — AIMessage with text and no tool_calls
                answer = message.content

    return {"answer": answer, "steps": steps}
