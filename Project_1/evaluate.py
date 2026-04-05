# evaluate.py — Multi-LLM evaluation for the RAG pipeline
# Run with:  python evaluate.py
#
# Measures three metrics per LLM without requiring OpenAI:
#   context_relevance  — avg cross-encoder score of retrieved chunks (0-1 normalised)
#   citation_rate      — % of answers that contain at least one [N] citation
#   grounded_rate      — % of answers NOT containing the "docs don't cover" fallback
#
# Compares multiple Ollama models side-by-side and prints a ranked table.

import re
import math
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from retriever import build_retriever
from chain import CITATION_PROMPT, format_docs_with_citations


# ── Models to compare ─────────────────────────────────────────────────────────
MODELS_TO_COMPARE = [
    "llama3.2",
    "mistral",
    "gemma3:1b",
]

# ── Evaluation questions ──────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    "What is the Transformer architecture and how does it differ from RNNs?",
    "How does the scaled dot-product attention mechanism work?",
    "What is multi-head attention and why is it used?",
    "What is the purpose of positional encoding in the Transformer?",
    "How does the encoder-decoder structure work in the Transformer?",
    "What BLEU scores did the Transformer achieve on WMT translation tasks?",
    "Why did the authors choose scaled dot-product over additive attention?",
]

FALLBACK_PHRASE = "don't cover this"


# ── Helpers ───────────────────────────────────────────────────────────────────

def sigmoid(x):
    """Map cross-encoder raw score (can be negative) to 0-1 range."""
    return 1 / (1 + math.exp(-x))


def build_chain_for_model(retriever, model_name: str):
    llm = ChatOllama(model=model_name, temperature=0)
    return (
        {
            "context":  retriever | RunnableLambda(format_docs_with_citations),
            "question": RunnablePassthrough(),
        }
        | CITATION_PROMPT
        | llm
        | StrOutputParser()
    )


# ── Evaluate one model ────────────────────────────────────────────────────────

def evaluate_model(model_name: str, retriever) -> dict:
    print(f"\n{'='*55}")
    print(f"  Evaluating model: {model_name}")
    print(f"{'='*55}")

    try:
        chain = build_chain_for_model(retriever, model_name)
    except Exception as e:
        print(f"  Could not load {model_name}: {e}")
        return None

    ctx_scores   = []
    citation_hits = 0
    grounded_hits = 0

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"  [{i}/{len(EVAL_QUESTIONS)}] {question[:60]}...", flush=True)
        try:
            docs   = retriever.invoke(question)
            answer = chain.invoke(question)
        except Exception as e:
            print(f"    Error: {e}", flush=True)
            continue

        # 1. Context relevance — average normalised cross-encoder score
        scores = [sigmoid(d.metadata["relevance_score"]) for d in docs if "relevance_score" in d.metadata]
        if scores:
            ctx_scores.append(sum(scores) / len(scores))

        # 2. Citation rate — does the answer contain at least one [N] reference?
        if re.search(r'\[\d+\]', answer):
            citation_hits += 1

        # 3. Grounded rate — answer is NOT the fallback "docs don't cover this"
        if FALLBACK_PHRASE.lower() not in answer.lower():
            grounded_hits += 1

        cited = "yes" if re.search(r'\[\d+\]', answer) else "no"
        grounded = "yes" if FALLBACK_PHRASE.lower() not in answer.lower() else "no"
        ctx_val = f"{ctx_scores[-1]:.3f}" if ctx_scores else "n/a"
        print(f"    ctx={ctx_val}  cited={cited}  grounded={grounded}", flush=True)

    n = len(EVAL_QUESTIONS)
    return {
        "context_relevance": round(sum(ctx_scores) / len(ctx_scores), 3) if ctx_scores else 0.0,
        "citation_rate":     round(citation_hits / n, 3),
        "grounded_rate":     round(grounded_hits / n, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n  Loading retriever (shared across all models)...")
    retriever = build_retriever()

    results = {}
    for model in MODELS_TO_COMPARE:
        scores = evaluate_model(model, retriever)
        if scores:
            results[model] = scores

    col = 18
    print("\n\n" + "=" * 72)
    print("  MULTI-LLM EVALUATION RESULTS")
    print("=" * 72)
    header = f"  {'Model':<{col}} {'Ctx.Relevance':>14} {'Citation%':>10} {'Grounded%':>10}"
    print(header)
    print("-" * 72)

    ranked = sorted(
        results.items(),
        key=lambda x: sum(x[1].values()) / 3,
        reverse=True,
    )

    for rank, (model, s) in enumerate(ranked, 1):
        avg = sum(s.values()) / 3
        print(
            f"  #{rank} {model:<{col-3}} "
            f"{s['context_relevance']:>14.3f} "
            f"{s['citation_rate']*100:>9.0f}% "
            f"{s['grounded_rate']*100:>9.0f}%   avg={avg:.3f}"
        )

    print("-" * 72)
    print("  Targets: ctx_relevance > 0.70 | citation% = 100 | grounded% > 80")
    print("  Scores range 0.0-1.0 (ctx) and 0-100% (rates). Higher is better.")
    print("=" * 72 + "\n")

    return results


if __name__ == "__main__":
    run_evaluation()
