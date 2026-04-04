# evaluate.py — Multi-LLM RAGAS evaluation for the RAG pipeline
# Run with:  python evaluate.py
#
# Measures three reference-free metrics per LLM:
#   faithfulness      — does the answer stay true to the retrieved context?
#   answer_relevancy  — does the answer actually address the question?
#   context_precision — are the retrieved chunks relevant to the question?
#
# Compares multiple Ollama models side-by-side and prints a ranked table.

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from retriever import build_retriever
from chain import CITATION_PROMPT, format_docs_with_citations
from config import EMBED_MODEL

# ── Configure RAGAS to use local Ollama instead of OpenAI ────────────────────
_ragas_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2", temperature=0))
_ragas_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
)
faithfulness.llm           = _ragas_llm
answer_relevancy.llm       = _ragas_llm
answer_relevancy.embeddings = _ragas_emb
context_precision.llm      = _ragas_llm


# ── Models to compare ─────────────────────────────────────────────────────────
# Add/remove any Ollama model you have pulled locally
MODELS_TO_COMPARE = [
    "llama3.2",
    # "mistral",    # run: ollama pull mistral
    # "gemma3:1b",  # run: ollama pull gemma3:1b
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


# ── Build chain for a given model ─────────────────────────────────────────────

def build_chain_for_model(retriever, model_name: str):
    llm = ChatOllama(model=model_name, temperature=0)
    chain = (
        {
            "context":  retriever | RunnableLambda(format_docs_with_citations),
            "question": RunnablePassthrough(),
        }
        | CITATION_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── Run RAGAS for one model ───────────────────────────────────────────────────

def evaluate_model(model_name: str, retriever) -> dict:
    print(f"\n{'='*55}")
    print(f"  Evaluating model: {model_name}")
    print(f"{'='*55}")

    try:
        chain = build_chain_for_model(retriever, model_name)
    except Exception as e:
        print(f"  Could not load {model_name}: {e}")
        return None

    records = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"  [{i}/{len(EVAL_QUESTIONS)}] {question[:60]}...")
        try:
            docs   = retriever.invoke(question)
            answer = chain.invoke(question)
        except Exception as e:
            print(f"  Error on question {i}: {e}")
            answer = ""
            docs   = []

        records["question"].append(question)
        records["answer"].append(answer)
        records["contexts"].append([d.page_content for d in docs])
        records["ground_truth"].append("")

    print(f"\n  Computing RAGAS metrics for {model_name}...")
    dataset = Dataset.from_dict(records)
    scores  = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return {
        "faithfulness":      round(float(scores["faithfulness"]),      3),
        "answer_relevancy":  round(float(scores["answer_relevancy"]),  3),
        "context_precision": round(float(scores["context_precision"]), 3),
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

    # ── Print comparison table ────────────────────────────────────────────────
    col = 18
    print("\n\n" + "=" * 70)
    print("  MULTI-LLM RAGAS COMPARISON")
    print("=" * 70)
    header = f"  {'Model':<{col}} {'Faithfulness':>14} {'Ans.Relevancy':>14} {'Ctx.Precision':>14}"
    print(header)
    print("-" * 70)

    # Sort by average score descending
    ranked = sorted(
        results.items(),
        key=lambda x: sum(x[1].values()) / 3,
        reverse=True,
    )

    for rank, (model, s) in enumerate(ranked, 1):
        avg = sum(s.values()) / 3
        print(
            f"  #{rank} {model:<{col-3}} "
            f"{s['faithfulness']:>14.3f} "
            f"{s['answer_relevancy']:>14.3f} "
            f"{s['context_precision']:>14.3f}   avg={avg:.3f}"
        )

    print("-" * 70)
    print("  Targets:           faithfulness > 0.80 | relevancy > 0.80 | precision > 0.75")
    print("  Scores range 0.0 - 1.0.  Higher is better.")
    print("=" * 70 + "\n")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
