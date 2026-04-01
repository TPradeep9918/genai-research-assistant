# evaluate.py — RAGAS evaluation for the RAG pipeline
# Run with:  python evaluate.py
#
# Measures three reference-free metrics:
#   faithfulness      — does the answer stay true to the retrieved context?
#   answer_relevancy  — does the answer actually address the question?
#   context_precision — are the retrieved chunks relevant to the question?
#
# No ground-truth answers required for these three metrics.

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from chain import get_chain


# ── Evaluation questions ──────────────────────────────────────────────────────
# Choose questions that span different parts of the indexed docs

EVAL_QUESTIONS = [
    "What is LCEL and why should I use it instead of legacy chains?",
    "How does RAG work in LangChain at a high level?",
    "What is the difference between a retriever and a vector store?",
    "How do I enable streaming responses in LangChain?",
    "What embedding models does LangChain natively support?",
    "What text splitters are available and when should I use each?",
    "How do I add source citations to RAG answers?",
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n  Loading RAG chain...")
    chain, retriever = get_chain()

    print(f"\n  Running {len(EVAL_QUESTIONS)} evaluation questions...\n")

    records = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],    # left empty — using reference-free metrics
    }

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"  [{i}/{len(EVAL_QUESTIONS)}] {question}")

        docs   = retriever.invoke(question)
        answer = chain.invoke(question)

        records["question"].append(question)
        records["answer"].append(answer)
        records["contexts"].append([d.page_content for d in docs])
        records["ground_truth"].append("")

    # ── Run RAGAS ─────────────────────────────────────────────────────────────
    print("\n  Computing RAGAS metrics...")

    dataset = Dataset.from_dict(records)

    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
    )

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  RAGAS Evaluation Results")
    print("="*50)
    print(f"  Faithfulness       : {scores['faithfulness']:.3f}  (target > 0.80)")
    print(f"  Answer relevancy   : {scores['answer_relevancy']:.3f}  (target > 0.80)")
    print(f"  Context precision  : {scores['context_precision']:.3f}  (target > 0.75)")
    print("="*50)
    print("\n  Scores range 0.0 – 1.0.  Higher is better.")
    print("  Low faithfulness = LLM hallucinating outside the retrieved context.")
    print("  Low answer_relevancy = answer drifts off-topic.")
    print("  Low context_precision = retriever returning irrelevant chunks.\n")

    return scores


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
