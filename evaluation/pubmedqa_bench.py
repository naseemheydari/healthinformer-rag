"""PubMedQA benchmark: accuracy on the expert-labeled yes/no/maybe split.

Loads the PubMedQA dataset ("qiaojin/PubMedQA", "pqa_labeled" split —
1,000 expert-annotated questions), feeds each question to the LLM
together with the paper's own gold context, extracts yes/no/maybe
from the generated answer, and compares to ground truth. Reports
overall accuracy against the 78% human ceiling from the PubMedQA
paper (Jin et al., 2019).

Important: PubMedQA is designed as a *reasoning-over-context* task.
Each item ships with the source paper's abstract passages in
`item["context"]["contexts"]` — that is the "gold context" the
question was written against. We use that here rather than routing
the question through our RAG retriever, because our 40K-chunk corpus
is built around 36 consumer-health categories and has no overlap
with PubMedQA's niche research questions (lace plant biology,
double balloon enteroscopy, etc.). Routing through our retriever
turns the benchmark into an unrelated out-of-corpus retrieval test.

Usage:
    from evaluation.pubmedqa_bench import run_pubmedqa_benchmark
    results_df = run_pubmedqa_benchmark(model="bedrock", max_questions=500)
"""

import re
import time
from typing import Any

import pandas as pd
from datasets import load_dataset

from config.settings import EVAL_RESULTS_DIR
from llm.base import BaseLLM

# Human expert ceiling from the PubMedQA paper (Jin et al., 2019)
HUMAN_CEILING: float = 0.780

# System prompt for the benchmark. Kept minimal and task-specific —
# we intentionally do NOT reuse the production system_prompt.txt here
# because that is tuned for plain-language consumer-health answers
# with citations and disclaimers. PubMedQA is a pure reasoning task
# and needs a deterministic yes/no/maybe verdict on one line so the
# extractor is trivial.
_PUBMEDQA_SYSTEM_PROMPT = (
    "You are a biomedical reasoning assistant. You will be given a "
    "research question and one or more passages from the abstract of "
    "the paper it came from. Read the passages carefully and decide "
    "whether they support a 'yes', 'no', or 'maybe' answer to the "
    "question. Brief reasoning is fine. Conclude with exactly one "
    "word on its own line: yes, no, or maybe."
)


def _build_user_prompt(question: str, context_passages: list[str]) -> str:
    """Format the PubMedQA question + its gold context passages."""
    numbered = "\n\n".join(
        f"[{i + 1}] {passage}" for i, passage in enumerate(context_passages)
    )
    return (
        f"## Context\n\n{numbered}\n\n"
        f"## Question\n\n{question}\n\n"
        f"Give brief reasoning, then on a final line write exactly "
        f"one word: yes, no, or maybe."
    )


def _extract_yes_no_maybe(answer: str) -> str:
    """Extract a yes/no/maybe verdict from the generated answer.

    The benchmark prompt instructs the model to end with exactly one
    word on its own line, so the canonical path is: take the last
    non-empty line, lowercase it, strip punctuation, and match. We
    keep a few fallbacks for stubborn cases.
    """
    text = answer.strip()
    if not text:
        return "maybe"

    # Primary: last non-empty line should be the verdict word.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].lower().strip(" .*:-—\"'`")
        if last in ("yes", "no", "maybe"):
            return last
        # Sometimes the model writes "Answer: yes" on the last line.
        m = re.search(r"\b(yes|no|maybe)\b\.?\s*$", last)
        if m:
            return m.group(1)

    # Fallback: explicit "the answer is ..." anywhere in the text.
    low = text.lower()
    m = re.search(
        r"\b(?:the answer is|answer:|conclusion:)\s*(yes|no|maybe)\b", low
    )
    if m:
        return m.group(1)

    # Last resort: majority vote on standalone yes/no/maybe tokens.
    yes_count = len(re.findall(r"\byes\b", low))
    no_count = len(re.findall(r"\bno\b", low))
    maybe_count = len(re.findall(r"\bmaybe\b", low))
    counts = {"yes": yes_count, "no": no_count, "maybe": maybe_count}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    return "maybe"


def _make_llm(model: str) -> BaseLLM:
    """Construct the requested LLM backend for the benchmark.

    We instantiate the LLM directly (not RAGChain) because we are
    intentionally bypassing retrieval for this benchmark.
    """
    if model == "groq":
        from llm.groq_client import GroqLLM
        return GroqLLM()
    elif model == "bedrock":
        from llm.bedrock_client import BedrockLLM
        return BedrockLLM()
    elif model == "bedrock-llama":
        from config.settings import BEDROCK_LLAMA_MODEL_ID
        from llm.bedrock_client import BedrockLLM
        return BedrockLLM(model_id=BEDROCK_LLAMA_MODEL_ID)
    raise ValueError(f"Unknown model: {model!r}. Use 'groq', 'bedrock', or 'bedrock-llama'.")


def run_pubmedqa_benchmark(
    model: str = "bedrock",
    max_questions: int = 500,
) -> pd.DataFrame:
    """Run PubMedQA benchmark and report accuracy.

    Uses each item's *gold context* (from the dataset) rather than
    retrieving from our ChromaDB store — PubMedQA is designed as a
    reasoning-over-given-context task, not an open-domain retrieval
    test, and our corpus does not cover its research topics.

    Args:
        model: LLM backend ("groq" or "bedrock").
        max_questions: Max questions to evaluate (default 500 per proposal).

    Returns:
        DataFrame with per-question results.
    """
    # ── Load PubMedQA dataset ───────────────────────────────────────────
    print("Loading PubMedQA dataset (pqa_labeled split)...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    total_available = len(dataset)
    n = min(max_questions, total_available)
    print(f"Loaded {total_available} questions, evaluating {n}")

    llm = _make_llm(model)
    print(f"Using {model} LLM directly (no retrieval — gold context from dataset)")

    # ── Evaluate each question ──────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    correct = 0

    print(f"\nRunning PubMedQA benchmark (model={model})...")
    for i in range(n):
        item = dataset[i]
        question = item["question"]
        ground_truth = item["final_decision"]  # "yes", "no", or "maybe"
        context_passages: list[str] = item["context"]["contexts"]

        print(f"  [{i + 1}/{n}] {question[:60]}...")
        start = time.time()
        try:
            user_prompt = _build_user_prompt(question, context_passages)
            # Low temperature + modest token budget — we need a short
            # reasoned verdict, not a long clinical essay.
            answer = llm.generate(
                prompt=user_prompt,
                max_tokens=300,
                temperature=0.0,
                system_prompt=_PUBMEDQA_SYSTEM_PROMPT,
            )
            elapsed = time.time() - start
            predicted = _extract_yes_no_maybe(answer)
            is_correct = predicted == ground_truth
            if is_correct:
                correct += 1

            rows.append({
                "pubid": item.get("pubid"),
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "answer": answer,
                "latency_sec": elapsed,
                "model": model,
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            rows.append({
                "pubid": item.get("pubid"),
                "question": question,
                "ground_truth": ground_truth,
                "predicted": "error",
                "correct": False,
                "answer": f"ERROR: {e}",
                "latency_sec": 0.0,
                "model": model,
            })

    # ── Save results ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = {"bedrock-llama": "llama"}.get(model, model)
    output_path = EVAL_RESULTS_DIR / f"pubmedqa_{file_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # ── Print summary ───────────────────────────────────────────────────
    accuracy = correct / len(df) if len(df) > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"PubMedQA Benchmark Results (model={model})")
    print(f"{'=' * 60}")
    print(f"  Questions evaluated:   {len(df)}")
    print(f"  Correct:               {correct}")
    print(f"  Accuracy:              {accuracy:.1%}")
    print(f"  Human ceiling:         {HUMAN_CEILING:.1%}")
    print(f"  Gap to human:          {HUMAN_CEILING - accuracy:+.1%}")
    print(f"  Avg latency:           {df['latency_sec'].mean():.1f}s")
    print(f"")
    # Breakdown by ground truth label
    for label in ["yes", "no", "maybe"]:
        subset = df[df["ground_truth"] == label]
        if len(subset) > 0:
            label_acc = subset["correct"].mean()
            print(f"  Accuracy ({label:5s}):       {label_acc:.1%}  ({len(subset)} questions)")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    # Quick test with a small sample
    run_pubmedqa_benchmark(model="bedrock", max_questions=10)
