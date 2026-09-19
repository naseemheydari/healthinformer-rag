"""Spot check: sample 50 questions, run through RAG, save for manual review.

Combines a mix of our curated test questions and PubMedQA questions to
produce a diverse sample for qualitative review. The output CSV is designed
for manual annotation of hallucination rate and answer quality.

Usage:
    from evaluation.spot_check import run_spot_check
    results_df = run_spot_check(model="groq", sample_size=50)
"""

import random
import time
from typing import Any

import pandas as pd
from datasets import load_dataset

from config.settings import EVAL_RESULTS_DIR, EVAL_SAMPLE_SIZE
from evaluation.test_questions import TEST_QUESTIONS
from pipeline.rag_chain import RAGChain


def _build_sample_pool(sample_size: int) -> list[dict[str, str]]:
    """Build a mixed sample pool from curated + PubMedQA questions.

    Splits roughly 60% curated / 40% PubMedQA so we cover our specific
    topic categories while also testing on external questions.
    """
    pool: list[dict[str, str]] = []

    # Add all curated questions
    for question, category, _ in TEST_QUESTIONS:
        pool.append({
            "question": question,
            "category": category,
            "source": "curated",
        })

    # Fill remaining slots with PubMedQA questions
    pubmedqa_needed = max(0, sample_size - len(pool))
    if pubmedqa_needed > 0:
        try:
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
            indices = random.sample(range(len(dataset)), min(pubmedqa_needed, len(dataset)))
            for idx in indices:
                item = dataset[idx]
                pool.append({
                    "question": item["question"],
                    "category": "PubMedQA",
                    "source": "pubmedqa",
                })
        except Exception as e:
            print(f"  Warning: Could not load PubMedQA: {e}")
            print(f"  Proceeding with {len(pool)} curated questions only.")

    # Shuffle and trim to sample_size
    random.shuffle(pool)
    return pool[:sample_size]


def run_spot_check(
    model: str = "groq",
    sample_size: int = EVAL_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Run spot check evaluation for manual review.

    Args:
        model: RAG chain LLM backend ("groq" or "bedrock").
        sample_size: Number of questions to sample (default 50).

    Returns:
        DataFrame with columns for manual annotation.
    """
    random.seed(42)  # Reproducible sampling
    sample = _build_sample_pool(sample_size)

    chain = RAGChain(model=model)

    # ── Run each question ───────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    print(f"\nSpot check: running {len(sample)} questions (model={model})...")
    for i, item in enumerate(sample, 1):
        question = item["question"]
        print(f"  [{i}/{len(sample)}] {question[:60]}...")
        start = time.time()
        try:
            result = chain.ask(question)
            elapsed = time.time() - start

            # Collect source PMIDs for traceability
            source_pmids = ", ".join(
                s["pmid"] for s in result["sources"]
            )
            source_urls = " | ".join(
                s["url"] for s in result["sources"]
            )

            rows.append({
                "question": question,
                "category": item["category"],
                "source_type": item["source"],
                "answer": result["answer"],
                "source_pmids": source_pmids,
                "source_urls": source_urls,
                "num_sources": len(result["sources"]),
                "latency_sec": elapsed,
                "model": model,
                # Columns for manual review (to be filled by human)
                "quality_1to5": "",
                "has_hallucination": "",
                "hallucination_notes": "",
                "reviewer": "",
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            rows.append({
                "question": question,
                "category": item["category"],
                "source_type": item["source"],
                "answer": f"ERROR: {e}",
                "source_pmids": "",
                "source_urls": "",
                "num_sources": 0,
                "latency_sec": 0.0,
                "model": model,
                "quality_1to5": "",
                "has_hallucination": "",
                "hallucination_notes": "",
                "reviewer": "",
            })

    # ── Save ────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = {"bedrock-llama": "llama"}.get(model, model)
    output_path = EVAL_RESULTS_DIR / f"spot_check_{file_name}.csv"
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")
    print(f"\n{'='*60}")
    print(f"Spot Check Summary (model={model})")
    print(f"{'='*60}")
    print(f"  Questions sampled:     {len(df)}")
    print(f"  Avg sources per answer:{df['num_sources'].mean():.1f}")
    print(f"  Avg latency:           {df['latency_sec'].mean():.1f}s")
    print(f"  Errors:                {(df['answer'].str.startswith('ERROR')).sum()}")
    print(f"")
    print(f"  Manual review columns ready:")
    print(f"    - quality_1to5: Rate overall quality 1 (poor) to 5 (excellent)")
    print(f"    - has_hallucination: yes/no — claim not supported by sources")
    print(f"    - hallucination_notes: Describe any hallucinated claims")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    run_spot_check(model="groq", sample_size=5)
