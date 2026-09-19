"""RAGAS evaluation: faithfulness, answer relevancy, context precision/recall.

Supports two modes:
  1. generate_answers() — runs the RAG chain, saves results to JSON.
  2. score_answers()    — loads saved JSON, runs RAGAS metrics.

This separation allows answer generation and RAGAS scoring to use different
LLM backends when desired.

METHODOLOGICAL NOTE — Evaluation circularity:
    When the evaluator LLM is from the same family as the generator LLM
    (e.g., both are Llama 3 via Groq), the evaluation may be biased —
    the model is effectively grading its own work. Using a *different*
    model family as evaluator (e.g., Bedrock Claude for evaluation)
    mitigates this. We flag this in the report.

Usage:
    # Generate answers (uses Groq tokens only)
    generate_answers(model="groq", top_k=8)

    # Score saved answers (uses Bedrock tokens only)
    df = score_answers(model="groq", evaluator="bedrock")

    # All-in-one (original behavior)
    df = run_ragas_evaluation(model="groq", evaluator="bedrock", top_k=8)
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.embeddings import Embeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    _AnswerRelevancy,
    _Faithfulness,
    _LLMContextPrecisionWithReference,
    _LLMContextRecall,
)

from config.settings import EVAL_RESULTS_DIR, GROQ_API_KEY, BEDROCK_MODEL_ID, AWS_REGION
from evaluation.test_questions import TEST_QUESTIONS
from pipeline.rag_chain import RAGChain


class _PubMedBERTEmbeddings(Embeddings):
    """Langchain-compatible wrapper around our PubMedBERT embedder.

    RAGAS AnswerRelevancy needs an embeddings model. Rather than
    defaulting to OpenAI, we reuse the same PubMedBERT model from
    our vectorstore — keeping everything local and free.
    """

    def __init__(self) -> None:
        from vectorstore.embedder import Embedder
        self._embedder = Embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._embedder.embed_batch(texts, show_progress=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._embedder.embed(text)
        return vec.tolist()


def _get_evaluator_llm(evaluator: str = "bedrock") -> LangchainLLMWrapper:
    """Create a RAGAS-compatible LLM wrapper for the evaluator.

    Args:
        evaluator: "groq" or "bedrock". Defaults to bedrock.
    """
    if evaluator == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.0,
            timeout=120,
            max_retries=3,
        )
    elif evaluator == "bedrock":
        from langchain_aws import ChatBedrockConverse
        llm = ChatBedrockConverse(
            model=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
            temperature=0.0,
            max_tokens=2048,
        )
    else:
        raise ValueError(f"Unknown evaluator: {evaluator!r}")

    return LangchainLLMWrapper(llm)


def _get_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """Create a RAGAS-compatible embeddings wrapper using PubMedBERT."""
    return LangchainEmbeddingsWrapper(_PubMedBERTEmbeddings())


# Map backend names to cleaner file names for output artifacts.
_FILE_NAMES: dict[str, str] = {
    "bedrock-llama": "llama",
}


def _generation_path(model: str) -> Path:
    """Standard path for saved generation JSON."""
    name = _FILE_NAMES.get(model, model)
    return EVAL_RESULTS_DIR / f"generated_{name}.json"


def generate_answers(
    model: str = "groq",
    top_k: int = 8,
    questions: list[tuple[str, str, str]] | None = None,
    resume: bool = True,
) -> Path:
    """Run the RAG chain on test questions and save results to JSON.

    Supports resuming from a partial run: if a generation file already
    exists for this model, previously answered questions are skipped.

    Args:
        model: LLM backend for generation ("groq" or "bedrock").
        top_k: Number of chunks to retrieve per query.
        questions: Override test questions. Defaults to TEST_QUESTIONS.
        resume: If True, load existing partial results and continue.

    Returns:
        Path to the saved JSON file.
    """
    questions = questions or TEST_QUESTIONS
    output_path = _generation_path(model)

    # Resume support: load existing results and skip completed questions
    results: list[dict[str, Any]] = []
    done_questions: set[str] = set()
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing.get("top_k") == top_k:
            results = existing["results"]
            done_questions = {r["question"] for r in results}
            print(f"\nResuming: {len(results)} questions already completed, "
                  f"{len(questions) - len(done_questions)} remaining.")

    chain = RAGChain(model=model, top_k=top_k)
    print(f"\nGenerating answers for {len(questions)} questions (model={model}, top_k={top_k})...")

    for i, (question, category, ground_truth) in enumerate(questions, 1):
        if question in done_questions:
            print(f"  [{i}/{len(questions)}] (cached) {question[:60]}")
            continue
        print(f"  [{i}/{len(questions)}] {question[:60]}...")
        start = time.time()
        try:
            result = chain.ask(question)
            elapsed = time.time() - start
            results.append({
                "question": question,
                "category": category,
                "ground_truth": ground_truth,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "latency_sec": round(elapsed, 3),
            })
            print(f"    OK ({elapsed:.1f}s)")
            # Save after each question so progress is never lost
            _save_generation(output_path, model, top_k, len(questions), results)
        except Exception as e:
            error_msg = str(e)
            print(f"    ERROR: {error_msg}")
            # Stop early on rate limit — remaining questions will all fail too
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                print(f"\n  Rate limit hit after {len(results)} questions. Saving partial results.")
                _save_generation(output_path, model, top_k, len(questions), results)
                break

    # Final save
    _save_generation(output_path, model, top_k, len(questions), results)
    print(f"\nGenerated answers saved to {output_path}")
    print(f"  {len(results)}/{len(questions)} questions succeeded")
    return output_path


def _save_generation(
    path: Path, model: str, top_k: int,
    total: int, results: list[dict[str, Any]],
) -> None:
    """Write generation results to JSON."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": model,
        "top_k": top_k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": total,
        "successful": len(results),
        "results": results,
    }
    path.write_text(json.dumps(output, indent=2))


def score_answers(
    model: str = "groq",
    evaluator: str = "bedrock",
    generation_path: Path | None = None,
) -> pd.DataFrame:
    """Load saved generation results and run RAGAS scoring.

    Args:
        model: Which model's generations to load (used for file naming).
        evaluator: LLM to use as RAGAS judge ("groq" or "bedrock").
        generation_path: Override path to generation JSON. Defaults to
            evaluation/results/generated_{model}.json.

    Returns:
        DataFrame with per-question RAGAS scores.
    """
    path = generation_path or _generation_path(model)
    if not path.exists():
        print(f"ERROR: No generation file found at {path}")
        print(f"  Run with --generate-only first: python run_eval.py --generate-only --model {model}")
        return pd.DataFrame()

    print(f"\nLoading generated answers from {path}...")
    data = json.loads(path.read_text())
    gen_results = data["results"]
    gen_model = data["model"]
    gen_top_k = data["top_k"]

    print(f"  Model: {gen_model}, Top-K: {gen_top_k}, Questions: {len(gen_results)}")

    # Build RAGAS samples
    samples: list[SingleTurnSample] = []
    for entry in gen_results:
        samples.append(SingleTurnSample(
            user_input=entry["question"],
            response=entry["answer"],
            retrieved_contexts=entry["contexts"],
            reference=entry["ground_truth"],
        ))

    if not samples:
        print("No samples to evaluate.")
        return pd.DataFrame()

    # Run RAGAS
    print(f"\nRunning RAGAS metrics (evaluator={evaluator})...")
    eval_llm = _get_evaluator_llm(evaluator)
    eval_embeddings = _get_evaluator_embeddings()

    metrics = [
        _Faithfulness(),
        # strictness=1: generate 1 question per answer (not 3).
        # Groq doesn't support n>1 in a single API call.
        _AnswerRelevancy(strictness=1),
        _LLMContextPrecisionWithReference(),
        _LLMContextRecall(),
    ]

    dataset = EvaluationDataset(samples=samples)
    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    # Build results DataFrame
    df = eval_result.to_pandas()

    # Add metadata columns
    df.insert(0, "question", [e["question"] for e in gen_results[:len(df)]])
    df.insert(1, "category", [e["category"] for e in gen_results[:len(df)]])
    df["latency_sec"] = [e["latency_sec"] for e in gen_results[:len(df)]]
    df["model"] = gen_model
    df["evaluator"] = evaluator

    # Save results
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = _FILE_NAMES.get(gen_model, gen_model)
    output_path = EVAL_RESULTS_DIR / f"ragas_{file_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    _print_summary(df, gen_model, evaluator)
    return df


def run_ragas_evaluation(
    model: str = "groq",
    evaluator: str = "bedrock",
    top_k: int = 8,
    questions: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """All-in-one: generate answers then score them with RAGAS.

    Args:
        model: RAG chain LLM backend ("groq" or "bedrock").
        evaluator: LLM to use as RAGAS judge ("groq" or "bedrock").
        top_k: Number of chunks to retrieve per query.
        questions: Override test questions. Defaults to TEST_QUESTIONS.

    Returns:
        DataFrame with per-question RAGAS scores.
    """
    generate_answers(model=model, top_k=top_k, questions=questions)
    return score_answers(model=model, evaluator=evaluator)


def _print_summary(df: pd.DataFrame, model: str, evaluator: str) -> None:
    """Print RAGAS score summary table."""
    print(f"\n{'='*60}")
    print(f"RAGAS Evaluation Summary (model={model}, evaluator={evaluator})")
    print(f"{'='*60}")
    metric_cols = [c for c in df.columns if c in [
        "faithfulness", "answer_relevancy",
        "context_precision", "llm_context_precision_with_reference",
        "LLMContextPrecisionWithReference",
        "context_recall", "llm_context_recall", "LLMContextRecall",
    ]]
    for col in metric_cols:
        mean_val = df[col].mean()
        print(f"  {col:45s} {mean_val:.4f}")
    avg_latency = df["latency_sec"].mean()
    print(f"  {'avg_latency_sec':45s} {avg_latency:.1f}")
    print(f"  {'questions_evaluated':45s} {len(df)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_ragas_evaluation(model="groq", evaluator="bedrock")
