"""Entry point: run all evaluations and print a final summary table.

Supports a split-phase workflow so generation and evaluation can be run separately:
  1. --generate-only: run RAG chain, save answers to JSON (uses generator tokens)
  2. --evaluate-only: load saved JSON, score with RAGAS (uses evaluator tokens)


Usage:
    python run_eval.py                                    # All-in-one (generate + evaluate)
    python run_eval.py --generate-only --model groq       # Generate only, save JSON
    python run_eval.py --generate-only --model bedrock    # Generate with Bedrock
    python run_eval.py --evaluate-only --evaluator bedrock  # Score saved results
    python run_eval.py --ragas-only                       # RAGAS only (all-in-one)
    python run_eval.py --model groq --top-k 8             # Override retrieval depth
    python run_eval.py --pubmedqa-only                    # Only PubMedQA benchmark
    python run_eval.py --spot-check-only                  # Only spot check
"""

import argparse
import sys

import pandas as pd

from config.settings import EVAL_RESULTS_DIR


# Proposal thresholds 
THRESHOLDS: dict[str, tuple[str, float]] = {
    "faithfulness":     ("RAGAS Faithfulness",           0.8),
    "answer_relevancy": ("RAGAS Answer Relevancy",       0.7),
    "context_precision": ("RAGAS Context Precision",     0.7),
    "llm_context_precision_with_reference": ("RAGAS Context Precision", 0.7),
    "LLMContextPrecisionWithReference": ("RAGAS Context Precision", 0.7),
    "context_recall":   ("RAGAS Context Recall",         0.6),
    "llm_context_recall": ("RAGAS Context Recall",       0.6),
    "LLMContextRecall": ("RAGAS Context Recall",         0.6),
}


def _print_threshold_table(ragas_df: pd.DataFrame | None, pubmedqa_df: pd.DataFrame | None) -> None:
    """Print a summary table comparing results to proposal thresholds."""
    print(f"\n{'='*70}")
    print(f"{'METRIC':<40} {'RESULT':>8} {'TARGET':>8} {'STATUS':>8}")
    print(f"{'-'*70}")

    if ragas_df is not None and not ragas_df.empty:
        seen_labels: set[str] = set()
        for col, (label, target) in THRESHOLDS.items():
            if col in ragas_df.columns and label not in seen_labels:
                val = ragas_df[col].mean()
                status = "PASS" if val >= target else "MISS"
                print(f"  {label:<38} {val:>7.3f} {target:>7.1f}  {'  ' + status}")
                seen_labels.add(label)

        # Latency
        avg_lat = ragas_df["latency_sec"].mean()
        lat_status = "PASS" if avg_lat < 10.0 else "MISS"
        print(f"  {'Response Latency (sec)':<38} {avg_lat:>7.1f} {'<10.0':>8}  {'  ' + lat_status}")

    if pubmedqa_df is not None and not pubmedqa_df.empty:
        accuracy = pubmedqa_df["correct"].mean()
        print(f"  {'PubMedQA Accuracy':<38} {accuracy:>7.1%} {'78.0%':>8}  {'  context'}")

    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HealthInformer evaluations")
    parser.add_argument("--model", type=str, default="groq",
                        choices=["groq", "bedrock", "bedrock-llama"],
                        help="LLM backend for answer generation (default: groq)")
    parser.add_argument("--evaluator", type=str, default="bedrock",
                        choices=["groq", "bedrock"],
                        help="LLM backend for RAGAS evaluation (default: bedrock)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of chunks to retrieve per question (default: 8)")

    # Phase flags
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate answers and save to JSON (no scoring)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Score previously generated answers (no generation)")

    # Eval-type flags
    parser.add_argument("--ragas-only", action="store_true",
                        help="Run only RAGAS evaluation")
    parser.add_argument("--pubmedqa-only", action="store_true",
                        help="Run only PubMedQA benchmark")
    parser.add_argument("--spot-check-only", action="store_true",
                        help="Run only spot check")
    parser.add_argument("--pubmedqa-max", type=int, default=500,
                        help="Max PubMedQA questions (default: 500)")
    parser.add_argument("--spot-check-size", type=int, default=50,
                        help="Spot check sample size (default: 50)")
    args = parser.parse_args()

    if args.generate_only and args.evaluate_only:
        print("ERROR: Cannot use --generate-only and --evaluate-only together.")
        sys.exit(1)

    # Determine which evals to run
    run_all = not (args.ragas_only or args.pubmedqa_only or args.spot_check_only
                   or args.generate_only or args.evaluate_only)
    run_ragas = run_all or args.ragas_only or args.generate_only or args.evaluate_only
    run_pubmedqa = run_all or args.pubmedqa_only
    run_spot = run_all or args.spot_check_only

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ragas_df = None
    pubmedqa_df = None

    # ── RAGAS ───────────────────────────────────────────────────────────
    if run_ragas:
        from evaluation.ragas_eval import generate_answers, score_answers, run_ragas_evaluation

        if args.generate_only:
            # Phase 1 only: generate and save
            print("\n" + "="*60)
            print("GENERATING ANSWERS (no scoring)")
            print("="*60)
            try:
                generate_answers(model=args.model, top_k=args.top_k)
            except Exception as e:
                print(f"Generation failed: {e}")

        elif args.evaluate_only:
            # Phase 2 only: load and score
            print("\n" + "="*60)
            print("SCORING SAVED ANSWERS (no generation)")
            print("="*60)
            try:
                ragas_df = score_answers(model=args.model, evaluator=args.evaluator)
            except Exception as e:
                print(f"Evaluation failed: {e}")

        else:
            # All-in-one
            print("\n" + "="*60)
            print("RUNNING RAGAS EVALUATION")
            print("="*60)
            try:
                ragas_df = run_ragas_evaluation(
                    model=args.model, evaluator=args.evaluator, top_k=args.top_k,
                )
            except Exception as e:
                print(f"RAGAS evaluation failed: {e}")

    # ── PubMedQA ────────────────────────────────────────────────────────
    if run_pubmedqa:
        print("\n" + "="*60)
        print("RUNNING PUBMEDQA BENCHMARK")
        print("="*60)
        from evaluation.pubmedqa_bench import run_pubmedqa_benchmark
        try:
            pubmedqa_df = run_pubmedqa_benchmark(
                model=args.model, max_questions=args.pubmedqa_max,
            )
        except Exception as e:
            print(f"PubMedQA benchmark failed: {e}")

    # ── Spot check ──────────────────────────────────────────────────────
    if run_spot:
        print("\n" + "="*60)
        print("RUNNING SPOT CHECK")
        print("="*60)
        from evaluation.spot_check import run_spot_check
        try:
            run_spot_check(
                model=args.model, sample_size=args.spot_check_size,
            )
        except Exception as e:
            print(f"Spot check failed: {e}")

    # ── Final summary ───────────────────────────────────────────────────
    if not args.generate_only:
        _print_threshold_table(ragas_df, pubmedqa_df)


if __name__ == "__main__":
    main()
