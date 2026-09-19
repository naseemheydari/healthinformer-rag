"""Entry point: ask a health question via the RAG pipeline.

Usage:
    python run_pipeline.py "What are the warning signs of diabetes?"
    python run_pipeline.py "How is hypertension treated?" --model bedrock
    python run_pipeline.py "What causes asthma?" --top-k 5
"""

import argparse
import sys
import time

from pipeline.rag_chain import RAGChain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask HealthInformer a health question",
    )
    parser.add_argument(
        "question",
        type=str,
        help="The health question to ask",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="groq",
        choices=["groq", "bedrock", "bedrock-llama"],
        help="LLM backend to use (default: groq)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: from settings)",
    )
    args = parser.parse_args()

    # Build the chain
    kwargs = {"model": args.model}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    chain = RAGChain(**kwargs)

    # Ask
    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"Model:    {args.model}")
    print(f"{'='*60}\n")

    start = time.time()
    result = chain.ask(args.question)
    elapsed = time.time() - start

    # Print answer
    print(result["answer"])

    # Print timing
    print(f"\n{'─'*60}")
    print(f"Response time: {elapsed:.1f}s | Model: {result['model']} | "
          f"Sources used: {len(result['sources'])}")


if __name__ == "__main__":
    main()
