"""Central configuration for the HealthInformer pipeline.

All tunable parameters live here — models, thresholds, paths, API settings.
Import from this module instead of hardcoding values elsewhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
CHROMA_DB_DIR: Path = PROJECT_ROOT / "vectorstore" / "chroma_db"
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
EVAL_RESULTS_DIR: Path = PROJECT_ROOT / "evaluation" / "results"

# ── PubMed API ──────────────────────────────────────────────────────────────
PUBMED_EMAIL: str = os.getenv("PUBMED_EMAIL", "")
PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_MAX_RESULTS_PER_QUERY: int = 200
PUBMED_BATCH_SIZE: int = 50  # IDs per efetch request
PUBMED_RATE_LIMIT_DELAY: float = 0.34  # seconds between requests (3/sec limit)

# ── Text Chunking ───────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS: int = 512
CHUNK_OVERLAP_SENTENCES: int = 1  # sentences of overlap between chunks

# ── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
EMBEDDING_DIM: int = 768

# ── ChromaDB ────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "healthinformer"

# ── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 8

# ── LLM Settings ────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS: int = 1024
GROQ_TEMPERATURE: float = 0.3

BEDROCK_MODEL_ID: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
BEDROCK_LLAMA_MODEL_ID: str = "us.meta.llama3-3-70b-instruct-v1:0"
BEDROCK_MAX_TOKENS: int = 1024
BEDROCK_TEMPERATURE: float = 0.3
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

# ── Evaluation ──────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE: int = 50  # for spot-check
RESPONSE_LATENCY_TARGET_SEC: float = 10.0
