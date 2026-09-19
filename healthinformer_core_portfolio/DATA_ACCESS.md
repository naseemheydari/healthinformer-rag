# Data Access Statement

This document describes how to access all data used in the HealthInformer project. Raw data is not redistributed in this repository, instead we provide scripts to reproduce it from public sources.

## PubMed Abstracts (Primary Corpus)

- **Source:** [PubMed E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- **Access:** Public, free, no API key required. NCBI asks that users provide a valid email address (set via `PUBMED_EMAIL` in `.env`).
- **License:** PubMed data is in the public domain (U.S. government work) and available for non-commercial research use.
- **Volume:** 124 search queries across 36 health topic categories, yielding 21,926 unique articles and 40,096 text chunks.

To reproduce the corpus:

```bash
python ingest_data.py
```

This fetches abstracts via the E-utilities API and chunks them using sentence-boundary splitting (~512 tokens per chunk with overlap). Output is saved to `data/raw/` (gitignored). The full ingestion takes approximately 30-45 minutes depending on network speed.

Search queries are defined in `config/constants.py`.

## PubMedQA Dataset (Evaluation Benchmark)

- **Source:** [PubMedQA](https://pubmedqa.github.io/) (Jin et al., 2019)
- **Access:** Open-source, downloaded automatically via the Hugging Face `datasets` library.
- **License:** MIT License
- **Usage:** 500 expert-labeled yes/no/maybe questions used for benchmark evaluation in `evaluation/pubmedqa_bench.py`.

No manual download is needed. The evaluation script loads it at runtime:

```python
from datasets import load_dataset
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
```

## Derived Data (Not Redistributed)

The following are generated locally and excluded from the repository via `.gitignore`:

| Artifact | Location | How to Reproduce |
|---|---|---|
| Raw articles and chunks | `data/raw/` | `python ingest_data.py` |
| ChromaDB vector store | `vectorstore/chroma_db/` | `python build_vectorstore.py` |

## Evaluation Results (Included)

Final evaluation outputs are included in `evaluation/results/` as JSON and CSV files. These are the artifacts used in the report and do not contain any protected or proprietary data.
