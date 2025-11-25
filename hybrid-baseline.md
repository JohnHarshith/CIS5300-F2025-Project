# Hybrid Baseline: BM25 + Sentence-BERT Fusion

## Overview

Milestone 3 introduces a hybrid retriever that combines lexical BM25 signals with semantic Sentence-BERT embeddings. BM25 excels at matching rare legal terminology, while SBERT captures paraphrases and cross-sentence semantics. The hybrid system fuses both views using Reciprocal Rank Fusion (default) or weighted score averaging.

## Key Ideas

- **Dual Indexing:** Documents are chunked into 500-word passages (50% overlap) and indexed by both BM25 and SBERT.
- **SBERT Model:** Defaults to `sentence-transformers/all-mpnet-base-v2`, but any Sentence-Transformers checkpoint can be supplied.
- **Fusion Strategies:**
  - `weighted` *(default)*: Weighted sum of min-max normalized scores (0.55 BM25 / 0.45 SBERT tuned on ContractNLI).
  - `rrf`: Reciprocal Rank Fusion over each retriever's top-N results.
- **Configurable Depth:** Control how many candidates each retriever contributes (`--fusion-depth`) and the RRF damping constant (`--rrf-k`).

## Usage

```bash
python hybrid-baseline.py \
    --corpus data_extracted/corpus \
    --benchmark data_extracted/benchmarks/contractnli.json \
    --output output/hybrid_predictions.json \
    --k 10 \
    --fusion-method weighted \
    --bm25-weight 0.55 \
    --dense-weight 0.45 \
    --fusion-depth 100 \
    --model-name sentence-transformers/all-mpnet-base-v2
```

### Arguments

- `--corpus`: Directory containing chunked corpus documents.
- `--benchmark`: Benchmark JSON with queries/snippets.
- `--output`: Where to save predictions JSON.
- `--k`: Number of passages per query (default 10).
- `--chunk-size`: Chunk length (default 500 words with 50% overlap).
- `--max-passages`: Optional cap for quick experiments.
- `--fusion-method`: `rrf` (default) or `weighted`.
- `--bm25-weight`, `--dense-weight`: Used when `fusion-method=weighted`.
- `--rrf-k`: Damping constant for RRF (default 60).
- `--fusion-depth`: Top-N per retriever considered (default 50).
- `--model-name`: Sentence-Transformers model (default `all-mpnet-base-v2`).
- `--dense-batch-size`: Encoding batch size for SBERT.
- `--device`: Force `cpu` or `cuda`.

## Sample Output

```json
[
  {
    "query": "Does the Agreement indicate that the Receiving Party has no rights ... ?",
    "retrieved_passages": [
      "Any and all proprietary rights...",
      "Confidential Information shall be and remain ...",
      "... additional passages ..."
    ]
  }
]
```

## Evaluation

After generating predictions:

```bash
python score.py output/hybrid_predictions.json \
    data_extracted/benchmarks/contractnli.json \
    --k 10 \
    --output output/hybrid_results.json
```

This computes Exact Match, Span F1, Recall@10, and nDCG@10 for the hybrid retriever. 
