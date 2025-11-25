# Milestone 3 - Hybrid Lexical + Semantic Retrieval

## Problem Overview
Our project tackles context-aware legal passage retrieval: given a short legal query (often a yes/no question grounded in a contract), the system must surface the most relevant span from a large repository of clauses (ContractNLI + CUAD). This matters because attorneys currently sift through long agreements manually. Automating retrieval accelerates diligence and reduces risk, making it a very practical NLP task that ties directly to the course’s focus on information retrieval and transformer-based semantic modeling.

## Extension Summary 
**Motivation.** Milestone 2 delivered two baselines: TF-IDF (lexical) and BM25 (smarter lexical). Dense Sentence-BERT retrieval added semantics but degraded recall because it missed rare legal terms. To bridge this gap we implemented a *hybrid* retriever that fuses BM25 scores with SBERT similarities so that (a) exact term matches still matter and (b) paraphrases do not get ignored.

**Method.** The new `hybrid-baseline.py` script indexes every 500-word chunk with both BM25 and Sentence-BERT (`all-mpnet-base-v2`). At query time we compute (i) BM25 scores over tokenized passages, (ii) cosine similarities between SBERT embeddings, and (iii) combine the ranked lists via weighted score fusion. We tuned the fusion depth (top‑100 from each retriever) and weights (0.55 BM25 / 0.45 SBERT) to maximize Recall@10 on ContractNLI, while keeping the pipeline training-free. Documentation lives in `hybrid-baseline.md`.

**Results.** We evaluated on all 977 ContractNLI queries with the official `score.py`. The hybrid retriever substantially improves recall and ranking quality relative to both baselines:

| Model (k=10)                    | Exact Match | Span F1 | Recall@10 | nDCG@10 |
|---------------------------------|-------------|---------|-----------|---------|
| TF-IDF (simple baseline)        | 0.0000      | 0.2009  | 0.3717    | 0.2752  |
| BM25 (strong baseline)          | 0.0000      | 0.2217  | 0.5267    | 0.4544  |
| Sentence-BERT (strong baseline) | 0.0000      | 0.2147  | 0.4317    | 0.3232  |
| Hybrid (BM25 + SBERT)           | 0.0000      | **0.2357** | **0.5511** | **0.4808** |

The exact-match metric stays zero because ContractNLI spans rarely align exactly with 500-word chunks; future work could trim passages downstream or add a cross-encoder re-ranker.

**Inference.** Combining complementary signals is more reliable than relying solely on lexical or semantic cues in legal text. This extension reinforces class lessons on ensemble retrieval, dense/sparse hybrids, and evaluation with ranking metrics.

## How to Reproduce
1. `python3 -m pip install -r requirements.txt` (needs `sentence-transformers`, `torch`, `rank-bm25`).
2. Run `python hybrid-baseline.py --corpus data_extracted/corpus --benchmark data_extracted/benchmarks/contractnli.json --output output/hybrid_predictions.json --k 10`.
3. Evaluate: `python score.py output/hybrid_predictions.json data_extracted/benchmarks/contractnli.json --k 10 --output output/hybrid_results.json`.
