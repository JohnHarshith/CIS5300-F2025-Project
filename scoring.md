# Evaluation Metrics Documentation

## Overview

This document describes the evaluation metrics used to assess the performance of our legal passage retrieval system. The evaluation script (`score.py`) computes four metrics: Exact Match, Span F1, Recall@10, and nDCG@10.

## Metrics

### 1. Exact Match (EM)

**Definition:** Exact Match measures whether the top retrieved passage exactly matches any of the gold standard answer passages (case-insensitive).

**Formula:**
```
EM = 1 if predicted_text.strip().lower() == gold_text.strip().lower() else 0
```

**Computation:** For each test query, we check if the top-1 retrieved passage exactly matches any gold answer. The metric is the average over all test queries.

**Range:** 0.0 to 1.0 (higher is better)

**Reference:** This is a standard metric used in question answering and retrieval tasks. Referring [Rajpurkar et al. (2016)](https://arxiv.org/abs/1606.05250) for its use in SQuAD.

---

### 2. Span F1

**Definition:** Span F1 measures token-level overlap between the predicted passage and the gold answer passage.

**Formula:**
```
Precision = |predicted_tokens ∩ gold_tokens| / |predicted_tokens|
Recall = |predicted_tokens ∩ gold_tokens| / |gold_tokens|
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Computation:** 
1. Tokenize both predicted and gold passages (case-insensitive, alphanumeric tokens only)
2. Compute intersection of token sets
3. Calculate precision and recall
4. Compute F1 score
5. For queries with multiple gold answers, use the maximum F1 score
6. Average F1 over all test queries

**Range:** 0.0 to 1.0 (higher is better)

**Reference:** This metric is widely used in extractive question answering. Referring [Rajpurkar et al. (2016)](https://arxiv.org/abs/1606.05250) for the SQuAD evaluation metric.

---

### 3. Recall@K

**Definition:** Recall@K measures the fraction of gold answer passages that appear in the top-K retrieved passages.

**Formula:**
```
Recall@K = |{gold_passages} ∩ {top_k_retrieved_passages}| / |{gold_passages}|
```

**Computation:**
1. For each test query, retrieve top-K passages
2. Check how many gold answer passages appear in the top-K (using substring matching)
3. Divide by the total number of gold passages for that query
4. Average over all test queries

**Range:** 0.0 to 1.0 (higher is better)

**Reference:** Recall@K is a standard information retrieval metric. We use K=10. Referring [Manning et al. (2008)](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf) for a comprehensive introduction to information retrieval metrics.

---

### 4. Normalized Discounted Cumulative Gain (nDCG@K)

**Definition:** nDCG@K measures the quality of the ranking by considering the position of relevant items, with higher positions weighted more heavily.

**Formula:**
```
DCG@K = Σ(rel_i / log2(i + 1)) for i in [1, K]
IDCG@K = Σ(1 / log2(i + 1)) for i in [1, min(num_relevant, K)]
nDCG@K = DCG@K / IDCG@K
```

Where:
- `rel_i` = 1 if passage at position i is relevant (matches a gold answer), 0 otherwise
- IDCG is the ideal DCG (all relevant items ranked first)

**Computation:**
1. For each test query, assign relevance scores (1 if passage matches gold, 0 otherwise)
2. Compute DCG@K using the relevance scores
3. Compute IDCG@K (ideal ranking)
4. Normalize: nDCG = DCG / IDCG
5. Average over all test queries

**Range:** 0.0 to 1.0 (higher is better)

**Reference:** nDCG is a standard ranking metric in information retrieval. Referring [Järvelin & Kekäläinen (2002)](https://dl.acm.org/doi/10.1145/582415.582418) for the original definition.

---

## Usage

### Command Line

```bash
python score.py <predictions.json> <gold_standard.json> [--k K] [--output output.json]
```

### Arguments

- `predictions.json`: Path to JSON file containing system predictions
- `gold_standard.json`: Path to JSON file containing gold standard answers
- `--k` (optional): Number of top results to consider for Recall@K and nDCG@K (default: 10)
- `--output` (optional): Path to save evaluation results JSON file

### Input Format

**Predictions JSON format:**
```json
[
  {
    "query": "Does the Agreement indicate that the Receiving Party has no rights to Confidential Information?",
    "retrieved_passages": [
      "passage text 1...",
      "passage text 2...",
      ...
    ]
  },
  ...
]
```

**Gold Standard JSON format:**
```json
{
  "tests": [
    {
      "query": "Does the Agreement indicate that the Receiving Party has no rights to Confidential Information?",
      "snippets": [
        {
          "file_path": "contractnli/file.txt",
          "span": [11461, 11963],
          "answer": "Any and all proprietary rights..."
        }
      ]
    },
    ...
  ]
}
```

### Example Output

```
Evaluation Results:
==========================
exact_match: 0.0000
span_f1: 0.2009
recall@10: 0.3717
ndcg@10: 0.2757
num_examples: 100.0000
==========================
```

### Example Command

```bash
# Evaluate predictions
python score.py output/tfidf_predictions.json data_extracted/benchmarks/contractnli.json --k 10

# Save results to file
python score.py output/bm25_predictions.json data_extracted/benchmarks/contractnli.json --k 10 --output output/bm25_results.json
```

---

## Implementation Details

### Passage Matching

For Recall@K and nDCG@K, we use substring matching to determine if a retrieved passage matches a gold answer:
- Normalize both texts (lowercase, strip whitespace)
- Check if gold passage is a substring of retrieved passage, or vice versa, or if they are exactly equal

This allows for some flexibility in matching while still requiring substantial overlap.

### Tokenization

For Span F1, we use NLTK's word tokenizer and filter to alphanumeric tokens only. Stopwords are not removed for F1 computation to preserve semantic content.

---

## References

1. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Reading Comprehension. *arXiv preprint arXiv:1606.05250*.

2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

3. Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422-446.

4. LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. [LegalBench Website](https://legalbench.readthedocs.io/)
