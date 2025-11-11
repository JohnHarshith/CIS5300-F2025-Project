# Strong Baseline: BM25 Retrieval

## Overview

The strong baseline uses **BM25 (Best Matching 25)**, a probabilistic ranking function used to estimate the relevance of documents to a given search query. BM25 is considered one of the most effective ranking functions for text retrieval and is widely used in information retrieval systems.

## Approach

### BM25 Algorithm

BM25 is a bag-of-words retrieval function that ranks documents based on the query terms appearing in each document. It is an improvement over TF-IDF and is based on probabilistic information retrieval.

**BM25 Score Formula:**
```
BM25(Q, D) = Σ IDF(q_i) × (f(q_i, D) × (k1 + 1)) / (f(q_i, D) + k1 × (1 - b + b × |D| / avgdl))
```

Where:
- `Q` is the query
- `D` is the document
- `q_i` is a query term
- `f(q_i, D)` is the frequency of term `q_i` in document `D`
- `|D|` is the length of document `D` (in words)
- `avgdl` is the average document length in the corpus
- `k1` and `b` are free parameters (typically k1=1.5, b=0.75)
- `IDF(q_i)` is the inverse document frequency of term `q_i`

**Inverse Document Frequency:**
```
IDF(q_i) = log((N - n(q_i) + 0.5) / (n(q_i) + 0.5))
```

Where:
- `N` is the total number of documents
- `n(q_i)` is the number of documents containing term `q_i`

### Key Advantages over TF-IDF

1. **Saturation:** BM25 has a saturation function that prevents very frequent terms from dominating the score
2. **Length Normalization:** Better handles documents of varying lengths through the `b` parameter
3. **Term Frequency Saturation:** The `k1` parameter controls how quickly term frequency saturates
4. **Probabilistic Foundation:** Based on probabilistic models of relevance

### Implementation Details

- **Parameters:**
  - `k1 = 1.5`: Controls term frequency saturation (higher = more weight to term frequency)
  - `b = 0.75`: Controls length normalization (1.0 = full normalization, 0.0 = no normalization)
- **Tokenization:** Uses NLTK word tokenizer with stopword removal
- **Preprocessing:** Lowercase text, remove non-alphanumeric tokens

### Retrieval Process

1. **Indexing:** 
   - Tokenize all passages in the corpus
   - Build BM25 index with tokenized passages
   - Compute average document length
   
2. **Query Processing:**
   - Tokenize query (remove stopwords)
   - Compute BM25 scores for all passages
   
3. **Ranking:** 
   - Sort passages by BM25 score (descending)
   - Return top-K passages

## Usage

### Python Code

```python
from strong_baseline import BM25Retriever

# Initializing retriever
retriever = BM25Retriever(k1=1.5, b=0.75)

# Indexing corpus
retriever.index(corpus_passages)

# Retrieving top-K passages for a query
results = retriever.retrieve("Does the Agreement indicate that...", k=10)
retrieved_passages = [passage for passage, score in results]
```

### Command Line

```bash
python strong-baseline.py \
    --corpus data_extracted/corpus \
    --benchmark data_extracted/benchmarks/contractnli.json \
    --output output/bm25_predictions.json \
    --k 10 \
    --k1 1.5 \
    --b 0.75
```

### Arguments

- `--corpus`: Directory containing corpus documents
- `--benchmark`: Path to benchmark JSON file with test queries
- `--output`: Path to save predictions JSON file
- `--k`: Number of passages to retrieve per query (default: 10)
- `--k1`: BM25 k1 parameter (default: 1.5)
- `--b`: BM25 b parameter (default: 0.75)
- `--chunk-size`: Size of text chunks in words (default: 500)

## Sample Output

```json
[
  {
    "query": "Does the Agreement indicate that the Receiving Party has no rights to Confidential Information?",
    "retrieved_passages": [
      "Any and all proprietary rights, including but not limited to rights to and in inventions...",
      "Confidential Information shall be and remain with the Participants respectively...",
      ...
    ]
  },
  ...
]
```

## Performance

When evaluated on the LegalBench-RAG test set (first 100 examples):

| Metric      | Score  |
|-------------|--------|
| Exact Match | 0.0000 |
| Span F1     | 0.2217 |
| Recall@10   | 0.5267 |
| nDCG@10     | 0.4544 |

## Why This is a Strong Baseline

BM25 is considered a "strong" baseline because:

1. **Proven Effectiveness:** BM25 has been shown to perform well on many IR tasks and is used in production systems (e.g., Elasticsearch, Lucene)
2. **State-of-the-Art for Lexical Retrieval:** For keyword-based retrieval, BM25 is often competitive with more complex neural methods
3. **No Training Required:** Unlike neural models, BM25 doesn't require training data
4. **Fast and Scalable:** Efficient implementation allows for retrieval over large corpora
5. **Well-Studied:** Extensive research has validated its effectiveness across domains

## Comparison with Simple Baseline (TF-IDF)

| Aspect                    | TF-IDF    | BM25             |
|---------------------------|-----------|------------------|
| Term Frequency Saturation | No        | Yes (via k1)     |
| Length Normalization      | Basic     | Advanced (via b) |
| Theoretical Foundation    | Heuristic | Probabilistic    |
| Typical Performance       | Good      | Better           |
| Parameter Tuning          | Minimal   | k1, b parameters |

## Limitations

1. **Lexical Matching Only:** Like TF-IDF, BM25 only matches on exact words and doesn't understand synonyms or semantic relationships
2. **No Semantic Understanding:** Doesn't capture semantic similarity between queries and documents
3. **Parameter Sensitivity:** Performance can vary with different k1 and b values (though defaults work well)
4. **Domain Specificity:** May not capture domain-specific terminology relationships without tuning

## References

1. Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. (Chapter 11: Probabilistic Information Retrieval)

3. Sparck Jones, K., Walker, S., & Robertson, S. E. (2000). A probabilistic model of information retrieval: development and comparative experiments. *Information Processing & Management*, 36(6), 779-808.

## Next Steps for Extensions

Potential improvements beyond BM25:
1. **Dense Retrieval:** Use semantic embeddings (BERT, sentence transformers) for semantic matching
2. **Hybrid Retrieval:** Combine BM25 with dense retrieval (e.g., ColBERT, RAG)
3. **Re-ranking:** Use cross-encoders to re-rank BM25 results
4. **Query Expansion:** Expand queries with synonyms or related terms
5. **Domain Adaptation:** Fine-tune embeddings on legal domain data
