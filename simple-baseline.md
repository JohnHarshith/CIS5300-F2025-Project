# Simple Baseline: TF-IDF Retrieval

## Overview

The simple baseline uses **TF-IDF (Term Frequency-Inverse Document Frequency)** with cosine similarity for passage retrieval. This is a classic information retrieval approach that measures the importance of terms in documents relative to the entire corpus.

## Approach

### TF-IDF

TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection of documents. It increases proportionally to the number of times a word appears in a document but is offset by the frequency of the word in the corpus.

**Term Frequency (TF):**
```
TF(t, d) = (number of times term t appears in document d) / (total number of terms in d)
```

**Inverse Document Frequency (IDF):**
```
IDF(t, D) = log(total number of documents / number of documents containing term t)
```

**TF-IDF Score:**
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

### Retrieval Process

1. **Indexing:** Build a TF-IDF vectorizer on the corpus of passages
2. **Query Processing:** Transform the query into a TF-IDF vector using the same vectorizer
3. **Similarity Computation:** Compute cosine similarity between query vector and all passage vectors
4. **Ranking:** Return top-K passages with highest similarity scores

### Implementation Details

- **Max Features:** 5000 (limits vocabulary size for efficiency)
- **Stopwords:** English stopwords are removed
- **N-grams:** Uses unigrams and bigrams (1-2 word sequences)
- **Normalization:** Text is lowercased before processing

## Usage

### Python Code

```python
from simple_baseline import TFIDFRetriever

# Initializing retriever
retriever = TFIDFRetriever(max_features=5000)

# Indexing corpus
retriever.index(corpus_passages)

# Retrieving top-K passages for a query
results = retriever.retrieve("Does the Agreement indicate that...", k=10)
retrieved_passages = [passage for passage, score in results]
```

### Command Line

```bash
python simple-baseline.py \
    --corpus data_extracted/corpus \
    --benchmark data_extracted/benchmarks/contractnli.json \
    --output output/tfidf_predictions.json \
    --k 10
```

### Arguments

- `--corpus`: Directory containing corpus documents
- `--benchmark`: Path to benchmark JSON file with test queries
- `--output`: Path to save predictions JSON file
- `--k`: Number of passages to retrieve per query (default: 10)
- `--max-features`: Maximum number of features for TF-IDF (default: 5000)
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
| Span F1     | 0.2009 |
| Recall@10   | 0.3717 |
| nDCG@10     | 0.2757 |

## Limitations

1. **Vocabulary Limitation:** Limited to 5000 features, which may miss important rare terms in legal documents
2. **No Semantic Understanding:** TF-IDF treats words as independent tokens and doesn't understand synonyms or semantic relationships
3. **Sparse Representation:** High-dimensional sparse vectors may not capture document similarity well for long documents
4. **No Context:** Doesn't consider word order or document structure beyond n-grams

## Why This is a Simple Baseline

This baseline is considered "simple" because:
- It uses standard, well-established IR techniques (TF-IDF has been used since the 1970s)
- It requires no training data or machine learning
- It's fast and easy to implement
- It serves as a lower bound for more sophisticated approaches

## Next Steps

For improvements, consider:
- Using BM25 (strong baseline) which is better suited for passage retrieval
- Incorporating semantic embeddings (e.g., BERT, sentence transformers)
- Fine-tuning on legal domain data
- Using cross-encoders for re-ranking
