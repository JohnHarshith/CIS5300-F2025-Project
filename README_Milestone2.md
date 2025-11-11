# Milestone 2: Evaluation Script and Baselines

This directory contains the implementation of evaluation metrics and baseline systems for the legal passage retrieval task.

## Files

### Evaluation Script
- `score.py`: Evaluation script that computes Exact Match, Span F1, Recall@10, and nDCG@10
- `scoring.md`: Detailed documentation of evaluation metrics

### Baselines
- `simple-baseline.py`: TF-IDF retrieval baseline
- `strong-baseline.py`: BM25 retrieval baseline
- `simple-baseline.md`: Documentation for TF-IDF baseline
- `strong-baseline.md`: Documentation for BM25 baseline

### Notebook
- `milestone2.ipynb`: Google Colab notebook with all code and examples

## Quick Start

### 1. Install Dependencies

```bash
pip install rank-bm25 nltk scikit-learn numpy pandas
```

### 2. Run Simple Baseline (TF-IDF)

```bash
python simple-baseline.py \
    --corpus data_extracted/corpus \
    --benchmark data_extracted/benchmarks/contractnli.json \
    --output output/tfidf_predictions.json \
    --k 10
```

### 3. Run Strong Baseline (BM25)

```bash
python strong-baseline.py \
    --corpus data_extracted/corpus \
    --benchmark data_extracted/benchmarks/contractnli.json \
    --output output/bm25_predictions.json \
    --k 10
```

### 4. Evaluate Results

```bash
# Evaluate TF-IDF baseline
python score.py output/tfidf_predictions.json data_extracted/benchmarks/contractnli.json --k 10

# Evaluate BM25 baseline
python score.py output/bm25_predictions.json data_extracted/benchmarks/contractnli.json --k 10 --output output/bm25_results.json
```

## Using Google Colab

1. Upload `milestone2.ipynb` to Google Colab
2. Upload our data files (benchmarks and corpus)
3. Update the paths in the configuration cell
4. Run all cells sequentially
5. Results will be saved in the `output/` directory

## Data Format

### Input: Benchmark JSON
```json
{
  "tests": [
    {
      "query": "Does the Agreement indicate that...",
      "snippets": [
        {
          "file_path": "contractnli/file.txt",
          "span": [11461, 11963],
          "answer": "Any and all proprietary rights..."
        }
      ]
    }
  ]
}
```

### Output: Predictions JSON
```json
[
  {
    "query": "Does the Agreement indicate that...",
    "retrieved_passages": [
      "passage text 1...",
      "passage text 2...",
      ...
    ]
  }
]
```

## Evaluation Metrics

- **Exact Match**: Top-1 retrieved passage exactly matches gold answer
- **Span F1**: Token-level F1 score between predicted and gold passages
- **Recall@10**: Fraction of gold passages found in top-10 results
- **nDCG@10**: Normalized discounted cumulative gain at rank 10

See `scoring.md` for detailed descriptions and formulas.

## Notes

- The baselines process documents by chunking them into 500-word segments with 50% overlap
- For large corpora, one may need to limit the number of passages indexed (refer notebook)
- Results will vary based on the test set size and corpus characteristics
