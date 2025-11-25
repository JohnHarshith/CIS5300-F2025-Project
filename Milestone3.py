"""
# Milestone 3: Hybrid BM25 + Sentence-BERT Retrieval

This notebook implements the hybrid retrieval extension that combines:
1. **BM25 (lexical retrieval)** - Captures exact term matches and rare legal keywords
2. **Sentence-BERT (semantic retrieval)** - Captures paraphrases and semantic similarity
3. **Fusion strategies** - Weighted combination/Reciprocal Rank Fusion (RRF)

The hybrid approach improves over individual baselines by leveraging complementary signals.

## Setup and Installation
"""

# Installing required packages
!pip install rank-bm25 nltk scikit-learn numpy pandas sentence-transformers torch -q

import os
import re
import nltk
import json
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Downloading stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

"""## Utilities Functions"""

def preprocess_text(text: str, lower: bool = True) -> str:
    """Preprocess text for retrieval."""
    if lower:
        text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Tokenize text."""
    text = preprocess_text(text, lower=True)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
    else:
        tokens = [t for t in tokens if t.isalnum()]
    return tokens

def load_test_benchmark(benchmark_path: str) -> List[dict]:
    """Load test benchmark JSON file."""
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    return data.get('tests', data)

def load_corpus_file(corpus_path: str) -> str:
    """Load a corpus text file."""
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def get_passage_from_span(text: str, span: List[int]) -> str:
    """Extract passage from text using character span."""
    start, end = span
    return text[start:end]

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size // 2)
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks if chunks else [text]

def prepare_corpus_from_benchmark(
    benchmark_path: str,
    corpus_dir: str,
    chunk_size: int = 500) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Prepare corpus by chunking documents and mapping queries to gold passages.

    Returns:
        corpus_passages: List of all passages in corpus
        query_to_gold: Mapping from query to list of gold answer passages
    """
    tests = load_test_benchmark(benchmark_path)
    corpus_passages = []
    query_to_gold = {}
    processed_files = set()

    for test in tests:
        query = test['query']
        gold_answers = []

        for snippet in test.get('snippets', []):
            file_path = snippet['file_path']
            span = snippet['span']

            full_path = os.path.join(corpus_dir, file_path)
            if os.path.exists(full_path):
                doc_text = load_corpus_file(full_path)
                gold_passage = get_passage_from_span(doc_text, span)
                gold_answers.append(gold_passage)

                if full_path not in processed_files:
                    chunks = chunk_text(doc_text, chunk_size)
                    corpus_passages.extend(chunks)
                    processed_files.add(full_path)

        if gold_answers:
            query_to_gold[query] = gold_answers

    return corpus_passages, query_to_gold

"""## Hybrid Retriever Implementation"""

class SentenceBERTEncoder:
    """Encodes passages/queries with Sentence-BERT."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus_texts: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None

    def index(self, passages: List[str]):
        """Encode and store corpus passages."""
        if not passages:
            raise ValueError("Cannot index empty passage list.")
        self.corpus_texts = passages
        embeddings = self.model.encode(
            passages,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        if not self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        self.corpus_embeddings = embeddings.astype(np.float32)

    def score(self, query: str) -> np.ndarray:
        """Score query against all corpus passages."""
        if self.corpus_embeddings is None:
            raise ValueError("Index before scoring.")
        query_emb = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )[0]
        if not self.normalize_embeddings:
            denom = np.linalg.norm(query_emb) + 1e-12
            query_emb = query_emb / denom
        scores = np.dot(self.corpus_embeddings, query_emb)
        return scores

class HybridRetriever:
    """
    Combines BM25 and Sentence-BERT scores via fusion.
    Supports two fusion methods: weighted combination and Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        dense_batch_size: int = 32,
        fusion_method: str = "weighted",
        bm25_weight: float = 0.55,
        dense_weight: float = 0.45,
        rrf_k: int = 60,
        fusion_depth: int = 100,
        device: Optional[str] = None,
    ):
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.fusion_method = fusion_method
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.fusion_depth = fusion_depth

        self.bm25 = None
        self.encoder = SentenceBERTEncoder(
            model_name=model_name,
            batch_size=dense_batch_size,
            device=device,
            normalize_embeddings=True,
        )
        self.corpus_texts: List[str] = []
        self.tokenized_passages: List[List[str]] = []

    def index(self, passages: List[str]):
        """Index passages with both BM25 and Sentence-BERT."""
        self.corpus_texts = passages
        self.tokenized_passages = [
            tokenize(p, remove_stopwords=True) for p in passages
        ]
        self.bm25 = BM25Okapi(
            self.tokenized_passages, k1=self.bm25_k1, b=self.bm25_b
        )
        self.encoder.index(passages)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if scores.size == 0:
            return scores
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-9:
            return np.ones_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def _fuse_rrf(
        self, bm25_scores: np.ndarray, dense_scores: np.ndarray
    ) -> Dict[int, float]:
        """Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[int, float] = defaultdict(float)

        bm25_rank = np.argsort(bm25_scores)[::-1][:self.fusion_depth]
        dense_rank = np.argsort(dense_scores)[::-1][:self.fusion_depth]

        for rank, idx in enumerate(bm25_rank):
            rrf_scores[int(idx)] += 1.0 / (self.rrf_k + rank + 1)

        for rank, idx in enumerate(dense_rank):
            rrf_scores[int(idx)] += 1.0 / (self.rrf_k + rank + 1)

        return rrf_scores

    def _fuse_weighted(
        self, bm25_scores: np.ndarray, dense_scores: np.ndarray
    ) -> Dict[int, float]:
        """Weighted combination of normalized scores."""
        bm25_norm = self._normalize_scores(bm25_scores)
        dense_norm = self._normalize_scores(dense_scores)
        combined = (
            self.bm25_weight * bm25_norm + self.dense_weight * dense_norm
        )
        return {int(idx): float(score) for idx, score in enumerate(combined)}

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k passages using hybrid fusion."""
        if self.bm25 is None or self.encoder.corpus_embeddings is None:
            raise ValueError("Retriever must be indexed before retrieval.")

        bm25_scores = self.bm25.get_scores(
            tokenize(query, remove_stopwords=True)
        )
        dense_scores = self.encoder.score(query)

        if self.fusion_method == "rrf":
            fused_scores = self._fuse_rrf(bm25_scores, dense_scores)
        else:
            fused_scores = self._fuse_weighted(bm25_scores, dense_scores)

        ranked = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]
        return [(self.corpus_texts[idx], score) for idx, score in ranked]

"""## Data Preparation"""

from google.colab import files
import zipfile
import io

# Uploading the ZIP file
uploaded = files.upload()

# Extracting the uploaded ZIP file
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(uploaded[filename]), 'r') as zip_ref:
            zip_ref.extractall('unzipped')
        print(f"Extracted {filename} to /content/unzipped")
    else:
        print(f"{filename} is not a ZIP file.")

BENCHMARK_PATH = '/content/unzipped/data_extracted/benchmarks/contractnli.json'
CORPUS_DIR = '/content/unzipped/data_extracted/corpus'
OUTPUT_DIR = '/content/output'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Loading benchmark
print(f"Loading benchmark from {BENCHMARK_PATH}...")
tests = load_test_benchmark(BENCHMARK_PATH)
print(f"Loaded {len(tests)} test cases")

# Preparing corpus
print(f"Preparing corpus from {CORPUS_DIR}...")
corpus_passages, query_to_gold = prepare_corpus_from_benchmark(
    BENCHMARK_PATH,
    CORPUS_DIR,
    chunk_size=500
)
print(f"Corpus size: {len(corpus_passages)} passages")
print(f"Number of test queries: {len(query_to_gold)}")

"""## Running Hybrid Baseline"""

import nltk

# Ensuring NLTK tokenizers are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

print("Hybrid Baseline: BM25 + Sentence-BERT Fusion")

if len(corpus_passages) > 0 and len(tests) > 0:
    # Initializing hybrid retriever with weighted fusion
    # Tuned weights: 0.55 BM25, 0.45 Sentence-BERT
    hybrid_retriever = HybridRetriever(
        fusion_method="weighted",
        bm25_weight=0.55,
        dense_weight=0.45,
        fusion_depth=100,
        model_name="sentence-transformers/all-mpnet-base-v2",
        dense_batch_size=32,
        device=None,  # Auto-detect GPU if available
    )

    # Indexing corpus
    print(f"Indexing {len(corpus_passages)} passages with BM25 and Sentence-BERT...")
    hybrid_retriever.index(corpus_passages)

    # Generating predictions
    num_tests = len(tests) # Processing all queries
    print(f"Processing {num_tests} test queries...")
    hybrid_predictions = []
    for i, test in enumerate(tests[:num_tests]):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_tests} queries...")
        query = test['query']
        results = hybrid_retriever.retrieve(query, k=10)
        retrieved_passages = [passage for passage, score in results]
        hybrid_predictions.append({
            'query': query,
            'retrieved_passages': retrieved_passages
        })

    # Saving predictions
    output_path = f'{OUTPUT_DIR}/hybrid_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(hybrid_predictions, f, indent=2)
    print(f"Predictions saved to {output_path}")
else:
    print("Skipping hybrid baseline: corpus or tests not loaded")
    hybrid_predictions = []

"""## Evaluation"""

# Evaluation functions (from score.py)
def exact_match(predicted: str, gold: str) -> bool:
    """Check if predicted text exactly matches gold text."""
    return predicted.strip().lower() == gold.strip().lower()

def span_f1(predicted: str, gold: str) -> float:
    """Compute F1 score based on token overlap."""
    pred_tokens = set(tokenize(predicted, remove_stopwords=False))
    gold_tokens = set(tokenize(gold, remove_stopwords=False))

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    intersection = pred_tokens & gold_tokens
    precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(intersection) / len(gold_tokens) if len(gold_tokens) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def recall_at_k(retrieved_passages: List[str], gold_passages: List[str], k: int = 10) -> float:
    """Compute Recall@K: fraction of gold passages found in top K."""
    if len(gold_passages) == 0:
        return 1.0 if len(retrieved_passages) == 0 else 0.0

    top_k = retrieved_passages[:k]
    gold_normalized = [preprocess_text(g).strip() for g in gold_passages]
    retrieved_normalized = [preprocess_text(r).strip() for r in top_k]

    matches = 0
    for gold in gold_normalized:
        for ret in retrieved_normalized:
            if gold in ret or ret in gold or gold == ret:
                matches += 1
                break

    return matches / len(gold_passages)

def ndcg_at_k(retrieved_passages: List[str], gold_passages: List[str], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG) at K."""
    if len(gold_passages) == 0:
        return 1.0 if len(retrieved_passages) == 0 else 0.0

    top_k = retrieved_passages[:k]
    gold_normalized = [preprocess_text(g).strip() for g in gold_passages]
    retrieved_normalized = [preprocess_text(r).strip() for r in top_k]

    relevances = []
    for ret in retrieved_normalized:
        is_relevant = False
        for gold in gold_normalized:
            if gold in ret or ret in gold or gold == ret:
                is_relevant = True
                break
        relevances.append(1.0 if is_relevant else 0.0)

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    num_relevant = int(min(sum(relevances), k))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg

def evaluate_retrieval(
    predictions: List[Dict],
    gold_standard: List[Dict],
    k: int = 10
) -> Dict[str, float]:
    """Evaluate retrieval system performance."""
    if len(predictions) != len(gold_standard):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(gold_standard)} gold examples")

    exact_matches = []
    span_f1_scores = []
    recall_at_k_scores = []
    ndcg_at_k_scores = []

    for pred, gold in zip(predictions, gold_standard):
        gold_answers = [snippet['answer'] for snippet in gold.get('snippets', [])]

        if len(gold_answers) == 0:
            continue

        retrieved = pred.get('retrieved_passages', [])

        if len(retrieved) == 0:
            exact_matches.append(0.0)
            span_f1_scores.append(0.0)
            recall_at_k_scores.append(0.0)
            ndcg_at_k_scores.append(0.0)
            continue

        top_pred = retrieved[0] if retrieved else ""
        em = any(exact_match(top_pred, gold_ans) for gold_ans in gold_answers)
        exact_matches.append(1.0 if em else 0.0)

        best_f1 = max([span_f1(top_pred, gold_ans) for gold_ans in gold_answers])
        span_f1_scores.append(best_f1)

        rec_k = recall_at_k(retrieved, gold_answers, k=k)
        recall_at_k_scores.append(rec_k)

        ndcg_k = ndcg_at_k(retrieved, gold_answers, k=k)
        ndcg_at_k_scores.append(ndcg_k)

    return {
        'exact_match': np.mean(exact_matches),
        'span_f1': np.mean(span_f1_scores),
        f'recall@{k}': np.mean(recall_at_k_scores),
        f'ndcg@{k}': np.mean(ndcg_at_k_scores),
        'num_examples': len(predictions)
    }

# Evaluating hybrid baseline predictions
if hybrid_predictions:
    hybrid_results = evaluate_retrieval(hybrid_predictions, tests[:len(hybrid_predictions)], k=10)
    print("=" * 60)
    print("HYBRID BASELINE RESULTS")
    print("=" * 60)
    for metric, value in hybrid_results.items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 60)

    # Save results
    results_path = f'{OUTPUT_DIR}/hybrid_results.json'
    with open(results_path, 'w') as f:
        json.dump(hybrid_results, f, indent=2)
    print(f"Results saved to {results_path}")
else:
    print("No predictions to evaluate. Please run the hybrid baseline cell first.")
