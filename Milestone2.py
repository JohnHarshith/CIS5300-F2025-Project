# Installing required packages
!pip install rank-bm25 nltk scikit-learn numpy pandas -q

import json
import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
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

def load_test_benchmark(benchmark_path: str) -> List[Dict]:
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

def load_training_data(csv_path: str, n_samples: int = None) -> pd.DataFrame:
    """Load training data from CSV."""
    df = pd.read_csv(csv_path)
    if n_samples:
        df = df.head(n_samples)
    return df

def preprocess_text(text: str, lower: bool = True, remove_stopwords: bool = False) -> str:
    """Preprocess text for retrieval."""
    if lower:
        text = text.lower()
    # Removing extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Tokenize text."""
    text = preprocess_text(text, lower=True)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
    else:
        tokens = [t for t in tokens if t.isalnum()]
    return tokens

def exact_match(predicted: str, gold: str) -> bool:
    """Check if predicted text exactly matches gold text."""
    return predicted.strip().lower() == gold.strip().lower()

def span_f1(predicted: str, gold: str) -> float:
    """
    Compute F1 score based on token overlap between predicted and gold spans.

    F1 = 2 * (precision * recall) / (precision + recall)
    precision = |predicted_tokens ∩ gold_tokens| / |predicted_tokens|
    recall = |predicted_tokens ∩ gold_tokens| / |gold_tokens|
    """
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

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def recall_at_k(retrieved_passages: List[str], gold_passages: List[str], k: int = 10) -> float:
    """
    Compute Recall@K: fraction of gold passages found in top K retrieved passages.

    Recall@K = |{gold_passages} ∩ {top_k_retrieved}| / |{gold_passages}|
    """
    if len(gold_passages) == 0:
        return 1.0 if len(retrieved_passages) == 0 else 0.0

    top_k = retrieved_passages[:k]

    # Normalizing and checking for matches
    gold_normalized = [preprocess_text(g).strip() for g in gold_passages]
    retrieved_normalized = [preprocess_text(r).strip() for r in top_k]

    # Counting how many gold passages appear in top K
    matches = 0
    for gold in gold_normalized:
        # Checking for exact or near-exact match
        for ret in retrieved_normalized:
            if gold in ret or ret in gold or gold == ret:
                matches += 1
                break

    return matches / len(gold_passages)

def ndcg_at_k(retrieved_passages: List[str], gold_passages: List[str], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) at K.

    DCG@K = sum(rel_i / log2(i+1)) for i in [1, K]
    nDCG@K = DCG@K / IDCG@K

    where rel_i = 1 if passage i is relevant (in gold), 0 otherwise
    """
    if len(gold_passages) == 0:
        return 1.0 if len(retrieved_passages) == 0 else 0.0

    top_k = retrieved_passages[:k]

    # Normalizing passages
    gold_normalized = [preprocess_text(g).strip() for g in gold_passages]
    retrieved_normalized = [preprocess_text(r).strip() for r in top_k]

    # Computing relevance scores
    relevances = []
    for ret in retrieved_normalized:
        is_relevant = False
        for gold in gold_normalized:
            if gold in ret or ret in gold or gold == ret:
                is_relevant = True
                break
        relevances.append(1.0 if is_relevant else 0.0)

    # Computing DCG@K
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

    # Computing IDCG@K (ideal: all relevant items first)
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
    """
    Evaluate retrieval system performance.

    Args:
        predictions: List of dicts with 'query', 'retrieved_passages' (list of top passages)
        gold_standard: List of dicts with 'query', 'snippets' (list with 'answer' field)
        k: Number of top results to consider for Recall@K and nDCG@K

    Returns:
        Dictionary with evaluation metrics
    """
    if len(predictions) != len(gold_standard):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(gold_standard)} gold examples")

    exact_matches = []
    span_f1_scores = []
    recall_at_k_scores = []
    ndcg_at_k_scores = []

    for pred, gold in zip(predictions, gold_standard):
        # Getting gold answers
        gold_answers = [snippet['answer'] for snippet in gold.get('snippets', [])]

        if len(gold_answers) == 0:
            continue

        # Getting retrieved passages
        retrieved = pred.get('retrieved_passages', [])

        if len(retrieved) == 0:
            exact_matches.append(0.0)
            span_f1_scores.append(0.0)
            recall_at_k_scores.append(0.0)
            ndcg_at_k_scores.append(0.0)
            continue

        # Exact Match: checking if top-1 matches any gold answer
        top_pred = retrieved[0] if retrieved else ""
        em = any(exact_match(top_pred, gold_ans) for gold_ans in gold_answers)
        exact_matches.append(1.0 if em else 0.0)

        # Span F1: average F1 with best matching gold answer
        best_f1 = max([span_f1(top_pred, gold_ans) for gold_ans in gold_answers])
        span_f1_scores.append(best_f1)

        # Recall@K
        rec_k = recall_at_k(retrieved, gold_answers, k=k)
        recall_at_k_scores.append(rec_k)

        # nDCG@K
        ndcg_k = ndcg_at_k(retrieved, gold_answers, k=k)
        ndcg_at_k_scores.append(ndcg_k)

    return {
        'exact_match': np.mean(exact_matches),
        'span_f1': np.mean(span_f1_scores),
        f'recall@{k}': np.mean(recall_at_k_scores),
        f'ndcg@{k}': np.mean(ndcg_at_k_scores),
        'num_examples': len(predictions)
    }

def load_predictions(predictions_path: str) -> List[Dict]:
    """Load predictions from JSON file."""
    with open(predictions_path, 'r') as f:
        return json.load(f)

def load_gold_standard(gold_path: str) -> List[Dict]:
    """Load gold standard from benchmark JSON file."""
    return load_test_benchmark(gold_path)

class TFIDFRetriever:
    """
    Simple TF-IDF based retrieval baseline.
    """

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.corpus_texts = []
        self.corpus_vectors = None

    def index(self, passages: List[str]):
        """Index a collection of passages."""
        self.corpus_texts = passages
        self.corpus_vectors = self.vectorizer.fit_transform(passages)

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k passages for a query."""
        if self.corpus_vectors is None:
            raise ValueError("Index must be built before retrieval")

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.corpus_vectors).flatten()

        top_indices = np.argsort(similarities)[::-1][:k]

        results = [(self.corpus_texts[i], similarities[i]) for i in top_indices]
        return results

class BM25Retriever:
    """
    BM25 (Best Matching 25) retrieval baseline.
    BM25 is a ranking function used to estimate the relevance of documents.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus_texts = []

    def index(self, passages: List[str]):
        """Index a collection of passages."""
        self.corpus_texts = passages
        # Tokenizing passages
        tokenized_passages = [tokenize(p, remove_stopwords=True) for p in passages]
        self.bm25 = BM25Okapi(tokenized_passages, k1=self.k1, b=self.b)

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k passages for a query."""
        if self.bm25 is None:
            raise ValueError("Index must be built before retrieval")

        query_tokens = tokenize(query, remove_stopwords=True)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:k]

        results = [(self.corpus_texts[i], scores[i]) for i in top_indices]
        return results

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size // 2):
        # 50% overlap
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks if chunks else [text]

def prepare_corpus_from_benchmark(
    benchmark_path: str,
    corpus_dir: str,
    chunk_size: int = 500
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Prepare corpus by chunking documents and mapping queries to gold passages.

    Returns:
        corpus_passages: List of all passages in corpus
        query_to_gold: Mapping from query to list of gold answer passages
    """
    tests = load_test_benchmark(benchmark_path)

    corpus_passages = []
    passage_to_idx = {}
    query_to_gold = {}
    processed_files = set()

    # Processing each test case
    for test in tests:
        query = test['query']
        gold_answers = []

        for snippet in test.get('snippets', []):
            file_path = snippet['file_path']
            span = snippet['span']
            answer = snippet['answer']

            # Constructing full path
            full_path = os.path.join(corpus_dir, file_path)

            if os.path.exists(full_path):
                # Loading document and extracting passage
                doc_text = load_corpus_file(full_path)
                gold_passage = get_passage_from_span(doc_text, span)
                gold_answers.append(gold_passage)

                # Chunk the document and adding to corpus (only once per file)
                if full_path not in processed_files:
                    # Splitting document into chunks
                    chunks = chunk_text(doc_text, chunk_size)
                    corpus_passages.extend(chunks)
                    processed_files.add(full_path)

        if gold_answers:
            query_to_gold[query] = gold_answers

    return corpus_passages, query_to_gold

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
TRAINING_DATA_PATH = '/content/unzipped/data_extracted/top_100000_data.csv'
OUTPUT_DIR = '/content/output'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Preparing corpus and test data
print("Preparing corpus and test data...")
print(f"Loading benchmark from: {BENCHMARK_PATH}")
print(f"Loading corpus from: {CORPUS_DIR}")

try:
    corpus_passages, query_to_gold = prepare_corpus_from_benchmark(
        BENCHMARK_PATH,
        CORPUS_DIR,
        chunk_size=500
    )
    print(f"Corpus size: {len(corpus_passages)} passages")
    print(f"Number of test queries: {len(query_to_gold)}")
except Exception as e:
    print(f"Error preparing corpus: {e}")
    print("Please ensure the paths are correct and files are uploaded to Colab")
    corpus_passages = []
    query_to_gold = {}

# Loading test benchmarks
try:
    tests = load_test_benchmark(BENCHMARK_PATH)
    print(f"Loaded {len(tests)} test cases")
    print(f"Sample query: {tests[0]['query'][:100]}...")
except Exception as e:
    print(f"Error loading test benchmark: {e}")
    tests = []

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

print("Simple Baseline: TF-IDF Retrieval")

if len(corpus_passages) > 0 and len(tests) > 0:
    # Initializing TF-IDF retriever
    tfidf_retriever = TFIDFRetriever(max_features=5000)

    # Indexing corpus (limiting to first 10000 passages)
    max_passages = min(10000, len(corpus_passages))
    print(f"Indexing {max_passages} passages...")
    tfidf_retriever.index(corpus_passages[:max_passages])

    # Generating predictions (processing first 100 for demo)
    num_tests = min(100, len(tests))
    print(f"Processing {num_tests} test queries...")
    tfidf_predictions = []
    for i, test in enumerate(tests[:num_tests]):
        if i % 10 == 0:
            print(f"  Processed {i}/{num_tests} queries...")
        query = test['query']
        results = tfidf_retriever.retrieve(query, k=10)
        retrieved_passages = [passage for passage, score in results]
        tfidf_predictions.append({
            'query': query,
            'retrieved_passages': retrieved_passages
        })

    # Evaluating
    tfidf_results = evaluate_retrieval(tfidf_predictions, tests[:num_tests], k=10)
    print("TF-IDF Baseline Results:")
    for metric, value in tfidf_results.items():
        print(f"  {metric}: {value:.4f}")

    # Saving predictions
    with open(f'{OUTPUT_DIR}/tfidf_predictions.json', 'w') as f:
        json.dump(tfidf_predictions, f, indent=2)
    print(f"Predictions saved to {OUTPUT_DIR}/tfidf_predictions.json")
else:
    print("Skipping TF-IDF baseline: corpus or tests not loaded")
    tfidf_results = {}

print("Strong Baseline: BM25 Retrieval")

if len(corpus_passages) > 0 and len(tests) > 0:
    # Initializing BM25 retriever
    bm25_retriever = BM25Retriever(k1=1.5, b=0.75)

    # Indexing corpus
    max_passages = min(10000, len(corpus_passages))
    print(f"Indexing {max_passages} passages...")
    bm25_retriever.index(corpus_passages[:max_passages])

    # Generating predictions
    num_tests = min(100, len(tests))
    print(f"Processing {num_tests} test queries...")
    bm25_predictions = []
    for i, test in enumerate(tests[:num_tests]):
        if i % 10 == 0:
            print(f"  Processed {i}/{num_tests} queries...")
        query = test['query']
        results = bm25_retriever.retrieve(query, k=10)
        retrieved_passages = [passage for passage, score in results]
        bm25_predictions.append({
            'query': query,
            'retrieved_passages': retrieved_passages
        })

    # Evaluating
    bm25_results = evaluate_retrieval(bm25_predictions, tests[:num_tests], k=10)
    print("BM25 Baseline Results:")
    for metric, value in bm25_results.items():
        print(f"  {metric}: {value:.4f}")

    # Saving predictions
    with open(f'{OUTPUT_DIR}/bm25_predictions.json', 'w') as f:
        json.dump(bm25_predictions, f, indent=2)
    print(f"Predictions saved to {OUTPUT_DIR}/bm25_predictions.json")
else:
    print("Skipping BM25 baseline: corpus or tests not loaded")
    bm25_results = {}

if tfidf_results and bm25_results:
    print("=" * 60)
    print("SUMMARY OF BASELINE PERFORMANCE")
    print("=" * 60)
    print(f"{'Metric':<15}{'TF-IDF':>10}{'BM25':>10}")
    print("-" * 60)
    print(f"{'Exact Match':<15}{tfidf_results.get('exact_match', 0):>10.4f}{bm25_results.get('exact_match', 0):>10.4f}")
    print(f"{'Span F1':<15}{tfidf_results.get('span_f1', 0):>10.4f}{bm25_results.get('span_f1', 0):>10.4f}")
    print(f"{'Recall@10':<15}{tfidf_results.get('recall@10', 0):>10.4f}{bm25_results.get('recall@10', 0):>10.4f}")
    print(f"{'nDCG@10':<15}{tfidf_results.get('ndcg@10', 0):>10.4f}{bm25_results.get('ndcg@10', 0):>10.4f}")
    print("=" * 60)

    summary = {
        'tfidf': tfidf_results,
        'bm25': bm25_results
    }
    with open(f'{OUTPUT_DIR}/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results summary saved to {OUTPUT_DIR}/results_summary.json")
else:
    print("Results not available. Please run the baseline cells above first.")
