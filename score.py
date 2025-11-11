"""
Evaluation script for legal passage retrieval.

Usage:
    python score.py <predictions.json> <gold_standard.json> [--k K] [--output output.json]
"""

import json
import argparse
import numpy as np
import re
from typing import List, Dict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def preprocess_text(text: str, lower: bool = True) -> str:
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

def load_test_benchmark(benchmark_path: str) -> List[Dict]:
    """Load test benchmark JSON file."""
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    return data.get('tests', data)

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

def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval system performance')
    parser.add_argument('predictions', type=str, help='Path to predictions JSON file')
    parser.add_argument('gold', type=str, help='Path to gold standard JSON file')
    parser.add_argument('--k', type=int, default=10, help='Number of top results to consider (default: 10)')
    parser.add_argument('--output', type=str, default=None, help='Path to save results JSON file')
    
    args = parser.parse_args()
    
    # Loading predictions and gold standard
    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loading gold standard from {args.gold}...")
    gold_standard = load_test_benchmark(args.gold)
    
    # Evaluating
    print(f"Evaluating with k={args.k}...")
    results = evaluate_retrieval(predictions, gold_standard, k=args.k)
    
    # Printing results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        if metric != 'num_examples':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
