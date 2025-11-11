"""
Strong Baseline: BM25 Retrieval

Usage:
    python strong-baseline.py --corpus <corpus_dir> --benchmark <benchmark.json> 
    --output <output.json> [--k 10] [--k1 1.5] [--b 0.75]
"""

import json
import argparse
import os
import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

    # 50% overlap
    for i in range(0, len(words), chunk_size // 2): 
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
) -> Tuple[List[str], dict]:
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

def main():
    parser = argparse.ArgumentParser(description='BM25 Retrieval Baseline')
    parser.add_argument('--corpus', type=str, required=True, help='Directory containing corpus documents')
    parser.add_argument('--benchmark', type=str, required=True, help='Path to benchmark JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to save predictions JSON file')
    parser.add_argument('--k', type=int, default=10, help='Number of passages to retrieve per query (default: 10)')
    parser.add_argument('--k1', type=float, default=1.5, help='BM25 k1 parameter (default: 1.5)')
    parser.add_argument('--b', type=float, default=0.75, help='BM25 b parameter (default: 0.75)')
    parser.add_argument('--chunk-size', type=int, default=500, help='Size of text chunks in words (default: 500)')
    
    args = parser.parse_args()
    
    # Loading benchmark
    print(f"Loading benchmark from {args.benchmark}...")
    tests = load_test_benchmark(args.benchmark)
    print(f"Loaded {len(tests)} test cases")
    
    # Preparing corpus
    print(f"Preparing corpus from {args.corpus}...")
    corpus_passages, query_to_gold = prepare_corpus_from_benchmark(
        args.benchmark,
        args.corpus,
        chunk_size=args.chunk_size
    )
    print(f"Corpus size: {len(corpus_passages)} passages")
    
    # Initializing BM25 retriever
    print(f"Initializing BM25 retriever with k1={args.k1}, b={args.b}...")
    retriever = BM25Retriever(k1=args.k1, b=args.b)
    
    # Indexing corpus
    print("Indexing corpus...")
    retriever.index(corpus_passages)
    
    # Generating predictions
    print(f"Generating predictions for {len(tests)} queries...")
    predictions = []
    for i, test in enumerate(tests):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(tests)} queries...")
        query = test['query']
        results = retriever.retrieve(query, k=args.k)
        retrieved_passages = [passage for passage, score in results]
        predictions.append({
            'query': query,
            'retrieved_passages': retrieved_passages
        })
    
    # Saving predictions
    print(f"Saving predictions to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Done! Generated predictions for {len(predictions)} queries.")

if __name__ == '__main__':
    main()
   