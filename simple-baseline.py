"""
Simple Baseline: TF-IDF Retrieval

Usage:
    python simple-baseline.py --corpus <corpus_dir> 
    --benchmark <benchmark.json> --output <output.json> [--k 10]
"""

import json
import argparse
import os
import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def main():
    parser = argparse.ArgumentParser(description='TF-IDF Retrieval Baseline')
    parser.add_argument('--corpus', type=str, required=True, help='Directory containing corpus documents')
    parser.add_argument('--benchmark', type=str, required=True, help='Path to benchmark JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to save predictions JSON file')
    parser.add_argument('--k', type=int, default=10, help='Number of passages to retrieve per query (default: 10)')
    parser.add_argument('--max-features', type=int, default=5000, help='Maximum number of features for TF-IDF (default: 5000)')
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
    
    # Initializing TF-IDF retriever
    print(f"Initializing TF-IDF retriever with max_features={args.max_features}...")
    retriever = TFIDFRetriever(max_features=args.max_features)
    
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
