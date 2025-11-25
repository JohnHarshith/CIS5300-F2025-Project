"""
Milestone 3 Extension: Hybrid BM25 + Sentence-BERT Retrieval

This script implements a fusion retriever that combines lexical BM25 scores
with semantic Sentence-BERT similarities via Reciprocal Rank Fusion (RRF) or
weighted score averaging.

Usage:
    python hybrid-baseline.py \
        --corpus data_extracted/corpus \
        --benchmark data_extracted/benchmarks/contractnli.json \
        --output output/hybrid_predictions.json \
        --k 10 \
        --fusion-method rrf \
        --model-name sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

def preprocess_text(text: str, lower: bool = True) -> str:
    if lower:
        text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    text = preprocess_text(text, lower=True)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
    else:
        tokens = [t for t in tokens if t.isalnum()]
    return tokens

def load_test_benchmark(benchmark_path: str) -> List[dict]:
    with open(benchmark_path, "r") as f:
        data = json.load(f)
    return data.get("tests", data)

def load_corpus_file(corpus_path: str) -> str:
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def get_passage_from_span(text: str, span: List[int]) -> str:
    start, end = span
    return text[start:end]

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    step = max(1, chunk_size // 2)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks if chunks else [text]

def prepare_corpus_from_benchmark(
    benchmark_path: str,
    corpus_dir: str,
    chunk_size: int = 500,
) -> Tuple[List[str], Dict[str, List[str]]]:
    tests = load_test_benchmark(benchmark_path)
    corpus_passages: List[str] = []
    query_to_gold: Dict[str, List[str]] = {}
    processed_files = set()

    for test in tests:
        query = test["query"]
        gold_answers = []

        for snippet in test.get("snippets", []):
            file_path = snippet["file_path"]
            span = snippet["span"]

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

class SentenceBERTEncoder:
    """Encodes passages/queries with Sentence-BERT."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 32,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus_texts: List[str] = []
        self.corpus_embeddings: np.ndarray | None = None

    def index(self, passages: List[str]):
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
        device: str | None = None,
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
        self.corpus_texts = passages
        self.tokenized_passages = [
            tokenize(p, remove_stopwords=True) for p in passages
        ]
        self.bm25 = BM25Okapi(
            self.tokenized_passages, k1=self.bm25_k1, b=self.bm25_b
        )
        self.encoder.index(passages)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
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
        rrf_scores: Dict[int, float] = defaultdict(float)

        bm25_rank = np.argsort(bm25_scores)[::-1][: self.fusion_depth]
        dense_rank = np.argsort(dense_scores)[::-1][: self.fusion_depth]

        for rank, idx in enumerate(bm25_rank):
            rrf_scores[int(idx)] += 1.0 / (self.rrf_k + rank + 1)

        for rank, idx in enumerate(dense_rank):
            rrf_scores[int(idx)] += 1.0 / (self.rrf_k + rank + 1)

        return rrf_scores

    def _fuse_weighted(
        self, bm25_scores: np.ndarray, dense_scores: np.ndarray
    ) -> Dict[int, float]:
        bm25_norm = self._normalize_scores(bm25_scores)
        dense_norm = self._normalize_scores(dense_scores)
        combined = (
            self.bm25_weight * bm25_norm + self.dense_weight * dense_norm
        )
        return {int(idx): float(score) for idx, score in enumerate(combined)}

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
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

def generate_predictions(
    retriever: HybridRetriever,
    tests: List[dict],
    k: int,
) -> List[dict]:
    predictions = []
    for i, test in enumerate(tests):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(tests)} queries...")
        results = retriever.retrieve(test["query"], k=k)
        retrieved_passages = [passage for passage, _ in results]
        predictions.append(
            {"query": test["query"], "retrieved_passages": retrieved_passages}
        )
    return predictions

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + Sentence-BERT Retrieval")
    parser.add_argument(
        "--corpus", type=str, required=True, help="Corpus directory")
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Benchmark JSON path")
    parser.add_argument(
        "--output", type=str, required=True, help="Predictions JSON output")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k passages to return per query",)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Word chunk size for documents",)
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="Optional cap on number of passages to index",)
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="weighted",
        choices=["rrf", "weighted"],
        help="Score fusion strategy",)
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.55,
        help="BM25 weight for weighted fusion",)
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.45,
        help="Dense weight for weighted fusion",)
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant (higher = flatter scores)",)
    parser.add_argument(
        "--fusion-depth",
        type=int,
        default=100,
        help="How many candidates per retriever to include in fusion",)
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-BERT checkpoint",)
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=32,
        help="Batch size for encoding passages",)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for Sentence-BERT (cpu or cuda)",)

    args = parser.parse_args()

    print(f"Loading benchmark from {args.benchmark} ...")
    tests = load_test_benchmark(args.benchmark)
    print(f"Loaded {len(tests)} test cases")

    print(f"Preparing corpus from {args.corpus} ...")
    corpus_passages, _ = prepare_corpus_from_benchmark(
        args.benchmark, args.corpus, chunk_size=args.chunk_size)
    if args.max_passages is not None:
        corpus_passages = corpus_passages[: args.max_passages]
    print(f"Total passages: {len(corpus_passages)}")

    if not corpus_passages:
        raise ValueError("Corpus is empty. Check paths and extraction.")

    retriever = HybridRetriever(
        fusion_method=args.fusion_method,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        rrf_k=args.rrf_k,
        fusion_depth=args.fusion_depth,
        model_name=args.model_name,
        dense_batch_size=args.dense_batch_size,
        device=args.device,)

    print("Indexing corpus with BM25 and Sentence-BERT ...")
    retriever.index(corpus_passages)

    print(f"Retrieving top-{args.k} passages per query...")
    predictions = generate_predictions(retriever, tests, k=args.k)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
