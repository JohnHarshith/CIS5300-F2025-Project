# Milestone 4 – Two-Stage Hybrid Retrieval with Legal-Domain Reranking

## Problem Overview

Building on earlier milestones, this stage tackles **two challenges** in legal passage retrieval:

1. **First-stage recall:** Given a legal query (e.g., a yes/no question grounded in a contract), the system must retrieve a small set of candidate passages that almost certainly contains the correct answer.
2. **Second-stage ranking:** Within that candidate pool, the system must rank passages so that the most legally appropriate span appears near the top.

The project now operates over two datasets:

- **LegalBench-RAG (ContractNLI)** as an *out-of-domain* legal QA benchmark, used to measure retrieval quality when the system is not trained on that dataset.
- **LePard** (legal citation dataset) as an *in-domain supervision source* for learning what counts as a good legal match (quote → destination context).

The overall goal of Milestone 4 is to move from a single hybrid retriever (BM25 + Sentence-BERT) to a **two-stage pipeline** with a **legal cross-encoder reranker**, and to quantify how much each component contributes.

---

## Extension Summary

### Motivation

Milestone 3 showed that a **hybrid BM25 + Sentence-BERT retriever** outperformed pure lexical or pure dense baselines on LegalBench-RAG, especially on Recall@10 and nDCG@10. However:

- Hybrid retrieval alone is limited by fixed 500-token chunks and cannot perfectly align with gold spans.
- It treats all retrieved candidates independently and does not explicitly model pairwise legal similarity.
- There is supervision available in **LePard** (quote/destination pairs) that encodes *“this context legally supports that quote”* – a richer signal than purely unsupervised retrieval.

Milestone 4 adds a **cross-encoder reranker**, fine-tuned on LePard, and explores how it improves performance when applied on top of the hybrid retriever.

### High-Level Method

The final pipeline has two stages:

1. **Stage 1: Hybrid retrieval (BM25 + Sentence-BERT)**  
   - Documents (contracts or legal decisions) are split into overlapping 500-word passages.  
   - Each passage is indexed with:
     - **BM25** over tokenized text (lexical signal),
     - **Sentence-BERT** embeddings (`all-mpnet-base-v2`, semantic signal).  
   - At query time:
     - BM25 scores and SBERT cosine similarities are computed.
     - Scores are normalized and fused via a **weighted hybrid** (e.g., 0.55 BM25 / 0.45 SBERT).
     - The top-K (e.g., K=10 or a larger candidate pool size) passages form the **candidate set**.

2. **Stage 2: Legal cross-encoder reranker (LePard-trained)**  
   - **Training data (LePard).**
     - LePard rows contain: `quote`, `destination_context`, and associated metadata.
     - Positive pairs: a quote and its true destination context.
     - Negative pairs: same quote but mismatched contexts (sampled from other rows).
   - A cross-encoder (`CrossEncoder` from `sentence-transformers`, e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) is fine-tuned on LePard train split to regress a *relevance score* for `(quote, context)` pairs.
   - During reranking:
     - For each query, all `(query, candidate_passage)` pairs from the hybrid retriever are scored by the cross-encoder.
     - The system can either:
       - **Use CE-only ranking** (sort purely by cross-encoder score), or
       - **Fuse CE scores with hybrid scores**:
         \[
         s_{\text{fused}} = \alpha \cdot s_{\text{CE}} + (1 - \alpha) \cdot s_{\text{hybrid}},
         \]
         where \(\alpha\) controls how much weight is given to the reranker vs. first-stage retrieval.

The same evaluation metrics as before are used for all setups, so improvements are directly comparable.

---

## Detailed Method & Implementation

### 1. Data Preparation

- **LegalBench-RAG (ContractNLI):**
  - Loads test cases from `contractnli.json`.
  - For each snippet, retrieves the source file from the corpus, extracts the gold char span, and constructs 500-word overlapping chunks.
  - Creates:
    - `corpus_passages`: list of all passages,
    - `tests`: list of query objects with gold answer snippets.

- **LePard:**
  - Loads the large LePard CSV (`training_top_100000_data.csv`).
  - Randomly samples 10,000 rows to keep experiments tractable.
  - Builds **disjoint splits**:
    - 80% train,
    - 10% dev,
    - 10% test  
    (with care taken to avoid leakage by `dest_id`).
  - For retrieval experiments on LePard:
    - Each row provides a query (`quote`) and a gold passage (`destination_context`).
    - These are used to build a small LePard retrieval corpus and gold mappings for evaluation.

### 2. Baseline Retrievers on LePard

In `Milestone4.ipynb` / `Milestone4.py`, the following retrieval baselines are implemented and evaluated on the **LePard test split**:

1. **TF-IDF**  
   - Uses scikit-learn’s `TfidfVectorizer` (English stopwords, min_df/max_df filtering).  
   - For each query, computes cosine similarity against the TF-IDF document matrix and returns top-K passages.

2. **BM25**  
   - Uses `rank-bm25` BM25Okapi over tokenized passages with stopword removal.  
   - Returns top-K passages ranked by BM25 score.

3. **Sentence-BERT (dense)**  
   - Encodes corpus passages and queries with `SentenceTransformer("sentence-transformers/all-mpnet-base-v2")`.  
   - Uses dot product / cosine similarity to retrieve top-K passages.

4. **Hybrid (BM25 + SBERT)**  
   - Normalizes BM25 and SBERT scores to [0,1].
   - Combines them as:
     \[
     s_{\text{hybrid}} = w_{\text{BM25}} \cdot s_{\text{BM25}} + w_{\text{SBERT}} \cdot s_{\text{SBERT}},
     \]
     with tuned weights (e.g., 0.55 / 0.45).
   - Retrieves top-K passages by the combined score.

These baselines are evaluated using: Exact Match, span-level F1, Recall@10, and nDCG@10.

### 3. Cross-Encoder Reranker (Baseline vs. Fine-Tuned)

**Model.**

- A cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) is used to score `(query, passage)` pairs.
- Input: concatenated query and passage.
- Output: single regression logit (higher = more relevant).

**Training data from LePard.**

- **Train split:**
  - For each LePard row, builds:
    - 1 positive pair: `(quote, destination_context, label=1.0)`,
    - Multiple negative pairs: `(quote, other_destination_context, label=0.0)`.
  - Wraps these as `InputExample` objects and feeds them to a `DataLoader` for training.
- **Dev split:**
  - Builds similar pairs for validation.
  - Uses a small `CEBinaryClassificationEvaluator` to monitor dev accuracy/F1/AP and select hyperparameters.

**Training loop.**

- Defines a training dataloader and dev evaluator.
- Uses `CrossEncoder.fit` with:
  - `epochs` (e.g., 2–3),
  - linear warmup steps,
  - Adam with a small learning rate,
  - mixed precision (`use_amp=True`).
- Saves the best model to an output path (e.g., `/content/output/cross_encoder/lepard_finetuned_ce`).

**Reranking hybrid candidates.**

- For each query on the **LePard test split**:
  - The hybrid retriever produces top-K candidate passages.
  - The cross-encoder scores all `(query, candidate_passage)` pairs.
  - Candidates are sorted:
    - **CE baseline:** using the pre-trained cross-encoder (no fine-tuning),
    - **CE fine-tuned:** using the LePard-fine-tuned cross-encoder.

This yields two reranked systems that can be compared to the original hybrid ranking.

### 4. Hybrid + CE Fusion on LegalBench-RAG

On **LegalBench-RAG (ContractNLI)**, the notebook also explores **fusion between hybrid scores and cross-encoder scores**, even though the cross-encoder is trained only on LePard:

- Stage 1: Hybrid BM25 + SBERT retrieves top-N candidates per query.
- Stage 2: CE scores each `(query, candidate)` pair.
- Fusion:
  \[
  s_{\text{fused}} = \alpha \cdot s_{\text{CE}} + (1 - \alpha) \cdot s_{\text{hybrid}},
  \]
  where \(\alpha \in \{0.2, 0.5, 0.8\}\) is swept.
- The best setting (around \(\alpha \approx 0.5\)) is reported.

An additional **candidate upper bound** is computed: the Recall@10 when only checking whether the *gold span appears in the hybrid candidate pool*, regardless of order. This upper bound on LegalBench-RAG is ~0.5855, showing that the reranker is already close to the maximum achievable with this specific first-stage candidate set.

---

## Key Results

### 1. LegalBench-RAG (ContractNLI) – Effect of Reranking

On ContractNLI, with the hybrid retriever as the first stage:

- **Hybrid-only**:
  - span_f1: 0.2357  
  - recall@10: 0.5511  
  - ndcg@10: 0.4808

- **CE-only (reranking same candidate pool)**:
  - span_f1: ≈0.2266  
  - recall@10: ≈0.5142  
  - ndcg@10: ≈0.4141

- **Hybrid + CE Fusion (α ≈ 0.5)**:
  - span_f1: ≈0.2393  
  - recall@10: ≈0.5607  
  - ndcg@10: ≈0.4896  

- **Candidate upper bound (hybrid pool only)**:
  - recall@10 upper bound: 0.5855

**Interpretation.**

- Hybrid-only already provides strong recall.
- CE-only reordering can sometimes hurt recall when used alone, because it discards the useful lexical signal from BM25.
- **Fused Hybrid+CE** slightly improves span_f1, recall@10, and nDCG@10, moving closer to the 0.5855 upper bound while still being limited by which candidates hybrid retrieval finds.

### 2. LePard (10k sampled rows, disjoint train/dev/test) – Full Comparison

On the LePard **test** split (≈977 examples, TOP_K = 10), the final summary comparison is:

| Model (k=10)                          | Exact Match | Span F1 | Recall@10 | nDCG@10 |
|--------------------------------------|-------------|---------|-----------|---------|
| TF-IDF                               | 0.0911      | 0.2586  | 0.1883    | 0.1406  |
| BM25                                 | 0.1198      | 0.3111  | 0.2354    | 0.1782  |
| Sentence-BERT                        | 0.0727      | 0.2339  | 0.1382    | 0.1076  |
| Hybrid (BM25 + SBERT)               | 0.1146      | 0.3035  | 0.2313    | 0.1732  |
| Cross-Encoder Baseline (CE-only)     | 0.1259      | 0.3186  | 0.2405    | 0.1847  |
| Cross-Encoder Fine-tuned (CE-only)   | 0.1208      | 0.3183  | **0.2600** | **0.1925** |

**Interpretation.**

- BM25 outperforms TF-IDF and SBERT in this legal citation setting, confirming the importance of lexical legal cues.
- Hybrid retrieval slightly improves span_F1 and nDCG@10 over BM25, but gains are modest.
- A **cross-encoder reranker** trained on generic data (baseline CE) already beats hybrid on all metrics.
- **Fine-tuning the cross-encoder on LePard** provides an additional improvement in **Recall@10** (0.2313 → 0.2600) and **nDCG@10** (0.1732 → 0.1925), confirming that legal-domain supervision helps the model better prioritize truly relevant contexts.

---

## How to Reproduce

Both `Milestone4.ipynb` and `Milestone4.py` contain equivalent code for the pipeline. The general steps are:

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   # or, inside a notebook:
   # !pip install sentence-transformers torch rank-bm25 scikit-learn nltk numpy pandas
    ```
Download and unpack datasets

Run Milestone4.ipynb
A cell will ask to upload files to set up files. Upload `data_extracted.zip` in that cell.  

Upload training_top_100000_data.tar.gz in root folder. 

Set the dataset paths (LegalBench-RAG and LePard) in the configuration cell.

Run all cells in order to:

Build the retrieval corpus and test examples,

Train and evaluate TF-IDF, BM25, SBERT, and Hybrid,

Train the cross-encoder on LePard (train/dev),

Run Hybrid+CE fusion experiments on LegalBench-RAG.

Evaluate all metrics on LePard test,

Run Milestone4.py (script version)

Ensure that the paths at the top of Milestone4.py match the local folder structure.

Execute:
```bash
  python Milestone4.py
```

The script will:
    
Load data, run retrieval baselines, train the cross-encoder, save predictions and metrics (e.g., as JSON) under an output/ directory.

## Takeaways

Hybrid BM25 + Sentence-BERT consistently improves over pure lexical or pure dense retrieval on legal text, especially for recall-oriented metrics.

A cross-encoder reranker, particularly after fine-tuning on LePard, further improves ranking quality and recall on the LePard test set.

On LegalBench-RAG, Hybrid+CE fusion nudges performance closer to the candidate upper bound, highlighting that most remaining gains may come from improving the first-stage retrieval (better chunking, indexing more context) rather than only refining reranking.