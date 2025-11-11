# Dataset Description

This file provides a structured description of the datasets used in our legal passage retrieval project.

---

## 1. Training Dataset: LePaRD

- **Source**: [Hugging Face - rmahari/LePaRD](https://huggingface.co/datasets/rmahari/LePaRD)
- **Format**: Hugging Face `DatasetDict`
- **Size**: ~22.7 million examples
- **Type**: Supervised legal citation-linked passage retrieval

### Description

LePaRD (Legal Passage Retrieval Dataset) links legal opinions that cite other cases to the specific passages being cited. It is constructed from a large corpus of U.S. court opinions. Each example includes a quoted passage from the citing document and the full destination context of the cited passage, making it ideal for training retrieval systems in the legal domain.

As per requirement, we will split this dataset into training and development dataset as 80%-20% split respectively. We have uploaded first 100000 rows in gradescope since dataset is too huge for gradescope. 

### Fields

| Field                   | Description                                                                  |
|-------------------------|------------------------------------------------------------------------------|
| `quote`                 | Quoted passage from the source opinion (used as a query)                     |
| `destination_context`   | Paragraph in the destination case where the quote appears (retrieval target) |
| `source_name`           | Name of the citing case                                                      |
| `dest_name`             | Name of the cited case                                                       |
| `source_cite`           | Citation of the source case                                                  |
| `dest_cite`             | Citation of the destination case                                             |
| `source_court`          | Court that issued the citing case                                            |
| `dest_court`            | Court that issued the cited case                                             |
| `source_date`           | Date of the citing case                                                      |
| `dest_date`             | Date of the cited case                                                       |
| `source_id` / `dest_id` | Unique case identifiers                                                      |
| `passage_id`            | Unique passage-level ID                                                      |

### Example

```json
{
  "dest_id": 4140271,
  "source_id": 3628546,
  "dest_date": "1934-10-17",
  "dest_court": "United States District Court for the District of Maryland",
  "dest_name": "United States ex rel. Pen Mar Co. v. J. L. Robinson Const. Co.",
  "dest_cite": "United States ex rel. Pen Mar Co. v. J. L. Robinson Const. Co., 8 F. Supp. 620 (1934)",
  "source_date": "1923-11-22",
  "source_court": "United States District Court for the Eastern District of New York",
  "source_name": "United States ex rel. Ganford Co. v. Conners",
  "source_cite": "United States ex rel. Ganford Co. v. Conners, 295 F. 521 (1923)",
  "passage_id": "3628546_1",
  "quote": "That in all suits instituted under the provisions of this section such personal notice of the pendency of such suits, informing them of their right to intervene as the court may order...",
  "destination_context": "A letter to the clerk of this court from Alfred G. Bennett, of the Bureau of Liquidations of the Insurance Department of the State of New York under date of December 29, 1933..."
}
```

### Loading Instructions

from datasets import load_dataset
ds = load_dataset("rmahari/LePaRD")

## 2. Testing Dataset: LegalBench-RAG

- **Source**: [Dropbox](https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&dl=0)
- **Format**: JSON + Raw Corpus Text (.txt)
- **Size**: ~300 natural language legal queries
- **Type**: Question-answer retrieval benchmark grounded in real legal documents

### Structure

The dataset consists of two directories:

#### 1. `benchmark/` (queries + answer spans)
- Contains `.json` files with a `"tests"` list.
- Each entry includes:
  - `query`: A natural language legal question.
  - `snippets`: A list of:
    - `file_path`: Path to a corpus document containing the answer.
    - `span`: Character indices (`start`, `end`) pointing to the relevant passage in the file.
    - `answer`: The answer text within the span.

##### Example

```json
{
  "query": "Does the Agreement indicate that the Receiving Party has no rights to Confidential Information?",
  "snippets": [
    {
      "file_path": "contractnli/CopAcc_NDA-and-ToP-Mentors_2.0_2017.txt",
      "span": [11461, 11963],
      "answer": "Any and all proprietary rights...shall be and remain with the Participants respectively..."
    }
  ]
}
```

#### 2. `corpus/` (raw documents)
- Directory of .txt files containing the full legal contracts and opinions.
- Used to test retrieval: models must find correct passages in these raw texts based on the query.

### Evaluation setup

- Input: Natural language query from benchmark/
- Target: Matching passage span (character-based) in the corresponding .txt document from corpus/
- Metrics: Recall@10, Exact Match, Span F1, and nDCG for ranked results.
