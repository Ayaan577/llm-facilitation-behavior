# Data

This folder contains the processed datasets used in the study. Raw AMI Meeting Corpus transcripts are not redistributed here; see the [AMI Corpus website](https://groups.inf.ed.ac.uk/ami/corpus/) for access.

## Files

### `features_processed.csv` — Main analysis dataset

796 rows (199 human + 199 × 3 LLM temperature conditions).

| Column | Type | Description |
|--------|------|-------------|
| `context_id` | int | Unique identifier for the facilitation move context |
| `source` | str | `human`, `llm_t03`, `llm_t07`, or `llm_t10` |
| `response_text` | str | The facilitation move text |
| `session_phase` | str | Design phase: `Ideate` or `Prototype/Evaluate` |
| `novel_score` | float | Semantic novelty: 1 − cosine_similarity(context, response) |
| `directive_score` | float | P(directive) from fine-tuned DistilBERT classifier |
| `specificity_score` | float | Composite: (NER density + TTR + mean_word_len / 10) / 3 |
| `empathy_score` | float | Empathy + hedging lexicon density / total tokens |
| `divergence_score` | float | Jensen–Shannon distance between LDA topic distributions |
| `phase_score` | float | Proportion of tokens matching phase-specific vocabulary |

### `llm_responses.csv` — Raw LLM outputs

Contains the original Llama 3.1 8B Instruct responses at T = 0.3, 0.7, and 1.0 before feature extraction.

### `directiveness_training.csv` — Classifier training data

197 synthetic examples (99 directive, 98 open/facilitative) used to fine-tune the DistilBERT directiveness classifier.

## Sample Sizes

| Source | N | Description |
|--------|---|-------------|
| `human` | 199 | Project Manager turns from AMI meetings |
| `llm_t03` | 199 | Llama 3.1 8B Instruct, temperature = 0.3 |
| `llm_t07` | 199 | Llama 3.1 8B Instruct, temperature = 0.7 |
| `llm_t10` | 199 | Llama 3.1 8B Instruct, temperature = 1.0 |

## Reproduction

To regenerate `features_processed.csv` from raw AMI data:

```bash
# 1. Download and extract AMI corpus
python scripts/01_preprocess.py

# 2. Generate LLM responses (requires GPU)
python scripts/02_generate_llm_responses.py

# 3. Compute features
python scripts/03_extract_features.py
```

## Data License

The AMI Meeting Corpus is distributed under a [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/). The processed feature scores and LLM responses in this repository are released under the MIT license.
