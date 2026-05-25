[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_llm_inference.ipynb)

# LLM vs. Human Facilitation Behavior in Collaborative Design Meetings

> **What Do LLMs Say That Human Facilitators Don't? A Computational Behavioral Analysis of AI vs. Human Collaborative Design Meeting Facilitation**
***

## Abstract

This study presents a quantitative behavioral comparison of human and LLM facilitation in collaborative design meetings. We analyze 199 matched facilitation move pairs from the AMI Meeting Corpus, comparing trained human facilitators (Project Managers) with Llama 3.1 8B Instruct across six behavioral dimensions: semantic novelty, directiveness, specificity, empathy, topic divergence, and phase appropriateness. Mann-Whitney U tests with Bonferroni correction reveal significant differences on four of six dimensions, with large effects for topic divergence (*r* = +0.58) and semantic novelty (*r* = −0.54). LLM facilitation is characterized by structured, coaching-style reframing; human facilitation by contextually embedded, semantically original responses.

***

## Key Findings

| Dimension | Human vs. LLM | Effect (*r*) | Magnitude | Practical Interpretation |
|-----------|---------------|-------------|-----------|--------------------------|
| **Divergence** | LLM higher | +0.58 | Large | LLM scores higher in 79.0% of pairs |
| **Novelty** | Human higher | −0.54 | Large | Human scores higher in 76.9% of pairs |
| **Phase Approp.** | LLM higher | +0.34 | Medium | LLM scores higher in 67.1% of pairs |
| **Specificity** | LLM higher | +0.25 | Small–medium | LLM scores higher in 62.7% of pairs |
| Directiveness | No difference detected | −0.005 | — | Underpowered (power = 0.05) |
| Empathy | No difference detected | −0.058 | — | Underpowered (power = 0.21) |

- LLM facilitation behavior is **largely temperature-stable**; only specificity (ρ = +0.12) and phase appropriateness (ρ = −0.10) show modest sensitivity
- **15/20** highest-divergence LLM responses use directive reframing ("Let's take a step back...")
- Results **replicate within the Ideate phase** subsample (*n* = 149), the largest segment

***

## Repository Structure

```
llm-facilitation-behavior/
├── data/
│   ├── features_processed.csv          # 796 rows × 10 cols (199 human + 597 LLM)
│   ├── llm_responses.csv               # Raw LLM-generated responses at 3 temperatures
│   ├── directiveness_training.csv      # 197 synthetic examples for DistilBERT
│   └── README.md                       # Column descriptions + reproduction notes
├── results/
│   ├── mannwhitney_results.csv         # Table 1: U tests + effect sizes + CIs
│   ├── spearman_temperature.csv        # Table 2: Temperature correlations
│   ├── effect_sizes.csv                # Table 3: Practical effect interpretation
│   └── descriptive_stats.csv           # Table 0: Word counts + vocab sizes
├── scripts/
│   ├── 01_preprocess.py                # AMI extraction + rule-based filtering
│   ├── 02_generate_llm_responses.py    # LLM inference (local GPU)
│   ├── 03_extract_features.py          # Six behavioral dimension scoring
│   ├── 04_statistical_analysis.py      # Mann-Whitney U, Bonferroni, power
│   ├── 04b_train_directiveness.py      # DistilBERT classifier training
│   ├── 05_generate_figures.py          # All publication figures
│   ├── run_pipeline_resampled.py       # End-to-end pipeline (stages 3–5)
│   └── compute_kappa.py               # Cohen's κ for inter-rater reliability
├── notebooks/
│   └── colab_llm_inference.ipynb       # Google Colab notebook (T4 GPU, 4-bit quantization)
├── figures/
│   ├── figure1_pca_biplot.png          # PCA of standardized behavioral features
│   ├── figure2_temperature_heatmap.png # Median scores × temperature (* = sig.)
│   ├── figure3_radar_profiles.png      # Normalized radar (Human + 3 LLM temps)
│   ├── figure4_correlation_matrix.png  # Spearman inter-dimension correlations
│   └── figure5_violin_distributions.png # Score distributions (Human vs LLM T=0.7)
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```

***

## Quickstart

```bash
# Clone
git clone https://github.com/Ayaan577/llm-facilitation-behavior.git
cd llm-facilitation-behavior

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run analysis (uses pre-computed features — no GPU needed)
python scripts/run_pipeline_resampled.py
```

This reproduces all statistical tests and figures from the pre-computed feature scores.

***

## Pipeline Overview

| # | Stage | Script | Input | Output | GPU? |
|---|-------|--------|-------|--------|------|
| 1 | **Extract** | `01_preprocess.py` | AMI XML transcripts | `facilitator_moves.csv` | No |
| 2 | **Generate** | `02_generate_llm_responses.py` | Facilitator moves | `llm_responses.csv` | **Yes** |
| 3 | **Score** | `03_extract_features.py` | LLM responses | `features_processed.csv` | No |
| 4 | **Analyze** | `04_statistical_analysis.py` | Feature scores | `results/*.csv` | No |
| 5 | **Visualize** | `05_generate_figures.py` | Feature scores | `figures/*.png` | No |

> **Quick path:** `run_pipeline_resampled.py` runs stages 3–5 end-to-end.
>
> **Full reproduction:** Requires a GPU with ≥8 GB VRAM. Use the [Colab notebook](notebooks/colab_llm_inference.ipynb) for free-tier T4 access.

***

## Data

### `features_processed.csv` — 796 rows

| Column | Type | Description |
|--------|------|-------------|
| `context_id` | int | Unique facilitation move context identifier |
| `source` | str | `human`, `llm_t03`, `llm_t07`, or `llm_t10` |
| `response_text` | str | Facilitation move text |
| `session_phase` | str | Design phase: `Ideate` or `Prototype/Evaluate` |
| `novel_score` | float | 1 − cos_sim(context, response) via all-MiniLM-L6-v2 |
| `directive_score` | float | P(directive) from fine-tuned DistilBERT |
| `specificity_score` | float | (NER density + TTR + mean_word_len/10) / 3 |
| `empathy_score` | float | Empathy + hedging lexicon density |
| `divergence_score` | float | Jensen–Shannon distance (15-topic LDA) |
| `phase_score` | float | Phase-specific vocabulary proportion |

Sample sizes: **199 human** PM turns × **3 temperatures** (0.3, 0.7, 1.0) = **597 LLM** responses.

Raw AMI transcripts: [groups.inf.ed.ac.uk/ami/corpus](https://groups.inf.ed.ac.uk/ami/corpus/)

***

## Statistical Results

### Mann-Whitney U Tests (Human vs. LLM at T = 0.7, Bonferroni-corrected)

| Dimension | Human Mdn | LLM Mdn | *r* | 95% CI | *p*_Bonf | Power |
|-----------|-----------|---------|-----|--------|----------|-------|
| Novelty | 0.742 | 0.593 | −0.54 | [−0.62, −0.45] | <0.001 | 1.00 |
| Directiveness | 0.021 | 0.036 | −0.005 | — | 1.000 | 0.05 |
| Specificity | 0.397 | 0.416 | +0.25 | [+0.14, +0.37] | <0.001 | 1.00 |
| Empathy | 0.000 | 0.000 | −0.058 | — | 1.000 | 0.21 |
| Divergence | 0.540 | 0.744 | +0.58 | [+0.48, +0.67] | <0.001 | 1.00 |
| Phase Approp. | 0.000 | 0.000 | +0.34 | [+0.26, +0.43] | <0.001 | 1.00 |

***

## Reproducibility

| Parameter | Value |
|-----------|-------|
| Random seeds | `numpy=42`, `sklearn=42`, `LDA=42`, `bootstrap=42` |
| LLM | Llama 3.1 8B Instruct, 4-bit quantization (bitsandbytes) |
| Hardware | Google Colab free-tier Tesla T4 (16 GB VRAM) |
| PCA | StandardScaler → PCA (sklearn, random_state=42) |
| Bootstrap CIs | Within-group resampling, 1000 iterations |
| Inter-rater reliability | κ = 0.81 (Cohen, 1960) |
| Python | 3.10+ |
| Approx. runtimes | Extraction: ~5 min · Inference: ~45 min (T4) · Analysis: ~30 sec |

***

## License

MIT License — see [LICENSE](LICENSE).
