# What Do LLMs Say That Human Facilitators Don't? A Computational Behavioral Analysis of AI vs. Human Collaborative Design Meeting Facilitation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_llm_inference.ipynb)

Official code and dataset repository for the paper:
> **What Do LLMs Say That Human Facilitators Don't? A Computational Behavioral Analysis of AI vs. Human Collaborative Design Meeting Facilitation**
>
> *Mohammed Azeez Khan, Aaron D'Souza, Ashutosh Mishra, and Amar Kumar Behera*

---

## Abstract

This study presents a quantitative behavioral comparison of human and LLM facilitation in collaborative design meetings. We analyze 199 matched facilitation move pairs from the AMI Meeting Corpus, comparing trained human facilitators (Project Managers) with Llama 3.1 8B Instruct across six behavioral dimensions: semantic novelty, directiveness, specificity, empathy, topic divergence, and phase appropriateness. Mann-Whitney U tests with Bonferroni correction reveal significant differences on four of six dimensions, with large effects for topic divergence (*r* = +0.58) and semantic novelty (*r* = −0.54). LLM facilitation is characterized by structured, coaching-style reframing; human facilitation by contextually embedded, semantically original responses.

---

## Key Findings

| Dimension | Human vs. LLM (T=0.7) | Effect (*r*) | Magnitude | Practical Interpretation |
|---|---|---|---|---|
| **Divergence** | LLM higher | +0.58 | Large | LLM scores higher in 79.0% of pairs |
| **Novelty** | Human higher | −0.54 | Large | Human scores higher in 76.9% of pairs |
| **Phase Approp.** | LLM higher | +0.34 | Medium | LLM scores higher in 67.1% of pairs |
| **Specificity** | LLM higher | +0.25 | Small–medium | LLM scores higher in 62.7% of pairs |
| Directiveness | No difference detected | −0.005 | — | Underpowered (*power* = 0.05) |
| Empathy | No difference detected | −0.058 | — | Underpowered (*power* = 0.21) |

---

## Repository Structure & Output File Map

```
llm-facilitation-behavior/
├── data/
│   ├── processed/
│   │   ├── facilitator_moves_candidates.csv # 2,685 candidate facilitation moves
│   │   ├── facilitator_moves_sampled_199.csv# 199 sampled facilitation moves
│   │   └── directiveness_training_data.csv # 197 synthetic training examples
│   ├── llm_responses.csv                   # 199 rows × 3 temps = 597 LLM responses
│   ├── features_processed.csv              # 796 rows × 10 cols (199 human + 597 LLM)
│   └── README.md                           # Dataset documentation
├── results/
│   ├── mannwhitney_results.csv             # 18 Mann-Whitney U test comparisons
│   ├── spearman_temperature.csv            # Temperature sensitivity correlations
│   ├── robustness_table6.csv               # Table 6: Length matching, partial r, MixedLM
│   ├── power_analysis.csv                  # Post-hoc power calculations
│   └── extreme_cases.csv                   # Extreme case pairs for qualitative inspection
├── scripts/
│   ├── 01_preprocess.py                    # Corpus parsing & stratified 199 sampling
│   ├── 02_generate_llm_responses.py        # Canonical Llama 3.1 8B Instruct inference
│   ├── 03_extract_features.py              # Six behavioral dimension scoring
│   ├── 04_statistical_analysis.py          # Main stats, correlations, power, extremes
│   ├── 04b_train_directiveness.py          # Fine-tune DistilBERT classifier
│   ├── 05_robustness_analysis.py          # Length matching, partial Spearman, MixedLM
│   ├── 06_generate_figures.py              # Figures 1–5 generation
│   └── validate_reproduction.py           # Automated 11-check reproducibility harness
├── notebooks/
│   └── colab_llm_inference.ipynb           # Free T4 GPU generation notebook on Colab
├── figures/
│   ├── figure1_violin_distributions.png    # Figure 1: Violin plots across 6 dimensions
│   ├── figure2_correlation_matrix.png      # Figure 2: Inter-dimension Spearman correlations
│   ├── figure3_pca_biplot.png              # Figure 3: PCA facilitation style space
│   ├── figure4_temperature_heatmap.png     # Figure 4: Temperature sensitivity heatmap
│   └── figure5_radar_profiles.png          # Figure 5: Radar profile comparison
├── validation/
│   ├── directiveness_annotation_task.csv  # Stratified 100 AMI move validation sample
│   ├── directiveness_annotation_filled.csv# Filled human annotations
│   ├── agreement_results.csv               # Cohen's kappa (κ = -0.0231) & accuracy metrics
│   └── compute_agreement.py                # Inter-rater agreement calculation
├── analysis/
│   └── provenance_notes.md                 # Provenance map & manuscript notes
├── REPRODUCIBILITY_MANIFEST.json           # SHA-256 hashes & reproduction parameters
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

---

## Reproduction Guide

### Quick Reproduce (No GPU Required — 30 Seconds)
To reproduce all statistical tables, robustness checks (Table 6), and publication figures from released feature scores:

```bash
git clone https://github.com/Ayaan577/llm-facilitation-behavior.git
cd llm-facilitation-behavior

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run analysis and figure generation
python scripts/04_statistical_analysis.py
python scripts/05_robustness_analysis.py
python scripts/06_generate_figures.py

# Run validation harness
python scripts/validate_reproduction.py
```

### Full Reproduction from Raw AMI Corpus
1. Download the AMI Meeting Corpus CSV files (`train.csv`, `validation.csv`, `test.csv`) into `../AMI_dataset/`.
2. Extract candidate moves and sampled contexts:
   ```bash
   python scripts/01_preprocess.py
   ```
3. Generate LLM responses (requires GPU with $\geq 8$ GB VRAM or Google Colab):
   ```bash
   python scripts/02_generate_llm_responses.py
   ```
   *Alternatively, run [`notebooks/colab_llm_inference.ipynb`](notebooks/colab_llm_inference.ipynb) on Google Colab T4 GPU.*
4. Extract behavioral feature vectors:
   ```bash
   python scripts/03_extract_features.py
   ```
5. Run statistical & robustness pipelines:
   ```bash
   python scripts/04_statistical_analysis.py
   python scripts/05_robustness_analysis.py
   python scripts/06_generate_figures.py
   python scripts/validate_reproduction.py
   ```

---

## Experimental & Computational Parameters

| Parameter | Value |
|---|---|
| **Corpus** | AMI Meeting Corpus (139 sessions) |
| **Candidate Moves** | 2,685 candidate facilitation moves |
| **Sampled Contexts** | 199 moves (stratified: 20% Empathize/Define, 50% Ideate, 30% Prototype/Evaluate) |
| **Context Window** | Preceding 5 turns (T-5 to T-1), max 180 words truncation |
| **LLM Model** | Meta Llama 3.1 8B Instruct (`meta-llama/Llama-3.1-8B-Instruct`) |
| **Quantization** | 4-bit `bitsandbytes` (NF4, double quant, float16 compute) |
| **Hardware** | NVIDIA Tesla T4 GPU (16 GB VRAM) on Google Colab |
| **Generation Settings** | `do_sample=True`, `top_p=0.9`, `max_new_tokens=150`, temperatures = `[0.3, 0.7, 1.0]` |
| **Quality Filter** | Min length 5 words, refusal detection, $<90\%$ n-gram overlap |
| **Random Seeds** | `numpy=42`, `sklearn=42`, `LDA=42`, `subsampling=42` |

---

## Behavioral Feature Definitions

1. **Semantic Novelty (`novel_score`):** $1 - \cos(\mathbf{e}_{\text{ctx}}, \mathbf{e}_{\text{resp}})$ using SentenceTransformer `all-MiniLM-L6-v2`. (Note: Manuscript equation writes $(1-\cos)/2$; existing dataset stores $1-\cos$, yielding Human Mdn = 0.742 vs LLM Mdn = 0.593).
2. **Directiveness (`directive_score`):** Probability $P(\text{directive})$ from fine-tuned DistilBERT model trained on 197 synthetic examples. Human validated on 100 AMI utterances ($\kappa = -0.0231$).
3. **Specificity (`specificity_score`):** Mean of NER density, Type-Token Ratio (TTR), and normalized average word length ($\min(\text{AWL}/10, 1)$).
4. **Empathy (`empathy_score`):** Density of empathy and hedging vocabulary terms.
5. **Topic Divergence (`divergence_score`):** Jensen-Shannon distance (`scipy.spatial.distance.jensenshannon`) between topic distributions from 15-topic LDA model.
6. **Phase Appropriateness (`phase_score`):** Proportion of phase-specific design thinking vocabulary tokens.

---

## Statistical Methodology

- **Main Test:** Mann-Whitney U test across 18 comparisons (6 dimensions $\times$ 3 temperatures) with Bonferroni correction ($\alpha = 0.05 / 18 = 0.00278$).
- **Temperature Correlations:** Spearman rank correlation $\rho$ between temperature ($0.3, 0.7, 1.0$) and feature scores.
- **Robustness (Table 6):**
  - *Length-Matched Subsampling:* 5 word-count bins, $n=72$ matched pairs.
  - *Partial Correlations:* Spearman partial $\rho$ controlling for response length.
  - *Linear Mixed-Effects:* `statsmodels` MixedLM with fixed effect `source` and random intercept `session_id`.

---

## Figure Mapping

- **Figure 1:** `figures/figure1_violin_distributions.png` — Score distributions for Human vs LLM T=0.7.
- **Figure 2:** `figures/figure2_correlation_matrix.png` — Inter-dimension Spearman correlation heatmap.
- **Figure 3:** `figures/figure3_pca_biplot.png` — PCA biplot of facilitation style space.
- **Figure 4:** `figures/figure4_temperature_heatmap.png` — Temperature correlation heatmap.
- **Figure 5:** `figures/figure5_radar_profiles.png` — Radar profiles comparing median facilitation styles.

---

## Citation

```bibtex
@article{khan2026llm_facilitation,
  title   = {What Do {LLMs} Say That Human Facilitators Don't? A Computational Behavioral Analysis of {AI} vs.\ Human Collaborative Design Meeting Facilitation},
  author  = {Khan, Mohammed Azeez and D'Souza, Aaron and Mishra, Ashutosh and Behera, Amar Kumar},
  journal = {Computers in Human Behavior},
  year    = {2026},
  url     = {https://github.com/Ayaan577/llm-facilitation-behavior}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
