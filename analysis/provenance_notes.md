# Paper-to-Code Provenance & Reproducibility Notes

**Study Title:** "What Do LLMs Say That Human Facilitators Don't? A Computational Behavioral Analysis of AI vs. Human Collaborative Design Meeting Facilitation"

---

## 1. Complete Pipeline Provenance Map

```
Raw AMI Corpus (train/validation/test splits)
    │
    ▼ [scripts/01_preprocess.py]
Candidate Facilitation Moves (2,685 candidate contexts)
    │   -> data/processed/facilitator_moves_candidates.csv
    ▼ [scripts/01_preprocess.py - Stratified Random Sampling]
Sampled Contexts (199 matched facilitation contexts)
    │   -> data/processed/facilitator_moves_sampled_199.csv
    ▼ [notebooks/colab_llm_inference.ipynb / scripts/02_generate_llm_responses.py]
LLM Response Generation (Llama 3.1 8B Instruct, 4-bit NF4, float16, T=0.3, 0.7, 1.0)
    │   -> data/llm_responses.csv (199 rows × 3 temps = 597 LLM responses)
    ▼ [scripts/03_extract_features.py]
Behavioral Feature Vector Extraction (6 dimensions × 4 sources = 796 feature rows)
    │   -> data/features_processed.csv
    ├──► [scripts/04_statistical_analysis.py]
    │        ├── Mann-Whitney U tests + Bonferroni -> results/mannwhitney_results.csv
    │        ├── Temperature Spearman correlations -> results/spearman_temperature.csv
    │        ├── Power analysis -> results/power_analysis.csv
    │        └── Extreme case analysis -> results/extreme_cases.csv
    │
    ├──► [scripts/05_robustness_analysis.py]
    │        ├── Length-matched subsampling (n=72) -> results/robustness_table6.csv
    │        ├── Partial Spearman correlations -> results/robustness_table6.csv
    │        └── Linear mixed-effects (MixedLM) -> results/mixed_effects_results.csv
    │
    └──► [scripts/06_generate_figures.py]
             ├── Figure 1: Violin Distributions -> figures/figure1_violin_distributions.png
             ├── Figure 2: Spearman Correlation Matrix -> figures/figure2_correlation_matrix.png
             ├── Figure 3: PCA Biplot -> figures/figure3_pca_biplot.png
             ├── Figure 4: Temperature Heatmap -> figures/figure4_temperature_heatmap.png
             └── Figure 5: Radar Profiles -> figures/figure5_radar_profiles.png
```

---

## 2. Documented Discrepancies & Resolutions

### Discrepancy 1: Semantic Novelty Formula Equation vs. Feature Data
- **Manuscript Text:** Defines $F_1 = (1 - \text{cosine\_similarity}) / 2$.
- **Result-Producing Data (`data/features_processed.csv`):** Stores $1 - \text{cosine\_similarity}$ directly, giving Human Median = 0.742 and LLM Median = 0.593 as reported in Table 1 of the paper.
- **Resolution:** The result-producing implementation ($1 - \text{cosine\_similarity}$) is preserved in `scripts/03_extract_features.py` to ensure that running feature extraction produces the exact numbers currently printed in Table 1 of the paper. Rescaling linearly by $1/2$ preserves rank order, Mann-Whitney U statistics ($U=30438.5$), rank-biserial $r = -0.537$, and p-values ($p < 0.001$), but changes median values (0.371 vs 0.296). Flagged for author review if manuscript equation or table values are updated in future revisions.

### Discrepancy 2: LLM Model & Generation Script
- **Stale Script (`scripts/02_generate_llm_responses.py`):** Referenced Ollama / Phi-3 Mini.
- **Published Experiment (`notebooks/colab_llm_inference.ipynb`):** Uses Meta Llama 3.1 8B Instruct with bitsandbytes 4-bit NF4 quantization, float16 compute, 180-word context window, top_p=0.9, max_new_tokens=150, temperatures 0.3, 0.7, 1.0.
- **Resolution:** Updated `scripts/02_generate_llm_responses.py` to match the canonical Llama 3.1 8B Instruct pipeline. Preserved the existing result-producing `data/llm_responses.csv` (597 responses) byte-for-byte.

### Discrepancy 3: Figure Numbering
- **Original Code:** Used inconsistent figure names (`figure1_pca.png`, `figure2_boxplots.png`, etc.).
- **Paper Manuscript:**
  - Figure 1 = Violin Distributions
  - Figure 2 = Spearman Correlation Matrix
  - Figure 3 = PCA Biplot
  - Figure 4 = Temperature Heatmap
  - Figure 5 = Radar Profiles
- **Resolution:** Synchronized `scripts/06_generate_figures.py` and output filenames in `figures/` to match manuscript numbering exactly.

---

## 3. Robustness Analysis Exposure (Table 6)
- **Status:** EXPOSED & VERIFIED
- **Script:** `scripts/05_robustness_analysis.py`
- **Outputs:** `results/robustness_table6.csv`, `results/length_robustness.csv`, `results/mixed_effects_results.csv`.
- **Verified Values:**
  - Length-matched ($n=72$ per group): Novelty $r = -0.309$ ($p=0.0245$), Specificity $r = +0.706$ ($p < 0.001$), Divergence $r = +0.499$ ($p < 0.001$).
  - Partial Spearman (controlling for length): Novelty $\rho = -0.231$, Specificity $\rho = +0.506$, Divergence $\rho = +0.412$.
  - MixedLM (session random intercept): Novelty $\beta = -0.1426$ ($p < 0.001$), Specificity $\beta = +0.0172$ ($p < 0.001$), Divergence $\beta = +0.1993$ ($p < 0.001$).
