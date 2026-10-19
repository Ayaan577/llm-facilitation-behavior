# Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ayaan577/llm-facilitation-behavior/blob/main/notebooks/colab_llm_inference.ipynb)

Official code and dataset repository for the paper:
> **Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison**
>
> *Mohammed Azeez Khan, Aaron Dsouza, Ashutosh Mishra, Ummul Faiz Z. Bibi, and Neha K. Nair*

---

## Abstract

Large language models are increasingly being explored as facilitators, but the behavioral differences between LLM-generated and human facilitation remain insufficiently characterized. This repository supports a context-matched computational comparison of human Project Manager facilitation turns from scenario-based AMI design meetings and Llama 3.1 8B Instruct responses across 199 paired contexts. Five confirmatory dimensions were tested with paired Wilcoxon signed-rank tests and one Bonferroni correction: a result is significant when its raw $p$ is below $0.010$, equivalently when $p_\text{Bonf}=\min(1,5p)$ is below $0.05$. Human facilitation was higher on contextual semantic distance ($r = 0.644$) and hedge-and-empathy lexical density ($r = 0.203$, raw $p = 0.00415$, $p_\text{Bonf} = 0.021$). Llama-generated facilitation was higher on topic divergence ($r = 0.703$), surface lexical elaboration ($r = 0.288$, length-confounded), and instruction-conditioned phase-vocabulary alignment ($r = 0.382$). Directiveness is exploratory only because domain validation failed ($\kappa = -0.02$). A secondary no-phase check left the Llama-higher phase contrast in place ($r = 0.294$) and did not establish spontaneous phase awareness. These are computational language measures, not measures of facilitation quality.

The compiled manuscript is [`paper/main.pdf`](paper/main.pdf).

---

## Key Findings

| Dimension | Direction (T=0.7) | Effect (*r*) | 95% CI |
|---|---|---|---|
| **Topic divergence** | LLM higher | 0.703 | [0.63, 0.76] |
| **Contextual distance** | Human higher | 0.644 | [0.56, 0.72] |
| **Phase-vocabulary alignment** | LLM higher | 0.382 | [0.25, 0.50] |
| **Lexical elaboration** | LLM higher | 0.288 | [0.16, 0.40] |
| **Empathy-and-hedge density** | Human higher; hedge proxy | 0.203 | [0.07, 0.34] |
| Directiveness (exploratory) | Not interpreted | 0.026 | [0.00, 0.17] |

---

## Repository Structure & Output File Map

```
llm-facilitation-behavior/
├── data/
│   ├── processed/
│   │   ├── facilitator_moves_candidates.csv
│   │   ├── facilitator_moves_sampled_199.csv
│   │   └── directiveness_training_data.csv
│   ├── llm_responses.csv
│   ├── llm_responses_nophase.csv
│   ├── features_processed.csv
│   └── README.md
├── results/
│   ├── mannwhitney_results.csv
│   ├── spearman_temperature.csv
│   ├── robustness_table6.csv
│   ├── power_analysis.csv
│   ├── phase_nophase_comparison.csv
│   ├── lda_topic_sensitivity.csv
│   ├── sensitivity_analyses.csv
│   ├── temperature_repeated.csv
│   └── extreme_cases.csv
├── scripts/
│   ├── 01_preprocess.py
│   ├── 02_generate_llm_responses.py
│   ├── 03_extract_features.py
│   ├── 04_statistical_analysis.py
│   ├── 04b_train_directiveness.py
│   ├── 05_robustness_analysis.py
│   ├── 06_generate_figures.py
│   ├── 07_phase_nophase.py
│   ├── 08_positioning_figures.py
│   └── validate_reproduction.py
├── notebooks/
│   └── colab_llm_inference.ipynb
├── figures/
│   ├── figure1_violin_distributions.png
│   ├── figure2_correlation_matrix.png
│   ├── figure3_pca_biplot.png
│   ├── figure4_temperature_heatmap.png
│   ├── figure5_radar_profiles.png
│   ├── figure6_effect_forest.png
│   └── figure7_nophase_means.png
├── paper/
│   └── main.pdf
├── validation/
│   ├── directiveness_annotation_task.csv
│   ├── directiveness_annotation_filled.csv
│   ├── agreement_results.csv
│   └── compute_agreement.py
├── analysis/
│   └── provenance_notes.md
├── REPRODUCIBILITY_MANIFEST.json
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
python scripts/07_phase_nophase.py
python scripts/08_positioning_figures.py

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
   python scripts/07_phase_nophase.py
   python scripts/08_positioning_figures.py
   python scripts/validate_reproduction.py
   ```

---

## Experimental & Computational Parameters

| Parameter | Value |
|---|---|
| **Corpus** | AMI scenario meetings. Candidate moves span 139 of the transcribed sessions |
| **Candidate Moves** | 2,685 candidate facilitation moves |
| **Sampled Contexts** | 199 moves (149 Ideate, 50 Prototype/Evaluate; no Empathize/Define turns) |
| **Context Window** | Preceding 5 turns (T-5 to T-1), max 180 words truncation |
| **LLM Model** | Meta Llama 3.1 8B Instruct (`meta-llama/Llama-3.1-8B-Instruct`) |
| **Quantization** | 4-bit `bitsandbytes` (NF4, double quant, float16 compute) |
| **Hardware** | Laptop: NVIDIA GeForce RTX 3050 Ti (4 GB) for analysis. Generation only: Tesla T4 (16 GB) on Google Colab |
| **Generation Settings** | `do_sample=True`, `top_p=0.9`, `max_new_tokens=150`, temperatures = `[0.3, 0.7, 1.0]` |
| **No-phase generation** | Same settings, phase line removed, seed 42. File: `data/llm_responses_nophase.csv`. Secondary only. |
| **Quality Filter** | At least 5 words; none of the refusal phrases `as an ai`, `i cannot`, `i'm unable`, `i am unable`, `i don't have`, `as a language model`, or `i'm just`; and, when the context has more than 10 words, rejection if more than 90% of the response word types also appear in the context |
| **Random Seeds** | Bootstrap intervals and LDA refits use seed 42. The no-phase generation used seed 42. A seed for the primary generation is not recorded. The 199 contexts are the stored response identifiers, not a new random draw |

---

## Behavioral Feature Definitions

1. **Contextual semantic distance (`novel_score`):** $1 - \cos(\mathbf{e}_{\text{ctx}}, \mathbf{e}_{\text{resp}})$ in `all-MiniLM-L6-v2` space (human median 0.742, LLM $T=0.7$ median 0.593). TF–IDF cosine distance does not reproduce this contrast. The result is specific to that embedding space.
2. **Directiveness (`directive_score`):** $P(\text{directive})$ from a DistilBERT classifier trained on 197 synthetic examples. Domain validation on 100 AMI utterances failed ($\kappa = -0.0231$). Exploratory only.
3. **Lexical elaboration** (column `specificity_score`): Mean of NER density, type–token ratio, and normalized average word length. A surface proxy, substantially associated with response length.
4. **Hedge and empathy lexical density (`empathy_score`):** Empathy-word and hedge-word hits divided by tokens. Hedge-dominated. Not psychological empathy.
5. **Topic divergence (`divergence_score`):** Jensen–Shannon distance between 15-topic LDA distributions. Greater divergence is not a creativity result.
6. **Phase-vocabulary alignment (`phase_score`):** Proportion of tokens matching the phase vocabulary. The primary contrast is instruction-conditioned.

---

## Statistical Methodology

- **Main Test:** Paired Wilcoxon signed-rank test, human vs. LLM at $T = 0.7$ ($n = 199$), Bonferroni correction over 5 confirmatory dimensions ($\alpha = 0.05 / 5 = 0.010$). Effect size $r = |Z|/\sqrt{n}$ uses the Wilcoxon $Z$, not a floored $p$-value. Directiveness is exploratory.
- **Temperature:** Friedman test within context across $T = 0.3, 0.7, 1.0$ (Bonferroni over 5 confirmatory dimensions). Spearman correlations on the stacked 597 rows are descriptive only.
- **Phase-label sensitivity:** `scripts/07_phase_nophase.py` rescores `data/llm_responses_nophase.csv` with the original phase-vocabulary function. At $T = 0.7$, no-phase responses remain higher than human turns ($r = 0.294$); label versus no label is not significant ($r = 0.123$, $p = 0.082$). This is not part of the confirmatory family.
- **Robustness:**
  - *Length-matched pairs:* relative word-count difference below 50% ($n=73$), paired Wilcoxon.
  - *Length association:* Spearman correlation between the paired difference and human word count.
  - *Linear mixed-effects:* `is_llm` fixed effect and random intercept for `context_id`.

---

## Figure Mapping

Filenames keep their original numbers. The preprint numbers them by order of appearance.

- **Preprint Figure 1:** `figures/figure6_effect_forest.png` — Signed effect sizes. Negative values are human-higher.
- **Preprint Figure 2:** `figures/figure1_violin_distributions.png` — Score distributions at $T = 0.7$.
- **Preprint Figure 3:** `figures/figure7_nophase_means.png` — Human, label-present, and no-label phase means.
- **Appendix:** `figures/figure2_correlation_matrix.png`, `figures/figure3_pca_biplot.png` (exploratory PCA of behavioral scores), `figures/figure4_temperature_heatmap.png`, and `figures/figure5_radar_profiles.png` (mean scores, one scale per dimension). These are not the main result.

---

## Citation

```bibtex
@misc{khan2026humanllmfacilitationcollaborative,
      title={Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison},
      author={Mohammed Azeez Khan and Aaron Dsouza and Ashutosh Mishra and Ummul Faiz Z. Bibi and Neha K. Nair},
      year={2026},
      eprint={2610.13082},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.13082},
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
