# Paper-to-Code Provenance

**Study title:** Human and LLM Facilitation in Collaborative Design Meetings: A Computational Behavioral Comparison

**Authoritative manuscript:** `paper/main_ieee.tex`  
**Superseded draft:** `paper/main_chb.tex` still reports the original unpaired Mann–Whitney analysis. Do not cite it.

The primary test is a paired Wilcoxon signed-rank comparison of human Project Manager turns and Llama 3.1 8B Instruct responses at temperature 0.7, with one Bonferroni correction across five confirmatory dimensions. `results/mannwhitney_results.csv` keeps its historical filename. It stores the paired Wilcoxon results.

## Pipeline

```
AMI Meeting Corpus
    │
    ▼ scripts/01_preprocess.py
2,685 candidate facilitation moves
199 sampled contexts (149 Ideate, 50 Prototype/Evaluate)
    │
    ▼ notebooks/colab_llm_inference.ipynb or scripts/02_generate_llm_responses.py
data/llm_responses.csv
199 contexts × temperatures 0.3, 0.7, 1.0
    │
    ▼ secondary generation, phase line removed, seed 42
data/llm_responses_nophase.csv
    │
    ▼ scripts/03_extract_features.py
data/features_processed.csv (796 rows)
    │
    ├── scripts/04_statistical_analysis.py
    │     paired Wilcoxon, descriptive temperature correlations,
    │     bootstrap intervals, extreme cases
    ├── scripts/05_robustness_analysis.py
    │     length matching (n = 73), duplicate-context removal,
    │     LDA topic counts, TF–IDF distance, mixed-effects models
    ├── scripts/07_phase_nophase.py
    │     secondary human vs no-label and label vs no-label tests
    ├── scripts/06_generate_figures.py
    └── scripts/08_positioning_figures.py
          plots already published effect sizes; it does not recompute tests
```

## What the current code matches

- Contextual semantic distance is $1 - \cos$, using `all-MiniLM-L6-v2`. The manuscript uses the same formula. An earlier draft divided by 2; that version is not the result-producing formula.
- Length-matched analysis uses pairs whose relative word-count difference is below 0.50. The current count is $n = 73$, not the earlier unpaired $n = 72$ analysis.
- Mixed-effects models use `is_llm` as a fixed effect and `context_id` as a random intercept.
- The no-phase file is a secondary sensitivity check. It is not a sixth confirmatory dimension.
- Figure 6 and Figure 7 are drawn from the published statistics by `scripts/08_positioning_figures.py`.

## Checked file hashes

`REPRODUCIBILITY_MANIFEST.json` lists SHA-256 hashes for the sampled contexts, both response files, the feature table, the result tables used in the manuscript, and the directiveness validation files. `scripts/validate_reproduction.py` checks those hashes.
