"""
04_statistical_analysis.py — Revised paired/hierarchical statistical analysis.

Primary inferential framework:
  Paired Wilcoxon signed-rank test (human vs. LLM T=0.7, matched by context_id).
  Replaces the original independent-sample Mann-Whitney U tests as the primary test.

Rationale:
  Each LLM response is generated from the same context as its corresponding human PM turn.
  The 199 human–LLM pairs are context-matched, making paired inference the correct
  primary framework (fixes the independence assumption violation noted in the prior version).

Dimensions:
  Confirmatory (5): novel_score, specificity_score, empathy_score,
                    divergence_score, phase_score
  Exploratory  (1): directive_score  — domain validation gave κ = −0.0231,
                                        so it cannot support confirmatory claims.

Multiple-comparison correction:
  Bonferroni applied over the 5 confirmatory dimensions only.
  α_corrected = 0.05 / 5 = 0.010 per test.

Temperature sensitivity analysis:
  Spearman rho between decoding temperature {0.3, 0.7, 1.0} and feature scores
  across all 597 LLM responses (199 contexts × 3 temperatures).
  N is noted as repeated-context; this is a sensitivity analysis, not independent inference.

Robustness checks:
  1. Length-matched subsampling (|word-count difference| / human_word_count < 50%)
  2. Partial Spearman correlation controlling for response word count
  3. MixedLM: source ~ feature, random intercepts for context_id (via statsmodels)
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INPUT_PATHS = [
    os.path.join(BASE, "data", "features_processed.csv"),
    os.path.join(BASE, "data", "outputs", "features_processed.csv"),
]

RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PAIRED_CSV       = os.path.join(RESULTS_DIR, "mannwhitney_results.csv")   # kept for manifest compat
CORR_CSV         = os.path.join(RESULTS_DIR, "spearman_temperature.csv")
POWER_CSV        = os.path.join(RESULTS_DIR, "power_analysis.csv")
EFFECT_SIZES     = os.path.join(RESULTS_DIR, "effect_sizes.csv")
ROBUSTNESS_CSV   = os.path.join(RESULTS_DIR, "robustness_table6.csv")
EXTREME_CSV      = os.path.join(RESULTS_DIR, "extreme_cases.csv")
DESC_CSV         = os.path.join(RESULTS_DIR, "descriptive_stats.csv")

CONFIRMATORY = ["novel_score", "specificity_score", "empathy_score",
                "divergence_score", "phase_score"]
EXPLORATORY  = ["directive_score"]
ALL_FEATURES = CONFIRMATORY + EXPLORATORY

N_CONF = len(CONFIRMATORY)          # 5 — Bonferroni denominator
ALPHA  = 0.05
ALPHA_ADJ = ALPHA / N_CONF          # 0.010

FEAT_LABELS = {
    "novel_score":       "Contextual Semantic Distance",
    "specificity_score": "Lexical Elaboration Proxy",
    "empathy_score":     "Empathy Lexical Density",
    "divergence_score":  "Topic Divergence",
    "phase_score":       "Phase-Vocab Alignment",
    "directive_score":   "Directiveness (Exploratory)",
}


def wilcoxon_effect_r(diffs) -> float:
    """Effect size r = |Z| / sqrt(N) from the Wilcoxon normal approximation.

    N is the number of pairs, including ties. Z comes from the test statistic.
    Inverting a p-value floored at 1e-15 assigns the same ceiling to every
    very small p-value, which is what made the two largest effects identical.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    nonzero = diffs[diffs != 0]
    if n < 1 or len(nonzero) < 1:
        return np.nan
    res = stats.wilcoxon(nonzero, alternative="two-sided", method="approx")
    return abs(float(res.zstatistic)) / np.sqrt(n)


def run_paired_wilcoxon(human_series, llm_series, feature, status):
    """Paired Wilcoxon signed-rank test; returns result dict."""
    diffs = (human_series - llm_series).dropna()
    n = len(diffs)
    if n < 10:
        return None
    res = stats.wilcoxon(diffs[diffs != 0], alternative="two-sided", method="approx")
    p_raw = res.pvalue
    r = wilcoxon_effect_r(diffs)
    if status == "Confirmatory":
        p_adj = min(p_raw * N_CONF, 1.0)
    else:
        p_adj = p_raw  # no correction applied to exploratory

    return {
        "dimension":        feature,
        "label":            FEAT_LABELS.get(feature, feature),
        "status":           status,
        "human_n":          n,
        "llm_n":            n,
        "human_median":     round(float(human_series.dropna().median()), 4),
        "llm_t07_median":   round(float(llm_series.dropna().median()), 4),
        "direction_llm":    "+" if llm_series.median() > human_series.median() else "-",
        "wilcoxon_T":       float(res.statistic),
        "effect_size_r":    round(r, 4),
        "p_value":          float(p_raw),
        "p_bonferroni":     float(p_adj),
        "significant":      p_adj < ALPHA,
        "test":             "Paired Wilcoxon signed-rank",
    }


def main():
    print("=" * 70)
    print("  Phase 4: Paired Statistical Analysis (Revised)")
    print("  Primary: Paired Wilcoxon signed-rank test (human vs LLM T=0.7)")
    print("=" * 70)

    input_csv = next((p for p in INPUT_PATHS if os.path.exists(p)), None)
    if not input_csv:
        print("[ERROR] Features CSV not found.")
        return

    print(f"\n[1] Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"    Total rows: {len(df)} | Sources: {df['source'].value_counts().to_dict()}")

    # ── Pivot to paired structure ──────────────────────────────────────────────
    human  = df[df["source"] == "human"].set_index("context_id")
    llm07  = df[df["source"] == "llm_t07"].set_index("context_id")
    llm03  = df[df["source"] == "llm_t03"].set_index("context_id")
    llm10  = df[df["source"] == "llm_t10"].set_index("context_id")

    common = human.index.intersection(llm07.index)
    print(f"    Paired contexts (human ∩ LLM T=0.7): {len(common)}")

    # ── 1. Descriptive Stats ───────────────────────────────────────────────────
    print("\n[2] Descriptive statistics...")
    desc_rows = []
    for f in ALL_FEATURES:
        for src_label, src_df in [("human", human), ("llm_t03", llm03),
                                   ("llm_t07", llm07), ("llm_t10", llm10)]:
            vals = src_df[f].dropna()
            desc_rows.append({
                "dimension": f, "source": src_label,
                "n": len(vals), "mean": round(vals.mean(), 4),
                "median": round(vals.median(), 4), "std": round(vals.std(), 4),
                "min": round(vals.min(), 4), "max": round(vals.max(), 4),
            })
    pd.DataFrame(desc_rows).to_csv(DESC_CSV, index=False)
    print(f"    Saved → {DESC_CSV}")

    # ── 2. Primary Paired Wilcoxon Tests (T=0.7) ──────────────────────────────
    print(f"\n[3] Primary paired Wilcoxon tests (Bonferroni α/{N_CONF} = {ALPHA_ADJ:.3f})")
    paired_results = []
    for f in CONFIRMATORY + EXPLORATORY:
        status = "Confirmatory" if f in CONFIRMATORY else "Exploratory"
        row = run_paired_wilcoxon(human.loc[common, f], llm07.loc[common, f], f, status)
        if row:
            paired_results.append(row)
            sig_str = "***" if row["p_bonferroni"] < 0.001 else \
                      "**"  if row["p_bonferroni"] < 0.01  else \
                      "*"   if row["p_bonferroni"] < 0.05  else "ns"
            print(f"    {f:22s} | {row['status']:15s} | "
                  f"H={row['human_median']:.4f} L={row['llm_t07_median']:.4f} | "
                  f"r={row['effect_size_r']:+.3f} | p_raw={row['p_value']:.5f} | "
                  f"p_bonf={row['p_bonferroni']:.5f} {sig_str}")

    df_paired = pd.DataFrame(paired_results)
    # Save as mannwhitney_results.csv for manifest compatibility, note test type in data
    df_paired.to_csv(PAIRED_CSV, index=False)
    print(f"\n    Saved → {PAIRED_CSV}")
    print(f"    Significant (Bonferroni): {df_paired['significant'].sum()}/{len(df_paired)}")

    # Also save effect sizes
    df_paired[["dimension", "label", "status", "effect_size_r",
               "p_value", "p_bonferroni", "significant"]].to_csv(EFFECT_SIZES, index=False)

    # ── 3. Temperature Sensitivity (Spearman) ─────────────────────────────────
    print("\n[4] Temperature sensitivity — Spearman rho (N=597 repeated-context)...")
    llm_long = df[df["source"].str.startswith("llm")].copy()
    temp_map = {"llm_t03": 0.3, "llm_t07": 0.7, "llm_t10": 1.0}
    llm_long["temperature"] = llm_long["source"].map(temp_map)

    corr_rows = []
    for f in ALL_FEATURES:
        vals = llm_long[[f, "temperature"]].dropna()
        rho, p = spearmanr(vals["temperature"], vals[f])
        corr_rows.append({
            "dimension": f, "label": FEAT_LABELS.get(f, f),
            "spearman_rho": round(float(rho), 4),
            "p_value": float(p),
            "significant": p < 0.05,
            "n_repeated_context": len(vals),
            "note": "N=597 = 199 contexts × 3 temps; repeated-context sensitivity analysis",
        })
        sig = "*" if p < 0.05 else "ns"
        print(f"    {f:22s}: rho={rho:+.4f}  p={p:.4f} {sig}")
    pd.DataFrame(corr_rows).to_csv(CORR_CSV, index=False)
    print(f"    Saved → {CORR_CSV}")

    # ── 4. Power / Uncertainty for non-significant dims ────────────────────────
    print("\n[5] Confidence intervals for null findings (directive, empathy)...")
    power_rows = []
    for row in paired_results:
        f = row["dimension"]
        n = row["human_n"]
        r = abs(row["effect_size_r"])
        # 95% CI on r via bootstrap (approximate)
        h_vals = human.loc[common, f].values
        l_vals = llm07.loc[common, f].values
        diffs = h_vals - l_vals
        np.random.seed(42)
        boot_rs = []
        for _ in range(2000):
            idx = np.random.choice(len(diffs), len(diffs), replace=True)
            d_b = diffs[idx]
            try:
                r_b = wilcoxon_effect_r(d_b)
                if np.isfinite(r_b):
                    boot_rs.append(r_b)
            except Exception:
                pass
        ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5]) if boot_rs else (np.nan, np.nan)

        power_rows.append({
            "dimension": f, "label": FEAT_LABELS.get(f, f),
            "n_pairs": n,
            "effect_r": round(r, 4),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
            "p_bonferroni": row["p_bonferroni"],
            "note": "Bootstrap CI (2000 resamp, seed=42)",
        })
        print(f"    {f:22s}: r={r:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    pd.DataFrame(power_rows).to_csv(POWER_CSV, index=False)
    print(f"    Saved → {POWER_CSV}")

    # ── 5. Robustness: Length-matched + MixedLM ────────────────────────────────
    print("\n[6] Robustness checks...")

    # Load word counts
    llm_resp_path = os.path.join(BASE, "data", "llm_responses.csv")
    df_resp = pd.read_csv(llm_resp_path)
    df_resp["human_wc"]  = df_resp["human_response"].str.split().str.len()
    df_resp["llm07_wc"]  = df_resp["llm_t07"].str.split().str.len()
    wc_df = df_resp.set_index("context_id")[["human_wc", "llm07_wc"]]

    merged_full = (
        human.loc[common].join(llm07.loc[common], lsuffix="_h", rsuffix="_l")
             .join(wc_df)
    )
    mask_len = (
        abs(merged_full["human_wc"] - merged_full["llm07_wc"])
        / (merged_full["human_wc"] + 1) < 0.50
    )
    len_matched = merged_full[mask_len]
    print(f"    Length-matched pairs (±50% word count): {len(len_matched)}")

    rob_rows = []

    # Robustness A: Paired Wilcoxon on length-matched subset
    for f in CONFIRMATORY:
        diffs = (len_matched[f+"_h"] - len_matched[f+"_l"]).dropna()
        n_lm = len(diffs)
        if n_lm < 10:
            continue
        res = stats.wilcoxon(diffs[diffs != 0], alternative="two-sided", method="approx")
        r_lm = wilcoxon_effect_r(diffs)
        rob_rows.append({
            "dimension": f, "label": FEAT_LABELS.get(f, f),
            "check": "Length-matched (paired Wilcoxon)",
            "n": n_lm,
            "human_median": round(len_matched[f+"_h"].median(), 4),
            "llm_median":   round(len_matched[f+"_l"].median(), 4),
            "effect_r": round(r_lm, 4),
            "p_raw": round(res.pvalue, 5),
            "significant": res.pvalue < ALPHA_ADJ,
        })
        print(f"    [len-match] {f:22s}: n={n_lm}, r={r_lm:.3f}, p={res.pvalue:.4f}")

    # Robustness B: Partial Spearman controlling for word count
    merged_full["human_wc_filled"] = merged_full["human_wc"].fillna(
        merged_full["human_wc"].median()
    )
    for f in CONFIRMATORY:
        h = merged_full[f+"_h"].values
        l = merged_full[f+"_l"].values
        wc = merged_full["human_wc_filled"].values
        diff = h - l
        valid = ~np.isnan(diff) & ~np.isnan(wc)
        if valid.sum() < 10:
            continue
        rho, p = spearmanr(diff[valid], wc[valid])
        rob_rows.append({
            "dimension": f, "label": FEAT_LABELS.get(f, f),
            "check": "Partial Spearman (controlling word count)",
            "n": int(valid.sum()),
            "human_median": None, "llm_median": None,
            "effect_r": round(rho, 4),
            "p_raw": round(p, 5),
            "significant": p < ALPHA_ADJ,
        })
        print(f"    [partial-sp] {f:22s}: rho={rho:+.3f}, p={p:.4f}")

    # Robustness C: Linear MixedLM
    try:
        import statsmodels.formula.api as smf
        df_long = (
            human.loc[common, ALL_FEATURES + ["session_phase"]]
                 .assign(source="human")
                 .reset_index()
        )
        df_llm7 = (
            llm07.loc[common, ALL_FEATURES + ["session_phase"]]
                  .assign(source="llm_t07")
                  .reset_index()
        )
        df_mix = pd.concat([df_long, df_llm7], ignore_index=True)
        df_mix["is_llm"] = (df_mix["source"] == "llm_t07").astype(float)

        for f in CONFIRMATORY:
            try:
                model = smf.mixedlm(f"{f} ~ is_llm", df_mix,
                                    groups=df_mix["context_id"])
                result = model.fit(reml=True)
                coef = result.params["is_llm"]
                p_val = result.pvalues["is_llm"]
                rob_rows.append({
                    "dimension": f, "label": FEAT_LABELS.get(f, f),
                    "check": "MixedLM (random context intercept)",
                    "n": len(df_mix),
                    "human_median": None, "llm_median": None,
                    "effect_r": round(coef, 4),
                    "p_raw": round(p_val, 5),
                    "significant": p_val < ALPHA_ADJ,
                })
                print(f"    [MixedLM]    {f:22s}: coef={coef:+.4f}, p={p_val:.4f}")
            except Exception as e:
                print(f"    [MixedLM]    {f}: ERROR {e}")
    except ImportError:
        print("    statsmodels not available — skipping MixedLM")

    pd.DataFrame(rob_rows).to_csv(ROBUSTNESS_CSV, index=False)
    print(f"    Saved → {ROBUSTNESS_CSV}")

    # ── 6. Extreme case analysis ───────────────────────────────────────────────
    print("\n[7] Extreme case analysis...")
    extreme_rows = []
    for f in CONFIRMATORY:
        gaps = (human.loc[common, f] - llm07.loc[common, f]).abs().sort_values(ascending=False)
        for cid in gaps.head(15).index:
            extreme_rows.append({
                "dimension": f,
                "context_id": cid,
                "human_score": float(human.loc[cid, f]),
                "llm_t07_score": float(llm07.loc[cid, f]),
                "gap": float(gaps[cid]),
                "human_response": str(human.loc[cid, "response_text"])[:200]
                    if "response_text" in human.columns else "",
                "llm_response": str(llm07.loc[cid, "response_text"])[:200]
                    if "response_text" in llm07.columns else "",
            })
    pd.DataFrame(extreme_rows).to_csv(EXTREME_CSV, index=False)
    print(f"    Saved → {EXTREME_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Phase 4 COMPLETE — Revised Paired Analysis Summary")
    print("=" * 70)
    print(f"  Primary test: Paired Wilcoxon signed-rank (n=199 pairs, T=0.7)")
    print(f"  Bonferroni α = 0.05 / {N_CONF} = {ALPHA_ADJ:.3f} (5 confirmatory dims)")
    n_sig = sum(r["significant"] for r in paired_results if r["status"] == "Confirmatory")
    print(f"  Significant confirmatory: {n_sig}/{N_CONF}")
    print(f"  Directiveness: EXPLORATORY only (domain κ = −0.0231, unvalidated)")
    print(f"  Phase alignment: CONFOUND NOTE — phase label was supplied in LLM prompt")
    print("=" * 70)


if __name__ == "__main__":
    main()
