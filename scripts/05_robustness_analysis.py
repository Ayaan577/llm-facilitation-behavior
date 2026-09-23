"""
05_robustness_analysis.py - Robustness Analyses for LLM Facilitation Study.
Implements:
1. Length-matched subsampling (n=72 per group, 5 word-count bins)
2. Partial Spearman correlations controlling for response length
3. Linear Mixed-Effects models (statsmodels MixedLM with session_id random intercepts)
4. Generates results/robustness_table6.csv

[PROVENANCE NOTE]:
Uses the historical result-producing dataset (data/features_processed.csv)
and session metadata to reproduce the exact Table 6 robustness metrics reported in the paper.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CSV = os.path.join(BASE, "data", "features_processed.csv")
MOVES_CSV    = os.path.join(BASE, "data", "processed", "facilitator_moves_sampled_199.csv")
RESULTS_DIR  = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_TABLE6 = os.path.join(RESULTS_DIR, "robustness_table6.csv")
OUT_LENGTH = os.path.join(RESULTS_DIR, "length_robustness.csv")
OUT_MIXED  = os.path.join(RESULTS_DIR, "mixed_effects_results.csv")

FEATURES = ["novel_score", "directive_score", "specificity_score",
            "empathy_score", "divergence_score", "phase_score"]


def partial_spearman(x, y, z):
    """
    Partial Spearman correlation between x and y controlling for z.
    Uses rank transformation + linear regression residuals.
    """
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    rz = pd.Series(z).rank()

    # Residuals of rx on rz
    slope_x, intercept_x = np.polyfit(rz, rx, 1)
    res_x = rx - (slope_x * rz + intercept_x)

    # Residuals of ry on rz
    slope_y, intercept_y = np.polyfit(rz, ry, 1)
    res_y = ry - (slope_y * rz + intercept_y)

    rho, p = spearmanr(res_x, res_y)
    return float(rho), float(p)


def run_length_matched_mw(df, feature, source_target="llm_t07", seed=42):
    """Create 5 word-count bins and sample proportionally for length matching."""
    df_h = df[df["source"] == "human"].copy()
    df_l = df[df["source"] == source_target].copy()

    df_h["word_count"] = df_h["response_text"].apply(lambda t: len(str(t).split()))
    df_l["word_count"] = df_l["response_text"].apply(lambda t: len(str(t).split()))

    combined_wc = pd.concat([df_h["word_count"], df_l["word_count"]])
    bins = pd.qcut(combined_wc, q=5, labels=False, duplicates="drop")

    df_h["bin"] = bins.loc[df_h.index]
    df_l["bin"] = bins.loc[df_l.index]

    np.random.seed(seed)
    sampled_h, sampled_l = [], []
    for b in range(5):
        h_bin = df_h[df_h["bin"] == b]
        l_bin = df_l[df_l["bin"] == b]
        min_n = min(len(h_bin), len(l_bin))
        if min_n > 0:
            sampled_h.append(h_bin.sample(n=min_n, random_state=seed))
            sampled_l.append(l_bin.sample(n=min_n, random_state=seed))

    if not sampled_h:
        return None

    sub_h = pd.concat(sampled_h)
    sub_l = pd.concat(sampled_l)

    h_vals = sub_h[feature].values
    l_vals = sub_l[feature].values

    stat, p = mannwhitneyu(h_vals, l_vals, alternative="two-sided")
    n1, n2 = len(h_vals), len(l_vals)
    r = 1 - (2 * stat) / (n1 * n2)
    p_bonf = min(p * 18, 1.0)

    return {
        "dimension": feature,
        "n_matched": len(sub_h),
        "rank_biserial_r": round(float(r), 4),
        "p_raw": float(p),
        "p_bonferroni": float(p_bonf),
        "significant": p_bonf < 0.05,
    }


def main():
    print("=" * 65)
    print("  Phase 5: Robustness Analysis (Table 6)")
    print("=" * 65)

    if not os.path.exists(FEATURES_CSV):
        print(f"[ERROR] Features file not found at {FEATURES_CSV}")
        return

    df_feat = pd.read_csv(FEATURES_CSV)
    df_feat["word_count"] = df_feat["response_text"].apply(lambda t: len(str(t).split()))
    print(f"Loaded {len(df_feat)} feature rows.")

    # Load session metadata if available
    df_moves = None
    if os.path.exists(MOVES_CSV):
        df_moves = pd.read_csv(MOVES_CSV)
        print(f"Loaded {len(df_moves)} move contexts with session metadata.")

    # Merge session_id
    if df_moves is not None and "session_id" in df_moves.columns:
        context_session_map = dict(zip(df_moves["context_id"], df_moves["session_id"]))
        df_feat["session_id"] = df_feat["context_id"].map(context_session_map)
    else:
        df_feat["session_id"] = "session_" + df_feat["context_id"].astype(str)

    table6_rows = []

    # 1. Length-matched subsampling (Human vs LLM T=0.7)
    print("\n[1] Length-matched subsampling (Human vs LLM T=0.7)...")
    for feat in FEATURES:
        res = run_length_matched_mw(df_feat, feat, source_target="llm_t07")
        if res:
            table6_rows.append({
                "dimension": feat,
                "analysis_type": "length_matched_subsampling",
                "sample_n": res["n_matched"],
                "effect_size": res["rank_biserial_r"],
                "p_bonferroni": res["p_bonferroni"],
                "significant": res["significant"],
                "note": f"n={res['n_matched']} per group matched by word count"
            })
            print(f"    {feat:20s}: r={res['rank_biserial_r']:+.3f}  p_bonf={res['p_bonferroni']:.4f}  {'✓' if res['significant'] else 'ns'}")

    # 2. Partial Spearman correlations controlling for length
    print("\n[2] Partial Spearman correlations (controlling for response length)...")
    df_pair = df_feat[df_feat["source"].isin(["human", "llm_t07"])].copy()
    df_pair["is_llm"] = (df_pair["source"] == "llm_t07").astype(int)

    for feat in FEATURES:
        rho, p = partial_spearman(df_pair["is_llm"], df_pair[feat], df_pair["word_count"])
        p_bonf = min(p * 18, 1.0)
        sig = p_bonf < 0.05
        table6_rows.append({
            "dimension": feat,
            "analysis_type": "partial_spearman_length_controlled",
            "sample_n": len(df_pair),
            "effect_size": round(rho, 4),
            "p_bonferroni": float(p_bonf),
            "significant": sig,
            "note": "Spearman partial rho controlling for word count"
        })
        print(f"    {feat:20s}: partial_rho={rho:+.3f}  p_bonf={p_bonf:.4f}  {'✓' if sig else 'ns'}")

    # 3. Linear Mixed-Effects Models (session random intercepts)
    print("\n[3] Linear Mixed-Effects Models (Fixed: source, Random: session_id)...")
    for feat in FEATURES:
        try:
            model = smf.mixedlm(f"{feat} ~ is_llm", df_pair, groups=df_pair["session_id"])
            mfit = model.fit()
            beta = float(mfit.params.get("is_llm", np.nan))
            pval = float(mfit.pvalues.get("is_llm", np.nan))
            se = float(mfit.bse.get("is_llm", np.nan))
            zval = float(mfit.tvalues.get("is_llm", np.nan))
            p_bonf = min(pval * 6, 1.0)
            sig = p_bonf < 0.05
            table6_rows.append({
                "dimension": feat,
                "analysis_type": "linear_mixed_effects_session_random",
                "sample_n": len(df_pair),
                "effect_size": round(beta, 4),
                "p_bonferroni": float(p_bonf),
                "significant": sig,
                "note": f"Beta={beta:.4f}, SE={se:.4f}, z={zval:.2f}"
            })
            print(f"    {feat:20s}: Beta={beta:+.4f}  z={zval:+.2f}  p_bonf={p_bonf:.4e}  {'✓' if sig else 'ns'}")
        except Exception as e:
            print(f"    {feat:20s}: MixedLM Error: {e}")

    df_t6 = pd.DataFrame(table6_rows)
    df_t6.to_csv(OUT_TABLE6, index=False)
    print(f"\n[4] Robustness Table 6 saved -> {OUT_TABLE6}")

    print("\n" + "=" * 65)
    print("  Phase 5 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
