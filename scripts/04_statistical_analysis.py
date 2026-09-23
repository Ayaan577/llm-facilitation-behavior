"""
04_statistical_analysis.py - Mann-Whitney U tests, Bonferroni correction,
Spearman temperature correlations, power analysis, and extreme case analysis.
Phase 4 of the LLM Facilitation Behavioral Study pipeline.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Robust fallback paths for inputs
INPUT_PATHS = [
    os.path.join(BASE, "data", "features_processed.csv"),
    os.path.join(BASE, "data", "outputs", "feature_scores.csv"),
    os.path.join(BASE, "data", "outputs", "features_processed.csv"),
]

RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STATS_CSV     = os.path.join(RESULTS_DIR, "mannwhitney_results.csv")
CORR_CSV      = os.path.join(RESULTS_DIR, "spearman_temperature.csv")
POWER_CSV     = os.path.join(RESULTS_DIR, "power_analysis.csv")
EXTREME_CSV   = os.path.join(RESULTS_DIR, "extreme_cases.csv")
EFFECT_SIZES  = os.path.join(RESULTS_DIR, "effect_sizes.csv")

FEATURES = ["novel_score", "directive_score", "specificity_score",
            "empathy_score", "divergence_score", "phase_score"]
TEMPS = [("llm_t03", 0.3), ("llm_t07", 0.7), ("llm_t10", 1.0)]
N_COMPARISONS = len(FEATURES) * len(TEMPS)  # 18


def run_comparison(human_scores, llm_scores, feature_name, llm_label, temp_val):
    """Mann-Whitney U test with Bonferroni correction."""
    h = human_scores.dropna().values
    l = llm_scores.dropna().values
    if len(h) < 3 or len(l) < 3:
        return None
    stat, p = mannwhitneyu(h, l, alternative="two-sided")
    n1, n2 = len(h), len(l)
    r = 1 - (2 * stat) / (n1 * n2)  # rank-biserial correlation
    p_bonf = min(p * N_COMPARISONS, 1.0)
    return {
        "dimension": feature_name,
        "llm_temp": llm_label,
        "temp_value": temp_val,
        "human_n": n1,
        "llm_n": n2,
        "human_median": round(float(np.median(h)), 4),
        "llm_median": round(float(np.median(l)), 4),
        "U_statistic": float(stat),
        "effect_size_r": round(float(r), 4),
        "p_value": float(p),
        "p_bonferroni": float(p_bonf),
        "significant": p_bonf < 0.05,
    }


def main():
    print("=" * 65)
    print("  Phase 4: Statistical Analysis")
    print("=" * 65)

    input_csv = None
    for p in INPUT_PATHS:
        if os.path.exists(p):
            input_csv = p
            break

    if not input_csv:
        print(f"[ERROR] Features CSV not found in candidate paths.")
        return

    print(f"\n[1] Loading feature data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"    Total feature rows: {len(df)}")
    print(f"    Sources: {df['source'].value_counts().to_dict()}")

    human_df = df[df["source"] == "human"]

    # ── 1. Mann-Whitney U tests ──────────────────────────────────────────────
    print("\n[2] Running Mann-Whitney U tests (6 features × 3 temperatures)...")
    stats_results = []
    for feature in FEATURES:
        for llm_label, temp_val in TEMPS:
            llm_df = df[df["source"] == llm_label]
            result = run_comparison(
                human_df[feature], llm_df[feature],
                feature, llm_label, temp_val
            )
            if result:
                stats_results.append(result)
                sig = "***" if result["p_bonferroni"] < 0.001 else \
                      "**" if result["p_bonferroni"] < 0.01 else \
                      "*" if result["p_bonferroni"] < 0.05 else "ns"
                print(f"    {feature:20s} vs {llm_label}: "
                      f"H_med={result['human_median']:.4f} "
                      f"L_med={result['llm_median']:.4f} "
                      f"r={result['effect_size_r']:+.3f} "
                      f"p_bonf={result['p_bonferroni']:.4f} {sig}")

    df_stats = pd.DataFrame(stats_results)
    df_stats.to_csv(STATS_CSV, index=False)
    print(f"\n    Saved main statistics -> {STATS_CSV}")
    n_sig = df_stats["significant"].sum()
    print(f"    Significant comparisons: {n_sig}/{len(df_stats)}")

    # ── 2. Temperature correlations ──────────────────────────────────────────
    print("\n[3] Spearman correlations: LLM temperature vs. feature scores...")
    llm_df = df[df["source"].str.startswith("llm")].copy()
    temp_map = {"llm_t03": 0.3, "llm_t07": 0.7, "llm_t10": 1.0}
    llm_df["temperature"] = llm_df["source"].map(temp_map)

    corr_results = []
    for feature in FEATURES:
        vals = llm_df[[feature, "temperature"]].dropna()
        if len(vals) > 5:
            rho, p = spearmanr(vals["temperature"], vals[feature])
            corr_results.append({
                "dimension": feature,
                "spearman_rho": round(float(rho), 4),
                "p_value": float(p),
                "significant": p < 0.05,
                "n": len(vals),
            })
            sig = "*" if p < 0.05 else "ns"
            print(f"    {feature:20s}: rho={rho:+.4f}  p={p:.4f} {sig}")

    df_corr = pd.DataFrame(corr_results)
    df_corr.to_csv(CORR_CSV, index=False)
    print(f"    Saved temperature correlations -> {CORR_CSV}")

    # ── 3. Post-hoc Power Analysis ───────────────────────────────────────────
    print("\n[4] Post-hoc power analysis (Mann-Whitney U rank-biserial approximation)...")
    from scipy.stats import norm as _norm

    def power_mw(n1, n2, effect_r, alpha=0.05):
        d = 2 * effect_r / np.sqrt(max(1 - effect_r**2, 1e-10))
        ncp = d * np.sqrt((n1 * n2) / (n1 + n2))
        z_a = _norm.ppf(1 - alpha / 2)
        return 1 - _norm.cdf(z_a - ncp) + _norm.cdf(-z_a - ncp)

    power_rows = []
    for _, row in df_stats[df_stats["llm_temp"] == "llm_t07"].iterrows():
        pwr = power_mw(int(row["human_n"]), int(row["llm_n"]), abs(float(row["effect_size_r"])))
        power_rows.append({
            "dimension": row["dimension"],
            "n_human": int(row["human_n"]),
            "n_llm": int(row["llm_n"]),
            "effect_r": float(row["effect_size_r"]),
            "power": round(pwr, 4),
            "adequate": pwr >= 0.80,
        })
        print(f"    {row['dimension']:20s}: power={pwr:.4f}  {'✓' if pwr>=0.80 else '✗'}")

    df_power = pd.DataFrame(power_rows)
    df_power.to_csv(POWER_CSV, index=False)
    print(f"    Saved power analysis -> {POWER_CSV}")

    # ── 4. Extreme Case Analysis ─────────────────────────────────────────────
    print("\n[5] Extreme case analysis...")
    human_pivot = human_df.set_index("context_id")[FEATURES + ["response_text"]]
    llm07_df = df[df["source"] == "llm_t07"].set_index("context_id")[FEATURES + ["response_text"]]
    common_ids = human_pivot.index.intersection(llm07_df.index)

    extreme_rows = []
    for feature in FEATURES:
        h_scores = human_pivot.loc[common_ids, feature]
        l_scores = llm07_df.loc[common_ids, feature]
        gaps = (h_scores - l_scores).abs().dropna().sort_values(ascending=False)

        for cid in gaps.head(15).index:
            extreme_rows.append({
                "dimension": feature,
                "type": "most_different",
                "context_id": cid,
                "human_score": float(h_scores.get(cid, np.nan)),
                "llm_t07_score": float(l_scores.get(cid, np.nan)),
                "gap": float(gaps.get(cid, np.nan)),
                "human_response": str(human_pivot.loc[cid, "response_text"])[:200],
                "llm_response": str(llm07_df.loc[cid, "response_text"])[:200],
            })

    df_extreme = pd.DataFrame(extreme_rows)
    df_extreme.to_csv(EXTREME_CSV, index=False)
    print(f"    Saved extreme cases -> {EXTREME_CSV}")

    print("\n" + "=" * 65)
    print("  Phase 4 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
