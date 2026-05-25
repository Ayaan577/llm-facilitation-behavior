"""
04_statistical_analysis.py - Mann-Whitney U tests, temperature correlations,
and extreme case analysis.
Phase 4 of the CHI 2027 pipeline.
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
INPUT_CSV = os.path.join(BASE, "data", "outputs", "feature_scores.csv")
STATS_CSV = os.path.join(BASE, "data", "outputs", "stats_results.csv")
CORR_CSV = os.path.join(BASE, "data", "outputs", "temperature_correlations.csv")
EXTREME_CSV = os.path.join(BASE, "data", "outputs", "extreme_cases.csv")

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
        "feature": feature_name,
        "llm_temp": llm_label,
        "temp_value": temp_val,
        "human_n": n1,
        "llm_n": n2,
        "human_median": float(np.median(h)),
        "human_mean": float(np.mean(h)),
        "llm_median": float(np.median(l)),
        "llm_mean": float(np.mean(l)),
        "U_statistic": float(stat),
        "p_value": float(p),
        "p_bonferroni": float(p_bonf),
        "effect_size_r": float(r),
        "significant": p_bonf < 0.05,
    }


def main():
    print("=" * 60)
    print("  Phase 4: Statistical Analysis")
    print("=" * 60)

    # Load features
    print(f"\n[1] Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"    Total rows: {len(df)}")
    print(f"    Sources: {df['source'].value_counts().to_dict()}")

    human_df = df[df["source"] == "human"]

    # ─── Mann-Whitney U tests ────────────────────────────────────────────
    print("\n[2] Running Mann-Whitney U tests (6 features x 3 temperatures)...")
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
    print(f"\n    Saved to {STATS_CSV}")
    n_sig = df_stats["significant"].sum()
    print(f"    Significant comparisons: {n_sig}/{len(df_stats)}")

    # ─── Temperature correlations ────────────────────────────────────────
    print("\n[3] Spearman correlations: temperature vs feature scores...")
    llm_df = df[df["source"].str.startswith("llm")]
    temp_map = {"llm_t03": 0.3, "llm_t07": 0.7, "llm_t10": 1.0}
    llm_df = llm_df.copy()
    llm_df["temperature"] = llm_df["source"].map(temp_map)

    corr_results = []
    for feature in FEATURES:
        vals = llm_df[[feature, "temperature"]].dropna()
        if len(vals) > 5:
            rho, p = spearmanr(vals["temperature"], vals[feature])
            corr_results.append({
                "feature": feature,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "significant": p < 0.05,
                "n": len(vals),
            })
            sig = "*" if p < 0.05 else "ns"
            print(f"    {feature:20s}: rho={rho:+.4f}  p={p:.4f} {sig}")

    df_corr = pd.DataFrame(corr_results)
    df_corr.to_csv(CORR_CSV, index=False)
    print(f"    Saved to {CORR_CSV}")

    # ─── Extreme case analysis ───────────────────────────────────────────
    print("\n[4] Extreme case analysis...")
    # Pivot to get human and llm_t07 side by side
    human_pivot = human_df.set_index("context_id")[FEATURES + ["response_text"]]
    llm07_df = df[df["source"] == "llm_t07"].set_index("context_id")[FEATURES + ["response_text"]]

    common_ids = human_pivot.index.intersection(llm07_df.index)
    print(f"    Common context_ids: {len(common_ids)}")

    extreme_rows = []
    for feature in FEATURES:
        h_scores = human_pivot.loc[common_ids, feature]
        l_scores = llm07_df.loc[common_ids, feature]
        gaps = (h_scores - l_scores).abs()
        gaps = gaps.dropna().sort_values(ascending=False)

        # Top 15 most different
        for cid in gaps.head(15).index:
            extreme_rows.append({
                "feature": feature,
                "type": "most_different",
                "context_id": cid,
                "human_score": float(h_scores.get(cid, np.nan)),
                "llm_t07_score": float(l_scores.get(cid, np.nan)),
                "gap": float(gaps.get(cid, np.nan)),
                "human_response": str(human_pivot.loc[cid, "response_text"])[:200],
                "llm_response": str(llm07_df.loc[cid, "response_text"])[:200],
            })

        # Top 15 most similar
        for cid in gaps.tail(15).index:
            extreme_rows.append({
                "feature": feature,
                "type": "most_similar",
                "context_id": cid,
                "human_score": float(h_scores.get(cid, np.nan)),
                "llm_t07_score": float(l_scores.get(cid, np.nan)),
                "gap": float(gaps.get(cid, np.nan)),
                "human_response": str(human_pivot.loc[cid, "response_text"])[:200],
                "llm_response": str(llm07_df.loc[cid, "response_text"])[:200],
            })

    df_extreme = pd.DataFrame(extreme_rows)
    df_extreme.to_csv(EXTREME_CSV, index=False)
    print(f"    Saved {len(df_extreme)} extreme cases to {EXTREME_CSV}")

    # ─── Power Analysis (Fix 6) ──────────────────────────────────────────
    print("\n[5] Post-hoc power analysis (Mann-Whitney U approximation)...")
    from scipy.stats import norm as _norm

    def power_mw(n1, n2, effect_r, alpha=0.05):
        """Approximate power for Mann-Whitney U via rank-biserial r → Cohen's d."""
        d = 2 * effect_r / np.sqrt(max(1 - effect_r**2, 1e-10))
        ncp = d * np.sqrt((n1 * n2) / (n1 + n2))
        z_a = _norm.ppf(1 - alpha / 2)
        return 1 - _norm.cdf(z_a - ncp) + _norm.cdf(-z_a - ncp)

    power_rows = []
    for _, row in df_stats[df_stats["llm_temp"] == "llm_t07"].iterrows():
        pwr = power_mw(int(row["human_n"]), int(row["llm_n"]),
                       abs(float(row["effect_size_r"])))
        power_rows.append({
            "feature": row["feature"],
            "n_human": int(row["human_n"]),
            "n_llm": int(row["llm_n"]),
            "effect_r": float(row["effect_size_r"]),
            "power": round(pwr, 4),
            "adequate": pwr >= 0.80,
        })
        print(f"    {row['feature']:20s}: power={pwr:.4f}  {'✓' if pwr>=0.80 else '✗'}")

    df_power = pd.DataFrame(power_rows)
    power_out = os.path.join(BASE, "data", "outputs", "power_analysis.csv")
    df_power.to_csv(power_out, index=False)
    print(f"    Power analysis saved to {power_out}")
    print(f"    Dimensions with power >= 0.80: {df_power['adequate'].sum()}/{len(df_power)}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Total feature vectors: {len(df)}")
    print(f"  Significant comparisons: {n_sig}/18")
    most_sig = df_stats.loc[df_stats["p_bonferroni"].idxmin()] if len(df_stats) > 0 else None
    if most_sig is not None:
        print(f"  Most significant: {most_sig['feature']} "
              f"(Human med={most_sig['human_median']:.4f}, "
              f"LLM med={most_sig['llm_median']:.4f}, "
              f"r={most_sig['effect_size_r']:.3f}, "
              f"p_bonf={most_sig['p_bonferroni']:.4e})")

    print("\n" + "=" * 60)
    print("  Phase 4 COMPLETE")
    print("=" * 60)




if __name__ == "__main__":
    main()
