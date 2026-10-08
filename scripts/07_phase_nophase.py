"""Secondary analysis: phase-vocabulary alignment without the phase label.

The original prompt included "Current design phase: {phase}".
llm_responses_nophase.csv was generated with that line removed.
Same model, temperatures, truncation, and phase scorer as scripts/03_extract_features.py.
This does not change the primary confirmatory family.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from importlib.machinery import SourceFileLoader

feat = SourceFileLoader(
    "feat03", os.path.join(BASE, "scripts", "03_extract_features.py")
).load_module()

PHASED = os.path.join(BASE, "data", "llm_responses.csv")
NOPHASE = os.path.join(BASE, "data", "llm_responses_nophase.csv")
FEATURES = os.path.join(BASE, "data", "features_processed.csv")
OUT = os.path.join(BASE, "results", "phase_nophase_comparison.csv")


def phase_score(text, phase):
    return feat.f6_phase_score(str(text), str(phase))


def effect_r(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    nonzero = diffs[diffs != 0]
    if n < 1 or len(nonzero) < 1:
        return np.nan, np.nan, n, 0
    res = stats.wilcoxon(nonzero, alternative="two-sided", method="approx")
    r = abs(float(res.zstatistic)) / np.sqrt(n)
    return r, float(res.pvalue), n, int(len(nonzero))


def bootstrap_r(diffs, seed=42, n_boot=2000):
    # Match scripts/04: numpy global seed, not a Generator.
    np.random.seed(seed)
    vals = []
    diffs = np.asarray(diffs, dtype=float)
    for _ in range(n_boot):
        idx = np.random.choice(len(diffs), len(diffs), replace=True)
        r, _, _, _ = effect_r(diffs[idx])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(name, a, b, family_m=None):
    """Paired contrast of a minus b. Positive mean means a is higher."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    r, p, n, n_nonzero = effect_r(diffs)
    lo, hi = bootstrap_r(diffs)
    p_adj = min(p * family_m, 1.0) if family_m and np.isfinite(p) else p
    return {
        "contrast": name,
        "n_pairs": n,
        "n_nonzero": n_nonzero,
        "mean_a": round(float(np.mean(a)), 4),
        "mean_b": round(float(np.mean(b)), 4),
        "median_a": round(float(np.median(a)), 4),
        "median_b": round(float(np.median(b)), 4),
        "mean_diff_a_minus_b": round(float(np.mean(diffs)), 4),
        "n_a_higher": int(np.sum(a > b)),
        "n_b_higher": int(np.sum(b > a)),
        "n_tie": int(np.sum(a == b)),
        "effect_r": round(float(r), 4) if np.isfinite(r) else np.nan,
        "ci_95_lo": round(lo, 4),
        "ci_95_hi": round(hi, 4),
        "p_raw": float(p) if np.isfinite(p) else np.nan,
        "p_bonferroni_secondary": float(p_adj) if np.isfinite(p_adj) else np.nan,
        "family_m": family_m if family_m else 1,
    }


def main():
    phased = pd.read_csv(PHASED)
    nophase = pd.read_csv(NOPHASE)
    features = pd.read_csv(FEATURES)

    print(f"phased rows {len(phased)} nophase rows {len(nophase)}")
    print("nophase columns", list(nophase.columns))
    print("prompt_included_phase", nophase["prompt_included_phase"].value_counts().to_dict())
    print("seed", nophase["seed"].value_counts().to_dict())
    for col in ("llm_t03_valid", "llm_t07_valid", "llm_t10_valid"):
        print(col, int(nophase[col].sum()), "/", len(nophase))

    ids_p = set(phased["context_id"])
    ids_n = set(nophase["context_id"])
    print("id overlap", len(ids_p & ids_n), "only phased", len(ids_p - ids_n), "only nophase", len(ids_n - ids_p))

    merged = phased.merge(nophase, on="context_id", suffixes=("_phased", "_nophase"))
    print("merged", len(merged))
    phase_match = (merged["session_phase_phased"] == merged["session_phase_nophase"]).all()
    human_match = (merged["human_response_phased"] == merged["human_response_nophase"]).all()
    ctx_match = (merged["context_text_phased"] == merged["context_text_nophase"]).all()
    print("phase labels match", bool(phase_match))
    print("human text match", bool(human_match))
    print("context text match", bool(ctx_match))

    phase = merged["session_phase_phased"]
    for label, text_col in (
        ("human", "human_response_phased"),
        ("phased_t03", "llm_t03_phased"),
        ("phased_t07", "llm_t07_phased"),
        ("phased_t10", "llm_t10_phased"),
        ("nophase_t03", "llm_t03_nophase"),
        ("nophase_t07", "llm_t07_nophase"),
        ("nophase_t10", "llm_t10_nophase"),
    ):
        merged[label] = [phase_score(t, p) for t, p in zip(merged[text_col], phase)]

    # Reproduce published phase scores from the stored responses.
    pub = features.pivot(index="context_id", columns="source", values="phase_score")
    check = merged.set_index("context_id")
    for src, col in (("human", "human"), ("llm_t07", "phased_t07"), ("llm_t03", "phased_t03"), ("llm_t10", "phased_t10")):
        both = pd.concat([pub[src].rename("pub"), check[col].rename("rescore")], axis=1).dropna()
        max_abs = (both["pub"] - both["rescore"]).abs().max()
        print(f"rescore vs published {src}: n={len(both)} max_abs={max_abs:.6f} "
              f"pub_mean={both['pub'].mean():.4f} rescore_mean={both['rescore'].mean():.4f}")

    rows = []
    # Pre-specified secondary family at T=0.7: two matched contrasts.
    rows.append(summarize(
        "human_vs_nophase_T0.7", merged["human"], merged["nophase_t07"], family_m=2
    ))
    rows.append(summarize(
        "phased_vs_nophase_T0.7", merged["phased_t07"], merged["nophase_t07"], family_m=2
    ))
    # Temperature strata of the conditioning contrast. Bonferroni across 3 temperatures.
    for temp, pcol, ncol in (
        ("0.3", "phased_t03", "nophase_t03"),
        ("0.7", "phased_t07", "nophase_t07"),
        ("1.0", "phased_t10", "nophase_t10"),
    ):
        rows.append(summarize(
            f"phased_vs_nophase_T{temp}_tempfamily", merged[pcol], merged[ncol], family_m=3
        ))
        rows.append(summarize(
            f"human_vs_nophase_T{temp}_unadjusted", merged["human"], merged[ncol], family_m=None
        ))

    # Original published contrast, rescored, for a sanity check.
    rows.append(summarize(
        "human_vs_phased_T0.7_rescore", merged["human"], merged["phased_t07"], family_m=None
    ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("wrote", OUT)

    for col in ("human", "phased_t07", "nophase_t07"):
        s = merged[col]
        print(f"{col}: mean={s.mean():.4f} median={s.median():.4f} "
              f"pct_zero={(s == 0).mean():.3f}")


if __name__ == "__main__":
    main()
