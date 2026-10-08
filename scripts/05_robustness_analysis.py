"""
05_robustness_analysis.py

Sensitivity checks that the paired primary analysis does not cover.
Does not overwrite results/robustness_table6.csv (that file is written by
04_statistical_analysis.py). The old unpaired length-matched Mann-Whitney
routine was removed because it ignored the matched context structure.

Checks:
1. Unique-context reanalysis (drop verbatim duplicate context + human turn).
2. Friedman tests of temperature within context, plus MixedLM.
3. LDA topic-count sensitivity for Jensen-Shannon divergence.
4. TF-IDF cosine distance as a second semantic-distance representation.
5. Joint pattern: human more distant in embedding space and less divergent in topic space.
6. Empathy dictionary: empathy-word hits versus hedge-word hits.
7. Whether stored phase labels can be rebuilt from turn position.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CSV = os.path.join(BASE, "data", "features_processed.csv")
RESP_CSV = os.path.join(BASE, "data", "llm_responses.csv")
CAND_CSV = os.path.join(BASE, "data", "processed", "facilitator_moves_candidates.csv")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIRMATORY = [
    "novel_score",
    "specificity_score",
    "empathy_score",
    "divergence_score",
    "phase_score",
]
EXPLORATORY = ["directive_score"]
ALL_FEATURES = CONFIRMATORY + EXPLORATORY
N_CONF = len(CONFIRMATORY)
ALPHA = 0.05

EMPATHY_WORDS = {
    "understand", "feel", "feeling", "difficult", "challenge", "challenging",
    "appreciate", "concern", "concerned", "sense", "noticed",
    "hear", "hearing", "recognize", "aware", "acknowledge", "respect",
    "frustrating", "exciting", "interesting", "curious", "wonder",
}
HEDGING_WORDS = {
    "perhaps", "might", "maybe", "could", "possibly", "seems", "appears",
    "wonder", "suppose", "imagine", "consider", "think", "feel", "believe",
}

LABELS = {
    "novel_score": "Contextual semantic distance",
    "specificity_score": "Lexical elaboration proxy",
    "empathy_score": "Empathy lexical density",
    "divergence_score": "Topic divergence",
    "phase_score": "Phase-vocabulary alignment",
    "directive_score": "Directiveness (exploratory)",
}


def effect_r(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    nonzero = diffs[diffs != 0]
    if n < 10 or len(nonzero) < 1:
        return np.nan, np.nan, n
    res = stats.wilcoxon(nonzero, alternative="two-sided", method="approx")
    return abs(float(res.zstatistic)) / np.sqrt(n), float(res.pvalue), n


def paired_row(human, llm, feature, analysis, ids):
    h = human.loc[ids, feature]
    l = llm.loc[ids, feature]
    r, p, n = effect_r(h.values - l.values)
    status = "Confirmatory" if feature in CONFIRMATORY else "Exploratory"
    p_adj = min(p * N_CONF, 1.0) if status == "Confirmatory" and np.isfinite(p) else p
    return {
        "analysis": analysis,
        "dimension": feature,
        "label": LABELS[feature],
        "status": status,
        "n_pairs": n,
        "human_median": round(float(h.median()), 4),
        "llm_median": round(float(l.median()), 4),
        "human_mean": round(float(h.mean()), 4),
        "llm_mean": round(float(l.mean()), 4),
        "effect_r": round(float(r), 4) if np.isfinite(r) else np.nan,
        "p_raw": float(p) if np.isfinite(p) else np.nan,
        "p_bonferroni": float(p_adj) if np.isfinite(p_adj) else np.nan,
        "significant_bonferroni": bool(p_adj < ALPHA) if np.isfinite(p_adj) else False,
        "direction": "human_higher" if h.mean() > l.mean() else "llm_higher",
    }


def topic_dist(model, matrix):
    dist = model.transform(matrix)
    dist = dist + 1e-10
    return dist / dist.sum(axis=1, keepdims=True)


def main():
    print("=" * 68)
    print("  Sensitivity checks (paired structure preserved)")
    print("=" * 68)

    feat = pd.read_csv(FEATURES_CSV)
    resp = pd.read_csv(RESP_CSV)
    human = feat[feat.source == "human"].set_index("context_id")
    llm = feat[feat.source == "llm_t07"].set_index("context_id")
    by_temp = {
        0.3: feat[feat.source == "llm_t03"].set_index("context_id"),
        0.7: llm,
        1.0: feat[feat.source == "llm_t10"].set_index("context_id"),
    }
    ids = human.index.intersection(llm.index).intersection(resp.set_index("context_id").index)
    ids = pd.Index(sorted(ids))
    resp = resp.set_index("context_id").loc[ids]

    rows = []

    # 1. Verbatim duplicate contexts
    print("\n[1] Unique context + human turn")
    keyed = resp.copy()
    keyed["dup_key"] = keyed["context_text"].astype(str) + " || " + keyed["human_response"].astype(str)
    unique_ids = keyed.sort_index().drop_duplicates("dup_key").index
    n_dup_groups = int(keyed["dup_key"].duplicated(keep=False).sum() / 2)
    print(f"    All pairs: {len(ids)} | unique pairs: {len(unique_ids)} | duplicate groups: {n_dup_groups}")
    for feature in ALL_FEATURES:
        rows.append(paired_row(human, llm, feature, "unique_context_wilcoxon", unique_ids))
        row = rows[-1]
        print(f"    {feature:22s} n={row['n_pairs']} r={row['effect_r']:.3f} "
              f"p_bonf={row['p_bonferroni']:.4g} {row['direction']}")

    # 2. Temperature: Friedman within context, then MixedLM
    print("\n[2] Temperature within context (Friedman)")
    temp_rows = []
    long_parts = []
    for temp, frame in by_temp.items():
        part = frame.loc[ids, ALL_FEATURES].copy()
        part["temperature"] = temp
        part["context_id"] = part.index
        long_parts.append(part.reset_index(drop=True))
    long = pd.concat(long_parts, ignore_index=True)

    for feature in ALL_FEATURES:
        samples = [by_temp[t].loc[ids, feature].astype(float).values for t in (0.3, 0.7, 1.0)]
        stat, p = stats.friedmanchisquare(*samples)
        status = "Confirmatory" if feature in CONFIRMATORY else "Exploratory"
        p_adj = min(p * N_CONF, 1.0) if status == "Confirmatory" else p
        medians = {f"median_t{str(t).replace('.', '')}": round(float(np.median(s)), 4)
                   for t, s in zip((0.3, 0.7, 1.0), samples)}
        temp_rows.append({
            "dimension": feature,
            "label": LABELS[feature],
            "status": status,
            "test": "Friedman",
            "n_contexts": len(ids),
            "statistic": round(float(stat), 4),
            "p_raw": float(p),
            "p_bonferroni": float(p_adj),
            "significant_bonferroni": bool(p_adj < ALPHA),
            **medians,
        })
        print(f"    {feature:22s} chi2={stat:.2f} p={p:.4g} p_bonf={p_adj:.4g}")

        try:
            sub = long[["context_id", "temperature", feature]].dropna()
            model = smf.mixedlm(
                f"{feature} ~ C(temperature, Treatment(reference=0.7))",
                sub,
                groups=sub["context_id"],
            )
            fit = model.fit(reml=True)
            for name in fit.params.index:
                if name == "Intercept" or name == "Group Var":
                    continue
                temp_rows.append({
                    "dimension": feature,
                    "label": LABELS[feature],
                    "status": status,
                    "test": "MixedLM_vs_T0.7",
                    "n_contexts": sub["context_id"].nunique(),
                    "statistic": round(float(fit.params[name]), 4),
                    "p_raw": float(fit.pvalues[name]),
                    "p_bonferroni": np.nan,
                    "significant_bonferroni": False,
                    "term": name,
                })
                print(f"      {name}: beta={fit.params[name]:+.4f} p={fit.pvalues[name]:.4g}")
        except Exception as exc:
            print(f"    MixedLM {feature}: {exc}")

    pd.DataFrame(temp_rows).to_csv(
        os.path.join(RESULTS_DIR, "temperature_repeated.csv"), index=False
    )

    # 3. LDA topic-count sensitivity
    print("\n[3] LDA topic-count sensitivity")
    context = resp["context_text"].astype(str)
    human_text = resp["human_response"].astype(str)
    llm_text = resp["llm_t07"].astype(str)
    corpus = pd.concat([
        context, human_text, llm_text,
        resp["llm_t03"].astype(str), resp["llm_t10"].astype(str),
    ], ignore_index=True)
    vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.9)
    matrix = vectorizer.fit_transform(corpus)
    n_docs = len(context)
    ctx_m = matrix[:n_docs]
    hum_m = matrix[n_docs: 2 * n_docs]
    llm_m = matrix[2 * n_docs: 3 * n_docs]

    lda_rows = []
    for k in (5, 10, 15, 20, 30):
        lda = LatentDirichletAllocation(
            n_components=k, random_state=42, learning_method="batch", max_iter=25
        )
        lda.fit(matrix)
        ctx_d = topic_dist(lda, ctx_m)
        hum_d = topic_dist(lda, hum_m)
        llm_d = topic_dist(lda, llm_m)
        h_js = np.array([jensenshannon(ctx_d[i], hum_d[i]) for i in range(n_docs)])
        l_js = np.array([jensenshannon(ctx_d[i], llm_d[i]) for i in range(n_docs)])
        r, p, n = effect_r(h_js - l_js)
        p_adj = min(p * len((5, 10, 15, 20, 30)), 1.0)
        lda_rows.append({
            "n_topics": k,
            "n_pairs": n,
            "human_median_js": round(float(np.median(h_js)), 4),
            "llm_median_js": round(float(np.median(l_js)), 4),
            "effect_r": round(float(r), 4),
            "p_raw": float(p),
            "p_bonferroni_across_k": float(p_adj),
            "llm_higher": bool(np.median(l_js) > np.median(h_js)),
            "significant": bool(p_adj < ALPHA),
        })
        print(f"    k={k:2d} Hmd={np.median(h_js):.3f} Lmd={np.median(l_js):.3f} "
              f"r={r:.3f} p_bonf={p_adj:.4g}")
    pd.DataFrame(lda_rows).to_csv(
        os.path.join(RESULTS_DIR, "lda_topic_sensitivity.csv"), index=False
    )

    # 4. TF-IDF semantic distance
    print("\n[4] TF-IDF contextual distance")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(pd.concat([context, human_text, llm_text], ignore_index=True))
    ctx_v = tfidf_matrix[:n_docs]
    hum_v = tfidf_matrix[n_docs: 2 * n_docs]
    llm_v = tfidf_matrix[2 * n_docs: 3 * n_docs]
    h_dist = 1 - np.array([
        cosine_similarity(ctx_v[i], hum_v[i])[0, 0] for i in range(n_docs)
    ])
    l_dist = 1 - np.array([
        cosine_similarity(ctx_v[i], llm_v[i])[0, 0] for i in range(n_docs)
    ])
    r, p, n = effect_r(h_dist - l_dist)
    rows.append({
        "analysis": "tfidf_semantic_distance",
        "dimension": "tfidf_distance",
        "label": "TF-IDF contextual distance",
        "status": "Sensitivity",
        "n_pairs": n,
        "human_median": round(float(np.median(h_dist)), 4),
        "llm_median": round(float(np.median(l_dist)), 4),
        "human_mean": round(float(np.mean(h_dist)), 4),
        "llm_mean": round(float(np.mean(l_dist)), 4),
        "effect_r": round(float(r), 4),
        "p_raw": float(p),
        "p_bonferroni": float(p),
        "significant_bonferroni": bool(p < ALPHA),
        "direction": "human_higher" if np.mean(h_dist) > np.mean(l_dist) else "llm_higher",
    })
    print(f"    r={r:.3f} p={p:.4g} direction={rows[-1]['direction']}")

    # 5. Joint pattern on the published embedding and LDA scores
    print("\n[5] Joint embedding-distance and topic-divergence pattern")
    h_nov = human.loc[ids, "novel_score"]
    l_nov = llm.loc[ids, "novel_score"]
    h_div = human.loc[ids, "divergence_score"]
    l_div = llm.loc[ids, "divergence_score"]
    both = ((h_nov > l_nov) & (h_div < l_div)).sum()
    print(f"    Pairs with human distance higher AND human topic divergence lower: "
          f"{int(both)}/{len(ids)} ({both / len(ids):.3f})")
    rows.append({
        "analysis": "joint_distance_and_divergence",
        "dimension": "novel_and_divergence",
        "label": "Human more distant and less topic-divergent",
        "status": "Sensitivity",
        "n_pairs": int(len(ids)),
        "human_median": np.nan,
        "llm_median": np.nan,
        "human_mean": np.nan,
        "llm_mean": np.nan,
        "effect_r": round(float(both / len(ids)), 4),
        "p_raw": np.nan,
        "p_bonferroni": np.nan,
        "significant_bonferroni": False,
        "direction": f"{int(both)}_of_{len(ids)}",
    })

    # 6. Empathy dictionary composition
    print("\n[6] Empathy versus hedge tokens")
    def counts(series):
        emp = hedge = 0
        for text in series.astype(str):
            tokens = text.lower().split()
            emp += sum(token in EMPATHY_WORDS and token not in HEDGING_WORDS for token in tokens)
            hedge += sum(token in HEDGING_WORDS for token in tokens)
        return emp, hedge

    for name, series in [("human", human_text), ("llm_t07", llm_text)]:
        emp, hedge = counts(series)
        print(f"    {name:10s} empathy-only hits={emp} hedge hits={hedge}")
        rows.append({
            "analysis": "empathy_dictionary",
            "dimension": name,
            "label": "Token hits",
            "status": "Sensitivity",
            "n_pairs": len(series),
            "human_median": emp,
            "llm_median": hedge,
            "human_mean": np.nan,
            "llm_mean": np.nan,
            "effect_r": round(emp / (emp + hedge), 4) if (emp + hedge) else np.nan,
            "p_raw": np.nan,
            "p_bonferroni": np.nan,
            "significant_bonferroni": False,
            "direction": "empathy_share_of_hits",
        })

    # 7. Phase position diagnostic
    print("\n[7] Phase label versus turn position")
    if os.path.exists(CAND_CSV):
        cand = pd.read_csv(CAND_CSV)
        session_max = cand.groupby("session_id")["turn_id"].max()
        samp = pd.read_csv(
            os.path.join(BASE, "data", "processed", "facilitator_moves_sampled_199.csv")
        )
        samp = samp[samp["context_id"].isin(ids)].copy()
        samp["session_max_turn"] = samp["session_id"].map(session_max)
        samp["pct_among_candidates"] = samp["turn_id"] / samp["session_max_turn"].clip(lower=1)
        print("    Stored phases:", samp["session_phase"].value_counts().to_dict())
        print(f"    Candidate-relative position: min={samp['pct_among_candidates'].min():.2f} "
              f"median={samp['pct_among_candidates'].median():.2f}")
        rows.append({
            "analysis": "phase_position_diagnostic",
            "dimension": "session_phase",
            "label": "Cannot rebuild full-session phase boundaries",
            "status": "Not run",
            "n_pairs": int(samp["session_phase"].eq("Empathize/Define").sum()),
            "human_median": round(float(samp["pct_among_candidates"].median()), 4),
            "llm_median": np.nan,
            "human_mean": np.nan,
            "llm_mean": np.nan,
            "effect_r": np.nan,
            "p_raw": np.nan,
            "p_bonferroni": np.nan,
            "significant_bonferroni": False,
            "direction": "session length in all turns is not stored",
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "sensitivity_analyses.csv"), index=False
    )
    print("\n  Wrote results/sensitivity_analyses.csv")
    print("  Wrote results/temperature_repeated.csv")
    print("  Wrote results/lda_topic_sensitivity.csv")
    print("=" * 68)


if __name__ == "__main__":
    main()
