"""
Step 1C+1D+1E: Rerun feature extraction, statistics (with bootstrap CIs),
and figure generation on the resampled data.

Run AFTER llm_responses_resampled.csv is placed in data/outputs/.
"""
import os, sys, warnings, time
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV  = os.path.join(BASE, "data", "outputs", "llm_responses_resampled.csv")
FEAT_CSV   = os.path.join(BASE, "data", "outputs", "feature_scores_resampled.csv")
STATS_CSV  = os.path.join(BASE, "data", "results", "stats_resampled.csv")
CORR_CSV   = os.path.join(BASE, "data", "results", "temperature_correlations.csv")
POWER_CSV  = os.path.join(BASE, "data", "results", "power_analysis.csv")
FIG_DIR    = os.path.join(BASE, "figures")
LDA_DIR    = os.path.join(BASE, "models", "lda_model")
DIR_DIR    = os.path.join(BASE, "models", "directiveness_classifier", "final")

for d in [os.path.dirname(STATS_CSV), FIG_DIR, LDA_DIR, DIR_DIR]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# LEXICONS
# ═══════════════════════════════════════════════════════════════
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
PHASE_VOCAB = {
    "Empathize/Define": ["user", "feel", "experience", "need", "observe",
                         "empathy", "problem", "who", "why", "pain"],
    "Ideate": ["idea", "what if", "imagine", "brainstorm", "how might",
               "explore", "possibilities", "creative", "alternative"],
    "Prototype/Evaluate": ["build", "test", "try", "make", "sketch",
                           "prototype", "criteria", "feedback", "evaluate"],
}

# ═══════════════════════════════════════════════════════════════
# FEATURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def f1_novelty(embedder, ctx, resp):
    from sklearn.metrics.pairwise import cosine_similarity
    c = embedder.encode([ctx]); r = embedder.encode([resp])
    return float(1 - cosine_similarity(c, r)[0][0])

_dir_tok = None; _dir_mod = None
def _load_dir():
    global _dir_tok, _dir_mod
    if _dir_mod is not None: return True
    if not os.path.exists(os.path.join(DIR_DIR, "config.json")): return False
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        _dir_tok = DistilBertTokenizer.from_pretrained(DIR_DIR)
        _dir_mod = DistilBertForSequenceClassification.from_pretrained(DIR_DIR)
        _dir_mod.eval(); return True
    except: return False

def f2_directiveness(text):
    if _load_dir():
        import torch
        inp = _dir_tok(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad(): logits = _dir_mod(**inp).logits
        return float(torch.softmax(logits, dim=1)[0][1])
    # Heuristic fallback
    tl = text.lower().strip(); score = 0.0
    for p in ["let's", "we need", "you should", "focus on", "move to", "start with"]:
        if tl.startswith(p): score += 0.3; break
    if tl.endswith("?"): score -= 0.3
    for p in ["what do you think", "how might", "what if"]:
        if p in tl: score -= 0.15
    return max(0.0, min(1.0, 0.5 + score))

def f3_specificity(nlp, text):
    doc = nlp(text)
    words = [t for t in doc if not t.is_space]
    if not words: return 0.0
    ner_d = len(doc.ents) / len(words)
    lemmas = [t.lemma_.lower() for t in words]
    ttr = len(set(lemmas)) / len(lemmas)
    awl = min(sum(len(t.text) for t in words) / len(words) / 10.0, 1.0)
    return float((ner_d + ttr + awl) / 3.0)

def f4_empathy(text):
    tokens = text.lower().split()
    if not tokens: return 0.0
    e = sum(1 for t in tokens if t in EMPATHY_WORDS)
    h = sum(1 for t in tokens if t in HEDGING_WORDS)
    return float((e + h) / len(tokens))

def f5_divergence(lda, dic, ctx, resp):
    def dist(t):
        bow = dic.doc2bow(t.lower().split())
        d = dict(lda.get_document_topics(bow, minimum_probability=0))
        a = np.array([d.get(i, 0.0) for i in range(lda.num_topics)]) + 1e-10
        return a / a.sum()
    return float(jensenshannon(dist(ctx), dist(resp)))

def f6_phase(text, phase):
    tokens = text.lower().split()
    if not tokens: return 0.0
    pw = PHASE_VOCAB.get(phase, [])
    hits = sum(1 for t in tokens if any(p in t for p in pw))
    return float(hits / len(tokens))

# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════

def bootstrap_r_ci(group1, group2, n_boot=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    boot_r = []
    for _ in range(n_boot):
        s1 = rng.choice(group1, size=len(group1), replace=True)
        s2 = rng.choice(group2, size=len(group2), replace=True)
        U, _ = mannwhitneyu(s1, s2, alternative='two-sided')
        r = 1 - (2 * U) / (len(s1) * len(s2))
        boot_r.append(r)
    lower = np.percentile(boot_r, (100 - ci) / 2)
    upper = np.percentile(boot_r, 100 - (100 - ci) / 2)
    return round(lower, 2), round(upper, 2)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Check input ──
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found.")
        print("Run generate_llm_responses_colab.ipynb on Colab first,")
        print("then place llm_responses_resampled.csv in data/outputs/")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Phases: {df['session_phase'].value_counts().to_dict()}")

    # ── Load NLP models ──
    print("\n[1] Loading NLP models...")
    from sentence_transformers import SentenceTransformer
    import spacy
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = spacy.load("en_core_web_sm")
    print("    sentence-transformers + spaCy loaded")

    # ── Train LDA ──
    print("\n[2] Training LDA...")
    from gensim.models import LdaModel
    from gensim.corpora import Dictionary
    all_texts = []
    for _, row in df.iterrows():
        for col in ["context_text", "human_response", "llm_t03", "llm_t07", "llm_t10"]:
            if col in df.columns:
                t = str(row.get(col, ""))
                if len(t.split()) > 2: all_texts.append(t)
    tokenized = [t.lower().split() for t in all_texts]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    corpus = [dictionary.doc2bow(d) for d in tokenized]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=15,
                   passes=10, random_state=42, chunksize=200)
    print(f"    LDA: 15 topics, vocab={len(dictionary)}")

    # ── Extract features ──
    print("\n[3] Extracting features...")
    sources = {"human": "human_response", "llm_t03": "llm_t03",
               "llm_t07": "llm_t07", "llm_t10": "llm_t10"}
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        if (i+1) % 20 == 0 or i == 0: print(f"    Row {i+1}/{len(df)}")
        ctx = str(row.get("context_text", ""))
        phase = str(row.get("session_phase", ""))
        cid = row.get("context_id", i)
        for sname, col in sources.items():
            resp = str(row.get(col, ""))
            if not resp or resp == "nan" or len(resp.split()) < 3: continue
            results.append({
                "context_id": cid, "source": sname, "response_text": resp,
                "session_phase": phase,
                "novel_score": round(f1_novelty(embedder, ctx, resp), 4),
                "directive_score": round(f2_directiveness(resp), 4),
                "specificity_score": round(f3_specificity(nlp, resp), 4),
                "empathy_score": round(f4_empathy(resp), 4),
                "divergence_score": round(f5_divergence(lda, dictionary, ctx, resp), 4),
                "phase_score": round(f6_phase(resp, phase), 4),
            })

    df_feat = pd.DataFrame(results)
    df_feat.to_csv(FEAT_CSV, index=False)
    print(f"\n    Saved {len(df_feat)} feature vectors to {FEAT_CSV}")
    FEATURES = ["novel_score", "directive_score", "specificity_score",
                "empathy_score", "divergence_score", "phase_score"]
    print("\n    Medians by source:")
    print(df_feat.groupby("source")[FEATURES].median().round(4).to_string())

    # ── Statistics ──
    print("\n[4] Mann-Whitney U tests (Bonferroni, 18 comparisons)...")
    TEMPS = [("llm_t03", 0.3), ("llm_t07", 0.7), ("llm_t10", 1.0)]
    N_COMP = len(FEATURES) * len(TEMPS)
    human_df = df_feat[df_feat["source"] == "human"]
    stats_rows = []

    for feat in FEATURES:
        for llm_lab, tval in TEMPS:
            llm_sub = df_feat[df_feat["source"] == llm_lab]
            h = human_df[feat].dropna().values
            l = llm_sub[feat].dropna().values
            if len(h) < 3 or len(l) < 3: continue
            U, p = mannwhitneyu(h, l, alternative="two-sided")
            n1, n2 = len(h), len(l)
            r = 1 - 2*U/(n1*n2)
            p_bonf = min(p * N_COMP, 1.0)
            sig = p_bonf < 0.05

            # Bootstrap CI for T=0.7
            ci_lo, ci_hi = np.nan, np.nan
            if llm_lab == "llm_t07" and sig:
                ci_lo, ci_hi = bootstrap_r_ci(h, l)

            star = "***" if p_bonf<0.001 else "**" if p_bonf<0.01 else "*" if p_bonf<0.05 else "ns"
            ci_str = f" [{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else ""
            print(f"    {feat:22s} vs {llm_lab}: H={np.median(h):.4f} L={np.median(l):.4f} "
                  f"r={r:+.3f}{ci_str} p_bonf={p_bonf:.4e} {star}")

            stats_rows.append({
                "feature": feat, "llm_temp": llm_lab, "temp_value": tval,
                "human_n": n1, "llm_n": n2,
                "human_median": float(np.median(h)), "human_mean": float(np.mean(h)),
                "llm_median": float(np.median(l)), "llm_mean": float(np.mean(l)),
                "U_statistic": float(U), "p_value": float(p),
                "p_bonferroni": float(p_bonf), "effect_size_r": float(r),
                "significant": sig,
                "ci_lower": float(ci_lo) if not np.isnan(ci_lo) else None,
                "ci_upper": float(ci_hi) if not np.isnan(ci_hi) else None,
            })

    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(STATS_CSV, index=False)
    n_sig = df_stats["significant"].sum()
    print(f"\n    Significant: {n_sig}/18. Saved to {STATS_CSV}")

    # ── Temperature correlations ──
    print("\n[5] Spearman temperature correlations...")
    llm_all = df_feat[df_feat["source"].str.startswith("llm")].copy()
    temp_map = {"llm_t03": 0.3, "llm_t07": 0.7, "llm_t10": 1.0}
    llm_all["temperature"] = llm_all["source"].map(temp_map)
    corr_rows = []
    for feat in FEATURES:
        vals = llm_all[[feat, "temperature"]].dropna()
        if len(vals) > 5:
            rho, p = spearmanr(vals["temperature"], vals[feat])
            star = "*" if p < 0.05 else "ns"
            print(f"    {feat:22s}: rho={rho:+.4f} p={p:.4f} {star}")
            corr_rows.append({"feature": feat, "spearman_rho": rho, "p_value": p, "significant": p < 0.05})
    pd.DataFrame(corr_rows).to_csv(CORR_CSV, index=False)

    # ── Power analysis ──
    print("\n[6] Post-hoc power analysis...")
    def power_mw(n1, n2, r, alpha=0.05):
        d = 2*r / np.sqrt(max(1-r**2, 1e-10))
        ncp = d * np.sqrt(n1*n2/(n1+n2))
        z_a = _norm.ppf(1 - alpha/2)
        return 1 - _norm.cdf(z_a - ncp) + _norm.cdf(-z_a - ncp)
    pwr_rows = []
    for _, row in df_stats[df_stats["llm_temp"] == "llm_t07"].iterrows():
        pwr = power_mw(int(row["human_n"]), int(row["llm_n"]), abs(float(row["effect_size_r"])))
        ok = pwr >= 0.80
        print(f"    {row['feature']:22s}: power={pwr:.4f} {'Y' if ok else 'N'}")
        pwr_rows.append({"feature": row["feature"], "effect_r": row["effect_size_r"],
                         "power": round(pwr, 4), "adequate": ok})
    pd.DataFrame(pwr_rows).to_csv(POWER_CSV, index=False)

    # ── Figures ──
    print("\n[7] Generating figures...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})
    colors = {"human": "#2563EB", "llm_t03": "#059669", "llm_t07": "#D97706", "llm_t10": "#DC2626"}
    labels_map = {"human": "Human", "llm_t03": "LLM T=0.3", "llm_t07": "LLM T=0.7", "llm_t10": "LLM T=1.0"}

    # Figure 1: PCA
    from sklearn.preprocessing import StandardScaler
    X = df_feat[FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    for src in ["human", "llm_t03", "llm_t07", "llm_t10"]:
        mask = df_feat["source"] == src
        ax.scatter(X2[mask, 0], X2[mask, 1], c=colors[src], label=labels_map[src],
                   alpha=0.5, s=30, marker="o" if src == "human" else "^")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(); ax.set_title("PCA of Behavioral Feature Space")
    fig.savefig(os.path.join(FIG_DIR, "pca_plot.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "figure1_pca.png"))
    plt.close()
    print(f"    Fig 1: PCA (PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)")
    print(f"    PC1 loadings: {dict(zip(FEATURES, [f'{v:+.3f}' for v in pca.components_[0]]))}")
    print(f"    PC2 loadings: {dict(zip(FEATURES, [f'{v:+.3f}' for v in pca.components_[1]]))}")

    # Figure 2: Heatmap
    medians = df_feat.groupby("source")[FEATURES].median()
    medians = medians.loc[["human", "llm_t03", "llm_t07", "llm_t10"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    short_names = [f.replace("_score", "").capitalize() for f in FEATURES]
    im = ax.imshow(medians.values.T, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Human", "T=0.3", "T=0.7", "T=1.0"])
    ax.set_yticks(range(6)); ax.set_yticklabels(short_names)
    for i in range(6):
        for j in range(4):
            ax.text(j, i, f"{medians.values.T[i,j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Median Score")
    ax.set_title("Median Feature Scores by Source")
    fig.savefig(os.path.join(FIG_DIR, "heatmap.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "figure3_temperature_heatmap.png"))
    plt.close()
    print("    Fig 2: Heatmap")

    # Figure 3: Radar (normalized)
    import matplotlib.patches as mpatches
    human_med_raw = df_feat[df_feat["source"]=="human"][FEATURES].median().values
    llm_med_raw = df_feat[df_feat["source"]=="llm_t07"][FEATURES].median().values
    # Normalize each axis by max(human, llm) so all axes are [0, 1]
    max_vals = np.maximum(human_med_raw, llm_med_raw)
    max_vals[max_vals == 0] = 1.0  # avoid division by zero
    human_med_norm = human_med_raw / max_vals
    llm_med_norm = llm_med_raw / max_vals
    angles = np.linspace(0, 2*np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    human_med_norm = list(human_med_norm) + [human_med_norm[0]]
    llm_med_norm = list(llm_med_norm) + [llm_med_norm[0]]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, human_med_norm, "o-", color="#2563EB", linewidth=2, label="Human")
    ax.fill(angles, human_med_norm, alpha=0.15, color="#2563EB")
    ax.plot(angles, llm_med_norm, "^--", color="#D97706", linewidth=2, label="LLM T=0.7")
    ax.fill(angles, llm_med_norm, alpha=0.15, color="#D97706")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(short_names, size=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    ax.set_title("Facilitation Behavioral Profiles (Normalized)", pad=20)
    fig.savefig(os.path.join(FIG_DIR, "radar.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "figure4_radar.png"))
    plt.close()
    print("    Fig 3: Radar (normalized)")

    # Copy PDFs to paper dir
    paper_dir = os.path.join(BASE, "paper")
    for fn in ["figure1_pca.png", "figure3_temperature_heatmap.png", "figure4_radar.png"]:
        src = os.path.join(FIG_DIR, fn)
        dst = os.path.join(paper_dir, fn)
        if os.path.exists(src):
            import shutil; shutil.copy2(src, dst)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} min")
    print(f"  Features: {FEAT_CSV}")
    print(f"  Stats:    {STATS_CSV}")
    print(f"  Figures:  {FIG_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
