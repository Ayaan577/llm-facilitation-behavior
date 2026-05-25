"""
05_generate_figures.py - Generate all paper figures using matplotlib.
Phase 5 of the CHI 2027 pipeline.
Figures: PCA scatter, box plots, temperature heatmap, radar chart.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CSV = os.path.join(BASE, "data", "outputs", "feature_scores.csv")
STATS_CSV    = os.path.join(BASE, "data", "outputs", "stats_results.csv")
CORR_CSV     = os.path.join(BASE, "data", "outputs", "temperature_correlations.csv")
FIG_DIR      = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = ["novel_score", "directive_score", "specificity_score",
            "empathy_score", "divergence_score", "phase_score"]
FEATURE_LABELS = {
    "novel_score":        "Semantic\nNovelty",
    "directive_score":    "Directiveness",
    "specificity_score":  "Specificity",
    "empathy_score":      "Empathy",
    "divergence_score":   "Topic\nDivergence",
    "phase_score":        "Phase\nAppropriateness",
}
FEATURE_LABELS_SHORT = {
    "novel_score":        "Novelty",
    "directive_score":    "Directiveness",
    "specificity_score":  "Specificity",
    "empathy_score":      "Empathy",
    "divergence_score":   "Divergence",
    "phase_score":        "Phase App.",
}
SOURCE_COLORS  = {"human": "#2196F3", "llm_t03": "#4CAF50",
                  "llm_t07": "#FF9800", "llm_t10": "#F44336"}
SOURCE_LABELS  = {"human": "Human", "llm_t03": "LLM T=0.3",
                  "llm_t07": "LLM T=0.7", "llm_t10": "LLM T=1.0"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ── Figure 1: PCA ─────────────────────────────────────────────────────────────
def figure1_pca(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    print("  Generating Figure 1: PCA Scatter Plot...")

    X = df[FEATURES].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(7, 5.5))
    handles = []
    for src in ["human", "llm_t03", "llm_t07", "llm_t10"]:
        mask = df["source"] == src
        sc = ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=SOURCE_COLORS[src], label=SOURCE_LABELS[src],
                        alpha=0.55, s=35, edgecolors="none")
        handles.append(mpatches.Patch(color=SOURCE_COLORS[src],
                                      label=SOURCE_LABELS[src]))

    # Confidence ellipses
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    for src in ["human", "llm_t07"]:
        mask = df["source"] == src
        pts = X_pca[mask]
        if len(pts) < 4:
            continue
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * 1.96 * np.sqrt(vals)
        ell = Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=theta,
                      color=SOURCE_COLORS[src], alpha=0.12)
        ax.add_patch(ell)

    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title("Facilitation Style Space (PCA): Human vs. LLM Responses")
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure1_pca.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}  (PC1={var1:.1f}%, PC2={var2:.1f}%)")


# ── Figure 2: Box plots ────────────────────────────────────────────────────────
def figure2_boxplots(df, df_stats):
    print("  Generating Figure 2: Box Plots...")
    min_p = df_stats.groupby("feature")["p_bonferroni"].min().sort_values()
    top3 = min_p.head(3).index.tolist()
    print(f"    Top 3 features: {top3}")

    sources = ["human", "llm_t03", "llm_t07", "llm_t10"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, feat in zip(axes, top3):
        data = [df[df["source"] == s][feat].dropna().values for s in sources]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4))
        for patch, src in zip(bp["boxes"], sources):
            patch.set_facecolor(SOURCE_COLORS[src])
            patch.set_alpha(0.75)
        ax.set_xticklabels([SOURCE_LABELS[s] for s in sources], rotation=20, ha="right")
        ax.set_title(FEATURE_LABELS_SHORT[feat], fontsize=12, fontweight="bold")
        ax.set_ylabel("Score")

    fig.suptitle("Human vs. LLM: Top 3 Most Significant Facilitation Dimensions",
                 fontsize=13, y=1.02)
    handles = [mpatches.Patch(color=SOURCE_COLORS[s], label=SOURCE_LABELS[s])
               for s in sources]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure2_boxplots.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ── Figure 3: Temperature heatmap ─────────────────────────────────────────────
def figure3_heatmap(df_corr):
    print("  Generating Figure 3: Temperature Heatmap...")
    feat_order = FEATURES
    df_c = df_corr.set_index("feature").reindex(feat_order)
    rhos = df_c["spearman_rho"].values.reshape(1, -1)
    pvals = df_c["p_value"].values

    fig, ax = plt.subplots(figsize=(10, 2.2))
    im = ax.imshow(rhos, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)

    labels = [FEATURE_LABELS_SHORT[f] for f in feat_order]
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0])
    ax.set_yticklabels(["Temperature"], fontsize=11)
    ax.set_title("Spearman Correlation: LLM Temperature vs. Facilitation Dimensions",
                 fontsize=12, pad=10)

    for j, (rho, p) in enumerate(zip(rhos[0], pvals)):
        sig = "*" if p < 0.05 else ""
        ax.text(j, 0, f"{rho:+.3f}{sig}", ha="center", va="center",
                fontsize=11, fontweight="bold" if p < 0.05 else "normal",
                color="white" if abs(rho) > 0.15 else "black")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure3_temperature_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ── Figure 4: Radar chart ─────────────────────────────────────────────────────
def figure4_radar(df):
    print("  Generating Figure 4: Radar Chart...")
    human_means  = df[df["source"] == "human"][FEATURES].mean().values
    llm07_means  = df[df["source"] == "llm_t07"][FEATURES].mean().values
    labels = [FEATURE_LABELS_SHORT[f] for f in FEATURES]
    N = len(FEATURES)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for vals, color, name in [(human_means, "#2196F3", "Human"),
                               (llm07_means, "#FF9800", "LLM T=0.7")]:
        v = vals.tolist() + vals[:1].tolist()
        ax.plot(angles, v, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, v, alpha=0.18, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, max(human_means.max(), llm07_means.max()) * 1.25 + 0.02)
    ax.set_title("Facilitation Style Profiles:\nHuman vs. LLM (T=0.7)",
                 size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), framealpha=0.9)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure4_radar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def main():
    print("=" * 60)
    print("  Phase 5: Generate Figures")
    print("=" * 60)

    df       = pd.read_csv(FEATURES_CSV)
    df_stats = pd.read_csv(STATS_CSV)
    df_corr  = pd.read_csv(CORR_CSV)

    figure1_pca(df)
    figure2_boxplots(df, df_stats)
    figure3_heatmap(df_corr)
    figure4_radar(df)

    print(f"\n  All figures saved to {FIG_DIR}")
    print("=" * 60)
    print("  Phase 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
