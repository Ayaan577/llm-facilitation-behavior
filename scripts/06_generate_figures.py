"""
06_generate_figures.py - Generate publication figures matching paper numbering.
Phase 6 of the LLM Facilitation Behavioral Study pipeline.

Figure mapping (matches manuscript numbering):
- Figure 1: Violin distributions (figure1_violin_distributions.png)
- Figure 2: Spearman correlation matrix (figure2_correlation_matrix.png)
- Figure 3: PCA biplot (figure3_pca_biplot.png)
- Figure 4: Temperature heatmap (figure4_temperature_heatmap.png)
- Figure 5: Radar profiles (figure5_radar_profiles.png)
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CSV = os.path.join(BASE, "data", "features_processed.csv")
STATS_CSV    = os.path.join(BASE, "results", "mannwhitney_results.csv")
CORR_CSV     = os.path.join(BASE, "results", "spearman_temperature.csv")
FIG_DIR      = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES = ["novel_score", "directive_score", "specificity_score",
            "empathy_score", "divergence_score", "phase_score"]

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
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


# ── Figure 1: Violin Distributions ───────────────────────────────────────────
def figure1_violin_distributions(df):
    print("  Generating Figure 1: Violin Distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    df_h = df[df["source"] == "human"]
    df_l = df[df["source"] == "llm_t07"]

    for idx, feat in enumerate(FEATURES):
        ax = axes[idx]
        h_vals = df_h[feat].dropna().values
        l_vals = df_l[feat].dropna().values

        parts = ax.violinplot([h_vals, l_vals], positions=[1, 2], showmedians=True, showextrema=False)
        colors = [SOURCE_COLORS["human"], SOURCE_COLORS["llm_t07"]]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Human", "LLM (T=0.7)"])
        ax.set_title(FEATURE_LABELS_SHORT[feat], fontweight="bold")
        ax.set_ylabel("Score")

    fig.suptitle("Figure 1: Behavioral Dimension Distributions (Human vs. LLM T=0.7)", fontsize=14, y=0.98)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure1_violin_distributions.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ── Figure 2: Spearman Correlation Matrix ─────────────────────────────────────
def figure2_correlation_matrix(df):
    print("  Generating Figure 2: Correlation Matrix...")
    from scipy.stats import spearmanr

    labels = [FEATURE_LABELS_SHORT[f] for f in FEATURES]
    corr_matrix = np.zeros((len(FEATURES), len(FEATURES)))

    for i, f1 in enumerate(FEATURES):
        for j, f2 in enumerate(FEATURES):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                rho, _ = spearmanr(df[f1].dropna(), df[f2].dropna())
                corr_matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Figure 2: Spearman Inter-Dimension Correlation Matrix", pad=12)

    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            val = corr_matrix[i, j]
            text_col = "white" if abs(val) > 0.35 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=text_col, fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure2_correlation_matrix.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ── Figure 3: PCA Biplot ──────────────────────────────────────────────────────
def figure3_pca_biplot(df):
    print("  Generating Figure 3: PCA Scatter / Biplot...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from matplotlib.patches import Ellipse

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
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=SOURCE_COLORS[src], label=SOURCE_LABELS[src],
                   alpha=0.55, s=35, edgecolors="none")
        handles.append(mpatches.Patch(color=SOURCE_COLORS[src], label=SOURCE_LABELS[src]))

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
    ax.set_title("Figure 3: Facilitation Style Space (PCA)")
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure3_pca_biplot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ── Figure 4: Temperature Heatmap ─────────────────────────────────────────────
def figure4_temperature_heatmap(df_corr):
    print("  Generating Figure 4: Temperature Heatmap...")
    df_c = df_corr.set_index("dimension").reindex(FEATURES)
    rhos = df_c["spearman_rho"].values.reshape(1, -1)
    pvals = df_c["p_value"].values

    fig, ax = plt.subplots(figsize=(10, 2.2))
    im = ax.imshow(rhos, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)

    labels = [FEATURE_LABELS_SHORT[f] for f in FEATURES]
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0])
    ax.set_yticklabels(["Temperature"], fontsize=10)
    ax.set_title("Figure 4: Spearman Correlation (LLM Temperature vs. Dimensions)", pad=10)

    for j, (rho, p) in enumerate(zip(rhos[0], pvals)):
        sig = "*" if p < 0.05 else ""
        ax.text(j, 0, f"{rho:+.3f}{sig}", ha="center", va="center",
                fontsize=10, fontweight="bold" if p < 0.05 else "normal",
                color="white" if abs(rho) > 0.15 else "black")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure4_temperature_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ── Figure 5: Radar Profiles ──────────────────────────────────────────────────
def figure5_radar_profiles(df):
    print("  Generating Figure 5: Radar Profiles...")
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
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, max(human_means.max(), llm07_means.max()) * 1.25 + 0.02)
    ax.set_title("Figure 5: Facilitation Style Profiles (Human vs. LLM T=0.7)", size=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), framealpha=0.9)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure5_radar_profiles.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def main():
    print("=" * 65)
    print("  Phase 6: Generate Figures (Synchronized Numbering)")
    print("=" * 65)

    if not os.path.exists(FEATURES_CSV):
        print(f"[ERROR] Features file not found at {FEATURES_CSV}")
        return

    df       = pd.read_csv(FEATURES_CSV)
    df_corr  = pd.read_csv(CORR_CSV) if os.path.exists(CORR_CSV) else None

    figure1_violin_distributions(df)
    if df_corr is not None:
        figure2_correlation_matrix(df)
        figure4_temperature_heatmap(df_corr)
    figure3_pca_biplot(df)
    figure5_radar_profiles(df)

    print(f"\n  All 5 publication figures saved to {FIG_DIR}")
    print("=" * 65)
    print("  Phase 6 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
