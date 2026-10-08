"""Presentation figures from already verified statistics. Does not recompute tests."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Signed r: negative = human higher, positive = Llama higher.
# Intervals are the published bootstrap CIs on r, signed with the known direction.
ROWS = [
    ("Contextual semantic\ndistance", -0.644, -0.72, -0.56),
    ("Lexical elaboration\nproxy", 0.288, 0.16, 0.40),
    ("Empathy-and-hedge\nlexical density", -0.203, -0.34, -0.07),
    ("Topic divergence", 0.703, 0.63, 0.76),
    ("Phase-vocabulary\nalignment", 0.382, 0.25, 0.50),
]


def forest():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ys = list(range(len(ROWS) - 1, -1, -1))
    for y, (label, r, lo, hi) in zip(ys, ROWS):
        color = "#1565C0" if r < 0 else "#E65100"
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(r, y, "o", color=color, ms=7, zorder=3)
        ax.text(0.82, y, f"{abs(r):.3f}", va="center", ha="left", fontsize=8, color="#333333")
    ax.axvline(0, color="#424242", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in ROWS], fontsize=9)
    ax.set_xlim(-0.95, 1.05)
    ax.set_xlabel("Paired effect size  r     human higher  ←    →  Llama higher", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure6_effect_forest.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


def nophase():
    labels = ["Human", "Llama\nlabel present", "Llama\nno phase label"]
    means = [0.0050, 0.0149, 0.0115]
    colors = ["#1565C0", "#E65100", "#FFCC80"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(range(3), means, color=colors, width=0.72)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Phase-vocabulary alignment", fontsize=9)
    ax.set_ylim(0, 0.022)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0006, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figure7_nophase_means.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)


if __name__ == "__main__":
    forest()
    nophase()
