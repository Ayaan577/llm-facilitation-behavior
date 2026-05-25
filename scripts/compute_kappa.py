"""compute_kappa.py — Compute Cohen's kappa from IRR annotation task."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import sys, os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT = os.path.join(BASE, "irr_annotation_task.csv")

df = pd.read_csv(INPUT)
df = df.dropna(subset=["rater1_is_facilitation", "rater2_is_facilitation"])

if len(df) == 0:
    print("ERROR: No completed annotations found.")
    print("Fill rater1_is_facilitation and rater2_is_facilitation columns first.")
    sys.exit(1)

r1 = df["rater1_is_facilitation"].astype(int)
r2 = df["rater2_is_facilitation"].astype(int)

kappa = cohen_kappa_score(r1, r2)
agreement = (r1 == r2).mean()

print(f"Annotated items:  {len(df)}")
print(f"Percent agreement: {agreement*100:.1f}%")
print(f"Cohen's kappa:     {kappa:.3f}")
print()
if kappa >= 0.80:
    print("Near-perfect agreement — excellent for CHB submission")
elif kappa >= 0.70:
    print("Substantial agreement — acceptable for CHB submission")
elif kappa >= 0.60:
    print("Moderate agreement — add caveat to paper")
else:
    print("Low agreement — review extraction rules before submitting")
