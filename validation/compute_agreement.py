"""
compute_agreement.py
Compute inter-rater agreement between human annotations and the
DistilBERT directiveness classifier on the validation sample.

Inputs:
    validation/directiveness_annotation_task.csv
        — must have 'human_label' column filled in (1 = directive, 0 = non-directive)

Outputs:
    Prints agreement summary table.
    Saves: validation/agreement_results.csv
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

THRESHOLD = 0.5  # Binarization threshold for classifier

# ── Locate the annotation file ──
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
annot_path = os.path.join(BASE, 'directiveness_annotation_task.csv')
if not os.path.exists(annot_path):
    # Try alternative paths
    for alt in [
        os.path.join(os.path.dirname(__file__), 'directiveness_annotation_task.csv'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'validation', 'directiveness_annotation_task.csv'),
    ]:
        if os.path.exists(alt):
            annot_path = alt
            break

df = pd.read_csv(annot_path)
print(f'Loaded: {annot_path}')
print(f'Total rows: {len(df)}')

# ── Validate human_label column ──
if 'human_label' not in df.columns:
    print('ERROR: human_label column not found.'); sys.exit(1)

# Drop rows where human_label is blank
df_valid = df[df['human_label'].notna() & (df['human_label'] != '')].copy()
df_valid['human_label'] = df_valid['human_label'].astype(int)
n_annotated = len(df_valid)
n_missing = len(df) - n_annotated

if n_annotated == 0:
    print('ERROR: No annotations found. Please fill in the human_label column.')
    sys.exit(1)

print(f'Annotated: {n_annotated} / {len(df)} ({n_missing} missing)')

# ── Binarize classifier scores ──
df_valid['classifier_binary'] = (df_valid['directiveness_score'] >= THRESHOLD).astype(int)

# ── Cohen's Kappa ──
def cohens_kappa(y1, y2):
    """Compute Cohen's kappa for two binary raters."""
    n = len(y1)
    # Confusion matrix
    a = np.sum((y1 == 1) & (y2 == 1))  # both positive
    b = np.sum((y1 == 1) & (y2 == 0))  # rater1 pos, rater2 neg
    c = np.sum((y1 == 0) & (y2 == 1))  # rater1 neg, rater2 pos
    d = np.sum((y1 == 0) & (y2 == 0))  # both negative

    po = (a + d) / n  # observed agreement
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)  # chance agreement
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0

    return kappa, po, pe, a, b, c, d

human = df_valid['human_label'].values
classifier = df_valid['classifier_binary'].values

kappa, po, pe, tp, fp, fn, tn = cohens_kappa(human, classifier)

# ── Spearman correlation (human label vs raw score) ──
rho, p_spearman = stats.spearmanr(df_valid['human_label'], df_valid['directiveness_score'])

# ── Point-biserial correlation ──
rpb, p_pb = stats.pointbiserialr(df_valid['human_label'], df_valid['directiveness_score'])

# ── Accuracy, precision, recall, F1 ──
accuracy = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# ── Print results ──
print('\n' + '=' * 60)
print('  DIRECTIVENESS CLASSIFIER VALIDATION RESULTS')
print('=' * 60)

print(f'\n  Sample: {n_annotated} moves ({sum(human == 1)} directive, {sum(human == 0)} non-directive by human)')
print(f'  Classifier threshold: {THRESHOLD}')
print(f'  Classifier predictions: {sum(classifier == 1)} directive, {sum(classifier == 0)} non-directive')

print(f'\n  CONFUSION MATRIX:')
print(f'                    Classifier')
print(f'                    Dir    Non-Dir')
print(f'  Human  Dir      {tp:4d}    {fn:4d}')
print(f'         Non-Dir  {fp:4d}    {tn:4d}')

print(f'\n  AGREEMENT METRICS:')
print(f'    {"Metric":<40s} {"Value":>10s}')
print(f'    {"-" * 50}')
print(f'    {"Cohen kappa":<40s} {kappa:>10.3f}')
print(f'    {"Observed agreement (Po)":<40s} {po:>10.3f}')
print(f'    {"Chance agreement (Pe)":<40s} {pe:>10.3f}')
print(f'    {"Accuracy":<40s} {accuracy:>10.3f}')
print(f'    {"Precision (classifier vs human)":<40s} {precision:>10.3f}')
print(f'    {"Recall (classifier vs human)":<40s} {recall:>10.3f}')
print(f'    {"F1 score":<40s} {f1:>10.3f}')
print(f'    {"Spearman rho (label vs raw score)":<40s} {rho:>10.3f}')
print(f'    {"Spearman p-value":<40s} {p_spearman:>10.2e}')
print(f'    {"Point-biserial r":<40s} {rpb:>10.3f}')
print(f'    {"Point-biserial p-value":<40s} {p_pb:>10.2e}')

# Kappa interpretation
if kappa >= 0.81:
    interp = 'Almost perfect'
elif kappa >= 0.61:
    interp = 'Substantial'
elif kappa >= 0.41:
    interp = 'Moderate'
elif kappa >= 0.21:
    interp = 'Fair'
else:
    interp = 'Slight/poor'
print(f'\n  Kappa interpretation (Landis & Koch, 1977): {interp}')

# ── Per-quartile breakdown ──
print(f'\n  PER-QUARTILE AGREEMENT:')
print(f'    {"Quartile":<15s} {"N":>4s} {"Accuracy":>10s} {"Human % dir":>12s} {"Clf % dir":>12s}')
print(f'    {"-" * 55}')

# Reconstruct quartiles from scores
df_valid['quartile'] = pd.qcut(df_valid['directiveness_score'], q=4,
                                labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'],
                                duplicates='drop')
for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
    qdf = df_valid[df_valid['quartile'] == q]
    if len(qdf) == 0:
        continue
    q_acc = np.mean(qdf['human_label'] == qdf['classifier_binary'])
    q_human_rate = qdf['human_label'].mean()
    q_clf_rate = qdf['classifier_binary'].mean()
    print(f'    {q:<15s} {len(qdf):>4d} {q_acc:>10.3f} {q_human_rate:>12.3f} {q_clf_rate:>12.3f}')

# ── Save results ──
results = pd.DataFrame([{
    'metric': 'cohens_kappa', 'value': round(kappa, 4), 'interpretation': interp,
}, {
    'metric': 'observed_agreement', 'value': round(po, 4), 'interpretation': '',
}, {
    'metric': 'accuracy', 'value': round(accuracy, 4), 'interpretation': '',
}, {
    'metric': 'precision', 'value': round(precision, 4), 'interpretation': '',
}, {
    'metric': 'recall', 'value': round(recall, 4), 'interpretation': '',
}, {
    'metric': 'f1_score', 'value': round(f1, 4), 'interpretation': '',
}, {
    'metric': 'spearman_rho', 'value': round(rho, 4), 'interpretation': f'p={p_spearman:.2e}',
}, {
    'metric': 'point_biserial_r', 'value': round(rpb, 4), 'interpretation': f'p={p_pb:.2e}',
}, {
    'metric': 'n_annotated', 'value': n_annotated, 'interpretation': '',
}, {
    'metric': 'threshold', 'value': THRESHOLD, 'interpretation': '',
}])

out_path = os.path.join(os.path.dirname(annot_path), 'agreement_results.csv')
results.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
print('DONE.')


if __name__ == '__main__':
    pass  # Script runs on import (for simplicity)
