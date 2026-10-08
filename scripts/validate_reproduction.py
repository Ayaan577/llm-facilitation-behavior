"""
validate_reproduction.py - Automated Reproducibility Validation Harness.
Verifies all 11 repository fidelity & paper reproduction constraints:

1. Exactly 199 human contexts
2. Exactly 597 LLM responses
3. Exactly 796 feature rows
4. Temperatures are 0.3, 0.7, 1.0
5. No duplicate context_id / temperature pairs
6. Every LLM response maps to a valid human context
7. Required feature columns exist (novel_score, directive_score, specificity_score, empathy_score, divergence_score, phase_score)
8. Main statistical output tables exist
9. Robustness Table 6 output exists
10. Manuscript figure files exist (figure1 to figure7)
11. Canonical file SHA-256 hashes match REPRODUCIBILITY_MANIFEST.json
"""
import os, sys, json, hashlib
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(BASE, "REPRODUCIBILITY_MANIFEST.json")

REQUIRED_FEATURES = ["novel_score", "directive_score", "specificity_score",
                     "empathy_score", "divergence_score", "phase_score"]

REQUIRED_FIGURES = [
    "figure1_violin_distributions.png",
    "figure2_correlation_matrix.png",
    "figure3_pca_biplot.png",
    "figure4_temperature_heatmap.png",
    "figure5_radar_profiles.png",
    "figure6_effect_forest.png",
    "figure7_nophase_means.png",
]

REQUIRED_RESULTS = [
    "mannwhitney_results.csv",
    "spearman_temperature.csv",
    "robustness_table6.csv",
    "power_analysis.csv",
    "extreme_cases.csv",
]


def check_hash(rel_path, expected_hash):
    full_path = os.path.join(BASE, rel_path)
    if not os.path.exists(full_path):
        return False, "File missing"
    h = hashlib.sha256()
    with open(full_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    actual_hash = h.hexdigest()
    if actual_hash.lower() == expected_hash.lower():
        return True, "Hash match"
    return False, f"Hash mismatch (actual={actual_hash[:8]}... expected={expected_hash[:8]}...)"


def main():
    print("=" * 65)
    print("  REPRODUCIBILITY VALIDATION HARNESS")
    print("=" * 65)

    passed_checks = 0
    total_checks = 11
    results_summary = []

    # --- Check 1: 199 Human Contexts ---
    moves_path = os.path.join(BASE, "data", "processed", "facilitator_moves_sampled_199.csv")
    if os.path.exists(moves_path):
        df_moves = pd.read_csv(moves_path)
        n_moves = df_moves["context_id"].nunique()
        if n_moves == 199:
            results_summary.append(("1. 199 Human Contexts", "PASS", f"Exactly {n_moves} contexts"))
            passed_checks += 1
        else:
            results_summary.append(("1. 199 Human Contexts", "FAIL", f"Found {n_moves} contexts (expected 199)"))
    else:
        results_summary.append(("1. 199 Human Contexts", "NOT VERIFIED", f"File missing: {moves_path}"))

    # --- Check 2: 597 LLM Responses ---
    resp_path = os.path.join(BASE, "data", "llm_responses.csv")
    if os.path.exists(resp_path):
        df_resp = pd.read_csv(resp_path)
        n_resp = len(df_resp) * 3  # 3 temperatures per context
        if len(df_resp) == 199 and all(c in df_resp.columns for c in ["llm_t03", "llm_t07", "llm_t10"]):
            results_summary.append(("2. 597 LLM Responses", "PASS", f"199 contexts × 3 temps = {n_resp} responses"))
            passed_checks += 1
        else:
            results_summary.append(("2. 597 LLM Responses", "FAIL", f"Unexpected structure (rows={len(df_resp)})"))
    else:
        results_summary.append(("2. 597 LLM Responses", "NOT VERIFIED", f"File missing: {resp_path}"))

    # --- Check 3: 796 Total Feature Rows ---
    feat_path = os.path.join(BASE, "data", "features_processed.csv")
    if os.path.exists(feat_path):
        df_feat = pd.read_csv(feat_path)
        if len(df_feat) == 796:
            results_summary.append(("3. 796 Feature Rows", "PASS", f"Exactly {len(df_feat)} feature rows"))
            passed_checks += 1
        else:
            results_summary.append(("3. 796 Feature Rows", "FAIL", f"Found {len(df_feat)} rows (expected 796)"))
    else:
        results_summary.append(("3. 796 Feature Rows", "NOT VERIFIED", f"File missing: {feat_path}"))

    # --- Check 4: Temperatures (0.3, 0.7, 1.0) ---
    if os.path.exists(feat_path):
        sources = set(df_feat["source"].unique())
        expected_sources = {"human", "llm_t03", "llm_t07", "llm_t10"}
        if expected_sources.issubset(sources):
            results_summary.append(("4. Temperatures 0.3/0.7/1.0", "PASS", f"Sources present: {sources}"))
            passed_checks += 1
        else:
            results_summary.append(("4. Temperatures 0.3/0.7/1.0", "FAIL", f"Found sources: {sources}"))
    else:
        results_summary.append(("4. Temperatures 0.3/0.7/1.0", "NOT VERIFIED", "Feature file missing"))

    # --- Check 5: No Duplicate Context/Temperature Pairs ---
    if os.path.exists(feat_path):
        dups = df_feat.duplicated(subset=["context_id", "source"]).sum()
        if dups == 0:
            results_summary.append(("5. No Duplicate Pairs", "PASS", "0 duplicates across context_id and source"))
            passed_checks += 1
        else:
            results_summary.append(("5. No Duplicate Pairs", "FAIL", f"Found {dups} duplicate pairs"))
    else:
        results_summary.append(("5. No Duplicate Pairs", "NOT VERIFIED", "Feature file missing"))

    # --- Check 6: LLM Responses Map to Valid Contexts ---
    if os.path.exists(resp_path) and os.path.exists(moves_path):
        resp_ids = set(df_resp["context_id"])
        move_ids = set(df_moves["context_id"])
        diff = resp_ids - move_ids
        if len(diff) == 0:
            results_summary.append(("6. Context Mapping Integrity", "PASS", "All 199 response context_ids map to valid human contexts"))
            passed_checks += 1
        else:
            results_summary.append(("6. Context Mapping Integrity", "FAIL", f"Unmapped context IDs: {diff}"))
    else:
        results_summary.append(("6. Context Mapping Integrity", "NOT VERIFIED", "Response or move file missing"))

    # --- Check 7: Required Feature Columns Exist ---
    if os.path.exists(feat_path):
        cols_present = all(c in df_feat.columns for c in REQUIRED_FEATURES)
        if cols_present:
            results_summary.append(("7. Feature Columns Exist", "PASS", f"All 6 features present: {REQUIRED_FEATURES}"))
            passed_checks += 1
        else:
            results_summary.append(("7. Feature Columns Exist", "FAIL", f"Missing columns in {df_feat.columns.tolist()}"))
    else:
        results_summary.append(("7. Feature Columns Exist", "NOT VERIFIED", "Feature file missing"))

    # --- Check 8: Main Statistical Output Tables Exist ---
    results_dir = os.path.join(BASE, "results")
    mw_path = os.path.join(results_dir, "mannwhitney_results.csv")
    corr_path = os.path.join(results_dir, "spearman_temperature.csv")
    if os.path.exists(mw_path) and os.path.exists(corr_path):
        df_mw = pd.read_csv(mw_path)
        if len(df_mw) >= 6:
            results_summary.append(("8. Statistical Output Tables", "PASS", f"mannwhitney_results.csv has {len(df_mw)} rows"))
            passed_checks += 1
        else:
            results_summary.append(("8. Statistical Output Tables", "FAIL", f"mannwhitney_results.csv has only {len(df_mw)} rows"))
    else:
        results_summary.append(("8. Statistical Output Tables", "NOT VERIFIED", "Statistical output files missing"))

    # --- Check 9: Table 6 Robustness Output Exists ---
    t6_path = os.path.join(results_dir, "robustness_table6.csv")
    if os.path.exists(t6_path):
        df_t6 = pd.read_csv(t6_path)
        if len(df_t6) >= 12:
            results_summary.append(("9. Robustness Table 6 Output", "PASS", f"robustness_table6.csv exists with {len(df_t6)} rows"))
            passed_checks += 1
        else:
            results_summary.append(("9. Robustness Table 6 Output", "FAIL", f"robustness_table6.csv has only {len(df_t6)} rows"))
    else:
        results_summary.append(("9. Robustness Table 6 Output", "NOT VERIFIED", f"File missing: {t6_path}"))

    # --- Check 10: All 5 Figure Files Exist ---
    fig_dir = os.path.join(BASE, "figures")
    missing_figs = [f for f in REQUIRED_FIGURES if not os.path.exists(os.path.join(fig_dir, f))]
    if not missing_figs:
        results_summary.append(("10. Manuscript Figures Present", "PASS", "figure1 through figure7 exist"))
        passed_checks += 1
    else:
        results_summary.append(("10. Manuscript Figures Present", "FAIL", f"Missing figures: {missing_figs}"))

    # --- Check 11: Manifest SHA-256 Hashes Match ---
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        hashes = manifest.get("sha256_hashes", {})
        hash_mismatches = []
        for rel_p, exp_h in hashes.items():
            ok, msg = check_hash(rel_p, exp_h)
            if not ok:
                hash_mismatches.append(f"{rel_p}: {msg}")
        if not hash_mismatches:
            results_summary.append(("11. SHA-256 Manifest Hashes", "PASS", f"All {len(hashes)} file hashes verified byte-for-byte"))
            passed_checks += 1
        else:
            results_summary.append(("11. SHA-256 Manifest Hashes", "FAIL", f"Mismatches: {hash_mismatches}"))
    else:
        results_summary.append(("11. SHA-256 Manifest Hashes", "NOT VERIFIED", f"Manifest missing: {MANIFEST_PATH}"))

    # --- Print Summary ---
    print("\n  VALIDATION RESULTS SUMMARY:")
    print("  " + "-" * 60)
    for title, status, note in results_summary:
        symbol = "✓ PASS" if status == "PASS" else ("✗ FAIL" if status == "FAIL" else "? NOT VERIFIED")
        print(f"  {title:<32s} [{symbol:^12s}] {note}")
    print("  " + "-" * 60)
    print(f"\n  Final Score: {passed_checks}/{total_checks} checks passed.")

    if passed_checks == total_checks:
        print("\n  >>> REPRODUCIBILITY VALIDATION: SUCCESS <<<")
    else:
        print("\n  >>> REPRODUCIBILITY VALIDATION: INCOMPLETE / ISSUES DETECTED <<<")

    return passed_checks == total_checks


if __name__ == "__main__":
    main()
