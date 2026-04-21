#!/usr/bin/env python3
"""
Analyze Exp1 results with 20 tasks (4 categories).
Compare Individual-history vs Group-history modes.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist

# =========================
# Configuration
# =========================

INDIVIDUAL_DIR = "results_exp1_20tasks_individual"
GROUP_DIR = "results_exp1_20tasks_all"

CATEGORIES = ["idea", "reasoning", "summarization", "creative_writing"]

# =========================
# Load SDI from embeddings
# =========================

def compute_sdi(embeddings: np.ndarray) -> float:
    """Compute Semantic Diversity Index (mean pairwise distance)."""
    if len(embeddings) < 2:
        return 0.0
    distances = pdist(embeddings)
    return np.mean(distances)

def load_sdi_data(results_dir: str) -> pd.DataFrame:
    """Load SDI values from embeddings."""
    embeddings_dir = os.path.join(results_dir, "embeddings")

    rows = []
    for filename in os.listdir(embeddings_dir):
        if not filename.endswith(".pkl"):
            continue

        # Parse filename: task_id_iterN.pkl
        parts = filename.replace(".pkl", "").rsplit("_iter", 1)
        task_id = parts[0]
        iteration = int(parts[1])

        # Determine category
        category = task_id.split("_")[0]
        if category == "summ":
            category = "summarization"
        elif category == "creative":
            category = "creative_writing"

        # Load embeddings and compute SDI
        with open(os.path.join(embeddings_dir, filename), "rb") as f:
            embeddings = pickle.load(f)

        sdi = compute_sdi(embeddings)

        rows.append({
            "task_id": task_id,
            "category": category,
            "iteration": iteration,
            "sdi": sdi,
        })

    return pd.DataFrame(rows)

# =========================
# Analysis
# =========================

def analyze_condition(df: pd.DataFrame, condition_name: str):
    """Analyze SDI for a single condition."""
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name}")
    print(f"{'='*60}")

    # Overall statistics
    baseline = df[df["iteration"] == 0]["sdi"].values
    final = df[df["iteration"] == 4]["sdi"].values

    baseline_mean = np.mean(baseline)
    final_mean = np.mean(final)
    change_pct = (final_mean - baseline_mean) / baseline_mean * 100

    print(f"\nOverall (all 20 tasks):")
    print(f"  Baseline SDI (t=0): {baseline_mean:.4f} (SD={np.std(baseline):.4f})")
    print(f"  Final SDI (t=4):    {final_mean:.4f} (SD={np.std(final):.4f})")
    print(f"  Change: {change_pct:+.1f}%")

    # Per-category statistics
    print(f"\nPer-category analysis:")
    category_results = []
    for cat in CATEGORIES:
        cat_df = df[df["category"] == cat]
        cat_baseline = cat_df[cat_df["iteration"] == 0]["sdi"].values
        cat_final = cat_df[cat_df["iteration"] == 4]["sdi"].values

        if len(cat_baseline) > 0 and len(cat_final) > 0:
            cat_baseline_mean = np.mean(cat_baseline)
            cat_final_mean = np.mean(cat_final)
            cat_change_pct = (cat_final_mean - cat_baseline_mean) / cat_baseline_mean * 100

            print(f"  {cat}:")
            print(f"    Baseline: {cat_baseline_mean:.4f}, Final: {cat_final_mean:.4f}, Change: {cat_change_pct:+.1f}%")

            category_results.append({
                "category": cat,
                "baseline_mean": cat_baseline_mean,
                "final_mean": cat_final_mean,
                "change_pct": cat_change_pct,
                "baseline_values": cat_baseline,
                "final_values": cat_final,
            })

    return {
        "overall": {
            "baseline_mean": baseline_mean,
            "final_mean": final_mean,
            "change_pct": change_pct,
            "baseline_values": baseline,
            "final_values": final,
        },
        "by_category": category_results,
    }

def compare_conditions(individual_results: dict, group_results: dict):
    """Statistical comparison between conditions."""
    print(f"\n{'='*60}")
    print("Statistical Comparison: Individual vs Group")
    print(f"{'='*60}")

    # Overall comparison
    ind_final = individual_results["overall"]["final_values"]
    grp_final = group_results["overall"]["final_values"]

    t_stat, p_value = stats.ttest_ind(ind_final, grp_final, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(ind_final, ddof=1) + np.var(grp_final, ddof=1)) / 2)
    cohens_d = (np.mean(ind_final) - np.mean(grp_final)) / pooled_std

    print(f"\nOverall (Final SDI):")
    print(f"  Individual: {np.mean(ind_final):.4f} (SD={np.std(ind_final):.4f})")
    print(f"  Group:      {np.mean(grp_final):.4f} (SD={np.std(grp_final):.4f})")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.2f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Per-category comparison
    print(f"\nPer-category comparison (Final SDI):")
    for cat in CATEGORIES:
        ind_cat = [r for r in individual_results["by_category"] if r["category"] == cat]
        grp_cat = [r for r in group_results["by_category"] if r["category"] == cat]

        if ind_cat and grp_cat:
            ind_final_cat = ind_cat[0]["final_values"]
            grp_final_cat = grp_cat[0]["final_values"]

            t_stat_cat, p_value_cat = stats.ttest_ind(ind_final_cat, grp_final_cat, equal_var=False)

            print(f"  {cat}:")
            print(f"    Individual change: {ind_cat[0]['change_pct']:+.1f}%")
            print(f"    Group change:      {grp_cat[0]['change_pct']:+.1f}%")
            print(f"    t={t_stat_cat:.2f}, p={p_value_cat:.4f}")

def main():
    print("Loading Individual-history data...")
    individual_df = load_sdi_data(INDIVIDUAL_DIR)
    print(f"  Loaded {len(individual_df)} data points")

    print("Loading Group-history data...")
    group_df = load_sdi_data(GROUP_DIR)
    print(f"  Loaded {len(group_df)} data points")

    # Analyze each condition
    individual_results = analyze_condition(individual_df, "Individual-history")
    group_results = analyze_condition(group_df, "Group-history (All)")

    # Compare conditions
    compare_conditions(individual_results, group_results)

    # Summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"\n{'Category':<20} {'Individual':>15} {'Group':>15} {'Difference':>15}")
    print("-" * 65)

    for cat in CATEGORIES:
        ind_cat = [r for r in individual_results["by_category"] if r["category"] == cat]
        grp_cat = [r for r in group_results["by_category"] if r["category"] == cat]

        if ind_cat and grp_cat:
            ind_change = ind_cat[0]["change_pct"]
            grp_change = grp_cat[0]["change_pct"]
            diff = ind_change - grp_change
            print(f"{cat:<20} {ind_change:>+14.1f}% {grp_change:>+14.1f}% {diff:>+14.1f}%")

    # Overall
    ind_overall = individual_results["overall"]["change_pct"]
    grp_overall = group_results["overall"]["change_pct"]
    diff_overall = ind_overall - grp_overall
    print("-" * 65)
    print(f"{'OVERALL':<20} {ind_overall:>+14.1f}% {grp_overall:>+14.1f}% {diff_overall:>+14.1f}%")

if __name__ == "__main__":
    main()
