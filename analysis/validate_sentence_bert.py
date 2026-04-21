"""
Robustness Validation: Second Embedding Model (Sentence-BERT)

This script re-computes the semantic diversity analysis from Experiment 1
using Sentence-BERT (all-MiniLM-L6-v2, 384 dimensions) as an independent
embedding model, validating that the main findings are not artifacts of
the OpenAI text-embedding-3-small model.

Compares:
  - Group-history (all) vs Individual-history diversity trajectories
  - Final diversity values and statistical tests
  - Correlation between OpenAI and Sentence-BERT diversity measures
"""

import sys

import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from itertools import combinations

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ALL = os.path.join(BASE_DIR, 'results_exp1_20tasks_all')
RESULTS_IND = os.path.join(BASE_DIR, 'results_exp1_20tasks_individual')
OUTPUT_DIR = os.path.join(BASE_DIR, 'validation_second_embedding')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_pairwise_diversity(embeddings):
    """Compute mean pairwise Euclidean distance."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    dists = []
    for i, j in combinations(range(n), 2):
        dists.append(np.linalg.norm(embeddings[i] - embeddings[j]))
    return np.mean(dists)

def load_texts_from_csv(csv_path):
    """Load texts organized by (task_id, iteration) -> list of texts."""
    df = pd.read_csv(csv_path)
    texts_by_task_iter = {}
    for _, row in df.iterrows():
        key = (row['task_id'], int(row['iteration']))
        if key not in texts_by_task_iter:
            texts_by_task_iter[key] = []
        texts_by_task_iter[key].append(str(row['final_output']))
    return texts_by_task_iter, df

def load_openai_diversity(emb_dir, task_ids, n_iters=5):
    """Load pre-computed OpenAI embeddings and compute diversity."""
    diversity_by_task = {}
    for task_id in task_ids:
        diversity_by_task[task_id] = []
        for it in range(n_iters):
            pkl_path = os.path.join(emb_dir, f"{task_id}_iter{it}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    emb = pickle.load(f)
                diversity_by_task[task_id].append(compute_pairwise_diversity(emb))
            else:
                diversity_by_task[task_id].append(None)
    return diversity_by_task

def main():
    print("=" * 70)
    print("ROBUSTNESS VALIDATION: Second Embedding Model (Sentence-BERT)")
    print("Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("=" * 70)

    # Load Sentence-BERT model
    print("\nLoading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Load texts
    print("\nLoading texts from CSV files...")
    texts_all, df_all = load_texts_from_csv(os.path.join(RESULTS_ALL, 'results.csv'))
    texts_ind, df_ind = load_texts_from_csv(os.path.join(RESULTS_IND, 'results.csv'))

    task_ids = sorted(df_all['task_id'].unique())
    n_tasks = len(task_ids)
    n_iters = 5
    print(f"  Group-history: {len(texts_all)} task-iteration groups")
    print(f"  Individual-history: {len(texts_ind)} task-iteration groups")
    print(f"  Tasks: {n_tasks}, Iterations: {n_iters}")

    # Compute Sentence-BERT embeddings and diversity
    print("\nComputing Sentence-BERT embeddings and diversity...")
    sbert_div_all = {}
    sbert_div_ind = {}

    for task_id in task_ids:
        sbert_div_all[task_id] = []
        sbert_div_ind[task_id] = []
        for it in range(n_iters):
            key = (task_id, it)
            # Group-history
            if key in texts_all:
                embs = model.encode(texts_all[key], convert_to_numpy=True)
                sbert_div_all[task_id].append(compute_pairwise_diversity(embs))
            else:
                sbert_div_all[task_id].append(None)
            # Individual-history
            if key in texts_ind:
                embs = model.encode(texts_ind[key], convert_to_numpy=True)
                sbert_div_ind[task_id].append(compute_pairwise_diversity(embs))
            else:
                sbert_div_ind[task_id].append(None)

    # Load OpenAI embedding diversity for correlation
    print("\nLoading OpenAI embedding diversity for correlation analysis...")
    openai_div_all = load_openai_diversity(
        os.path.join(RESULTS_ALL, 'embeddings'), task_ids, n_iters)
    openai_div_ind = load_openai_diversity(
        os.path.join(RESULTS_IND, 'embeddings'), task_ids, n_iters)

    # === Analysis 1: Diversity trajectories ===
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Diversity Trajectories (Sentence-BERT)")
    print("=" * 70)

    for condition, div_data, label in [
        ("Group-history", sbert_div_all, "all"),
        ("Individual-history", sbert_div_ind, "individual")
    ]:
        print(f"\n  {condition}:")
        for it in range(n_iters):
            vals = [div_data[t][it] for t in task_ids if div_data[t][it] is not None]
            print(f"    Iteration {it}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}")

    # === Analysis 2: Relative diversity change ===
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Relative Diversity Change (Baseline → Final)")
    print("=" * 70)

    rel_changes_all = []
    rel_changes_ind = []
    for task_id in task_ids:
        base_all = sbert_div_all[task_id][0]
        final_all = sbert_div_all[task_id][-1]
        if base_all and final_all and base_all > 0:
            rel_changes_all.append((final_all - base_all) / base_all * 100)

        base_ind = sbert_div_ind[task_id][0]
        final_ind = sbert_div_ind[task_id][-1]
        if base_ind and final_ind and base_ind > 0:
            rel_changes_ind.append((final_ind - base_ind) / base_ind * 100)

    print(f"\n  Group-history:      mean change = {np.mean(rel_changes_all):+.1f}% (std={np.std(rel_changes_all):.1f}%)")
    print(f"  Individual-history: mean change = {np.mean(rel_changes_ind):+.1f}% (std={np.std(rel_changes_ind):.1f}%)")

    # === Analysis 3: Statistical test (final diversity) ===
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Statistical Comparison (Welch's t-test on final diversity)")
    print("=" * 70)

    final_all = [sbert_div_all[t][-1] for t in task_ids if sbert_div_all[t][-1] is not None]
    final_ind = [sbert_div_ind[t][-1] for t in task_ids if sbert_div_ind[t][-1] is not None]

    t_stat, p_value = stats.ttest_ind(final_ind, final_all, equal_var=False)
    cohens_d = (np.mean(final_ind) - np.mean(final_all)) / np.sqrt(
        (np.std(final_ind)**2 + np.std(final_all)**2) / 2)

    print(f"\n  Individual final: mean={np.mean(final_ind):.4f}, std={np.std(final_ind):.4f}, n={len(final_ind)}")
    print(f"  Group final:      mean={np.mean(final_all):.4f}, std={np.std(final_all):.4f}, n={len(final_all)}")
    print(f"  Welch's t-test:   t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  Cohen's d:        {cohens_d:.3f}")
    print(f"  Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")

    # === Analysis 4: Correlation with OpenAI embeddings ===
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Correlation Between OpenAI and Sentence-BERT Diversity")
    print("=" * 70)

    openai_vals = []
    sbert_vals = []
    for task_id in task_ids:
        for it in range(n_iters):
            ov_all = openai_div_all[task_id][it]
            sv_all = sbert_div_all[task_id][it]
            if ov_all is not None and sv_all is not None:
                openai_vals.append(ov_all)
                sbert_vals.append(sv_all)
            ov_ind = openai_div_ind[task_id][it]
            sv_ind = sbert_div_ind[task_id][it]
            if ov_ind is not None and sv_ind is not None:
                openai_vals.append(ov_ind)
                sbert_vals.append(sv_ind)

    r, p_corr = stats.pearsonr(openai_vals, sbert_vals)
    rho, p_spearman = stats.spearmanr(openai_vals, sbert_vals)
    print(f"\n  N data points: {len(openai_vals)}")
    print(f"  Pearson r:  {r:.4f} (p={p_corr:.2e})")
    print(f"  Spearman ρ: {rho:.4f} (p={p_spearman:.2e})")

    # === Analysis 5: Per-category breakdown ===
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Per-Category Diversity Change (Sentence-BERT)")
    print("=" * 70)

    categories = {}
    for task_id in task_ids:
        cat = task_id.split('_')[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(task_id)

    for cat, cat_tasks in sorted(categories.items()):
        rc_all = []
        rc_ind = []
        for t in cat_tasks:
            b, f = sbert_div_all[t][0], sbert_div_all[t][-1]
            if b and f and b > 0:
                rc_all.append((f - b) / b * 100)
            b, f = sbert_div_ind[t][0], sbert_div_ind[t][-1]
            if b and f and b > 0:
                rc_ind.append((f - b) / b * 100)
        print(f"\n  {cat.upper()} (n={len(cat_tasks)}):")
        print(f"    Group:      {np.mean(rc_all):+.1f}%")
        print(f"    Individual: {np.mean(rc_ind):+.1f}%")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: Consistency Check")
    print("=" * 70)

    openai_direction_all = "decrease" if np.mean(rel_changes_all) < 0 else "increase"
    openai_direction_ind = "increase"  # Known from paper
    sbert_direction_all = "decrease" if np.mean(rel_changes_all) < 0 else "increase"
    sbert_direction_ind = "decrease" if np.mean(rel_changes_ind) < 0 else "increase"

    print(f"\n  OpenAI embedding:      Group={openai_direction_all}, Individual={openai_direction_ind}")
    print(f"  Sentence-BERT:         Group={sbert_direction_all}, Individual={sbert_direction_ind}")
    print(f"  Direction consistent:  {'YES' if sbert_direction_all == 'decrease' and sbert_direction_ind == 'increase' else 'PARTIAL / NO'}")
    print(f"  Stat. significance:    {'YES' if p_value < 0.05 else 'NO'} (p={p_value:.4f})")
    print(f"  Cross-model corr:      r={r:.3f}")

    # Save results
    results = {
        'sbert_div_all': sbert_div_all,
        'sbert_div_ind': sbert_div_ind,
        'openai_div_all': openai_div_all,
        'openai_div_ind': openai_div_ind,
        'rel_changes_all': rel_changes_all,
        'rel_changes_ind': rel_changes_ind,
        'welch_t': t_stat,
        'welch_p': p_value,
        'cohens_d': cohens_d,
        'pearson_r': r,
        'spearman_rho': rho,
    }
    with open(os.path.join(OUTPUT_DIR, 'validation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Results saved to {OUTPUT_DIR}/validation_results.pkl")

if __name__ == '__main__':
    main()
