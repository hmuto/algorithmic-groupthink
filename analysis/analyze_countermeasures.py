#!/usr/bin/env python3
"""
Analyze countermeasure experiment results.
Compares Baseline (group-history), Diversity Prompt, and Adversarial Sampling.
"""

import os
import pickle
import csv
from collections import defaultdict
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 1.2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "embedding_cache")
FIG_DIR = os.path.join(BASE_DIR, "template", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_cached_embeddings(exp_name, workflow, task_id, iteration):
    """Load cached embeddings for a specific configuration."""
    cache_key = f"{exp_name}_{workflow}_{task_id}_{iteration}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def compute_sdi(embeddings):
    """Compute Semantic Diversity Index."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    upper_tri = distances[np.triu_indices(n, k=1)]
    return np.mean(upper_tri)


def get_sdi_by_iteration(exp_name, workflow="self-refine", num_tasks=5, num_iterations=4):
    """Get SDI values organized by iteration."""
    sdi_by_iter = defaultdict(list)

    for task_id in range(num_tasks):
        for iteration in range(num_iterations):
            embeddings = load_cached_embeddings(exp_name, workflow, task_id, iteration)
            if embeddings is not None:
                sdi = compute_sdi(embeddings)
                sdi_by_iter[iteration].append(sdi)

    return dict(sdi_by_iter)


def analyze_results():
    """Analyze and compare all countermeasure results."""
    print("="*70)
    print("COUNTERMEASURE EXPERIMENT ANALYSIS")
    print("="*70)

    # Define experiments to analyze
    experiments = {
        "Baseline (Group)": ("exp1_all", "self-refine"),
        "Diversity Prompt": ("exp3_diversity_prompt", "self-refine"),
        "Adversarial": ("exp3_adversarial", "self-refine"),
    }

    results = {}
    for name, (exp_name, workflow) in experiments.items():
        sdi_data = get_sdi_by_iteration(exp_name, workflow)
        if sdi_data:
            results[name] = sdi_data
            print(f"\n{name}:")
            for it in sorted(sdi_data.keys()):
                mean_sdi = np.mean(sdi_data[it])
                std_sdi = np.std(sdi_data[it])
                print(f"  Iteration {it}: SDI = {mean_sdi:.4f} ± {std_sdi:.4f} (n={len(sdi_data[it])})")
        else:
            print(f"\n{name}: No data found")

    # Calculate relative changes
    print("\n" + "-"*50)
    print("RELATIVE CHANGE FROM BASELINE (Iteration 0 → 3)")
    print("-"*50)

    for name, sdi_data in results.items():
        if 0 in sdi_data and 3 in sdi_data:
            baseline = np.mean(sdi_data[0])
            final = np.mean(sdi_data[3])
            change = (final - baseline) / baseline * 100
            print(f"  {name}: {change:+.1f}%")

    # Statistical comparison
    print("\n" + "-"*50)
    print("STATISTICAL COMPARISONS (Final Iteration)")
    print("-"*50)

    if "Baseline (Group)" in results and "Diversity Prompt" in results:
        if 3 in results["Baseline (Group)"] and 3 in results["Diversity Prompt"]:
            baseline_final = results["Baseline (Group)"][3]
            diversity_final = results["Diversity Prompt"][3]
            t_stat, p_val = stats.ttest_ind(diversity_final, baseline_final, equal_var=False)
            print(f"\n  Diversity Prompt vs Baseline:")
            print(f"    t = {t_stat:.3f}, p = {p_val:.4f}")

    if "Baseline (Group)" in results and "Adversarial" in results:
        if 3 in results["Baseline (Group)"] and 3 in results["Adversarial"]:
            baseline_final = results["Baseline (Group)"][3]
            adversarial_final = results["Adversarial"][3]
            t_stat, p_val = stats.ttest_ind(adversarial_final, baseline_final, equal_var=False)
            print(f"\n  Adversarial vs Baseline:")
            print(f"    t = {t_stat:.3f}, p = {p_val:.4f}")

    return results


def create_figure5_countermeasures(results):
    """Create Figure 5: Countermeasure comparison."""
    print("\nCreating Figure 5: Countermeasure comparison...")

    if len(results) < 2:
        print("  [WARN] Not enough data to create figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Colors
    colors = {
        "Baseline (Group)": "#e74c3c",
        "Diversity Prompt": "#3498db",
        "Adversarial": "#2ecc71",
    }
    markers = {
        "Baseline (Group)": "s",
        "Diversity Prompt": "o",
        "Adversarial": "^",
    }

    # Panel A: SDI trajectory
    ax1 = axes[0]
    for name, sdi_data in results.items():
        iterations = sorted(sdi_data.keys())
        means = [np.mean(sdi_data[it]) for it in iterations]
        sems = [np.std(sdi_data[it]) / np.sqrt(len(sdi_data[it])) for it in iterations]

        ax1.errorbar(iterations, means, yerr=sems,
                    marker=markers.get(name, 'o'), markersize=10,
                    label=name, linewidth=2.5, capsize=5,
                    color=colors.get(name, '#666666'),
                    markerfacecolor='white', markeredgewidth=2)

    ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Semantic Diversity Index (SDI)', fontsize=13, fontweight='bold')
    ax1.set_title('(A) SDI Trajectory by Countermeasure', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.grid(True, alpha=0.3)

    # Panel B: Relative change bar chart
    ax2 = axes[1]

    names = []
    rel_changes = []
    rel_sems = []

    for name, sdi_data in results.items():
        if 0 in sdi_data and 3 in sdi_data:
            baseline = np.mean(sdi_data[0])
            final_values = sdi_data[3]
            rel_values = [(v - baseline) / baseline * 100 for v in final_values]
            names.append(name.replace(" (Group)", "\n(Group)"))
            rel_changes.append(np.mean(rel_values))
            rel_sems.append(np.std(rel_values) / np.sqrt(len(rel_values)))

    x_pos = range(len(names))
    bar_colors = [colors.get(n.replace("\n", " "), '#666666') for n in names]

    bars = ax2.bar(x_pos, rel_changes, yerr=rel_sems,
                   color=bar_colors, edgecolor='black', linewidth=1.5,
                   capsize=8, error_kw={'linewidth': 2})

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, fontsize=11)
    ax2.set_ylabel('Relative SDI Change (%)', fontsize=13, fontweight='bold')
    ax2.set_title('(B) Final SDI Change from Baseline', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, rel_changes):
        height = bar.get_height()
        offset = 2 if height > 0 else -4
        ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_countermeasures.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig5_countermeasures.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(FIG_DIR, 'fig5_countermeasures.pdf')}")


def create_table3_countermeasures(results):
    """Create Table 3: Countermeasure summary statistics."""
    print("\nCreating Table 3: Countermeasure summary...")

    table_rows = []
    baseline_final = None

    for name, sdi_data in results.items():
        if 0 in sdi_data and 3 in sdi_data:
            baseline_sdi = np.mean(sdi_data[0])
            baseline_std = np.std(sdi_data[0])
            final_sdi = np.mean(sdi_data[3])
            final_std = np.std(sdi_data[3])
            change = (final_sdi - baseline_sdi) / baseline_sdi * 100
            n = len(sdi_data[3])

            if "Baseline" in name:
                baseline_final = sdi_data[3]

            table_rows.append({
                'name': name,
                'baseline_sdi': baseline_sdi,
                'baseline_std': baseline_std,
                'final_sdi': final_sdi,
                'final_std': final_std,
                'change': change,
                'n': n,
                'final_values': sdi_data[3]
            })

    # Build LaTeX table
    table_content = """\\begin{table}[ht]
\\centering
\\caption{Summary of Experiment 3: Countermeasure Effectiveness}
\\label{tab:exp3_summary}
\\begin{tabular}{lccccl}
\\hline
\\textbf{Condition} & \\textbf{Baseline SDI} & \\textbf{Final SDI} & \\textbf{Change (\\%)} & \\textbf{n} & \\textbf{vs Baseline} \\\\
\\hline
"""

    for row in table_rows:
        # Calculate statistical comparison vs baseline
        if baseline_final is not None and "Baseline" not in row['name']:
            t_stat, p_val = stats.ttest_ind(row['final_values'], baseline_final, equal_var=False)
            if p_val < 0.001:
                sig = "p < 0.001***"
            elif p_val < 0.01:
                sig = f"p = {p_val:.3f}**"
            elif p_val < 0.05:
                sig = f"p = {p_val:.3f}*"
            else:
                sig = f"p = {p_val:.3f}"
        else:
            sig = "---"

        table_content += f"{row['name']} & {row['baseline_sdi']:.3f} $\\pm$ {row['baseline_std']:.3f} & "
        table_content += f"{row['final_sdi']:.3f} $\\pm$ {row['final_std']:.3f} & "
        table_content += f"{row['change']:+.1f} & {row['n']} & {sig} \\\\\n"

    table_content += """\\hline
\\end{tabular}
\\end{table}
"""

    with open(os.path.join(FIG_DIR, "table3_countermeasures.tex"), "w") as f:
        f.write(table_content)

    print(f"  Saved: {os.path.join(FIG_DIR, 'table3_countermeasures.tex')}")


def main():
    results = analyze_results()

    if results:
        create_figure5_countermeasures(results)
        create_table3_countermeasures(results)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
