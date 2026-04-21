#!/usr/bin/env python3
"""
Create publication-ready figures for the Algorithmic Groupthink paper.
Formatted according to Scientific Reports guidelines:
- Helvetica font for all text
- Panel labels: lowercase bold (a, b, c)
- Labels: only first letter capitalized
- White background, minimal decoration
- Line width >= 1pt
"""

import os
import pickle
from collections import defaultdict
import numpy as np
from scipy import stats
import csv

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# Scientific Reports formatting
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# NPG (Nature Publishing Group) official color palette
# Source: ggsci package - https://nanx.me/ggsci/reference/pal_npg.html
NPG_COLORS = {
    'red': '#E64B35',         # NPG Red
    'blue': '#4DBBD5',        # NPG Blue/Cyan
    'green': '#00A087',       # NPG Green/Teal
    'purple': '#3C5488',      # NPG Dark Blue/Purple
    'orange': '#F39B7F',      # NPG Orange/Salmon
    'lavender': '#8491B4',    # NPG Lavender
    'mint': '#91D1C2',        # NPG Mint
    'darkred': '#DC0000',     # NPG Dark Red
    'brown': '#7E6148',       # NPG Brown
    'yellow': '#B09C85',      # NPG Tan/Yellow
}

# Map conditions to NPG colors
COLOR_INDIVIDUAL = NPG_COLORS['blue']   # Blue/Cyan for Individual
COLOR_GROUP = NPG_COLORS['red']         # Red for Group
COLORS_RATIO = [NPG_COLORS['blue'], NPG_COLORS['mint'], NPG_COLORS['orange'], NPG_COLORS['red']]  # Blue to red gradient

# Colors for t-SNE iteration comparison
COLOR_T0 = NPG_COLORS['purple']   # Purple - Iteration 0
COLOR_T3 = NPG_COLORS['orange']   # Orange - Iteration 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "embedding_cache")
FIG_DIR = os.path.join(BASE_DIR, "template", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_results(results_dir):
    csv_path = os.path.join(results_dir, "results.csv")
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["task_id"] = int(row["task_id"])
            row["iteration"] = int(row["iteration"])
            row["candidate"] = int(row["candidate"])
            results.append(row)
    return results


def load_cached_embeddings(exp_name, workflow, task_id, iteration):
    cache_key = f"{exp_name}_{workflow}_{task_id}_{iteration}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def compute_sdi(embeddings):
    n = len(embeddings)
    if n < 2:
        return 0.0
    diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    upper_tri = distances[np.triu_indices(n, k=1)]
    return np.mean(upper_tri)


def get_sdi_values(exp_name, results_dir):
    results = load_results(results_dir)
    groups = set()
    for row in results:
        groups.add((row["workflow"], row["task_id"], row["iteration"]))

    sdi_by_iteration = defaultdict(list)
    for workflow, task_id, iteration in groups:
        embeddings = load_cached_embeddings(exp_name, workflow, task_id, iteration)
        if embeddings is not None:
            sdi = compute_sdi(embeddings)
            sdi_by_iteration[iteration].append(sdi)

    return dict(sdi_by_iteration)


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold lowercase panel label (a, b, c, etc.)"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left')


def format_thousands(x):
    """Format numbers with comma separators for thousands"""
    if x >= 1000:
        return f'{x:,.0f}'
    return str(x)


def create_figure1_exp1_comparison():
    """Figure 1: Individual vs Group comparison with statistics."""
    print("Creating Figure 1: Exp1 comparison...")

    individual_sdi = get_sdi_values("exp1_individual",
                                    os.path.join(BASE_DIR, "results_exp1_individual"))
    all_sdi = get_sdi_values("exp1_all",
                             os.path.join(BASE_DIR, "results_exp1_all"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    iterations = sorted(individual_sdi.keys())

    # Calculate statistics
    ind_means = [np.mean(individual_sdi[i]) for i in iterations]
    ind_sems = [np.std(individual_sdi[i]) / np.sqrt(len(individual_sdi[i])) for i in iterations]
    all_means = [np.mean(all_sdi[i]) for i in iterations]
    all_sems = [np.std(all_sdi[i]) / np.sqrt(len(all_sdi[i])) for i in iterations]

    # Panel a: Semantic diversity trajectory
    ax1 = axes[0]
    add_panel_label(ax1, 'a')

    ax1.errorbar(iterations, ind_means, yerr=ind_sems, marker='o', markersize=7,
                 label='Individual-history', linewidth=1.5, capsize=4,
                 color=COLOR_INDIVIDUAL, markerfacecolor='white', markeredgewidth=1.5)
    ax1.errorbar(iterations, all_means, yerr=all_sems, marker='s', markersize=7,
                 label='Group-history', linewidth=1.5, capsize=4,
                 color=COLOR_GROUP, markerfacecolor='white', markeredgewidth=1.5)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Semantic diversity\n(mean pairwise distance)')
    ax1.legend(frameon=False, loc='upper right')
    ax1.set_xticks(iterations)
    ax1.set_ylim(0.2, 0.45)

    # Panel b: Relative change (bar chart)
    ax2 = axes[1]
    add_panel_label(ax2, 'b')

    final_iter = max(iterations)

    # Calculate relative change
    ind_baseline = np.mean(individual_sdi[0])
    all_baseline = np.mean(all_sdi[0])
    ind_final_rel = (np.mean(individual_sdi[final_iter]) - ind_baseline) / ind_baseline * 100
    all_final_rel = (np.mean(all_sdi[final_iter]) - all_baseline) / all_baseline * 100

    # Calculate SEM for relative change
    ind_rel_values = [(v - ind_baseline) / ind_baseline * 100 for v in individual_sdi[final_iter]]
    all_rel_values = [(v - all_baseline) / all_baseline * 100 for v in all_sdi[final_iter]]
    ind_rel_sem = np.std(ind_rel_values) / np.sqrt(len(ind_rel_values))
    all_rel_sem = np.std(all_rel_values) / np.sqrt(len(all_rel_values))

    x_pos = [0, 1]
    bars = ax2.bar(x_pos, [ind_final_rel, all_final_rel],
                   yerr=[ind_rel_sem, all_rel_sem],
                   color=[COLOR_INDIVIDUAL, COLOR_GROUP],
                   edgecolor='black', linewidth=1.0,
                   capsize=5, error_kw={'linewidth': 1.5}, width=0.6)

    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Relative diversity change (%)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Individual-\nhistory', 'Group-\nhistory'])
    ax2.set_ylim(-45, 30)  # Expanded range to accommodate labels

    # Statistics
    t_stat, p_value = stats.ttest_ind(individual_sdi[final_iter], all_sdi[final_iter], equal_var=False)
    n1, n2 = len(individual_sdi[final_iter]), len(all_sdi[final_iter])
    var1 = np.var(individual_sdi[final_iter], ddof=1)
    var2 = np.var(all_sdi[final_iter], ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (np.mean(individual_sdi[final_iter]) - np.mean(all_sdi[final_iter])) / pooled_std

    # Add significance marker first to know its position
    y_sig_base = max(ind_final_rel + ind_rel_sem, all_final_rel + all_rel_sem) + 4
    ax2.plot([0, 0, 1, 1], [y_sig_base, y_sig_base + 2, y_sig_base + 2, y_sig_base], 'k-', lw=1.0)
    ax2.text(0.5, y_sig_base + 3, '***', ha='center', va='bottom', fontsize=10)

    # Add value labels - position considering significance marker for positive values
    sems = [ind_rel_sem, all_rel_sem]
    for i, (bar, val, sem) in enumerate(zip(bars, [ind_final_rel, all_final_rel], sems)):
        if val > 0:
            # Place label above significance marker to avoid overlap
            y_pos = y_sig_base + 8  # Above the significance marker
            va = 'bottom'
        else:
            y_pos = val - sem - 2  # Below the error bar
            va = 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:+.1f}%', ha='center', va=va, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_exp1_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig1_exp1_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_exp1_comparison.pdf")


def create_figure2_share_ratio():
    """Figure 2: Share ratio analysis with statistics."""
    print("Creating Figure 2: Share ratio analysis...")

    ratio_stats = {}
    for ratio in [0.0, 0.25, 0.5, 0.75]:
        dir_name = f"results_exp2_ratio_{str(ratio).replace('.', '_')}"
        exp_name = f"exp2_ratio_{ratio}"
        sdi_data = get_sdi_values(exp_name, os.path.join(BASE_DIR, dir_name))
        if sdi_data:
            ratio_stats[ratio] = sdi_data

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel a: Semantic diversity trajectories
    ax1 = axes[0]
    add_panel_label(ax1, 'a')

    for idx, (ratio, sdi_data) in enumerate(sorted(ratio_stats.items())):
        iterations = sorted(sdi_data.keys())
        means = [np.mean(sdi_data[i]) for i in iterations]
        sems = [np.std(sdi_data[i]) / np.sqrt(len(sdi_data[i])) for i in iterations]

        ax1.errorbar(iterations, means, yerr=sems, marker='o', markersize=6,
                     label=f'{ratio:.0%}', linewidth=1.5, capsize=3,
                     color=COLORS_RATIO[idx], markerfacecolor='white', markeredgewidth=1.5)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Semantic diversity\n(mean pairwise distance)')
    ax1.legend(title='Share ratio', frameon=False, loc='lower left')
    ax1.set_xticks([0, 1, 2, 3])

    # Panel b: Final relative diversity vs share ratio
    ax2 = axes[1]
    add_panel_label(ax2, 'b')

    ratios = sorted(ratio_stats.keys())
    final_rel_changes = []
    final_rel_sems = []

    for ratio in ratios:
        sdi_data = ratio_stats[ratio]
        iterations = sorted(sdi_data.keys())
        baseline = np.mean(sdi_data[0])
        final_iter = max(iterations)
        final_rel = [((v - baseline) / baseline * 100) for v in sdi_data[final_iter]]
        final_rel_changes.append(np.mean(final_rel))
        final_rel_sems.append(np.std(final_rel) / np.sqrt(len(final_rel)))

    bars = ax2.bar(range(len(ratios)), final_rel_changes, yerr=final_rel_sems,
                   color=COLORS_RATIO, edgecolor='black', linewidth=1.0,
                   capsize=4, width=0.6)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax2.set_xticks(range(len(ratios)))
    ax2.set_xticklabels([f'{r:.0%}' for r in ratios])
    ax2.set_xlabel('Share ratio')
    ax2.set_ylabel('Relative diversity change (%)')
    ax2.set_ylim(-25, 20)

    # Add value labels - position based on error bar height to avoid overlap
    for i, (bar, val, sem) in enumerate(zip(bars, final_rel_changes, final_rel_sems)):
        if val > 0:
            y_pos = val + sem + 1.5  # Above the error bar
            va = 'bottom'
        else:
            y_pos = val - sem - 1.5  # Below the error bar
            va = 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:+.1f}%', ha='center', va=va, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_share_ratio.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig2_share_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_share_ratio.pdf")


def create_figure3_mechanism():
    """Figure 3: Mechanism analysis - t-SNE and distribution."""
    print("Creating Figure 3: Mechanism analysis...")

    from sklearn.manifold import TSNE

    # Load embeddings for both conditions from all workflows
    workflows = ['gen-critic-sum', 'gen-critic', 'parallel-merge', 'parallel', 'self-refine']
    individual_embeddings = {'t0': [], 't3': []}
    group_embeddings = {'t0': [], 't3': []}

    # Load Individual condition from all workflows
    for workflow in workflows:
        for task_id in range(5):
            for t, label in [(0, 't0'), (3, 't3')]:
                emb = load_cached_embeddings("exp1_individual", workflow, task_id, t)
                if emb is not None:
                    individual_embeddings[label].extend(emb)

    # Load Group condition from all workflows
    for workflow in workflows:
        for task_id in range(5):
            for t, label in [(0, 't0'), (3, 't3')]:
                emb = load_cached_embeddings("exp1_all", workflow, task_id, t)
                if emb is not None:
                    group_embeddings[label].extend(emb)

    print(f"  Individual embeddings: t0={len(individual_embeddings['t0'])}, t3={len(individual_embeddings['t3'])}")
    print(f"  Group embeddings: t0={len(group_embeddings['t0'])}, t3={len(group_embeddings['t3'])}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel a: Individual t-SNE
    ax1 = axes[0, 0]
    add_panel_label(ax1, 'a')

    ind_all = np.array(individual_embeddings['t0'] + individual_embeddings['t3'])
    has_legend = False
    if len(ind_all) > 0:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(ind_all)-1))
        ind_tsne = tsne.fit_transform(ind_all)
        n_t0 = len(individual_embeddings['t0'])

        # NPG colors: purple circles for t=0, orange triangles for t=3
        ax1.scatter(ind_tsne[:n_t0, 0], ind_tsne[:n_t0, 1],
                   facecolors='none', alpha=0.8, s=30, label='Iteration 0',
                   edgecolors=COLOR_T0, linewidths=0.8)
        ax1.scatter(ind_tsne[n_t0:, 0], ind_tsne[n_t0:, 1],
                   facecolors='none', alpha=0.9, s=50, marker='^', label='Iteration 3',
                   edgecolors=COLOR_T3, linewidths=0.8)
        has_legend = True

    ax1.set_xlabel('t-SNE dimension 1')
    ax1.set_ylabel('t-SNE dimension 2')
    ax1.set_title('Individual-history', fontsize=11)
    if has_legend:
        ax1.legend(frameon=False, loc='upper right')

    # Panel b: Group t-SNE
    ax2 = axes[0, 1]
    add_panel_label(ax2, 'b')

    grp_all = np.array(group_embeddings['t0'] + group_embeddings['t3'])
    has_legend = False
    if len(grp_all) > 0:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(grp_all)-1))
        grp_tsne = tsne.fit_transform(grp_all)
        n_t0 = len(group_embeddings['t0'])

        # NPG colors: purple circles for t=0, orange triangles for t=3
        ax2.scatter(grp_tsne[:n_t0, 0], grp_tsne[:n_t0, 1],
                   facecolors='none', alpha=0.8, s=30, label='Iteration 0',
                   edgecolors=COLOR_T0, linewidths=0.8)
        ax2.scatter(grp_tsne[n_t0:, 0], grp_tsne[n_t0:, 1],
                   facecolors='none', alpha=0.9, s=50, marker='^', label='Iteration 3',
                   edgecolors=COLOR_T3, linewidths=0.8)
        has_legend = True

    ax2.set_xlabel('t-SNE dimension 1')
    ax2.set_ylabel('t-SNE dimension 2')
    ax2.set_title('Group-history', fontsize=11)
    if has_legend:
        ax2.legend(frameon=False, loc='upper right')

    # Panel c: Mean distance from centroid over iterations
    ax3 = axes[1, 0]
    add_panel_label(ax3, 'c')

    # Calculate mean distance from centroid for each iteration
    def calc_centroid_distances(embeddings_by_iter):
        """Calculate mean distance from centroid for each iteration."""
        distances_by_iter = {}
        for iteration, emb_list in embeddings_by_iter.items():
            if len(emb_list) > 0:
                emb_array = np.array(emb_list)
                centroid = np.mean(emb_array, axis=0)
                distances = np.sqrt(np.sum((emb_array - centroid) ** 2, axis=1))
                distances_by_iter[iteration] = distances
        return distances_by_iter

    # Load embeddings for all iterations for both conditions
    ind_emb_by_iter = {}
    grp_emb_by_iter = {}

    for t in range(4):  # iterations 0-3
        ind_emb_by_iter[t] = []
        grp_emb_by_iter[t] = []
        for workflow in workflows:
            for task_id in range(5):
                emb = load_cached_embeddings("exp1_individual", workflow, task_id, t)
                if emb is not None:
                    ind_emb_by_iter[t].extend(emb)
                emb = load_cached_embeddings("exp1_all", workflow, task_id, t)
                if emb is not None:
                    grp_emb_by_iter[t].extend(emb)

    ind_centroid_dist = calc_centroid_distances(ind_emb_by_iter)
    grp_centroid_dist = calc_centroid_distances(grp_emb_by_iter)

    iterations = sorted(ind_centroid_dist.keys())
    ind_means = [np.mean(ind_centroid_dist[i]) for i in iterations]
    grp_means = [np.mean(grp_centroid_dist[i]) for i in iterations]

    # Normalize to baseline (iteration 0)
    ind_baseline = ind_means[0]
    grp_baseline = grp_means[0]
    ind_norm = [m / ind_baseline for m in ind_means]
    grp_norm = [m / grp_baseline for m in grp_means]
    ind_sems = [np.std(ind_centroid_dist[i]) / np.sqrt(len(ind_centroid_dist[i])) / ind_baseline for i in iterations]
    grp_sems = [np.std(grp_centroid_dist[i]) / np.sqrt(len(grp_centroid_dist[i])) / grp_baseline for i in iterations]

    ax3.errorbar(iterations, ind_norm, yerr=ind_sems, fmt='o-', color=COLOR_INDIVIDUAL,
                 label='Individual-history', linewidth=1.5, markersize=6, capsize=3,
                 markerfacecolor='white', markeredgewidth=1.5)
    ax3.errorbar(iterations, grp_norm, yerr=grp_sems, fmt='s-', color=COLOR_GROUP,
                 label='Group-history', linewidth=1.5, markersize=6, capsize=3,
                 markerfacecolor='white', markeredgewidth=1.5)
    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Normalized centroid distance')
    ax3.legend(frameon=False, loc='upper right')
    ax3.set_xticks(iterations)

    # Panel d: Pairwise distance distribution - Focus on t=3 comparison with violin plot
    ax4 = axes[1, 1]
    add_panel_label(ax4, 'd')

    # Calculate pairwise distances at t=0 and t=3
    def get_pairwise_distances(embeddings_list):
        if len(embeddings_list) < 2:
            return np.array([])
        all_emb = np.array(embeddings_list)
        n = len(all_emb)
        diff = all_emb[:, np.newaxis, :] - all_emb[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances[np.triu_indices(n, k=1)]

    ind_t3_dist = get_pairwise_distances(individual_embeddings['t3'])
    grp_t3_dist = get_pairwise_distances(group_embeddings['t3'])

    # Only create plot if we have data
    has_data = len(ind_t3_dist) > 0 and len(grp_t3_dist) > 0

    if has_data:
        # Create violin plot for t=3 comparison
        positions = [1, 2]
        data = [ind_t3_dist, grp_t3_dist]

        vp = ax4.violinplot(data, positions=positions, showmeans=True, showmedians=True)

        # Color the violin plots
        colors = [COLOR_INDIVIDUAL, COLOR_GROUP]
        for i, (body, color) in enumerate(zip(vp['bodies'], colors)):
            body.set_facecolor(color)
            body.set_alpha(0.6)
            body.set_edgecolor('black')
            body.set_linewidth(1)

        # Style the lines
        for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if partname in vp:
                vp[partname].set_edgecolor('black')
                vp[partname].set_linewidth(1)

        # Add individual points (subsampled for visibility)
        np.random.seed(42)
        subsample = 500
        ind_sub = np.random.choice(ind_t3_dist, min(subsample, len(ind_t3_dist)), replace=False)
        grp_sub = np.random.choice(grp_t3_dist, min(subsample, len(grp_t3_dist)), replace=False)

        ax4.scatter(np.random.normal(1, 0.05, len(ind_sub)), ind_sub,
                   alpha=0.1, s=3, color=COLOR_INDIVIDUAL)
        ax4.scatter(np.random.normal(2, 0.05, len(grp_sub)), grp_sub,
                   alpha=0.1, s=3, color=COLOR_GROUP)

        # Add statistical annotation
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(ind_t3_dist, grp_t3_dist)

        # Draw significance bar
        y_max = max(np.percentile(ind_t3_dist, 99), np.percentile(grp_t3_dist, 99))
        y_bar = y_max * 1.05
        ax4.plot([1, 1, 2, 2], [y_bar, y_bar*1.02, y_bar*1.02, y_bar], 'k-', linewidth=1)
        ax4.text(1.5, y_bar*1.04, '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*'),
                ha='center', va='bottom', fontsize=12)

        # Add mean values as text
        ind_mean = np.mean(ind_t3_dist)
        grp_mean = np.mean(grp_t3_dist)
        pct_diff = ((ind_mean - grp_mean) / ind_mean) * 100

        # Add mean markers (horizontal lines) for emphasis
        ax4.hlines(ind_mean, 0.7, 1.3, colors=COLOR_INDIVIDUAL, linewidth=2, linestyle='--', label=f'Individual μ={ind_mean:.4f}')
        ax4.hlines(grp_mean, 1.7, 2.3, colors=COLOR_GROUP, linewidth=2, linestyle='--', label=f'Group μ={grp_mean:.4f}')

        # Add percentage difference annotation
        ax4.annotate(f'{pct_diff:.1f}% lower',
                    xy=(2, grp_mean), xytext=(2.5, (ind_mean + grp_mean)/2),
                    fontsize=9, ha='left', va='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))

        # Set x-tick labels
        ax4.set_xticks(positions)
        ax4.set_xticklabels(['Individual\n(Iter. 3)', 'Group\n(Iter. 3)'])
        ax4.set_xlim(0.4, 3.0)

        # Narrow Y-axis range to emphasize the difference
        y_min = min(np.percentile(ind_t3_dist, 1), np.percentile(grp_t3_dist, 1))
        y_max = max(np.percentile(ind_t3_dist, 99), np.percentile(grp_t3_dist, 99))
        margin = (y_max - y_min) * 0.15
        ax4.set_ylim(y_min - margin, y_max + margin * 2)

    else:
        ax4.text(0.5, 0.5, 'No embedding data\navailable', ha='center', va='center',
                transform=ax4.transAxes, fontsize=10)

    ax4.set_ylabel('Pairwise distance at Iteration 3')
    ax4.set_title('Distribution comparison', fontsize=11)
    ax4.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_mechanism.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig3_mechanism.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_mechanism.pdf")


def create_figure4_convergence():
    """Figure 4: Convergence dynamics."""
    print("Creating Figure 4: Convergence dynamics...")

    individual_sdi = get_sdi_values("exp1_individual", os.path.join(BASE_DIR, "results_exp1_individual"))
    all_sdi = get_sdi_values("exp1_all", os.path.join(BASE_DIR, "results_exp1_all"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    iterations = sorted(individual_sdi.keys())

    # Panel a: Cumulative diversity change
    ax1 = axes[0]
    add_panel_label(ax1, 'a')

    ind_baseline = np.mean(individual_sdi[0])
    all_baseline = np.mean(all_sdi[0])

    ind_cum = [(np.mean(individual_sdi[i]) - ind_baseline) / ind_baseline * 100 for i in iterations]
    all_cum = [(np.mean(all_sdi[i]) - all_baseline) / all_baseline * 100 for i in iterations]

    # Calculate SEM for cumulative change
    ind_cum_sems = [np.std(individual_sdi[i]) / np.sqrt(len(individual_sdi[i])) / ind_baseline * 100 for i in iterations]
    all_cum_sems = [np.std(all_sdi[i]) / np.sqrt(len(all_sdi[i])) / all_baseline * 100 for i in iterations]

    ax1.errorbar(iterations, ind_cum, yerr=ind_cum_sems, fmt='o-', color=COLOR_INDIVIDUAL,
             label='Individual-history', linewidth=1.5, markersize=6, capsize=3,
             markerfacecolor='white', markeredgewidth=1.5)
    ax1.errorbar(iterations, all_cum, yerr=all_cum_sems, fmt='s-', color=COLOR_GROUP,
             label='Group-history', linewidth=1.5, markersize=6, capsize=3,
             markerfacecolor='white', markeredgewidth=1.5)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cumulative diversity change (%)')
    ax1.legend(frameon=False, loc='lower left')
    ax1.set_xticks(iterations)

    # Panel b: Per-iteration delta
    ax2 = axes[1]
    add_panel_label(ax2, 'b')

    ind_deltas = []
    all_deltas = []
    ind_delta_sems = []
    all_delta_sems = []
    for i in range(1, len(iterations)):
        prev_i = iterations[i-1]
        curr_i = iterations[i]
        # Calculate delta for each task
        ind_task_deltas = [individual_sdi[curr_i][j] - individual_sdi[prev_i][j]
                          for j in range(min(len(individual_sdi[curr_i]), len(individual_sdi[prev_i])))]
        all_task_deltas = [all_sdi[curr_i][j] - all_sdi[prev_i][j]
                          for j in range(min(len(all_sdi[curr_i]), len(all_sdi[prev_i])))]
        ind_deltas.append(np.mean(ind_task_deltas))
        all_deltas.append(np.mean(all_task_deltas))
        ind_delta_sems.append(np.std(ind_task_deltas) / np.sqrt(len(ind_task_deltas)))
        all_delta_sems.append(np.std(all_task_deltas) / np.sqrt(len(all_task_deltas)))

    x = np.arange(len(ind_deltas))
    width = 0.35

    ax2.bar(x - width/2, ind_deltas, width, color=COLOR_INDIVIDUAL,
            label='Individual-history', edgecolor='black', linewidth=1.0,
            yerr=ind_delta_sems, capsize=3, error_kw={'elinewidth': 1.5})
    ax2.bar(x + width/2, all_deltas, width, color=COLOR_GROUP,
            label='Group-history', edgecolor='black', linewidth=1.0,
            yerr=all_delta_sems, capsize=3, error_kw={'elinewidth': 1.5})
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Iteration transition')
    ax2.set_ylabel('Per-iteration diversity change')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{i} to {i+1}' for i in range(len(ind_deltas))])
    ax2.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_convergence.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig4_convergence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_convergence.pdf")


def create_figure5_countermeasures():
    """Figure 5: Countermeasure effectiveness."""
    print("Creating Figure 5: Countermeasures...")

    # Load countermeasure data - baseline is exp1_all (group-history)
    baseline_sdi = get_sdi_values("exp1_all", os.path.join(BASE_DIR, "results_exp1_all"))
    diversity_sdi = get_sdi_values("exp3_diversity_prompt", os.path.join(BASE_DIR, "results_exp3_diversity_prompt"))
    adversarial_sdi = get_sdi_values("exp3_adversarial", os.path.join(BASE_DIR, "results_exp3_adversarial"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel a: SDI trajectories
    ax1 = axes[0]
    add_panel_label(ax1, 'a')

    colors = {'Baseline': COLOR_GROUP, 'Diversity prompt': NPG_COLORS['green'], 'Adversarial': NPG_COLORS['lavender']}
    markers = {'Baseline': 's', 'Diversity prompt': 'o', 'Adversarial': '^'}

    for name, sdi_data, color, marker in [
        ('Baseline', baseline_sdi, colors['Baseline'], markers['Baseline']),
        ('Diversity prompt', diversity_sdi, colors['Diversity prompt'], markers['Diversity prompt']),
        ('Adversarial', adversarial_sdi, colors['Adversarial'], markers['Adversarial'])
    ]:
        if sdi_data:
            iterations = sorted(sdi_data.keys())
            means = [np.mean(sdi_data[i]) for i in iterations]
            sems = [np.std(sdi_data[i]) / np.sqrt(len(sdi_data[i])) for i in iterations]
            ax1.errorbar(iterations, means, yerr=sems, marker=marker, markersize=6,
                        label=name, linewidth=1.5, capsize=3, color=color,
                        markerfacecolor='white', markeredgewidth=1.5)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Semantic diversity\n(mean pairwise distance)')
    ax1.legend(frameon=False, loc='upper right')
    ax1.set_xticks([0, 1, 2, 3])

    # Panel b: Final relative change
    ax2 = axes[1]
    add_panel_label(ax2, 'b')

    conditions = []
    changes = []
    sems = []
    bar_colors = []

    for name, sdi_data, color in [
        ('Baseline', baseline_sdi, colors['Baseline']),
        ('Diversity\nprompt', diversity_sdi, colors['Diversity prompt']),
        ('Adversarial', adversarial_sdi, colors['Adversarial'])
    ]:
        if sdi_data:
            iterations = sorted(sdi_data.keys())
            baseline = np.mean(sdi_data[0])
            final = np.mean(sdi_data[max(iterations)])
            rel_change = (final - baseline) / baseline * 100
            rel_values = [(v - baseline) / baseline * 100 for v in sdi_data[max(iterations)]]

            conditions.append(name)
            changes.append(rel_change)
            sems.append(np.std(rel_values) / np.sqrt(len(rel_values)))
            bar_colors.append(color)

    bars = ax2.bar(range(len(conditions)), changes, yerr=sems,
                   color=bar_colors, edgecolor='black', linewidth=1.0,
                   capsize=4, width=0.6)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(conditions)
    ax2.set_ylabel('Relative diversity change (%)')
    ax2.set_ylim(-40, 10)

    # Add value labels - position based on error bar height to avoid overlap
    for bar, val, sem in zip(bars, changes, sems):
        if val > 0:
            y_pos = val + sem + 1.5  # Above the error bar
            va = 'bottom'
        else:
            y_pos = val - sem - 1.5  # Below the error bar
            va = 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:+.1f}%', ha='center', va=va, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_countermeasures.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "fig5_countermeasures.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig5_countermeasures.pdf")


def main():
    print("\n" + "="*70)
    print("Creating Scientific Reports-formatted figures")
    print("="*70)

    create_figure1_exp1_comparison()
    create_figure2_share_ratio()
    create_figure3_mechanism()
    create_figure4_convergence()
    create_figure5_countermeasures()

    print("\n" + "="*70)
    print("All figures created successfully!")
    print(f"Output directory: {FIG_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
