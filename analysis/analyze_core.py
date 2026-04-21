
import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Dict, Any
import openai
from collections import Counter
from scipy.spatial.distance import jensenshannon

"""
Analysis script for "Algorithmic Groupthink" paper.

Features:
1. Computes Semantic Diversity Index (SDI) using OpenAI Embeddings.
2. Generates publication-ready figures:
   - Fig 1: SDI Trajectory (Line plot)
   - Fig 2: PCA Visualization (Scatter plot)
   - Fig 3: Team Size Effect (Bar plot)
3. Performs basic statistical tests (t-test).

Usage:
  export OPENAI_API_KEY=...
  python analyze.py --input results_v2/results.csv --output analysis_output
"""

# =========================
# Configuration
# =========================
EMBEDDING_MODEL = "text-embedding-3-large"
SNS_STYLE = "whitegrid"
PALETTE = "viridis"

def get_embeddings(texts: List[str], batch_size: int = 50) -> np.ndarray:
    """
    Fetch embeddings from OpenAI API in batches.
    Returns a numpy array of shape (N, D).
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = []
    
    # Remove empty strings to avoid API errors, but keep track of indices if needed
    # For simplicity, we assume valid texts.
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            # Sort by index to ensure order is preserved
            batch_embs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
            embeddings.extend(batch_embs)
            print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts...")
        except Exception as e:
            print(f"Error embedding batch {i}: {e}")
            # Fill with zeros or handle appropriately
            embeddings.extend([[0.0]*3072] * len(batch)) 
            
    return np.array(embeddings)

def compute_sdi(embeddings: np.ndarray) -> float:
    """
    Compute Semantic Diversity Index (SDI).
    SDI = Mean pairwise Euclidean distance.
    """
    if len(embeddings) < 2:
        return 0.0
    dists = euclidean_distances(embeddings)
    # Take upper triangle to avoid duplicates and self-distances
    tri_indices = np.triu_indices(len(embeddings), k=1)
    return np.mean(dists[tri_indices])

def compute_vendi_score(embeddings: np.ndarray, kernel: str = "linear") -> float:
    """
    Compute Vendi Score (effective number of unique elements).
    Reference: Friedman et al. (2023) "The Vendi Score: A Diversity Evaluation Metric for Machine Learning"
    
    Vendi Score = exp( - sum(lambda_i * log(lambda_i)) )
    where lambda_i are eigenvalues of the normalized kernel matrix K/N.
    """
    if len(embeddings) < 2:
        return 1.0
    
    X = embeddings
    N = X.shape[0]
    
    # Normalize embeddings to unit length for cosine similarity (linear kernel)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Compute Kernel Matrix K (Cosine Similarity)
    K = X_norm @ X_norm.T
    
    # Normalize Kernel Matrix
    K_bar = K / N
    
    # Compute eigenvalues (symmetric matrix, so eigh is faster/stable)
    eigvals = np.linalg.eigvalsh(K_bar)
    
    # Filter out small numerical noise and normalize to sum to 1 (probability distribution)
    eigvals = eigvals[eigvals > 1e-10]
    # Theoretically sum(eigvals) should be 1 for K/N, but re-normalize to be safe
    eigvals = eigvals / eigvals.sum()
    
    # Compute Shannon Entropy of eigenvalues
    entropy = -np.sum(eigvals * np.log(eigvals))
    
    # Vendi Score
    return np.exp(entropy)

def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """
    Compute Distinct-n: ratio of unique n-grams to total n-grams.
    Higher values indicate more lexical diversity.
    
    Args:
        texts: List of generated texts
        n: n-gram size (default: 2 for bigrams)
    
    Returns:
        Distinct-n score (0.0 to 1.0)
    """
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    
    if len(all_ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    return unique_ngrams / total_ngrams

def compute_jsd(texts: List[str]) -> float:
    """
    Compute average Jensen-Shannon Divergence between individual text distributions
    and the mean distribution.
    
    Returns:
        Average JSD score
    """
    if len(texts) < 2:
        return 0.0
    
    # Build vocabulary
    vocab = set()
    for text in texts:
        vocab.update(text.lower().split())
    
    if len(vocab) == 0:
        return 0.0
    
    vocab = sorted(vocab)
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    
    # Convert texts to probability distributions
    distributions = []
    for text in texts:
        counts = Counter(text.lower().split())
        total = sum(counts.values())
        if total == 0:
            continue
        dist = np.zeros(len(vocab))
        for word, count in counts.items():
            dist[vocab_to_idx[word]] = count / total
        distributions.append(dist)
    
    if len(distributions) < 2:
        return 0.0
    
    # Compute mean distribution
    mean_dist = np.mean(distributions, axis=0)
    
    # Compute average JSD to mean
    jsds = [jensenshannon(dist, mean_dist) for dist in distributions]
    
    return np.mean(jsds)


def compute_drift_from_initial(initial_texts: List[str], current_texts: List[str]) -> float:
    """
    Compute Jensen-Shannon Divergence between the aggregated unigram distribution 
    of the initial texts (Iteration 0) and the current texts (Iteration k).
    This measures how far the group has drifted from the starting point.
    """
    if not initial_texts or not current_texts:
        return 0.0

    def get_dist(texts):
        counts = Counter()
        for t in texts:
            counts.update(t.lower().replace("\n", " ").split())
        return counts

    p = get_dist(initial_texts)
    q = get_dist(current_texts)
    
    # Union vocabulary
    vocab = set(p.keys()) | set(q.keys())
    if not vocab:
        return 0.0
    
    vocab = sorted(vocab)
    p_vec = np.array([p[w] for w in vocab], dtype=np.float64)
    q_vec = np.array([q[w] for w in vocab], dtype=np.float64)
    
    # Normalize
    p_prob = p_vec / (p_vec.sum() + 1e-12)
    q_prob = q_vec / (q_vec.sum() + 1e-12)
    m = 0.5 * (p_prob + q_prob)
    
    def kl_div(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * (np.log(a[mask]) - np.log(b[mask]))).sum())
    
    return 0.5 * kl_div(p_prob, m) + 0.5 * kl_div(q_prob, m)


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load results CSV and return DataFrame.
    """
    df = pd.read_csv(csv_path)
    # Filter out empty outputs
    df = df[df['final_output'].notna() & (df['final_output'] != "")]
    
    # Fix: Unescape newlines that were escaped in sim_v2.py
    # The CSV contains literal "\n" characters (backslash + n), we need actual newlines
    df['final_output'] = df['final_output'].apply(lambda x: x.replace("\\n", "\n") if isinstance(x, str) else x)
    
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results_v2/results.csv")
    parser.add_argument("--output", type=str, default="analysis_output")
    parser.add_argument("--cache", type=str, default="embeddings_cache.npy", help="Path to save/load embeddings cache")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows.")

    # --- 1. Embeddings ---
    # Check if cache exists to save money/time
    if os.path.exists(args.cache):
        print(f"Loading embeddings from cache: {args.cache}")
        all_embeddings = np.load(args.cache)
        if len(all_embeddings) != len(df):
            print("Cache size mismatch! Re-computing embeddings...")
            all_embeddings = get_embeddings(df['final_output'].tolist())
            np.save(args.cache, all_embeddings)
    else:
        print("Computing embeddings (this may cost API credits)...")
        all_embeddings = get_embeddings(df['final_output'].tolist())
        np.save(args.cache, all_embeddings)

    # Add embeddings to DF (as a column of lists is tricky, we'll use indices)
    # Better: Compute SDI per group and store in a summary DF
    
    # --- 2. Compute SDI per (Model, Workflow, Task, Iteration) ---
    print("Computing SDI for each group...")
    results = []
    
    # Group by experimental conditions
    grouped = df.groupby(['model_family', 'model', 'workflow', 'task_id', 'iteration'])
    
    # Map original indices to embeddings
    # We need to know which row in DF corresponds to which index in all_embeddings
    # Since we didn't shuffle DF, it should be 1:1
    
    for name, group in grouped:
        indices = group.index.tolist()
        group_embs = all_embeddings[indices] # This works if df index was reset or matches
        # If df was filtered, indices might not match array positions. 
        # Let's be safer:
        # We need to map df integer location (iloc) to embedding index.
        # But df is filtered. 
        # Correct approach: 
        # 1. Get list of texts. 2. Embed. 3. Assign embeddings back to DF.
        pass

    # Re-approach for safety:
    # Assign embedding index to DF
    df['emb_idx'] = range(len(df))
    
    sdi_records = []
    
    # Pre-compute Iteration 0 texts for drift calculation
    initial_texts_map = {}
    iter0_df = df[df['iteration'] == 0]
    for (family, model, workflow, task_id), group in iter0_df.groupby(['model_family', 'model', 'workflow', 'task_id']):
        initial_texts_map[(family, model, workflow, task_id)] = group['final_output'].tolist()
    
    grouped = df.groupby(['model_family', 'model', 'workflow', 'task_id', 'iteration'])
    for (family, model, workflow, task_id, iteration), group in grouped:
        indices = group['emb_idx'].values
        group_embs = all_embeddings[indices]
        group_texts = group['final_output'].tolist()
        
        sdi = compute_sdi(group_embs)
        vendi = compute_vendi_score(group_embs)
        distinct_2 = compute_distinct_n(group_texts, n=2)
        distinct_2 = compute_distinct_n(group_texts, n=2)
        jsd = compute_jsd(group_texts)
        
        # Compute Drift
        drift = 0.0
        if iteration > 0:
            initial_texts = initial_texts_map.get((family, model, workflow, task_id))
            if initial_texts:
                drift = compute_drift_from_initial(initial_texts, group_texts)
        
        sdi_records.append({
            'model_family': family,
            'model': model,
            'workflow': workflow,
            'task_id': task_id,
            'iteration': iteration,
            'sdi': sdi,
            'vendi': vendi,
            'distinct_2': distinct_2,
            'distinct_2': distinct_2,
            'jsd': jsd,
            'drift': drift,
            'n_samples': len(group)
        })
    
    df_sdi = pd.DataFrame(sdi_records)
    df_sdi.to_csv(os.path.join(args.output, "sdi_summary.csv"), index=False)
    print("SDI summary saved.")

    # --- 3. Visualization ---
    sns.set_style(SNS_STYLE)
    
    # Figure 1: SDI Trajectory (Aggregated over tasks)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_sdi, 
        x="iteration", 
        y="sdi", 
        hue="workflow", 
        style="model_family",
        markers=True, 
        dashes=False,
        linewidth=2.5
    )
    plt.title("Collapse of Semantic Diversity Across Iterations", fontsize=14)
    plt.xlabel("Iteration Round", fontsize=12)
    plt.ylabel("Semantic Diversity Index (SDI)", fontsize=12)
    plt.legend(title="Workflow", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "fig1_sdi_trajectory.png"), dpi=300)
    plt.close()
    print("Figure 1 generated.")

    # Figure 2: PCA Visualization (Example Task)
    # Pick the first task and 'gen-critic' workflow for demo
    example_task_id = 0 
    example_workflow = "gen-critic"
    
    subset = df[
        (df['task_id'] == example_task_id) & 
        (df['workflow'] == example_workflow) &
        (df['iteration'].isin([0, 3])) # Compare start vs end
    ].copy()
    
    if not subset.empty:
        subset_indices = subset['emb_idx'].values
        subset_embs = all_embeddings[subset_indices]
        
        if len(subset) < 2:
            print(f"Skipping Figure 2 (Not enough data points for PCA: {len(subset)})")
        else:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(subset_embs)
            
            subset['pca_x'] = coords[:, 0]
            subset['pca_y'] = coords[:, 1]
            
            plt.figure(figsize=(8, 8))
            sns.scatterplot(
                data=subset,
                x='pca_x',
                y='pca_y',
                hue='iteration',
                palette={0: 'blue', 3: 'red'} if 3 in subset['iteration'].values else None,
                s=100,
                alpha=0.7
            )
            plt.title(f"PCA of Semantic Space: Task {example_task_id}", fontsize=14)
            plt.xlabel("PC1", fontsize=12)
            plt.ylabel("PC2", fontsize=12)
            plt.legend(title="Iteration")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, "fig2_pca_collapse.png"), dpi=300)
            plt.close()
            print("Figure 2 generated.")
    else:
        print("Skipping Figure 2 (No data for example task).")

    # Figure 3: Team Size / Workflow Comparison (Bar plot at Iteration 3)
    # We can interpret workflows as proxies for team complexity
    # self-refine (1 agent), gen-critic (2 agents), gen-critic-sum (3 agents)
    
    # Use the last iteration available in the data
    max_iter = df_sdi['iteration'].max()
    df_final = df_sdi[df_sdi['iteration'] == max_iter].copy()
    print(f"Generating Figure 3 using data from Iteration {max_iter}...")
    
    # Map workflows to "Team Size" label if applicable
    workflow_map = {
        "self-refine": "1 Agent (Self)",
        "gen-critic": "2 Agents (Dyad)",
        "gen-critic-sum": "3 Agents (Triad)",
        "parallel-merge": "Parallel (Ensemble)",
        "divergence-keeper": "Intervention",
        "parallel": "Baseline (Parallel)",
        "cyclic": "Cyclic (Ring)",
        "debate": "Debate"
    }
    
    # We're using iteration 2 for comparison due to sample availability
    target_iter = 2
    if target_iter in df_sdi['iteration'].values:
        subset_iter = df_sdi[df_sdi['iteration'] == target_iter].copy()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=subset_iter,
            x='workflow',
            y='sdi',
            hue='model_family',
            ci='sd',  # Show standard deviation
            palette=PALETTE
        )
        plt.title(f"Team Size Effect on Diversity (Iteration {target_iter})", fontsize=14)
        plt.xlabel("Workflow Type", fontsize=12)
        plt.ylabel("Semantic Diversity Index (SDI)", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Model Family")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "fig3_team_size.png"), dpi=300)
        plt.close()
        print("Figure 3 generated.")
    else:
        print(f"Skipping Figure 3 (No data for iteration {target_iter}).")

    # Figure 4: Distinct-2 Trajectory
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_sdi, 
        x="iteration", 
        y="distinct_2", 
        hue="workflow", 
        style="model_family",
        markers=True, 
        dashes=False,
        linewidth=2.5
    )
    plt.title("Lexical Diversity (Distinct-2) Across Iterations", fontsize=14)
    plt.xlabel("Iteration Round", fontsize=12)
    plt.ylabel("Distinct-2 Score", fontsize=12)
    plt.legend(title="Workflow", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "fig4_distinct2_trajectory.png"), dpi=300)
    plt.close()
    print("Figure 4 (Distinct-2) generated.")

    # Figure 5: JSD Trajectory
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_sdi, 
        x="iteration", 
        y="jsd", 
        hue="workflow", 
        style="model_family",
        markers=True, 
        dashes=False,
        linewidth=2.5
    )
    plt.title("Distributional Diversity (JSD) Across Iterations", fontsize=14)
    plt.xlabel("Iteration Round", fontsize=12)
    plt.ylabel("Jensen-Shannon Divergence", fontsize=12)
    plt.legend(title="Workflow", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "fig5_jsd_trajectory.png"), dpi=300)
    plt.close()
    print("Figure 5 (JSD) generated.")

    # Figure 6: Semantic Drift Trajectory
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_sdi, 
        x="iteration", 
        y="drift", 
        hue="workflow", 
        style="model_family",
        markers=True, 
        dashes=False,
        linewidth=2.5
    )
    plt.title("Semantic Drift from Initial State", fontsize=14)
    plt.xlabel("Iteration Round", fontsize=12)
    plt.ylabel("Jensen-Shannon Divergence (Drift)", fontsize=12)
    plt.legend(title="Workflow", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "fig6_drift_trajectory.png"), dpi=300)
    plt.close()
    print("Figure 6 (Drift) generated.")

    print("Analysis complete.")

if __name__ == "__main__":
    main()
