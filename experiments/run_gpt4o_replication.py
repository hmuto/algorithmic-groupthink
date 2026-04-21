#!/usr/bin/env python3
"""
Reviewer Response: Experiment 1 Replication with Current Model (gpt-4o)
=======================================================================
Addresses reviewer concern #4: Model temporal limitation.

Replicates the core finding (group-history vs individual-history) using
gpt-4o (current generation, 2025+) instead of gpt-4o-mini (Dec 2024).

Small-scale: 5 tasks (1 per category), 5 candidates, 5 iterations.
gen-critic-sum workflow, temperature 1.0.
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from itertools import combinations
import numpy as np


from openai import OpenAI

# ===== Configuration =====
MODEL = "gpt-4o"  # Current generation model
TEMPERATURE = 1.0
N_CANDIDATES = 5
N_ITERATIONS = 5  # k=0 to k=4 (matching Experiment 1)

# 5 representative tasks (1 from each category)
TASKS = {
    "idea_0": ("idea", "Generate 5 novel product ideas that help reduce food waste at home."),
    "reasoning_0": ("reasoning", "Explain why traffic congestion occurs in large cities and propose 3 countermeasures."),
    "summ_0": ("summarization", "Summarize the key challenges of AI ethics in autonomous driving."),
    "creative_0": ("creative_writing", "Write a short story about a city where AI agents and humans co-create art."),
    "reasoning_1": ("reasoning", "Analyze the trade-offs of remote work vs. in-person work for a software team."),
}

# ===== Prompts (matching paper exactly) =====
STANDARD_SYSTEM = "You are a helpful AI assistant."

CRITIC_SYSTEM = (
    "You are a critical reviewer and editor. "
    "Your job is to provide constructive criticism to improve the quality of the draft. "
    "Focus on: 1) Clarity and Coherence, 2) Originality, 3) Completeness."
)

SUMMARIZER_SYSTEM = (
    "You are a summarization expert. "
    "Merge and compress the following content into one concise, well-structured response "
    "without losing key nuances."
)


def call_api(system_prompt, user_prompt, model=MODEL):
    """Call OpenAI API with retry."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def gen_critic_sum(task, previous_outputs, candidate_id, iteration):
    """Run Generator-Critic-Summarizer workflow."""
    # Generator
    if iteration == 0:
        gen_prompt = f"Task: {task}\n\nPlease generate a comprehensive and high-quality response to the above task."
    else:
        refs = "\n---\n".join(previous_outputs)
        gen_prompt = (
            f"Task: {task}\n\n"
            f"Previous outputs from candidates:\n{refs}\n\n"
            f"Based on these previous outputs, generate an improved response."
        )

    gen_output = call_api(STANDARD_SYSTEM, gen_prompt)

    # Critic
    critic_prompt = (
        f"Task: {task}\n\n"
        f"Original Draft:\n{gen_output}\n\n"
        f"Provide a critique of the draft, listing 3 specific areas for improvement. "
        f"Then, rewrite the draft incorporating your own feedback."
    )
    critic_output = call_api(CRITIC_SYSTEM, critic_prompt)

    # Summarizer
    sum_prompt = f"Revised content:\n{critic_output}"
    final_output = call_api(SUMMARIZER_SYSTEM, sum_prompt)

    return final_output


def compute_diversity(texts):
    """Compute embeddings and mean pairwise Euclidean distance."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embeddings = np.array([d.embedding for d in resp.data])
    dists = []
    for i, j in combinations(range(len(embeddings)), 2):
        dists.append(np.linalg.norm(embeddings[i] - embeddings[j]))
    return np.mean(dists), embeddings


def run_condition(condition_name, reference_mode, outdir):
    """Run individual or group history condition."""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "results.csv")
    diversity_path = os.path.join(outdir, "diversity.json")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'reference_mode', 'task_id', 'category', 'task',
                       'iteration', 'candidate', 'final_output'])

        outputs = {}  # outputs[task_id][iteration] = [cand0, cand1, ...]
        diversity_results = {}

        for task_id, (category, task_text) in sorted(TASKS.items()):
            outputs[task_id] = {}
            diversity_results[task_id] = []
            print(f"\n  [{condition_name}] Task: {task_id}")

            for iteration in range(N_ITERATIONS):
                outputs[task_id][iteration] = []

                for candidate in range(N_CANDIDATES):
                    # Build previous outputs based on reference mode
                    if iteration == 0:
                        previous = []
                    elif reference_mode == "individual":
                        # Only own previous output
                        prev_outputs = outputs[task_id].get(iteration - 1, [])
                        if candidate < len(prev_outputs):
                            previous = [prev_outputs[candidate]]
                        else:
                            previous = []
                    else:  # "all" / group-history
                        previous = outputs[task_id].get(iteration - 1, [])

                    result = gen_critic_sum(task_text, previous, candidate, iteration)
                    outputs[task_id][iteration].append(result)

                    writer.writerow([
                        datetime.now().isoformat(),
                        reference_mode,
                        task_id,
                        category,
                        task_text,
                        iteration,
                        candidate,
                        result.replace('\n', '\\n'),
                    ])
                    f.flush()
                    print(f"    iter={iteration} cand={candidate} [OK, len={len(result)}]")

                # Compute diversity
                if len(outputs[task_id][iteration]) == N_CANDIDATES:
                    div_val, _ = compute_diversity(outputs[task_id][iteration])
                    diversity_results[task_id].append(div_val)
                    print(f"    → Diversity at iter {iteration}: {div_val:.4f}")

    with open(diversity_path, 'w') as f:
        json.dump(diversity_results, f, indent=2)

    return diversity_results


def analyze_results(div_individual, div_group):
    """Compare individual vs group history."""
    from scipy import stats

    print("\n" + "=" * 70)
    print(f"ANALYSIS: Current Model Replication ({MODEL})")
    print("=" * 70)

    rel_ind = []
    rel_grp = []

    for task_id in sorted(TASKS.keys()):
        if task_id in div_individual and len(div_individual[task_id]) >= 2:
            b, f = div_individual[task_id][0], div_individual[task_id][-1]
            if b > 0:
                rel_ind.append((f - b) / b * 100)

        if task_id in div_group and len(div_group[task_id]) >= 2:
            b, f = div_group[task_id][0], div_group[task_id][-1]
            if b > 0:
                rel_grp.append((f - b) / b * 100)

    print(f"\n  Individual-history (n={len(rel_ind)}):")
    print(f"    Mean relative change: {np.mean(rel_ind):+.1f}% (std={np.std(rel_ind):.1f}%)")
    print(f"\n  Group-history (n={len(rel_grp)}):")
    print(f"    Mean relative change: {np.mean(rel_grp):+.1f}% (std={np.std(rel_grp):.1f}%)")

    # Final diversity comparison
    final_ind = [div_individual[t][-1] for t in sorted(TASKS.keys())
                 if t in div_individual and len(div_individual[t]) >= 2]
    final_grp = [div_group[t][-1] for t in sorted(TASKS.keys())
                 if t in div_group and len(div_group[t]) >= 2]

    if len(final_ind) > 1 and len(final_grp) > 1:
        t_stat, p_value = stats.ttest_ind(final_ind, final_grp, equal_var=False)
        pooled_std = np.sqrt((np.std(final_ind)**2 + np.std(final_grp)**2) / 2)
        cohens_d = (np.mean(final_ind) - np.mean(final_grp)) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Individual final: mean={np.mean(final_ind):.4f}")
        print(f"  Group final:      mean={np.mean(final_grp):.4f}")
        print(f"  Welch's t-test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")

    # Direction check
    ind_direction = "increase" if np.mean(rel_ind) > 0 else "decrease"
    grp_direction = "increase" if np.mean(rel_grp) > 0 else "decrease"
    consistent = ind_direction == "increase" and grp_direction == "decrease"

    print(f"\n  Direction consistent with paper findings: {'YES' if consistent else 'NO'}")
    print(f"    Individual: {ind_direction} ({np.mean(rel_ind):+.1f}%)")
    print(f"    Group: {grp_direction} ({np.mean(rel_grp):+.1f}%)")

    # Comparison with original gpt-4o-mini results
    print(f"\n  Comparison with original gpt-4o-mini results:")
    print(f"    gpt-4o-mini: Individual +10.3%, Group -5.9%")
    print(f"    {MODEL}:     Individual {np.mean(rel_ind):+.1f}%, Group {np.mean(rel_grp):+.1f}%")


def main():
    print("=" * 70)
    print(f"Reviewer Experiment: Current Model Replication ({MODEL})")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Tasks: {len(TASKS)}, Candidates: {N_CANDIDATES}, Iterations: {N_ITERATIONS}")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Run individual-history
    print("\n>>> Running Individual-History condition...")
    div_ind = run_condition(
        "individual",
        reference_mode="individual",
        outdir=os.path.join(base_dir, "results_reviewer_gpt4o_individual")
    )

    # Run group-history
    print("\n>>> Running Group-History condition...")
    div_grp = run_condition(
        "group",
        reference_mode="all",
        outdir=os.path.join(base_dir, "results_reviewer_gpt4o_all")
    )

    # Analyze
    analyze_results(div_ind, div_grp)


if __name__ == '__main__':
    main()
