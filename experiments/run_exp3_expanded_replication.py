#!/usr/bin/env python3
"""
Reviewer Response: Experiment 3 Expanded (n=20 tasks)
=====================================================
Addresses reviewer concern #2: Sample size expansion for Diversity Prompt.

Runs two conditions under group-history mode with gen-critic-sum workflow:
  1. Baseline (standard prompts)
  2. Diversity Prompt (explicit diversity instructions)

Uses the same 20 tasks as Experiment 1 for direct comparability.
Temperature = 1.0 (matching paper's main experiments).
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from itertools import combinations
import numpy as np

# Ensure we can import openai

from openai import OpenAI

# ===== Configuration =====
MODEL = "gpt-4o-mini"
TEMPERATURE = 1.0
N_CANDIDATES = 5
N_ITERATIONS = 4  # k=0 to k=3 (matching Experiment 3)
N_WORKERS = 1  # Sequential for reliability

# 20 tasks from the paper (same as Experiment 1)
TASKS = {
    "idea_0": ("idea", "Generate 5 novel product ideas that help reduce food waste at home."),
    "idea_1": ("idea", "Generate 5 new interaction concepts for supporting remote teamwork."),
    "idea_2": ("idea", "Generate 5 services that use AI to support elderly people living alone."),
    "idea_3": ("idea", "Generate 5 ideas for playful urban installations using light and sound."),
    "idea_4": ("idea", "Generate 5 ideas for improving the experience of public transportation."),
    "reasoning_0": ("reasoning", "Explain why traffic congestion occurs in large cities and propose 3 countermeasures."),
    "reasoning_1": ("reasoning", "Analyze the trade-offs of remote work vs. in-person work for a software team."),
    "reasoning_2": ("reasoning", "Explain the main causes of climate change and propose realistic mitigation steps."),
    "reasoning_3": ("reasoning", "Compare subscription-based and one-time purchase business models."),
    "reasoning_4": ("reasoning", "Analyze risks and benefits of using AI chatbots in customer support."),
    "summ_0": ("summarization", "Summarize the key challenges of AI ethics in autonomous driving."),
    "summ_1": ("summarization", "Summarize main usability issues in mobile banking applications."),
    "summ_2": ("summarization", "Summarize the advantages and disadvantages of online education."),
    "summ_3": ("summarization", "Summarize the key properties of human-centered design."),
    "summ_4": ("summarization", "Summarize typical barriers to adopting new technologies in organizations."),
    "creative_0": ("creative_writing", "Write a short story about a city where AI agents and humans co-create art."),
    "creative_1": ("creative_writing", "Write a dialogue between two AI agents arguing about creativity."),
    "creative_2": ("creative_writing", "Write a short story about a future classroom using AI tutors."),
    "creative_3": ("creative_writing", "Write a short story about a day in the life of an AI facilitator."),
    "creative_4": ("creative_writing", "Write a short story about a researcher studying AI homogenization."),
}

# ===== Prompts =====
STANDARD_SYSTEM = "You are a helpful AI assistant."

DIVERSITY_SYSTEM = (
    "You are an AI assistant focused on generating DIVERSE and UNIQUE responses. "
    "Your goal is to produce outputs that are distinctly different from previous responses. "
    "Avoid common patterns, clichés, and generic solutions. "
    "Prioritize novelty and creative thinking over conventional approaches."
)

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


def gen_critic_sum(task, previous_outputs, candidate_id, iteration, use_diversity_prompt=False):
    """Run Generator-Critic-Summarizer workflow."""
    system = DIVERSITY_SYSTEM if use_diversity_prompt else STANDARD_SYSTEM

    # Generator
    if iteration == 0:
        gen_prompt = f"Task: {task}\n\nPlease generate a comprehensive and high-quality response to the above task."
    else:
        refs = "\n---\n".join(previous_outputs)
        if use_diversity_prompt:
            gen_prompt = (
                f"Task: {task}\n\n"
                f"Previous outputs from candidates:\n{refs}\n\n"
                f"Based on these previous outputs, generate an improved response. "
                f"Your response MUST be substantially different from the previous outputs shown above. "
                f"Generate a response that takes a unique angle, uses different examples, or proposes unconventional solutions."
            )
        else:
            gen_prompt = (
                f"Task: {task}\n\n"
                f"Previous outputs from candidates:\n{refs}\n\n"
                f"Based on these previous outputs, generate an improved response."
            )

    gen_output = call_api(system, gen_prompt)

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
    n = len(embeddings)
    dists = []
    for i, j in combinations(range(n), 2):
        dists.append(np.linalg.norm(embeddings[i] - embeddings[j]))
    return np.mean(dists), embeddings


def run_condition(condition_name, use_diversity_prompt, outdir):
    """Run a full experimental condition."""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "results.csv")
    diversity_path = os.path.join(outdir, "diversity.json")

    # Check for resume
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row['task_id'], int(row['iteration']), int(row['candidate'])))

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not completed:
            writer.writerow(['timestamp', 'condition', 'task_id', 'category', 'task',
                           'iteration', 'candidate', 'final_output'])

        # Store outputs for reference passing
        # outputs[task_id][iteration] = [candidate_0_output, candidate_1_output, ...]
        outputs = {}
        diversity_results = {}

        for task_id, (category, task_text) in sorted(TASKS.items()):
            outputs[task_id] = {}
            diversity_results[task_id] = []
            print(f"\n  [{condition_name}] Task: {task_id}")

            for iteration in range(N_ITERATIONS):
                outputs[task_id][iteration] = []

                for candidate in range(N_CANDIDATES):
                    if (task_id, iteration, candidate) in completed:
                        print(f"    iter={iteration} cand={candidate} [SKIP - already done]")
                        # Need to load from CSV for reference
                        continue

                    # Build previous outputs (group-history: all candidates from previous iteration)
                    if iteration == 0:
                        previous = []
                    else:
                        previous = outputs[task_id].get(iteration - 1, [])

                    result = gen_critic_sum(
                        task_text, previous, candidate, iteration,
                        use_diversity_prompt=use_diversity_prompt
                    )
                    outputs[task_id][iteration].append(result)

                    writer.writerow([
                        datetime.now().isoformat(),
                        condition_name,
                        task_id,
                        category,
                        task_text,
                        iteration,
                        candidate,
                        result.replace('\n', '\\n'),
                    ])
                    f.flush()
                    print(f"    iter={iteration} cand={candidate} [OK, len={len(result)}]")

                # Compute diversity for this iteration
                if len(outputs[task_id][iteration]) == N_CANDIDATES:
                    div_val, _ = compute_diversity(outputs[task_id][iteration])
                    diversity_results[task_id].append(div_val)
                    print(f"    → Diversity at iter {iteration}: {div_val:.4f}")

    # Save diversity results
    with open(diversity_path, 'w') as f:
        json.dump(diversity_results, f, indent=2)

    return diversity_results


def analyze_results(div_baseline, div_diversity):
    """Compare baseline and diversity prompt conditions."""
    from scipy import stats

    print("\n" + "=" * 70)
    print("ANALYSIS: Expanded Experiment 3 (n=20 tasks)")
    print("=" * 70)

    # Relative diversity change per task
    rel_baseline = []
    rel_diversity = []

    for task_id in sorted(TASKS.keys()):
        if task_id in div_baseline and len(div_baseline[task_id]) >= 2:
            base = div_baseline[task_id][0]
            final = div_baseline[task_id][-1]
            if base > 0:
                rel_baseline.append((final - base) / base * 100)

        if task_id in div_diversity and len(div_diversity[task_id]) >= 2:
            base = div_diversity[task_id][0]
            final = div_diversity[task_id][-1]
            if base > 0:
                rel_diversity.append((final - base) / base * 100)

    print(f"\n  Baseline (n={len(rel_baseline)}):")
    print(f"    Mean relative change: {np.mean(rel_baseline):+.1f}% (std={np.std(rel_baseline):.1f}%)")
    print(f"\n  Diversity Prompt (n={len(rel_diversity)}):")
    print(f"    Mean relative change: {np.mean(rel_diversity):+.1f}% (std={np.std(rel_diversity):.1f}%)")

    # Statistical test
    if len(rel_baseline) > 1 and len(rel_diversity) > 1:
        t_stat, p_value = stats.ttest_ind(rel_diversity, rel_baseline, equal_var=False)
        pooled_std = np.sqrt((np.std(rel_diversity)**2 + np.std(rel_baseline)**2) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(rel_diversity) - np.mean(rel_baseline)) / pooled_std
        else:
            cohens_d = 0

        print(f"\n  Welch's t-test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")

    # Per-category breakdown
    print("\n  Per-category breakdown:")
    categories = {}
    for task_id in sorted(TASKS.keys()):
        cat = TASKS[task_id][0]
        if cat not in categories:
            categories[cat] = {'baseline': [], 'diversity': []}

        if task_id in div_baseline and len(div_baseline[task_id]) >= 2:
            b, f = div_baseline[task_id][0], div_baseline[task_id][-1]
            if b > 0:
                categories[cat]['baseline'].append((f - b) / b * 100)

        if task_id in div_diversity and len(div_diversity[task_id]) >= 2:
            b, f = div_diversity[task_id][0], div_diversity[task_id][-1]
            if b > 0:
                categories[cat]['diversity'].append((f - b) / b * 100)

    for cat in sorted(categories.keys()):
        bl = categories[cat]['baseline']
        dv = categories[cat]['diversity']
        print(f"    {cat}: Baseline {np.mean(bl):+.1f}%, Diversity {np.mean(dv):+.1f}%")


def main():
    print("=" * 70)
    print("Reviewer Experiment: Expanded Countermeasure Test")
    print(f"Model: {MODEL}, Temperature: {TEMPERATURE}")
    print(f"Tasks: {len(TASKS)}, Candidates: {N_CANDIDATES}, Iterations: {N_ITERATIONS}")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Run baseline
    print("\n>>> Running Baseline condition...")
    div_baseline = run_condition(
        "baseline",
        use_diversity_prompt=False,
        outdir=os.path.join(base_dir, "results_reviewer_exp3_baseline")
    )

    # Run diversity prompt
    print("\n>>> Running Diversity Prompt condition...")
    div_diversity = run_condition(
        "diversity_prompt",
        use_diversity_prompt=True,
        outdir=os.path.join(base_dir, "results_reviewer_exp3_diversity")
    )

    # Analyze
    analyze_results(div_baseline, div_diversity)


if __name__ == '__main__':
    main()
