#!/usr/bin/env python3
"""
LLM-as-Judge Quality Evaluation

Evaluates the quality of generated outputs using GPT-4o-mini as a judge.
Compares quality across conditions to assess the quality-diversity tradeoff.
"""

import os
import csv
import json
import time
import random
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVALUATION_PROMPT = """You are an expert evaluator assessing the quality of creative ideation responses.

Task: {task}

Response to evaluate:
{response}

Please rate this response on the following criteria (1-5 scale):

1. **Creativity** (1-5): How novel and innovative are the ideas? Do they go beyond obvious solutions?
2. **Relevance** (1-5): How well do the ideas address the given task?
3. **Practicality** (1-5): How feasible are the ideas to implement?
4. **Completeness** (1-5): How thorough and well-developed are the ideas?
5. **Clarity** (1-5): How clear and well-articulated is the response?

Return your evaluation as a JSON object with the following format:
{{
    "creativity": <score>,
    "relevance": <score>,
    "practicality": <score>,
    "completeness": <score>,
    "clarity": <score>,
    "overall": <average of all scores>,
    "brief_rationale": "<1-2 sentence explanation>"
}}

Return ONLY the JSON object, no other text."""


def load_results(results_dir: str) -> List[Dict]:
    """Load results from CSV file."""
    csv_path = os.path.join(results_dir, "results.csv")
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["task_id"] = int(row["task_id"])
            row["iteration"] = int(row["iteration"])
            row["candidate"] = int(row["candidate"])
            row["final_output"] = row["final_output"].replace("\\n", "\n")
            results.append(row)
    return results


def get_final_iteration_samples(results: List[Dict], max_iteration: int = None) -> List[Dict]:
    """Get samples from the final iteration only."""
    if max_iteration is None:
        max_iteration = max(r["iteration"] for r in results)

    return [r for r in results if r["iteration"] == max_iteration]


def call_openai_for_evaluation(task: str, response: str) -> Dict:
    """Call OpenAI API to evaluate a response."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    prompt = EVALUATION_PROMPT.format(task=task, response=response)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic for consistency
            max_tokens=500
        )

        content = resp.choices[0].message.content.strip()

        # Parse JSON response
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        evaluation = json.loads(content)
        return evaluation

    except Exception as e:
        print(f"    [ERROR] Evaluation failed: {e}")
        return None


def evaluate_condition(results_dir: str, condition_name: str, sample_size: int = 10) -> List[Dict]:
    """Evaluate a sample of outputs from a condition."""
    print(f"\nEvaluating: {condition_name}")

    results = load_results(results_dir)
    final_samples = get_final_iteration_samples(results)

    # Random sample for evaluation
    if len(final_samples) > sample_size:
        samples = random.sample(final_samples, sample_size)
    else:
        samples = final_samples

    evaluations = []
    for i, sample in enumerate(samples):
        print(f"  Evaluating sample {i+1}/{len(samples)}...")

        eval_result = call_openai_for_evaluation(
            task=sample["task"],
            response=sample["final_output"]
        )

        if eval_result:
            eval_result["condition"] = condition_name
            eval_result["task_id"] = sample["task_id"]
            eval_result["candidate"] = sample["candidate"]
            evaluations.append(eval_result)

        time.sleep(0.5)  # Rate limiting

    return evaluations


def analyze_quality_results(all_evaluations: Dict[str, List[Dict]]):
    """Analyze and compare quality results across conditions."""
    print("\n" + "="*70)
    print("QUALITY EVALUATION RESULTS")
    print("="*70)

    metrics = ["creativity", "relevance", "practicality", "completeness", "clarity", "overall"]

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"{'Condition':<25} {'Creativity':<12} {'Relevance':<12} {'Overall':<12} {'n':<5}")
    print("-" * 70)

    condition_stats = {}
    for condition, evals in all_evaluations.items():
        if not evals:
            continue

        stats_dict = {}
        for metric in metrics:
            values = [e[metric] for e in evals if metric in e]
            if values:
                stats_dict[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "n": len(values),
                    "values": values
                }

        condition_stats[condition] = stats_dict

        creativity = stats_dict.get("creativity", {})
        relevance = stats_dict.get("relevance", {})
        overall = stats_dict.get("overall", {})

        print(f"{condition:<25} "
              f"{creativity.get('mean', 0):.2f}±{creativity.get('std', 0):.2f}  "
              f"{relevance.get('mean', 0):.2f}±{relevance.get('std', 0):.2f}  "
              f"{overall.get('mean', 0):.2f}±{overall.get('std', 0):.2f}  "
              f"{creativity.get('n', 0):<5}")

    # Statistical comparisons
    print("\n--- Statistical Comparisons (Overall Quality) ---")

    conditions = list(condition_stats.keys())
    if len(conditions) >= 2:
        # Compare each condition to baseline if available
        baseline = None
        for c in conditions:
            if "Baseline" in c or "baseline" in c:
                baseline = c
                break

        if baseline is None:
            baseline = conditions[0]

        baseline_values = condition_stats[baseline].get("overall", {}).get("values", [])

        for condition in conditions:
            if condition == baseline:
                continue

            other_values = condition_stats[condition].get("overall", {}).get("values", [])

            if len(baseline_values) > 1 and len(other_values) > 1:
                t_stat, p_val = stats.ttest_ind(other_values, baseline_values, equal_var=False)

                mean_diff = np.mean(other_values) - np.mean(baseline_values)

                sig = ""
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"

                print(f"  {condition} vs {baseline}:")
                print(f"    Mean diff: {mean_diff:+.2f}, t = {t_stat:.2f}, p = {p_val:.3f}{sig}")

    return condition_stats


def main():
    print("="*70)
    print("LLM-as-Judge Quality Evaluation")
    print("="*70)

    # Conditions to evaluate
    conditions = {
        "Baseline (Group)": os.path.join(BASE_DIR, "results_exp1_all"),
        "Individual": os.path.join(BASE_DIR, "results_exp1_individual"),
        "Diversity Prompt": os.path.join(BASE_DIR, "results_exp3_diversity_prompt"),
    }

    # Evaluate each condition
    all_evaluations = {}
    sample_size = 15  # Evaluate 15 samples per condition

    for condition_name, results_dir in conditions.items():
        if os.path.exists(os.path.join(results_dir, "results.csv")):
            evals = evaluate_condition(results_dir, condition_name, sample_size)
            all_evaluations[condition_name] = evals
        else:
            print(f"  [SKIP] {condition_name}: Results not found")

    # Analyze results
    stats = analyze_quality_results(all_evaluations)

    # Save results
    output_path = os.path.join(BASE_DIR, "quality_evaluation_results.json")
    with open(output_path, "w") as f:
        # Convert numpy types to native Python types
        serializable = {}
        for cond, evals in all_evaluations.items():
            serializable[cond] = evals

        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
