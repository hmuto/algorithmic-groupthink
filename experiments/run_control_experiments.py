#!/usr/bin/env python3
"""
Control Experiments for Algorithmic Groupthink

This script runs control experiments to rule out alternative explanations:

Control 1: Random Reference (Input Length Hypothesis)
    - Agents receive same-length context but from UNRELATED tasks
    - If input length causes collapse, diversity should still decrease
    - If shared CONTENT causes collapse, diversity should be preserved

Control 2: Independent Repeated (Natural Convergence Hypothesis)
    - Agents generate independently each iteration (no iteration, no reference)
    - Tests whether same model/prompt naturally produces similar outputs
    - SDI should remain stable if there's no natural convergence

Control 3: Shuffled History (Temporal Dependency)
    - Agents receive previous outputs but in shuffled order
    - Tests whether specific temporal sequence matters
"""

import os
import csv
import json
import time
import pickle
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple
import threading
import numpy as np

# =========================
# Configuration
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "embedding_cache")

TASKS = [
    "Generate 5 novel product ideas that help reduce food waste at home.",
    "Generate 5 new interaction concepts for supporting remote teamwork.",
    "Generate 5 services that use AI to support elderly people living alone.",
    "Generate 5 ideas for playful urban installations using light and sound.",
    "Generate 5 ideas for improving the experience of public transportation."
]

# Unrelated filler texts for Control 1 (Random Reference)
# These are generic texts that are unrelated to the creative tasks
FILLER_TEXTS = [
    """The history of coffee dates back to the 15th century in Yemen. Coffee beans are actually seeds from the Coffea plant. The two most common species are Arabica and Robusta. Coffee cultivation spread from Africa to the Middle East, then to Europe and the Americas.""",
    """The periodic table organizes chemical elements by atomic number. Dmitri Mendeleev created the first widely recognized version in 1869. Elements in the same column share similar chemical properties. There are currently 118 confirmed elements.""",
    """The Great Wall of China stretches over 13,000 miles. Construction began in the 7th century BC. The wall was built to protect against invasions from the north. Today it is one of the most visited tourist attractions in the world.""",
    """Photosynthesis is the process by which plants convert light energy into chemical energy. Chlorophyll in plant cells absorbs sunlight. Carbon dioxide and water are converted to glucose and oxygen. This process is essential for life on Earth.""",
    """The human brain contains approximately 86 billion neurons. These neurons communicate through electrical and chemical signals. The brain uses about 20% of the body's total energy. Different regions of the brain control different functions.""",
]

# =========================
# API Utilities
# =========================

def retry_with_backoff(func, retries=3, backoff_in_seconds=1):
    def wrapper(*args, **kwargs):
        x = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if x == retries:
                    raise e
                sleep_time = (backoff_in_seconds * 2 ** x + random.uniform(0, 1))
                time.sleep(sleep_time)
                x += 1
    return wrapper


def call_openai_chat(model: str, messages: List[Dict[str, str]], temperature: float = 1.0, seed: int = None) -> Tuple[str, Any]:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if seed is not None:
        kwargs["seed"] = seed

    @retry_with_backoff
    def _make_call():
        return client.chat.completions.create(**kwargs)

    resp = _make_call()
    text = resp.choices[0].message.content
    return text, resp


def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embedding for a text using OpenAI's embedding API."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    @retry_with_backoff
    def _make_call():
        return client.embeddings.create(input=text, model=model)

    resp = _make_call()
    return np.array(resp.data[0].embedding)


# =========================
# Control Experiment Classes
# =========================

class ControlExperiment:
    def __init__(
        self,
        exp_name: str,
        control_type: str,  # "random_reference", "independent", "shuffled"
        outdir: str,
        tasks: List[str] = TASKS,
        iterations: int = 4,
        candidates: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        seed: int = 42,
    ):
        self.exp_name = exp_name
        self.control_type = control_type
        self.outdir = outdir
        self.tasks = tasks
        self.iterations = iterations
        self.candidates = candidates
        self.model = model
        self.temperature = temperature
        self.seed = seed

        os.makedirs(outdir, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

        self.csvfile = os.path.join(outdir, "results.csv")
        self.logfile = os.path.join(outdir, "run_log.jsonl")

        self._lock = threading.Lock()
        self.chain_state: Dict[Tuple[int, int], str] = {}  # (task_id, candidate) -> latest_text
        self.done_index = set()

        self._load_done_index()
        self._init_csv()

    def _load_done_index(self):
        if not os.path.exists(self.csvfile):
            return
        try:
            with open(self.csvfile, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (int(row["task_id"]), int(row["iteration"]), int(row["candidate"]))
                    self.done_index.add(key)
                    chain_key = (int(row["task_id"]), int(row["candidate"]))
                    output = row["final_output"].replace("\\n", "\n")
                    self.chain_state[chain_key] = output
        except Exception as e:
            print(f"[WARN] Failed to load state: {e}")

    def _init_csv(self):
        if not os.path.exists(self.csvfile):
            with open(self.csvfile, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "exp_name", "control_type", "workflow", "task_id", "task",
                    "iteration", "candidate", "final_output"
                ])

    def _save_row(self, row: List[Any]):
        with self._lock:
            with open(self.csvfile, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def _log_json(self, record: Dict[str, Any]):
        with self._lock:
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _format_previous_outputs(self, previous_outputs: List[str]) -> str:
        if not previous_outputs:
            return ""
        if len(previous_outputs) == 1:
            return f"Previous Output:\n{previous_outputs[0]}\n\n"
        formatted = "\n\n".join(
            f"--- Previous Response {i+1} ---\n{text}"
            for i, text in enumerate(previous_outputs)
        )
        return f"Previous Outputs from All Candidates:\n{formatted}\n\n"

    def _generate(self, task: str, iteration: int, candidate: int, context: str) -> str:
        """Generate a response with given context."""
        content = (
            f"Task: {task}\n"
            f"Iteration: {iteration}\n"
            f"Candidate: {candidate}\n\n"
            f"{context}"
            "Generate a comprehensive and high-quality response to the above task."
        )
        messages = [{"role": "user", "content": content}]
        text, _ = call_openai_chat(self.model, messages, self.temperature, self.seed)
        return text

    def _get_context_for_control(self, task_id: int, iteration: int, candidate: int) -> str:
        """Get appropriate context based on control type."""

        if self.control_type == "random_reference":
            # Control 1: Random unrelated reference (same input length, different content)
            if iteration == 0:
                return ""
            # Use filler texts instead of actual previous outputs
            num_refs = self.candidates
            selected_fillers = random.sample(FILLER_TEXTS * 2, min(num_refs, len(FILLER_TEXTS)))
            return self._format_previous_outputs(selected_fillers)

        elif self.control_type == "independent":
            # Control 2: No reference at all (completely independent each iteration)
            return ""

        elif self.control_type == "shuffled":
            # Control 3: Shuffled history (same content, different order)
            if iteration == 0:
                return ""
            previous_outputs = []
            for c in range(self.candidates):
                chain_key = (task_id, c)
                if chain_key in self.chain_state:
                    previous_outputs.append(self.chain_state[chain_key])
            # Shuffle the order
            random.shuffle(previous_outputs)
            return self._format_previous_outputs(previous_outputs)

        else:
            raise ValueError(f"Unknown control type: {self.control_type}")

    def _run_single_job(self, task_id: int, task: str, iteration: int, candidate: int) -> str:
        """Run a single generation job."""
        print(f"  Running: task={task_id}, iter={iteration}, cand={candidate}, control={self.control_type}")

        context = self._get_context_for_control(task_id, iteration, candidate)
        output = self._generate(task, iteration, candidate, context)

        # Save to CSV
        output_safe = output.replace("\n", "\\n")
        self._save_row([
            datetime.now().isoformat(),
            self.exp_name,
            self.control_type,
            "self-refine",
            task_id,
            task,
            iteration,
            candidate,
            output_safe
        ])

        # Log
        self._log_json({
            "timestamp": datetime.now().isoformat(),
            "exp_name": self.exp_name,
            "control_type": self.control_type,
            "task_id": task_id,
            "iteration": iteration,
            "candidate": candidate,
            "output_length": len(output),
            "context_length": len(context)
        })

        return output

    def run(self):
        """Run the control experiment."""
        print(f"\n{'='*60}")
        print(f"Starting Control Experiment: {self.exp_name}")
        print(f"Control Type: {self.control_type}")
        print(f"Tasks: {len(self.tasks)}, Iterations: {self.iterations}, Candidates: {self.candidates}")
        print(f"{'='*60}\n")

        for iteration in range(self.iterations):
            print(f"\n--- Iteration {iteration} ---")

            for task_id, task in enumerate(self.tasks):
                for candidate in range(self.candidates):
                    key = (task_id, iteration, candidate)
                    if key in self.done_index:
                        print(f"  Skipping (done): task={task_id}, iter={iteration}, cand={candidate}")
                        continue

                    # Generate
                    output = self._run_single_job(task_id, task, iteration, candidate)

                    # Update state (needed for shuffled control)
                    chain_key = (task_id, candidate)
                    with self._lock:
                        self.chain_state[chain_key] = output

                    # Small delay to avoid rate limiting
                    time.sleep(0.1)

        print(f"\n{'='*60}")
        print(f"Control Experiment {self.exp_name} completed!")
        print(f"Results saved to: {self.outdir}")
        print(f"{'='*60}\n")


def compute_embeddings_and_cache(exp_name: str, results_dir: str):
    """Compute and cache embeddings for experiment results."""
    print(f"\nComputing embeddings for {exp_name}...")

    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] Results file not found: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by (task_id, iteration)
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = (int(row["task_id"]), int(row["iteration"]))
        groups[key].append(row)

    for (task_id, iteration), group_rows in groups.items():
        cache_key = f"{exp_name}_self-refine_{task_id}_{iteration}"
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

        if os.path.exists(cache_path):
            print(f"  Skipping (cached): {cache_key}")
            continue

        print(f"  Computing: {cache_key} ({len(group_rows)} samples)")

        embeddings = []
        for row in group_rows:
            text = row["final_output"].replace("\\n", "\n")
            emb = get_embedding(text)
            embeddings.append(emb)

        embeddings_array = np.array(embeddings)

        with open(cache_path, "wb") as f:
            pickle.dump(embeddings_array, f)

        time.sleep(0.1)  # Rate limiting

    print(f"  Embeddings cached for {exp_name}")


def analyze_control_results():
    """Analyze and compare control experiment results."""
    print("\n" + "="*70)
    print("CONTROL EXPERIMENT ANALYSIS")
    print("="*70)

    from collections import defaultdict

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

    experiments = {
        "Baseline (Group)": "exp1_all",
        "Individual": "exp1_individual",
        "Control: Random Ref": "control_random_reference",
        "Control: Independent": "control_independent",
        "Control: Shuffled": "control_shuffled",
    }

    results = {}
    for name, exp_name in experiments.items():
        sdi_by_iter = defaultdict(list)
        for task_id in range(5):
            for iteration in range(4):
                embeddings = load_cached_embeddings(exp_name, "self-refine", task_id, iteration)
                if embeddings is not None:
                    sdi = compute_sdi(embeddings)
                    sdi_by_iter[iteration].append(sdi)

        if sdi_by_iter:
            results[name] = dict(sdi_by_iter)
            print(f"\n{name}:")
            for it in sorted(sdi_by_iter.keys()):
                mean_sdi = np.mean(sdi_by_iter[it])
                std_sdi = np.std(sdi_by_iter[it])
                print(f"  Iteration {it}: SDI = {mean_sdi:.4f} ± {std_sdi:.4f}")

    # Calculate relative changes
    print("\n" + "-"*50)
    print("RELATIVE CHANGE (Iteration 0 → 3)")
    print("-"*50)

    for name, sdi_data in results.items():
        if 0 in sdi_data and 3 in sdi_data:
            baseline = np.mean(sdi_data[0])
            final = np.mean(sdi_data[3])
            change = (final - baseline) / baseline * 100
            print(f"  {name}: {change:+.1f}%")

    # Statistical comparisons
    print("\n" + "-"*50)
    print("STATISTICAL COMPARISONS (Final Iteration SDI)")
    print("-"*50)

    from scipy import stats

    baseline_final = results.get("Baseline (Group)", {}).get(3, [])

    for name in ["Control: Random Ref", "Control: Independent", "Control: Shuffled"]:
        if name in results and 3 in results[name]:
            control_final = results[name][3]
            if len(baseline_final) > 1 and len(control_final) > 1:
                t_stat, p_val = stats.ttest_ind(control_final, baseline_final, equal_var=False)
                print(f"\n  {name} vs Baseline (Group):")
                print(f"    t = {t_stat:.3f}, p = {p_val:.4f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run control experiments")
    parser.add_argument("--control", type=str, required=True,
                        choices=["random_reference", "independent", "shuffled", "all", "analyze"],
                        help="Control experiment type")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=5)
    args = parser.parse_args()

    if args.control == "analyze":
        analyze_control_results()
        return

    controls_to_run = []
    if args.control == "all":
        controls_to_run = ["random_reference", "independent", "shuffled"]
    else:
        controls_to_run = [args.control]

    for control_type in controls_to_run:
        exp_name = f"control_{control_type}"
        outdir = os.path.join(BASE_DIR, f"results_{exp_name}")

        exp = ControlExperiment(
            exp_name=exp_name,
            control_type=control_type,
            outdir=outdir,
            iterations=args.iterations,
            candidates=args.candidates,
        )
        exp.run()

        # Compute embeddings after experiment
        compute_embeddings_and_cache(exp_name, outdir)

    # Analyze all results
    analyze_control_results()

    print("\n" + "="*60)
    print("All control experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()
