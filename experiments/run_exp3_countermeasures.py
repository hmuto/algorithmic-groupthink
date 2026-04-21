#!/usr/bin/env python3
"""
Countermeasure Experiments for Algorithmic Groupthink

This script runs experiments testing two countermeasures:
1. Diversity Prompt: Explicit instruction to generate diverse responses
2. Adversarial Sampling: Select responses that maximize distance from existing outputs

Both experiments use group-history (all) mode as the baseline.
"""

import os
import csv
import json
import time
import pickle
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# =========================
# Prompts
# =========================

PROMPTS = {
    # Standard prompt (baseline - same as original experiment)
    "standard_system": "",
    "standard_user": (
        "Task: {task}\n"
        "Iteration: {iteration}\n"
        "Candidate: {candidate}\n\n"
        "{previous_context}"
        "Generate a comprehensive and high-quality response to the above task."
    ),

    # Diversity Prompt: Explicit instruction to be different
    "diversity_system": (
        "You are an AI assistant focused on generating DIVERSE and UNIQUE responses.\n"
        "Your goal is to produce outputs that are distinctly different from previous responses.\n"
        "Avoid common patterns, clichés, and generic solutions.\n"
        "Prioritize novelty and creative thinking over conventional approaches."
    ),
    "diversity_user": (
        "Task: {task}\n"
        "Iteration: {iteration}\n"
        "Candidate: {candidate}\n\n"
        "{previous_context}"
        "IMPORTANT: Your response MUST be substantially different from the previous outputs shown above.\n"
        "Generate a response that takes a unique angle, uses different examples, or proposes unconventional solutions.\n"
        "DO NOT simply rephrase or slightly modify what others have said. Be creative and distinct."
    ),
}

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
# Experiment Classes
# =========================

class CountermeasureExperiment:
    def __init__(
        self,
        exp_name: str,
        mode: str,  # "baseline", "diversity_prompt", "adversarial"
        outdir: str,
        tasks: List[str] = TASKS,
        iterations: int = 4,
        candidates: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        seed: int = 42,
        num_workers: int = 1,
        adversarial_candidates: int = 3,  # Number of candidates to generate for adversarial selection
    ):
        self.exp_name = exp_name
        self.mode = mode
        self.outdir = outdir
        self.tasks = tasks
        self.iterations = iterations
        self.candidates = candidates
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.num_workers = num_workers
        self.adversarial_candidates = adversarial_candidates

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
                    "timestamp", "exp_name", "mode", "workflow", "task_id", "task",
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

    def _generate_standard(self, task: str, iteration: int, candidate: int, previous_outputs: List[str]) -> str:
        """Standard generation (baseline)."""
        prev_context = self._format_previous_outputs(previous_outputs)
        content = PROMPTS["standard_user"].format(
            task=task, iteration=iteration, candidate=candidate, previous_context=prev_context
        )
        messages = [{"role": "user", "content": content}]
        text, _ = call_openai_chat(self.model, messages, self.temperature, self.seed)
        return text

    def _generate_diversity_prompt(self, task: str, iteration: int, candidate: int, previous_outputs: List[str]) -> str:
        """Generation with explicit diversity instruction."""
        prev_context = self._format_previous_outputs(previous_outputs)
        messages = [
            {"role": "system", "content": PROMPTS["diversity_system"]},
            {"role": "user", "content": PROMPTS["diversity_user"].format(
                task=task, iteration=iteration, candidate=candidate, previous_context=prev_context
            )}
        ]
        text, _ = call_openai_chat(self.model, messages, self.temperature, self.seed)
        return text

    def _generate_adversarial(self, task: str, iteration: int, candidate: int, previous_outputs: List[str]) -> str:
        """Adversarial sampling: generate multiple candidates and select the most distant."""
        prev_context = self._format_previous_outputs(previous_outputs)

        # Generate multiple candidates
        candidates_texts = []
        for i in range(self.adversarial_candidates):
            content = PROMPTS["standard_user"].format(
                task=task, iteration=iteration, candidate=candidate, previous_context=prev_context
            )
            messages = [{"role": "user", "content": content}]
            # Vary seed slightly for each candidate
            text, _ = call_openai_chat(self.model, messages, self.temperature, self.seed + i * 100)
            candidates_texts.append(text)

        # If no previous outputs, just return the first candidate
        if not previous_outputs:
            return candidates_texts[0]

        # Get embeddings for all previous outputs and candidates
        prev_embeddings = [get_embedding(p) for p in previous_outputs]
        cand_embeddings = [get_embedding(c) for c in candidates_texts]

        # Calculate minimum distance to any previous output for each candidate
        min_distances = []
        for cand_emb in cand_embeddings:
            distances = [np.linalg.norm(cand_emb - prev_emb) for prev_emb in prev_embeddings]
            min_dist = min(distances)  # Minimum distance to any previous output
            min_distances.append(min_dist)

        # Select the candidate with maximum minimum distance (most different from all previous)
        best_idx = np.argmax(min_distances)

        print(f"    [Adversarial] Min distances: {[f'{d:.3f}' for d in min_distances]}, selected idx={best_idx}")

        return candidates_texts[best_idx]

    def _run_single_job(self, task_id: int, task: str, iteration: int, candidate: int, previous_outputs: List[str]) -> str:
        """Run a single generation job."""
        print(f"  Running: task={task_id}, iter={iteration}, cand={candidate}, mode={self.mode}")

        if self.mode == "baseline":
            output = self._generate_standard(task, iteration, candidate, previous_outputs)
        elif self.mode == "diversity_prompt":
            output = self._generate_diversity_prompt(task, iteration, candidate, previous_outputs)
        elif self.mode == "adversarial":
            output = self._generate_adversarial(task, iteration, candidate, previous_outputs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Save to CSV
        output_safe = output.replace("\n", "\\n")
        self._save_row([
            datetime.now().isoformat(),
            self.exp_name,
            self.mode,
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
            "mode": self.mode,
            "task_id": task_id,
            "iteration": iteration,
            "candidate": candidate,
            "output_length": len(output)
        })

        return output

    def run(self):
        """Run the experiment."""
        print(f"\n{'='*60}")
        print(f"Starting Experiment: {self.exp_name}")
        print(f"Mode: {self.mode}")
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

                    # Gather previous outputs (group-history mode)
                    previous_outputs = []
                    if iteration > 0:
                        for c in range(self.candidates):
                            chain_key = (task_id, c)
                            if chain_key in self.chain_state:
                                previous_outputs.append(self.chain_state[chain_key])

                    # Generate
                    output = self._run_single_job(task_id, task, iteration, candidate, previous_outputs)

                    # Update state
                    chain_key = (task_id, candidate)
                    with self._lock:
                        self.chain_state[chain_key] = output

                    # Small delay to avoid rate limiting
                    time.sleep(0.1)

        print(f"\n{'='*60}")
        print(f"Experiment {self.exp_name} completed!")
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run countermeasure experiments")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "diversity_prompt", "adversarial", "all"],
                        help="Experiment mode")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--adversarial-candidates", type=int, default=3,
                        help="Number of candidates to generate for adversarial selection")
    args = parser.parse_args()

    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["baseline", "diversity_prompt", "adversarial"]
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        exp_name = f"exp3_{mode}"
        outdir = os.path.join(BASE_DIR, f"results_{exp_name}")

        exp = CountermeasureExperiment(
            exp_name=exp_name,
            mode=mode,
            outdir=outdir,
            iterations=args.iterations,
            candidates=args.candidates,
            adversarial_candidates=args.adversarial_candidates,
        )
        exp.run()

        # Compute embeddings after experiment
        compute_embeddings_and_cache(exp_name, outdir)

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()
