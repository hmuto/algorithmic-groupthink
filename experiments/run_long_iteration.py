#!/usr/bin/env python3
"""
Long-iteration experiment to test whether individual-history mode
also shows convergence over extended iterations (k=0 to k=9).

This experiment tests the hypothesis that even without shared context,
agents may converge to their own outputs through self-reinforcement loops.
"""

import os
import csv
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pickle

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =========================
# Configuration
# =========================

OPENAI_MODEL_CHAT = "gpt-4o-mini"
OPENAI_MODEL_EMBED = "text-embedding-3-small"
TEMPERATURE = 1.0
TOP_P = 1.0

# 5 diverse tasks (1 from each category)
TASKS = [
    ("idea_0", "idea", "Generate 5 novel product ideas that help reduce food waste at home."),
    ("reasoning_0", "reasoning", "Explain why traffic congestion occurs in large cities and propose 3 countermeasures."),
    ("summ_0", "summarization", "Summarize the key challenges of AI ethics in autonomous driving."),
    ("creative_0", "creative_writing", "Write a short story about a city where AI agents and humans co-create art."),
    ("idea_4", "idea", "Generate 5 ideas for improving the experience of public transportation."),
]

# =========================
# OpenAI Client
# =========================

client = OpenAI()

def call_llm(messages: List[Dict], seed: int = 42) -> str:
    """Call OpenAI chat completion."""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL_CHAT,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=seed,
    )
    return resp.choices[0].message.content.strip()

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings from OpenAI."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    resp = client.embeddings.create(
        model=OPENAI_MODEL_EMBED,
        input=texts,
    )
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

# =========================
# Workflow: Gen-Critic-Sum
# =========================

PROMPTS = {
    "gen_initial": (
        "Task: {task}\n\n"
        "Please generate a comprehensive and high-quality response to the above task."
    ),
    "gen_refine": (
        "Task: {task}\n\n"
        "Previous outputs from candidates:\n{previous_outputs}\n\n"
        "Based on these previous outputs, generate an improved response."
    ),
    "critic_system": (
        "You are a critical reviewer and editor.\n"
        "Your job is to provide constructive criticism to improve the quality of the draft.\n"
        "Focus on: 1) Clarity and Coherence, 2) Originality, 3) Completeness."
    ),
    "critic_user": (
        "Task: {task}\n\n"
        "Original Draft:\n{draft}\n\n"
        "Provide a critique of the draft, listing 3 specific areas for improvement.\n"
        "Then, rewrite the draft incorporating your own feedback."
    ),
    "summarizer_system": (
        "You are a summarization expert.\n"
        "Merge and compress the following content into one concise, well-structured response "
        "without losing key nuances."
    ),
    "summarizer_user": "Revised content:\n{draft}",
}

def run_gen_critic_sum(task: str, previous_outputs: Optional[List[str]], seed: int) -> str:
    """Run Generator -> Critic -> Summarizer pipeline."""

    # Generator
    if previous_outputs:
        prev_text = "\n\n".join([f"Candidate {i+1}:\n{out}" for i, out in enumerate(previous_outputs)])
        gen_prompt = PROMPTS["gen_refine"].format(task=task, previous_outputs=prev_text)
    else:
        gen_prompt = PROMPTS["gen_initial"].format(task=task)

    gen_messages = [{"role": "user", "content": gen_prompt}]
    gen_output = call_llm(gen_messages, seed=seed)

    # Critic
    critic_messages = [
        {"role": "system", "content": PROMPTS["critic_system"]},
        {"role": "user", "content": PROMPTS["critic_user"].format(task=task, draft=gen_output)}
    ]
    critic_output = call_llm(critic_messages, seed=seed)

    # Summarizer
    sum_messages = [
        {"role": "system", "content": PROMPTS["summarizer_system"]},
        {"role": "user", "content": PROMPTS["summarizer_user"].format(draft=critic_output)}
    ]
    sum_output = call_llm(sum_messages, seed=seed)

    return sum_output


class LongIterationRunner:
    def __init__(
        self,
        reference_mode: str = "individual",
        n_candidates: int = 5,
        n_iterations: int = 10,
        outdir: str = "results_long_iteration",
        seed: int = 42,
        num_workers: int = 5,
    ):
        self.reference_mode = reference_mode
        self.n_candidates = n_candidates
        self.n_iterations = n_iterations
        self.outdir = outdir
        self.seed = seed
        self.num_workers = num_workers

        self.tasks = TASKS
        self.chain_state: Dict[Tuple, str] = {}
        self._lock = threading.Lock()

        os.makedirs(outdir, exist_ok=True)

        self.results_file = os.path.join(outdir, "results.csv")
        self.embeddings_dir = os.path.join(outdir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Initialize CSV
        with open(self.results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "reference_mode", "task_id", "category", "task",
                "iteration", "candidate", "final_output"
            ])

    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def save_result(self, task_id: str, category: str, task: str,
                    iteration: int, candidate: int, output: str):
        with self._lock:
            with open(self.results_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.reference_mode,
                    task_id,
                    category,
                    task,
                    iteration,
                    candidate,
                    output
                ])

    def save_embeddings(self, task_id: str, iteration: int, embeddings: np.ndarray):
        path = os.path.join(self.embeddings_dir, f"{task_id}_iter{iteration}.pkl")
        with open(path, "wb") as f:
            pickle.dump(embeddings, f)

    def get_previous_outputs(self, task_id: str, candidate: int) -> Optional[List[str]]:
        """Get previous outputs based on reference mode."""
        if self.reference_mode == "individual":
            key = (task_id, candidate)
            if key in self.chain_state:
                return [self.chain_state[key]]
            return None
        else:  # "all" (group-history)
            outputs = []
            for c in range(self.n_candidates):
                key = (task_id, c)
                if key in self.chain_state:
                    outputs.append(self.chain_state[key])
            return outputs if outputs else None

    def run_single_job(self, task_id: str, category: str, task: str,
                       iteration: int, candidate: int) -> str:
        """Run a single generation job."""
        prev_outputs = self.get_previous_outputs(task_id, candidate) if iteration > 0 else None
        seed = self.seed + hash((task_id, iteration, candidate)) % 10000

        output = run_gen_critic_sum(task, prev_outputs, seed)

        with self._lock:
            self.chain_state[(task_id, candidate)] = output

        return output

    def run(self):
        self.log(f"Starting Long Iteration Experiment")
        self.log(f"  Reference mode: {self.reference_mode}")
        self.log(f"  Candidates: {self.n_candidates}")
        self.log(f"  Iterations: {self.n_iterations}")
        self.log(f"  Tasks: {len(self.tasks)}")
        self.log(f"  Output: {self.outdir}")

        for task_id, category, task_text in self.tasks:
            self.log(f"\n=== Task: {task_id} ({category}) ===")
            self.chain_state.clear()

            for iteration in range(self.n_iterations):
                self.log(f"  Iteration {iteration}/{self.n_iterations-1}")

                jobs = []
                for candidate in range(self.n_candidates):
                    jobs.append((task_id, category, task_text, iteration, candidate))

                outputs = []
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = {
                        executor.submit(self.run_single_job, *job): job
                        for job in jobs
                    }

                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            output = future.result()
                            outputs.append(output)
                            self.save_result(
                                task_id=job[0],
                                category=job[1],
                                task=job[2],
                                iteration=job[3],
                                candidate=job[4],
                                output=output
                            )
                        except Exception as e:
                            self.log(f"    [ERROR] Job failed: {e}")

                if outputs:
                    try:
                        embeddings = get_embeddings(outputs)
                        self.save_embeddings(task_id, iteration, embeddings)

                        if len(outputs) > 1:
                            from scipy.spatial.distance import pdist
                            distances = pdist(embeddings)
                            sdi = np.mean(distances)
                            self.log(f"    SDI: {sdi:.4f}")
                    except Exception as e:
                        self.log(f"    [WARN] Failed to compute embeddings: {e}")

        self.log(f"\n=== Experiment complete ===")
        self.log(f"Results saved to: {self.results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Long Iteration Experiment")
    parser.add_argument("--reference-mode", type=str, default="individual",
                        choices=["individual", "all"],
                        help="Reference mode: 'individual' or 'all' (group-history)")
    parser.add_argument("--outdir", type=str, default="results_long_iteration",
                        help="Output directory")
    parser.add_argument("--candidates", type=int, default=5,
                        help="Number of candidates per iteration")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations (0 to n-1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel workers")

    args = parser.parse_args()

    runner = LongIterationRunner(
        reference_mode=args.reference_mode,
        n_candidates=args.candidates,
        n_iterations=args.iterations,
        outdir=args.outdir,
        seed=args.seed,
        num_workers=args.workers,
    )

    runner.run()
