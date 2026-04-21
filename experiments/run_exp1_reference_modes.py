#!/usr/bin/env python3
"""
Exp1 with 20 diverse tasks (4 categories x 5 tasks each)
Compares group-history vs individual-history reference modes.

Usage:
    # Run individual-history condition
    python run_exp1_20tasks.py --reference-mode individual --outdir results_exp1_20tasks_individual

    # Run group-history condition
    python run_exp1_20tasks.py --reference-mode all --outdir results_exp1_20tasks_all
"""

import os
import csv
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
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

# =========================
# 20 Tasks (4 categories x 5 each)
# =========================

def build_tasks() -> List[Tuple[str, str, str]]:
    """
    Returns list of (task_id, category, task_text).
    4 categories: idea, reasoning, summarization, creative_writing
    """
    tasks = []

    # Category 1: Idea Generation (open-ended, creative)
    idea_tasks = [
        "Generate 5 novel product ideas that help reduce food waste at home.",
        "Generate 5 new interaction concepts for supporting remote teamwork.",
        "Generate 5 services that use AI to support elderly people living alone.",
        "Generate 5 ideas for playful urban installations using light and sound.",
        "Generate 5 ideas for improving the experience of public transportation."
    ]
    for i, t in enumerate(idea_tasks):
        tasks.append((f"idea_{i}", "idea", t))

    # Category 2: Reasoning/Analysis (structured thinking)
    reasoning_tasks = [
        "Explain why traffic congestion occurs in large cities and propose 3 countermeasures.",
        "Analyze the trade-offs of remote work vs. in-person work for a software team.",
        "Explain the main causes of climate change and propose realistic mitigation steps.",
        "Compare subscription-based and one-time purchase business models.",
        "Analyze risks and benefits of using AI chatbots in customer support."
    ]
    for i, t in enumerate(reasoning_tasks):
        tasks.append((f"reasoning_{i}", "reasoning", t))

    # Category 3: Summarization (synthesis)
    summarization_tasks = [
        "Summarize the key challenges of AI ethics in autonomous driving.",
        "Summarize main usability issues in mobile banking applications.",
        "Summarize the advantages and disadvantages of online education.",
        "Summarize the key properties of human-centered design.",
        "Summarize typical barriers to adopting new technologies in organizations."
    ]
    for i, t in enumerate(summarization_tasks):
        tasks.append((f"summ_{i}", "summarization", t))

    # Category 4: Creative Writing (narrative)
    creative_tasks = [
        "Write a short story about a city where AI agents and humans co-create art.",
        "Write a dialogue between two AI agents arguing about creativity.",
        "Write a short story about a future classroom using AI tutors.",
        "Write a short story about a day in the life of an AI facilitator.",
        "Write a short story about a researcher studying AI homogenization."
    ]
    for i, t in enumerate(creative_tasks):
        tasks.append((f"creative_{i}", "creative_writing", t))

    return tasks

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
# Workflow: Gen-Critic-Sum (same as paper)
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

# =========================
# Experiment Runner
# =========================

class Exp1Runner:
    def __init__(
        self,
        reference_mode: str = "individual",
        n_candidates: int = 5,
        n_iterations: int = 5,
        outdir: str = "results_exp1_20tasks",
        seed: int = 42,
        num_workers: int = 8,
    ):
        self.reference_mode = reference_mode
        self.n_candidates = n_candidates
        self.n_iterations = n_iterations
        self.outdir = outdir
        self.seed = seed
        self.num_workers = num_workers

        self.tasks = build_tasks()
        self.chain_state: Dict[Tuple, str] = {}  # (task_id, candidate) -> output
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
            # Only see own previous output
            key = (task_id, candidate)
            if key in self.chain_state:
                return [self.chain_state[key]]
            return None
        else:  # "all" (group-history)
            # See all candidates' previous outputs
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

        # Update chain state
        with self._lock:
            self.chain_state[(task_id, candidate)] = output

        return output

    def run(self):
        self.log(f"Starting Exp1 with 20 tasks")
        self.log(f"  Reference mode: {self.reference_mode}")
        self.log(f"  Candidates: {self.n_candidates}")
        self.log(f"  Iterations: {self.n_iterations}")
        self.log(f"  Output: {self.outdir}")

        for task_id, category, task_text in self.tasks:
            self.log(f"\n=== Task: {task_id} ({category}) ===")
            self.chain_state.clear()  # Reset chain state for each task

            for iteration in range(self.n_iterations):
                self.log(f"  Iteration {iteration}/{self.n_iterations-1}")

                # Run all candidates in parallel
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

                # Compute and save embeddings for this iteration
                if outputs:
                    try:
                        embeddings = get_embeddings(outputs)
                        self.save_embeddings(task_id, iteration, embeddings)

                        # Compute SDI for logging
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

    parser = argparse.ArgumentParser(description="Run Exp1 with 20 tasks")
    parser.add_argument("--reference-mode", type=str, default="individual",
                        choices=["individual", "all"],
                        help="Reference mode: 'individual' or 'all' (group-history)")
    parser.add_argument("--outdir", type=str, default="results_exp1_20tasks",
                        help="Output directory")
    parser.add_argument("--candidates", type=int, default=5,
                        help="Number of candidates per iteration")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations (0 to n-1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")

    args = parser.parse_args()

    runner = Exp1Runner(
        reference_mode=args.reference_mode,
        n_candidates=args.candidates,
        n_iterations=args.iterations,
        outdir=args.outdir,
        seed=args.seed,
        num_workers=args.workers,
    )

    runner.run()
