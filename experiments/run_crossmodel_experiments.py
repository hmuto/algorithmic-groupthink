#!/usr/bin/env python3
"""
Multi-Model Experiments for Algorithmic Groupthink

This script runs the core experiment (Group vs Individual) across multiple LLM providers
to test the generalizability of Algorithmic Groupthink phenomenon.

Supported models:
- OpenAI: gpt-4o-mini
- Anthropic: claude-3-5-haiku-latest
- Google: gemini-2.0-flash-exp
"""

import os
import csv

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
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

# Model configurations
MODEL_CONFIGS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "temperature": 1.0,
    },
    "claude-3-5-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-5-haiku-latest",
        "temperature": 1.0,
    },
    "gemini-2.0-flash": {
        "provider": "google",
        "model_id": "gemini-2.0-flash-exp",
        "temperature": 1.0,
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "model_id": "gemini-2.5-flash",
        "temperature": 1.0,
    },
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
                print(f"    [Retry {x+1}/{retries}] Error: {e}, waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                x += 1
    return wrapper


def call_openai(model: str, messages: List[Dict[str, str]], temperature: float = 1.0) -> str:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    @retry_with_backoff
    def _make_call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    resp = _make_call()
    return resp.choices[0].message.content


def call_anthropic(model: str, messages: List[Dict[str, str]], temperature: float = 1.0) -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    # Convert messages format
    system_msg = None
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            user_messages.append(msg)

    @retry_with_backoff
    def _make_call():
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "messages": user_messages,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg
        return client.messages.create(**kwargs)

    resp = _make_call()
    return resp.content[0].text


def call_google(model: str, messages: List[Dict[str, str]], temperature: float = 1.0) -> str:
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=api_key)

    # Convert messages to Google format
    system_instruction = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            contents.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            contents.append({"role": "model", "parts": [msg["content"]]})

    @retry_with_backoff
    def _make_call():
        model_obj = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )
        response = model_obj.generate_content(contents)
        return response

    resp = _make_call()
    return resp.text


def call_llm(provider: str, model: str, messages: List[Dict[str, str]], temperature: float = 1.0) -> str:
    """Unified LLM call interface."""
    if provider == "openai":
        return call_openai(model, messages, temperature)
    elif provider == "anthropic":
        return call_anthropic(model, messages, temperature)
    elif provider == "google":
        return call_google(model, messages, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embedding using OpenAI's embedding API (used for all models)."""
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
# Experiment Class
# =========================

class MultiModelExperiment:
    def __init__(
        self,
        model_name: str,
        reference_mode: str,  # "individual" or "all"
        outdir: str,
        tasks: List[str] = TASKS,
        iterations: int = 4,
        candidates: int = 5,
    ):
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        self.reference_mode = reference_mode
        self.outdir = outdir
        self.tasks = tasks
        self.iterations = iterations
        self.candidates = candidates

        os.makedirs(outdir, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

        self.csvfile = os.path.join(outdir, "results.csv")
        self.logfile = os.path.join(outdir, "run_log.jsonl")

        self._lock = threading.Lock()
        self.chain_state: Dict[Tuple[int, int], str] = {}
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
                    "timestamp", "model_name", "provider", "reference_mode",
                    "task_id", "task", "iteration", "candidate", "final_output"
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

    def _get_previous_outputs(self, task_id: int, iteration: int, candidate: int) -> List[str]:
        """Get previous outputs based on reference mode."""
        if iteration == 0:
            return []

        if self.reference_mode == "individual":
            # Only own previous output
            chain_key = (task_id, candidate)
            if chain_key in self.chain_state:
                return [self.chain_state[chain_key]]
            return []

        elif self.reference_mode == "all":
            # All candidates' previous outputs
            outputs = []
            for c in range(self.candidates):
                chain_key = (task_id, c)
                if chain_key in self.chain_state:
                    outputs.append(self.chain_state[chain_key])
            return outputs

        else:
            raise ValueError(f"Unknown reference mode: {self.reference_mode}")

    def _generate(self, task: str, iteration: int, candidate: int, previous_outputs: List[str]) -> str:
        prev_context = self._format_previous_outputs(previous_outputs)
        content = (
            f"Task: {task}\n"
            f"Iteration: {iteration}\n"
            f"Candidate: {candidate}\n\n"
            f"{prev_context}"
            "Generate a comprehensive and high-quality response to the above task."
        )
        messages = [{"role": "user", "content": content}]

        return call_llm(
            provider=self.model_config["provider"],
            model=self.model_config["model_id"],
            messages=messages,
            temperature=self.model_config["temperature"],
        )

    def _run_single_job(self, task_id: int, task: str, iteration: int, candidate: int) -> str:
        print(f"  [{self.model_name}] task={task_id}, iter={iteration}, cand={candidate}")

        previous_outputs = self._get_previous_outputs(task_id, iteration, candidate)
        output = self._generate(task, iteration, candidate, previous_outputs)

        # Save to CSV
        output_safe = output.replace("\n", "\\n")
        self._save_row([
            datetime.now().isoformat(),
            self.model_name,
            self.model_config["provider"],
            self.reference_mode,
            task_id,
            task,
            iteration,
            candidate,
            output_safe
        ])

        # Log
        self._log_json({
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "provider": self.model_config["provider"],
            "reference_mode": self.reference_mode,
            "task_id": task_id,
            "iteration": iteration,
            "candidate": candidate,
            "output_length": len(output)
        })

        return output

    def run(self):
        print(f"\n{'='*60}")
        print(f"Multi-Model Experiment: {self.model_name}")
        print(f"Reference Mode: {self.reference_mode}")
        print(f"Provider: {self.model_config['provider']}")
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

                    output = self._run_single_job(task_id, task, iteration, candidate)

                    # Update state
                    chain_key = (task_id, candidate)
                    with self._lock:
                        self.chain_state[chain_key] = output

                    time.sleep(0.2)  # Rate limiting

        print(f"\n{'='*60}")
        print(f"Experiment completed: {self.model_name} ({self.reference_mode})")
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

        time.sleep(0.1)

    print(f"  Embeddings cached for {exp_name}")


def analyze_results():
    """Analyze and compare results across models."""
    print("\n" + "="*70)
    print("CROSS-MODEL ANALYSIS")
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

    # Define experiments to analyze
    experiments = {}
    for model_name in MODEL_CONFIGS.keys():
        for mode in ["individual", "all"]:
            exp_name = f"multimodel_{model_name}_{mode}"
            experiments[f"{model_name} ({mode})"] = exp_name

    # Also include original experiments
    experiments["gpt-4o-mini (all) [original]"] = "exp1_all"
    experiments["gpt-4o-mini (individual) [original]"] = "exp1_individual"

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

    # Print results by model
    print("\n" + "-"*50)
    print("SDI BY MODEL AND MODE")
    print("-"*50)

    for name in sorted(results.keys()):
        sdi_data = results[name]
        if 0 in sdi_data and 3 in sdi_data:
            baseline = np.mean(sdi_data[0])
            final = np.mean(sdi_data[3])
            change = (final - baseline) / baseline * 100
            print(f"\n{name}:")
            print(f"  Iteration 0: SDI = {baseline:.4f}")
            print(f"  Iteration 3: SDI = {final:.4f}")
            print(f"  Change: {change:+.1f}%")

    # Summary table
    print("\n" + "-"*50)
    print("SUMMARY: SDI CHANGE BY MODEL")
    print("-"*50)
    print(f"{'Model':<30} {'Individual':>12} {'Group':>12} {'Difference':>12}")
    print("-"*66)

    for model_name in MODEL_CONFIGS.keys():
        ind_name = f"{model_name} (individual)"
        grp_name = f"{model_name} (all)"

        ind_change = None
        grp_change = None

        if ind_name in results and 0 in results[ind_name] and 3 in results[ind_name]:
            baseline = np.mean(results[ind_name][0])
            final = np.mean(results[ind_name][3])
            ind_change = (final - baseline) / baseline * 100

        if grp_name in results and 0 in results[grp_name] and 3 in results[grp_name]:
            baseline = np.mean(results[grp_name][0])
            final = np.mean(results[grp_name][3])
            grp_change = (final - baseline) / baseline * 100

        ind_str = f"{ind_change:+.1f}%" if ind_change is not None else "N/A"
        grp_str = f"{grp_change:+.1f}%" if grp_change is not None else "N/A"
        diff_str = ""
        if ind_change is not None and grp_change is not None:
            diff_str = f"{ind_change - grp_change:+.1f}%"

        print(f"{model_name:<30} {ind_str:>12} {grp_str:>12} {diff_str:>12}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-model experiments")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()) + ["all", "analyze"],
                        help="Model to run or 'all' for all models or 'analyze' to analyze results")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["individual", "all", "both"],
                        help="Reference mode")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=5)
    args = parser.parse_args()

    if args.model == "analyze":
        analyze_results()
        return

    models_to_run = []
    if args.model == "all":
        models_to_run = list(MODEL_CONFIGS.keys())
    else:
        models_to_run = [args.model]

    modes_to_run = []
    if args.mode == "both":
        modes_to_run = ["individual", "all"]
    else:
        modes_to_run = [args.mode]

    for model_name in models_to_run:
        for mode in modes_to_run:
            exp_name = f"multimodel_{model_name}_{mode}"
            outdir = os.path.join(BASE_DIR, f"results_{exp_name}")

            exp = MultiModelExperiment(
                model_name=model_name,
                reference_mode=mode,
                outdir=outdir,
                iterations=args.iterations,
                candidates=args.candidates,
            )
            exp.run()

            compute_embeddings_and_cache(exp_name, outdir)

    # Analyze all results
    analyze_results()

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()
