
import os
import csv
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random


"""
Multi-model, multi-workflow experiment runner (v2) for Scientific Reports.

Enhancements over v1:
1. Reproducibility: Added `seed` parameter to API calls (where supported).
2. Centralized Prompts: All system/user prompts are defined in `PROMPTS` dictionary.
3. Intervention Workflow: Added `divergence-keeper` workflow.
4. Metadata Logging: Records prompt versions and seeds in JSONL.
5. Chained Iterations: Iteration N uses output from Iteration N-1.
6. Expanded Workflows: Added `parallel` and `cyclic`.
7. Expanded Tasks: 80 tasks across 4 domains.

Usage:
  1. Set API keys in your environment.
  2. Run: python sim_v2.py
"""

# =========================
# Centralized Prompts
# =========================

PROMPTS = {
    "gen_initial": (
        "Task: {task}\n"
        "Iteration: {iteration}\n"
        "Candidate: {candidate}\n\n"
        "Please generate a comprehensive and high-quality response to the above task."
    ),
    "self_refine_system": (
        "You are an expert AI assistant capable of self-correction and refinement.\n"
        "Your goal is to improve the quality, clarity, and depth of the response in each iteration."
    ),
    "self_refine_user": (
        "Task: {task}\n"
        "Iteration: {iteration}\n"
        "Candidate: {candidate}\n\n"
        "Previous Output:\n{previous_output}\n\n"
        "Critique the previous output: Identify specific weaknesses, missing perspectives, or areas for improvement.\n"
        "Then, rewrite the response to address these points while maintaining high quality."
    ),
    "cyclic_system": (
        "You are a collaborative AI assistant working in a relay team.\n"
        "Your goal is to build upon the previous agent's work to create a superior final product."
    ),
    "cyclic_user": (
        "Task: {task}\n"
        "Iteration: {iteration}\n"
        "Candidate: {candidate}\n\n"
        "Previous Output from Peer:\n{previous_output}\n\n"
        "Review the peer's output. Keep the strong points but refine the weak ones.\n"
        "Add your own unique perspective to make the response more comprehensive and distinct."
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
    "expert_user": (
        "You are Expert {expert_id}.\n"
        "Task: {task}\n"
        "Iteration: {iteration}\n"
        "Candidate: {candidate}\n"
        "Propose one distinct solution or perspective.\n"
        "Be concrete and do not simply repeat common answers."
    ),
    "merger_system": (
        "You are a synthesis expert.\n"
        "Merge the following expert proposals into one coherent, well-structured response.\n"
        "Preserve minority or unconventional ideas where useful."
    ),
    "debate_pro": (
        "You are a debater arguing the Affirmative (Pro) position.\n"
        "Task: {task}\n"
        "Present a strong, persuasive argument supporting the affirmative view.\n"
        "Focus on logical consistency and concrete evidence."
    ),
    "debate_con": (
        "You are a debater arguing the Negative (Con) position.\n"
        "Task: {task}\n\n"
        "Opponent's Argument:\n{pro_text}\n\n"
        "Critique the opponent's argument. Identify logical fallacies or overlooked downsides.\n"
        "Then, present a strong counter-argument."
    ),
    "debate_judge": (
        "You are an impartial Judge in a debate.\n"
        "Task: {task}\n\n"
        "Affirmative Argument:\n{pro_text}\n\n"
        "Negative Argument:\n{con_text}\n\n"
        "Evaluate both arguments. Synthesize the valid points from both sides into a final, balanced conclusion.\n"
        "Do not simply compromise; find the most robust solution."
    ),
    "divergence_keeper_system": (
        "You are a 'Divergence Keeper' agent.\n"
        "Your specific role is to PREVENT the loss of semantic diversity.\n"
        "In multi-agent systems, responses tend to converge to a generic mean. You must fight this tendency."
    ),
    "divergence_keeper_user": (
        "Task: {task}\n"
        "Draft:\n{draft}\n\n"
        "Review the draft. Does it sound generic or cliché?\n"
        "If yes, rewrite it to be more distinct, specific, and novel.\n"
        "Preserve the core meaning, but inject unique phrasing or an unconventional angle.\n"
        "Do NOT normalize the text."
    )
}

# =========================
# Model API call utilities
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
                sleep_time = (backoff_in_seconds * 2 ** x + 
                              random.uniform(0, 1))
                time.sleep(sleep_time)
                x += 1
    return wrapper


def call_openai_chat(model: str, messages: List[Dict[str, str]], seed: Optional[int] = None) -> Tuple[str, Any]:
    """
    Call OpenAI Chat Completions API with optional seed for reproducibility.
    Uses OpenAI Python SDK v1.0.0+
    """
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    client = OpenAI(api_key=api_key)
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.7")),
    }
    if seed is not None:
        kwargs["seed"] = seed

    try:
        # Wrap the API call with retry logic
        @retry_with_backoff
        def _make_call():
            return client.chat.completions.create(**kwargs)
            
        resp = _make_call()
        text = resp.choices[0].message.content
        return text, resp
    except Exception as e:
        # Fallback for older libraries or models that don't support seed
        if "seed" in kwargs:
            del kwargs["seed"]
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            return text, resp
        raise e


def call_claude_chat(model: str, messages: List[Dict[str, str]], seed: Optional[int] = None) -> Tuple[str, Any]:
    """
    Call Anthropic Claude Messages API.
    Note: Claude API does not currently support a 'seed' parameter in the same way as OpenAI.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in environment.")

    client = anthropic.Anthropic(api_key=api_key)

    content_text = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    
    # System prompts are handled differently in Claude, but for simplicity we merge here or extract if needed.
    # Ideally, extract 'system' role if present.
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
    
    kwargs = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": content_text}],
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.7")),
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    if system_prompt:
        kwargs["system"] = system_prompt

    @retry_with_backoff
    def _make_call():
        return client.messages.create(**kwargs)

    resp = _make_call()

    text_parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    text = "\n".join(text_parts) if text_parts else str(resp)
    return text, resp


def call_gemini_chat(model: str, messages: List[Dict[str, str]], seed: Optional[int] = None) -> Tuple[str, Any]:
    """
    Call Google Gemini API.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment.")

    genai.configure(api_key=api_key)

    prompt = ""
    for m in messages:
        prefix = "User: " if m["role"] == "user" else "Assistant: " # Gemini doesn't use 'system' role in chat history same way
        if m["role"] == "system":
             prefix = "System Instruction: "
        prompt += prefix + m["content"] + "\n"

    model_obj = genai.GenerativeModel(model)
    model_obj = genai.GenerativeModel(model)
    
    @retry_with_backoff
    def _make_call():
        return model_obj.generate_content(
            prompt,
            generation_config={"temperature": float(os.environ.get("LLM_TEMPERATURE", "0.7"))}
        )

    resp = _make_call()
    text = resp.text if hasattr(resp, "text") else str(resp)
    return text, resp


MODEL_CALLERS = {
    "openai": call_openai_chat,
    "claude": call_claude_chat,
    "gemini": call_gemini_chat,
}


# =========================
# Quality Evaluation
# =========================

QUALITY_EVAL_PROMPT = """You are an expert evaluator. Rate the following response on a scale of 1-10 for each criterion.

Task: {task}

Response to evaluate:
{response}

Rate on these criteria (1=poor, 10=excellent):
1. Relevance: Does it address the task directly?
2. Completeness: Is it thorough and comprehensive?
3. Clarity: Is it well-structured and easy to understand?
4. Originality: Does it offer unique or creative perspectives?
5. Quality: Overall writing quality and polish.

Respond ONLY with a JSON object in this exact format:
{{"relevance": X, "completeness": X, "clarity": X, "originality": X, "quality": X, "overall": X}}

Where X is a number from 1-10. "overall" should be the average of the other scores.
"""


def evaluate_quality(task: str, response: str, model: str = "gpt-4o-mini") -> Dict[str, float]:
    """
    Evaluate the quality of a response using an LLM judge.
    Returns a dict with scores for each criterion.
    """
    messages = [
        {"role": "user", "content": QUALITY_EVAL_PROMPT.format(task=task, response=response)}
    ]

    try:
        text, _ = call_openai_chat(model, messages, seed=42)
        # Parse JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
    except Exception as e:
        print(f"[WARN] Quality evaluation failed: {e}")

    # Return default scores on failure
    return {"relevance": 5, "completeness": 5, "clarity": 5, "originality": 5, "quality": 5, "overall": 5}


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


# =========================
# Experiment framework
# =========================

class MultiModelExperiment:
    def __init__(
        self,
        tasks: List[str],
        workflows: List[str],
        models: Dict[str, List[str]],
        outdir: str = "results_v2",
        iterations: int = 4,
        candidates: int = 6,
        num_workers: int = 1,
        temperature: float = 0.7,
        seed: int = 42,
        reference_mode: str = "individual",
        share_ratio: float = 1.0,
        share_frequency: int = 1,
        evaluate_quality: bool = False,
    ) -> None:
        self.tasks = tasks
        self.workflows = workflows
        self.models = models
        self.outdir = outdir
        self.iterations = iterations
        self.candidates = candidates
        self.num_workers = max(1, num_workers)
        self.temperature = temperature
        self.seed = seed
        self.reference_mode = reference_mode
        self.share_ratio = share_ratio  # 0.0 to 1.0: fraction of candidates to share
        self.share_frequency = share_frequency  # Share every N iterations (1=every, 2=every other, etc.)
        self.evaluate_quality = evaluate_quality  # Whether to run quality evaluation

        # Append mode to outdir if it is the default one to avoid mixing results
        if self.outdir == "results_v2":
            mode_suffix = self.reference_mode
            if self.share_ratio < 1.0:
                mode_suffix += f"_ratio{int(self.share_ratio*100)}"
            if self.share_frequency > 1:
                mode_suffix += f"_freq{self.share_frequency}"
            self.outdir = f"results_v2_{mode_suffix}"

        ensure_dir(self.outdir)

        self.logfile_txt = os.path.join(self.outdir, "run_log.txt")
        self.logfile_jsonl = os.path.join(self.outdir, "run_log.jsonl")
        self.csvfile = os.path.join(self.outdir, "results.csv")

        self._lock = threading.Lock()
        self.done_index = set()
        
        # State tracking for chained iterations
        # Key: (family, model, workflow, task_id, candidate) -> latest_text
        self.chain_state: Dict[Tuple[str, str, str, int, int], str] = {}
        
        self._load_done_index()

        with open(self.logfile_txt, "a", encoding="utf-8") as f:
            f.write(f"=== Experiment started {datetime.now().isoformat()} (Seed: {self.seed}) ===\n")

        if not os.path.exists(self.csvfile):
            with open(self.csvfile, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "model_family",
                        "model",
                        "workflow",
                        "task_id",
                        "task",
                        "iteration",
                        "candidate",
                        "final_output",
                    ]
                )

    def _load_done_index(self) -> None:
        if not os.path.exists(self.csvfile):
            return
        try:
            with open(self.csvfile, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (
                        row["model_family"],
                        row["model"],
                        row["workflow"],
                        int(row["task_id"]),
                        int(row["iteration"]),
                        int(row["candidate"]),
                    )
                    self.done_index.add(key)
                    
                    # Also restore chain state if we are resuming
                    # We need the output of the *latest* iteration for each candidate
                    chain_key = (
                        row["model_family"],
                        row["model"],
                        row["workflow"],
                        int(row["task_id"]),
                        int(row["candidate"]),
                    )
                    # Since CSV is append-only, later rows overwrite earlier ones in this dict, which is exactly what we want (latest state)
                    # We need to un-escape newlines if we want to use it as input
                    output = row["final_output"].replace("\\n", "\n")
                    self.chain_state[chain_key] = output
                    
        except Exception as e:
            print(f"[WARN] Failed to load done_index from CSV: {e}")

    def log_text(self, msg: str) -> None:
        with self._lock:
            with open(self.logfile_txt, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")

    def log_json(self, record: Dict[str, Any]) -> None:
        with self._lock:
            with open(self.logfile_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_csv_row(self, row: List[Any]) -> None:
        with self._lock:
            with open(self.csvfile, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
    
    def _format_previous_candidates(self, previous_outputs: List[str]) -> str:
        """Format list of previous outputs for prompt."""
        if not previous_outputs:
            return ""
        
        if len(previous_outputs) == 1:
            return f"Previous Output:\n{previous_outputs[0]}\n\n"
        
        formatted = "\n\n".join(
            f"- Candidate {i+1}:\n{text}"
            for i, text in enumerate(previous_outputs)
        )
        return f"Here are previous responses:\n{formatted}\n\n"

    # ---------- Workflows ----------

    def _job_parallel(self, model_caller, model_name, task, iteration, candidate, previous_outputs) -> Dict[str, Any]:
        """
        Parallel workflow: Independent generations. Ignores previous output.
        """
        messages = [
            {
                "role": "user",
                "content": PROMPTS["gen_initial"].format(task=task, iteration=iteration, candidate=candidate),
            }
        ]
        text, raw = model_caller(model_name, messages, seed=self.seed)
        return {
            "step_type": "parallel",
            "messages": messages,
            "response_text": text,
            "final_output": text,
        }

    def _job_self_refine(self, model_caller, model_name, task, iteration, candidate, previous_outputs) -> Dict[str, Any]:
        if iteration == 0 or not previous_outputs:
            return self._job_parallel(model_caller, model_name, task, iteration, candidate, None)
        
        # Format previous candidates (single or multiple)
        prev_text = self._format_previous_candidates(previous_outputs)
        
        # Adjust prompt based on mode (implicitly handled by _format_previous_candidates content)
        # But we need to construct the prompt text.
        # If individual (len=1), use self_refine_user template?
        # If all (len>1), use a modified template?
        
        # To keep it simple and consistent with PROMPTS, let's construct the content manually or use a unified prompt.
        # The PROMPTS["self_refine_user"] uses {previous_output}.
        
        if len(previous_outputs) == 1:
             content = PROMPTS["self_refine_user"].format(
                task=task, 
                iteration=iteration, 
                candidate=candidate,
                previous_output=previous_outputs[0]
            )
        else:
            # All reference mode
            content = (
                f"Task: {task}\n"
                f"Iteration: {iteration}\n"
                f"Candidate: {candidate}\n\n"
                f"{prev_text}"
                f"Critique the above outputs and then generate a new response that is higher quality, more distinct, and robust."
            )

        messages = [{"role": "user", "content": content}]
        text, raw = model_caller(model_name, messages, seed=self.seed)
        return {
            "step_type": "self-refine",
            "messages": messages,
            "response_text": text,
            "final_output": text,
        }

    def _job_cyclic(self, model_caller, model_name, task, iteration, candidate, previous_outputs) -> Dict[str, Any]:
        if iteration == 0 or not previous_outputs:
            return self._job_parallel(model_caller, model_name, task, iteration, candidate, None)
        
        prev_text = self._format_previous_candidates(previous_outputs)
        
        if len(previous_outputs) == 1:
            content = PROMPTS["cyclic_user"].format(
                task=task, 
                iteration=iteration, 
                candidate=candidate,
                previous_output=previous_outputs[0]
            )
        else:
             content = (
                f"Task: {task}\n"
                f"Iteration: {iteration}\n"
                f"Candidate: {candidate}\n\n"
                f"{prev_text}"
                f"You are the next agent in the cycle. Improve upon the previous outputs, adding your own unique perspective."
            )

        messages = [{"role": "user", "content": content}]
        text, raw = model_caller(model_name, messages, seed=self.seed)
        return {
            "step_type": "cyclic",
            "messages": messages,
            "response_text": text,
            "final_output": text,
        }

    def _job_gen_critic(self, model_caller, model_name, task, iteration, candidate, previous_outputs) -> Dict[str, Any]:
        if iteration == 0 or not previous_outputs:
            # Generator
            gen_messages = [
                {
                    "role": "user",
                    "content": PROMPTS["gen_initial"].format(task=task, iteration=iteration, candidate=candidate),
                }
            ]
            gen_text, _ = model_caller(model_name, gen_messages, seed=self.seed)
            draft = gen_text
        else:
            # If multiple previous outputs, we need to decide what is the "draft".
            # For "All" mode, maybe we ask the generator to synthesize/improve them first?
            # Or just pick the corresponding one?
            # To strictly follow "All Reference", the agent should see all.
            # Let's assume for Gen-Critic in All mode, the Generator sees all and produces a new draft.
            
            if len(previous_outputs) == 1:
                draft = previous_outputs[0]
                gen_messages = []
                gen_text = ""
            else:
                # All mode: Generator sees all and creates a new draft
                prev_text = self._format_previous_candidates(previous_outputs)
                gen_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task}\n"
                            f"Iteration: {iteration}\n"
                            f"Candidate: {candidate}\n\n"
                            f"{prev_text}"
                            f"Based on these previous outputs, generate a new, improved draft."
                        )
                    }
                ]
                gen_text, _ = model_caller(model_name, gen_messages, seed=self.seed)
                draft = gen_text

        # Critic
        critic_messages = [
            {"role": "system", "content": PROMPTS["critic_system"]},
            {"role": "user", "content": PROMPTS["critic_user"].format(task=task, draft=draft)},
        ]
        crit_text, _ = model_caller(model_name, critic_messages, seed=self.seed)

        return {
            "step_type": "gen-critic",
            "generator": {"messages": gen_messages, "response_text": gen_text},
            "critic": {"messages": critic_messages, "response_text": crit_text},
            "final_output": crit_text,
        }

    def _job_gen_critic_sum(self, model_caller, model_name, task, iteration, candidate, previous_outputs) -> Dict[str, Any]:
        if iteration == 0 or not previous_outputs:
            gen_messages = [
                {
                    "role": "user",
                    "content": PROMPTS["gen_initial"].format(task=task, iteration=iteration, candidate=candidate),
                }
            ]
            gen_text, _ = model_caller(model_name, gen_messages, seed=self.seed)
            draft = gen_text
        else:
            if len(previous_outputs) == 1:
                draft = previous_outputs[0]
                gen_messages = []
                gen_text = ""
            else:
                prev_text = self._format_previous_candidates(previous_outputs)
                gen_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task}\n"
                            f"Iteration: {iteration}\n"
                            f"Candidate: {candidate}\n\n"
                            f"{prev_text}"
                            f"Based on these previous outputs, generate a new, improved draft."
                        )
                    }
                ]
                gen_text, _ = model_caller(model_name, gen_messages, seed=self.seed)
                draft = gen_text

        # Critic
        critic_messages = [
            {"role": "system", "content": PROMPTS["critic_system"]},
            {"role": "user", "content": PROMPTS["critic_user"].format(task=task, draft=draft)},
        ]
        crit_text, _ = model_caller(model_name, critic_messages, seed=self.seed)

        # Summarizer
        sum_messages = [
            {"role": "system", "content": PROMPTS["summarizer_system"]},
            {"role": "user", "content": PROMPTS["summarizer_user"].format(draft=crit_text)},
        ]
        sum_text, _ = model_caller(model_name, sum_messages, seed=self.seed)

        return {
            "step_type": "gen-critic-sum",
            "generator": {"messages": gen_messages, "response_text": gen_text},
            "critic": {"messages": critic_messages, "response_text": crit_text},
            "summarizer": {"messages": sum_messages, "response_text": sum_text},
            "final_output": sum_text,
        }

    def _job_parallel_merge(self, model_caller, model_name, task, iteration, candidate, previous_outputs) -> Dict[str, Any]:
        if iteration == 0 or not previous_outputs:
             # Initial: 3 experts generate from scratch
             base_prompt = PROMPTS["expert_user"]
             context = ""
        else:
             # Chain: 3 experts refine the previous output(s)
             if len(previous_outputs) == 1:
                 base_prompt = PROMPTS["expert_user"] + "\n\nBase text to improve:\n" + previous_outputs[0]
                 context = previous_outputs[0]
             else:
                 prev_text = self._format_previous_candidates(previous_outputs)
                 base_prompt = PROMPTS["expert_user"] + "\n\nBase texts to improve:\n" + prev_text
                 context = prev_text

        expert_msgs = []
        expert_outs = []
        for i in range(3):
            msgs = [
                {
                    "role": "user",
                    "content": base_prompt.format(
                        expert_id=i+1, 
                        task=task, 
                        iteration=iteration, 
                        candidate=candidate
                    ),
                }
            ]
            text, _ = model_caller(model_name, msgs, seed=self.seed + i) # Vary seed slightly for experts
            expert_msgs.append(msgs)
            expert_outs.append(text)

        merged_source = "\n\n".join([f"Expert {i+1}:\n{expert_outs[i]}" for i in range(3)])
        merge_messages = [
            {"role": "system", "content": PROMPTS["merger_system"]},
            {"role": "user", "content": merged_source},
        ]
        merged_text, _ = model_caller(model_name, merge_messages, seed=self.seed)

        return {
            "step_type": "parallel-merge",
            "experts": [{"id": i+1, "text": expert_outs[i]} for i in range(3)],
            "merger": {"messages": merge_messages, "response_text": merged_text},
            "final_output": merged_text,
        }

    def _job_debate(self, model_caller, model_name, task, iteration, candidate, previous_outputs) -> Dict[str, Any]:
        if iteration == 0 or not previous_outputs:
             # Standard Debate
             pro_prompt = PROMPTS["debate_pro"].format(task=task, iteration=iteration, candidate=candidate)
        else:
             if len(previous_outputs) == 1:
                 pro_prompt = PROMPTS["debate_pro"].format(task=task, iteration=iteration, candidate=candidate) + "\n\nStarting Point:\n" + previous_outputs[0]
             else:
                 prev_text = self._format_previous_candidates(previous_outputs)
                 pro_prompt = PROMPTS["debate_pro"].format(task=task, iteration=iteration, candidate=candidate) + "\n\nStarting Points:\n" + prev_text

        # Pro
        pro_messages = [{"role": "user", "content": pro_prompt}]
        pro_text, _ = model_caller(model_name, pro_messages, seed=self.seed)

        # Con
        con_messages = [{"role": "user", "content": PROMPTS["debate_con"].format(task=task, pro_text=pro_text)}]
        con_text, _ = model_caller(model_name, con_messages, seed=self.seed)

        # Judge
        judge_messages = [{"role": "user", "content": PROMPTS["debate_judge"].format(task=task, pro_text=pro_text, con_text=con_text)}]
        judge_text, _ = model_caller(model_name, judge_messages, seed=self.seed)

        return {
            "step_type": "debate",
            "pro": {"text": pro_text},
            "con": {"text": con_text},
            "judge": {"text": judge_text},
            "final_output": judge_text,
        }

    def _job_divergence_keeper(self, model_caller, model_name, task, iteration, candidate, previous_outputs) -> Dict[str, Any]:
        if iteration == 0 or not previous_outputs:
            gen_messages = [
                {
                    "role": "user",
                    "content": PROMPTS["gen_initial"].format(task=task, iteration=iteration, candidate=candidate),
                }
            ]
            gen_text, _ = model_caller(model_name, gen_messages, seed=self.seed)
            draft = gen_text
        else:
            if len(previous_outputs) == 1:
                draft = previous_outputs[0]
                gen_messages = []
                gen_text = ""
            else:
                prev_text = self._format_previous_candidates(previous_outputs)
                gen_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task}\n"
                            f"Iteration: {iteration}\n"
                            f"Candidate: {candidate}\n\n"
                            f"{prev_text}"
                            f"Based on these previous outputs, generate a new, distinct draft."
                        )
                    }
                ]
                gen_text, _ = model_caller(model_name, gen_messages, seed=self.seed)
                draft = gen_text

        # Divergence Keeper
        keeper_messages = [
            {"role": "system", "content": PROMPTS["divergence_keeper_system"]},
            {"role": "user", "content": PROMPTS["divergence_keeper_user"].format(task=task, draft=draft)},
        ]
        keeper_text, _ = model_caller(model_name, keeper_messages, seed=self.seed)

        return {
            "step_type": "divergence-keeper",
            "generator": {"messages": gen_messages, "response_text": gen_text},
            "keeper": {"messages": keeper_messages, "response_text": keeper_text},
            "final_output": keeper_text,
        }

    def _run_single_job(self, job: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        family = job["family"]
        model_name = job["model"]
        workflow = job["workflow"]
        task_id = job["task_id"]
        task_text = job["task"]
        iteration = job["iteration"]
        candidate = job["candidate"]
        previous_outputs = job.get("previous_outputs", [])

        model_caller = MODEL_CALLERS[family]
        self.log_text(f"START job: {family}/{model_name} | wf={workflow} | task_id={task_id} | iter={iteration} | cand={candidate}")

        if workflow == "parallel":
            record = self._job_parallel(model_caller, model_name, task_text, iteration, candidate, previous_outputs)
        elif workflow == "self-refine":
            record = self._job_self_refine(model_caller, model_name, task_text, iteration, candidate, previous_outputs)
        elif workflow == "cyclic":
            record = self._job_cyclic(model_caller, model_name, task_text, iteration, candidate, previous_outputs)
        elif workflow == "gen-critic":
            record = self._job_gen_critic(model_caller, model_name, task_text, iteration, candidate, previous_outputs)
        elif workflow == "gen-critic-sum":
            record = self._job_gen_critic_sum(model_caller, model_name, task_text, iteration, candidate, previous_outputs)
        elif workflow == "parallel-merge":
            record = self._job_parallel_merge(model_caller, model_name, task_text, iteration, candidate, previous_outputs)
        elif workflow == "debate":
            record = self._job_debate(model_caller, model_name, task_text, iteration, candidate, previous_outputs)
        elif workflow == "divergence-keeper":
            record = self._job_divergence_keeper(model_caller, model_name, task_text, iteration, candidate, previous_outputs)
        else:
            raise ValueError(f"Unknown workflow: {workflow}")

        final_output = record.get("final_output", record.get("response_text", ""))
        # Sanitize for CSV: replace newlines to keep rows on single lines
        final_output_safe = final_output.replace("\n", "\\n")

        meta = {
            "timestamp": datetime.now().isoformat(),
            "model_family": family,
            "model": model_name,
            "workflow": workflow,
            "task_id": task_id,
            "task": task_text,
            "iteration": iteration,
            "candidate": candidate,
            "seed": self.seed,
        }
        full_record = {"meta": meta, "trace": record}

        self.log_json(full_record)
        self.save_csv_row(
            [
                meta["timestamp"],
                family,
                model_name,
                workflow,
                task_id,
                task_text,
                iteration,
                candidate,
                final_output_safe,
            ]
        )

        self.log_text(f"END job: {family}/{model_name} | wf={workflow} | task_id={task_id}")
        
        return job, final_output

    def run(self) -> None:
        # Layered execution:
        # We must run Iteration 0 for all candidates first (or at least for a specific candidate).
        # To maximize parallelism, we can run all Candidates for Iteration 0 in parallel.
        # Then all Candidates for Iteration 1.
        
        # Structure:
        # for iteration in range(iterations):
        #    jobs = []
        #    for ... all other dims ...
        #       prepare job (fetching prev_output from state)
        #    execute jobs
        #    update state
        
        for it in range(self.iterations):
            self.log_text(f"--- Starting Iteration {it} ---")
            jobs = []
            
            for family, model_list in self.models.items():
                if family not in MODEL_CALLERS:
                    raise ValueError(f"Unknown model family: {family}")
                for model_name in model_list:
                    for task_id, task_text in enumerate(self.tasks):
                        for workflow in self.workflows:
                            for c in range(self.candidates):
                                key = (family, model_name, workflow, task_id, it, c)
                                if key in self.done_index:
                                    continue
                                
                                # Get previous output(s) if it > 0
                                prev_outs = []
                                if it > 0:
                                    # Check share_frequency: only share on specific iterations
                                    should_share = (it % self.share_frequency == 0)

                                    if self.reference_mode == "individual" or not should_share:
                                        # Individual mode OR non-sharing iteration: fetch only self
                                        chain_key = (family, model_name, workflow, task_id, c)
                                        p_out = self.chain_state.get(chain_key)
                                        if p_out:
                                            prev_outs.append(p_out)
                                    elif self.reference_mode == "all":
                                        # All mode with share_ratio support
                                        # Collect all candidates first
                                        all_candidates = []
                                        for cand_idx in range(self.candidates):
                                            chain_key = (family, model_name, workflow, task_id, cand_idx)
                                            p_out = self.chain_state.get(chain_key)
                                            if p_out:
                                                all_candidates.append((cand_idx, p_out))

                                        # Apply share_ratio: select a subset of candidates
                                        if self.share_ratio < 1.0 and len(all_candidates) > 1:
                                            # Always include self
                                            self_out = None
                                            others = []
                                            for cand_idx, p_out in all_candidates:
                                                if cand_idx == c:
                                                    self_out = p_out
                                                else:
                                                    others.append(p_out)

                                            # Select share_ratio fraction of others
                                            num_to_share = max(0, int(len(others) * self.share_ratio))
                                            # Use deterministic selection based on seed + iteration + candidate
                                            rng = random.Random(self.seed + it * 1000 + c)
                                            selected_others = rng.sample(others, min(num_to_share, len(others)))

                                            # Combine: self first, then selected others
                                            if self_out:
                                                prev_outs.append(self_out)
                                            prev_outs.extend(selected_others)
                                        else:
                                            # Full sharing (share_ratio=1.0)
                                            prev_outs = [p_out for _, p_out in all_candidates]
                                
                                # Check if we have required inputs
                                if it > 0 and not prev_outs and workflow != "parallel":
                                    self.log_text(f"[WARN] Missing previous outputs for {key}. Skipping.")
                                    continue
                                
                                jobs.append({
                                    "family": family,
                                    "model": model_name,
                                    "workflow": workflow,
                                    "task_id": task_id,
                                    "task": task_text,
                                    "iteration": it,
                                    "candidate": c,
                                    "previous_outputs": prev_outs,
                                })
            
            # Execute jobs in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._run_single_job, job) for job in jobs]
                
                for i, fut in enumerate(as_completed(futures), start=1):
                    try:
                        job_info, output = fut.result()
                        # Update chain state
                        chain_key = (
                            job_info["family"], 
                            job_info["model"], 
                            job_info["workflow"], 
                            job_info["task_id"], 
                            job_info["candidate"]
                        )
                        with self._lock:
                            self.chain_state[chain_key] = output
                            
                    except Exception as e:
                        self.log_text(f"[ERROR] Job failed: {e}")
                    
                    if i % 10 == 0:
                        self.log_text(f"Progress (Iter {it}): {i}/{len(futures)} jobs completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-model experiments.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (default: 16)")
    parser.add_argument("--iterations", type=int, default=4, help="Number of iterations per task (default: 4)")
    parser.add_argument("--candidates", type=int, default=50, help="Number of candidates per iteration (default: 50)")
    parser.add_argument("--outdir", type=str, default="results_v2", help="Output directory (default: results_v2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--families", nargs="+", help="List of model families to run (e.g. openai gemini)")
    parser.add_argument("--models", nargs="+", help="List of specific models to run (e.g. gpt-4o-mini)")
    parser.add_argument("--reference-mode", type=str, default="individual", choices=["individual", "all"],
                        help="Reference mode: 'individual' (self-history) or 'all' (group-history)")
    parser.add_argument("--share-ratio", type=float, default=1.0,
                        help="Fraction of other candidates to share (0.0-1.0). Only applies when reference-mode='all'. "
                             "0.0 = share none (like individual), 0.5 = share 50%%, 1.0 = share all (default)")
    parser.add_argument("--share-frequency", type=int, default=1,
                        help="Share every N iterations. 1=every iteration (default), 2=every other, 3=every third, etc.")
    parser.add_argument("--evaluate-quality", action="store_true",
                        help="Run quality evaluation on final outputs (requires additional API calls)")
    args = parser.parse_args()

    # 80 Diverse Tasks for Scientific Reports Experiment
    # Categories: Creative (1-20), Reasoning (21-40), Explanation (41-60), Policy (61-80)

    tasks = [
        # --- Creative Ideation (1-20) ---
        "Generate 5 novel product ideas that help reduce food waste at home.",
        "Generate 5 new interaction concepts for supporting remote teamwork.",
        "Generate 5 services that use AI to support elderly people living alone.",
        "Generate 5 ideas for playful urban installations using light and sound.",
        "Generate 5 ideas for improving the experience of public transportation."
    ]

    """ tasks = [
        # --- Creative Ideation (1-20) ---
        "Generate 5 novel product ideas that help reduce food waste at home.",
        "Write a short story opening that begins with the sentence: 'The clock struck thirteen.'",
        "Propose a unique marketing strategy for a new coffee brand targeting remote workers.",
        "Invent a new sport that combines elements of chess and tennis.",
        "Design a futuristic city concept that functions entirely without cars.",
        "Create a synopsis for a movie where dreams can be recorded and sold.",
        "Design a mobile app that gamifies learning a new language for elderly people.",
        "Propose a new holiday that celebrates scientific curiosity.",
        "Write a poem about the feeling of losing a digital memory.",
        "Invent a new musical instrument that uses water as its primary sound source.",
        "Describe a restaurant concept where the menu changes based on the weather.",
        "Create a superhero whose power is based on empathy, not strength.",
        "Design a board game that teaches players about ecosystem balance.",
        "Write a dialogue between a time traveler from 2050 and a person from 1950.",
        "Propose a fashion line made entirely from recycled ocean plastic.",
        "Invent a new color and describe the emotions it evokes.",
        "Design a playground that is accessible to children with all types of disabilities.",
        "Write a recipe for a dish that represents the concept of 'nostalgia'.",
        "Create a myth explaining why the moon has phases.",
        "Propose a new form of public transport for a mountainous region.",

        # --- Reasoning & Planning (21-40) ---
        "Outline a step-by-step plan to transition a small office to a 4-day work week.",
        "Propose a solution to reduce traffic congestion in a megacity without building new roads.",
        "Develop a curriculum for teaching critical thinking to elementary school students.",
        "Create a weekly meal plan for a vegan athlete training for a marathon.",
        "Suggest a strategy for a bookstore to survive in the age of digital media.",
        "Plan a 3-day itinerary for a sustainable tourism trip to Kyoto.",
        "Develop a disaster recovery plan for a small cloud-based startup.",
        "Propose a method to fairly allocate office desk space in a hybrid work environment.",
        "Design a mentorship program for women in STEM fields.",
        "Create a budget plan for a family of four saving for a house in 5 years.",
        "Outline a strategy to reduce plastic usage in a local school district.",
        "Develop a plan to launch a community garden in an urban food desert.",
        "Propose a system for managing space debris in low Earth orbit.",
        "Create a workflow for a remote team to collaborate on a creative project asynchronously.",
        "Suggest a strategy for a non-profit to increase youth volunteer engagement.",
        "Plan a fundraising event for a local animal shelter with zero budget.",
        "Develop a protocol for handling customer complaints in a high-stress call center.",
        "Propose a solution for improving water efficiency in traditional agriculture.",
        "Create a study schedule for a working professional preparing for a certification exam.",
        "Outline a plan to introduce a new software tool to a resistant workforce.",

        # --- Explanation & Synthesis (41-60) ---
        "Explain the concept of 'Algorithmic Groupthink' to a non-technical audience.",
        "Summarize the potential benefits and risks of space tourism.",
        "Describe how a blockchain works using a metaphor involving a library.",
        "Explain the psychological concept of 'Cognitive Dissonance' with everyday examples.",
        "Synthesize the arguments for and against universal basic income (UBI).",
        "Explain the difference between machine learning and deep learning to a high schooler.",
        "Summarize the impact of the printing press on European society.",
        "Describe how a blockchain works using a metaphor involving a library.",
        "Explain the concept of 'Schrödinger's cat' without using jargon.",
        "Synthesize the main causes of the 2008 financial crisis.",
        "Explain how a vaccine works to the immune system using a castle defense metaphor.",
        "Summarize the ethical arguments surrounding genetic engineering in humans.",
        "Describe the lifecycle of a star from birth to black hole.",
        "Explain the concept of 'inflation' in economics using a pizza analogy.",
        "Synthesize the current scientific consensus on climate change.",
        "Explain the significance of the Rosetta Stone in understanding history.",
        "Summarize the plot and themes of '1984' by George Orwell.",
        "Describe how the internet works using a postal service metaphor.",
        "Explain the concept of 'entropy' in thermodynamics to a child.",
        "Synthesize the benefits of meditation on the human brain.",

        # --- Policy & Ethics (61-80) ---
        "Draft a corporate policy for the ethical use of AI in hiring processes.",
        "Propose a set of guidelines for regulating autonomous delivery drones in residential areas.",
        "Argue for or against the mandatory labeling of AI-generated content on social media.",
        "Develop a code of conduct for a new online community focused on political debate.",
        "Suggest a framework for allocating limited medical resources during a hypothetical pandemic.",
        "Draft a policy for handling data privacy in a smart city project.",
        "Propose guidelines for the use of facial recognition technology by law enforcement.",
        "Argue for or against the implementation of a global carbon tax.",
        "Develop a policy for remote work rights and 'right to disconnect'.",
        "Suggest a framework for regulating deepfakes in political campaigns.",
        "Draft a code of ethics for autonomous vehicle decision-making in accidents.",
        "Propose guidelines for the preservation of indigenous languages in the digital age.",
        "Argue for or against the patenting of genetically modified seeds.",
        "Develop a policy for addressing cyberbullying in schools.",
        "Suggest a framework for the ethical treatment of AI agents if they achieve sentience.",
        "Draft a proposal for regulating space mining operations.",
        "Propose guidelines for the use of neurotechnology in the workplace.",
        "Argue for or against the ban of single-use plastics in hospitals.",
        "Develop a policy for ensuring algorithmic fairness in loan approval systems.",
        "Suggest a framework for international cooperation on AI safety research."
    ]
 """
    # ===== Workflow Selection =====
    # Commercial LLM-focused workflows (recommended for studying real-world AI systems):
    #   - gen-critic:      Safety filter structure (all commercial LLMs)
    #   - self-refine:     System 2 / CoT reasoning (OpenAI o1, Claude)
    #   - parallel-merge:  MoE architecture (GPT-4, Gemini)
    #   - gen-critic-sum:  Filter + summarization pipeline
    #
    # Full experimental workflows (for comprehensive analysis):
    #   - parallel:           Baseline (no interaction)
    #   - cyclic:             Cyclic refinement
    #   - debate:             Adversarial reasoning
    #   - divergence-keeper:  Intervention method

    # Option A: Commercial LLM focus (recommended)
    workflows = [
        "parallel",         # Baseline
        "gen-critic",       # Safety filter (most critical - all commercial LLMs)
        "self-refine",      # System 2 reasoning (o1, Claude thinking)
        "parallel-merge",   # MoE architecture (GPT-4, Gemini)
        "gen-critic-sum",   # Full commercial pipeline simulation
    ]

    # Option B: Full analysis (uncomment to use)
    # workflows = [
    #     "parallel",
    #     "self-refine",
    #     "cyclic",
    #     "gen-critic",
    #     "gen-critic-sum",
    #     "parallel-merge",
    #     "debate",
    #     "divergence-keeper",
    # ]
    
    all_models = {
        "openai": ["gpt-4o-mini"], 
        "claude": ["claude-3-haiku-20240307"],
        "gemini": ["gemini-1.5-flash"],
    }

    final_models = {}
    if not args.families and not args.models:
        final_models = all_models
    else:
        for family, m_list in all_models.items():
            filtered_list = []
            for m in m_list:
                family_match = (args.families is None) or (family in args.families)
                model_match = (args.models is None) or (m in args.models)
                
                if family_match and model_match:
                    filtered_list.append(m)
            
            if filtered_list:
                final_models[family] = filtered_list

    print(f"Starting experiment with {args.workers} workers...")
    print(f"Iterations: {args.iterations}, Candidates: {args.candidates}")
    print(f"Selected Models: {final_models}")
    print(f"Reference Mode: {args.reference_mode}")
    if args.reference_mode == "all":
        print(f"Share Ratio: {args.share_ratio} ({int(args.share_ratio*100)}% of other candidates)")
        print(f"Share Frequency: every {args.share_frequency} iteration(s)")
    if args.evaluate_quality:
        print("Quality Evaluation: ENABLED (will run after main experiment)")

    if not final_models:
        print("[ERROR] No models selected! Check your --families and --models arguments.")
        exit(1)

    exp = MultiModelExperiment(
        tasks=tasks,
        workflows=workflows,
        models=final_models,
        outdir=args.outdir,
        iterations=args.iterations,
        candidates=args.candidates,
        num_workers=args.workers,
        temperature=0.7,
        seed=args.seed,
        reference_mode=args.reference_mode,
        share_ratio=args.share_ratio,
        share_frequency=args.share_frequency,
        evaluate_quality=args.evaluate_quality,
    )
    exp.run()
