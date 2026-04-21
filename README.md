# Algorithmic Groupthink: Code and Data
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19676464.svg)](https://doi.org/10.5281/zenodo.19676464)

This repository contains the code, data, and analysis scripts for the paper:

> Muto, H., & Ogi, T. *Algorithmic Groupthink: Measuring and Mitigating Semantic Convergence in Multi-agent LLM Systems.*

## What is in this repository

- `src/` — Core simulation framework
- `experiments/` — Scripts for running each experiment reported in the paper
- `analysis/` — Scripts for analysis, statistical tests, and figure generation
- `data/raw_outputs/` — Text outputs from all experiments (CSV format)
- `docs/` — Supplementary documentation

## Quick start

### Requirements

- Python 3.11+
- OpenAI API key (for GPT-4o-mini and GPT-4o)
- Anthropic API key (for Claude-3.5-Haiku)
- Google API key (for Gemini-2.0-Flash)

### Installation

```bash
git clone https://github.com/hmuto/algorithmic-groupthink.git
cd algorithmic-groupthink
pip install -r requirements.txt
```

### Setting API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

## Reproducing the main experiments

### Experiment 1: Reference mode comparison (20 tasks)

```bash
python experiments/run_exp1_reference_modes.py --mode individual
python experiments/run_exp1_reference_modes.py --mode all
python analysis/analyze_exp1.py
```

Expected output: individual-history shows diversity increase (+10.3%), group-history shows decrease (−5.9%).

### Experiment 2: Dose-response (share ratio)

```bash
for ratio in 0.0 0.25 0.5 0.75 1.0; do
  python src/simulation.py --reference-mode all --share-ratio $ratio \
    --outdir results_exp2_ratio_${ratio//./_}
done
```

### Experiment 3: Countermeasures

```bash
python experiments/run_exp3_countermeasures.py --mode diversity_prompt
python experiments/run_exp3_countermeasures.py --mode adversarial
python analysis/analyze_countermeasures.py
```

### Cross-model replication

```bash
python experiments/run_crossmodel_experiments.py --model claude-3-5-haiku
python experiments/run_crossmodel_experiments.py --model gemini-2.0-flash
python experiments/run_gpt4o_replication.py
```

### Control experiments

```bash
python experiments/run_control_experiments.py
```

### Sentence-BERT robustness validation

```bash
python analysis/validate_sentence_bert.py
```

## Raw data format

Each CSV file under `data/raw_outputs/` has the following columns:

| Column | Description |
| --- | --- |
| `timestamp` | ISO 8601 timestamp of generation |
| `reference_mode` | `individual` or `all` (or share ratio for Exp. 2) |
| `task_id` | Task identifier (e.g., `idea_0`, `creative_3`) |
| `category` | `idea`, `reasoning`, `summarization`, or `creative_writing` |
| `task` | Full task prompt given to agents |
| `iteration` | Iteration index (k = 0, 1, 2, ..., K) |
| `candidate` | Agent index (0--4, five agents per task) |
| `final_output` | Raw text output after the Gen-Critic-Sum workflow |

## Models and accessed dates

| Model | Provider | Accessed | Temperature |
| --- | --- | --- | --- |
| GPT-4o-mini | OpenAI | December 2024 | 1.0 |
| Claude-3.5-Haiku | Anthropic | January 2025 | 1.0 |
| Gemini-2.0-Flash-Exp | Google | January 2025 | 1.0 |
| GPT-4o | OpenAI | March 2026 | 1.0 |
| text-embedding-3-small | OpenAI | (embeddings) | — |
| all-MiniLM-L6-v2 (Sentence-BERT) | Local | (robustness check) | — |

## Task set (20 open-ended tasks)

### Idea generation
1. Generate 5 novel product ideas that help reduce food waste at home.
2. Generate 5 new interaction concepts for supporting remote teamwork.
3. Generate 5 services that use AI to support elderly people living alone.
4. Generate 5 ideas for playful urban installations using light and sound.
5. Generate 5 ideas for improving the experience of public transportation.

### Reasoning / analysis
6. Explain why traffic congestion occurs in large cities and propose 3 countermeasures.
7. Analyze the trade-offs of remote work vs. in-person work for a software team.
8. Explain the main causes of climate change and propose realistic mitigation steps.
9. Compare subscription-based and one-time purchase business models.
10. Analyze risks and benefits of using AI chatbots in customer support.

### Summarization
11. Summarize the key challenges of AI ethics in autonomous driving.
12. Summarize main usability issues in mobile banking applications.
13. Summarize the advantages and disadvantages of online education.
14. Summarize the key properties of human-centered design.
15. Summarize typical barriers to adopting new technologies in organizations.

### Creative writing
16. Write a short story about a city where AI agents and humans co-create art.
17. Write a dialogue between two AI agents arguing about creativity.
18. Write a short story about a future classroom using AI tutors.
19. Write a short story about a day in the life of an AI facilitator.
20. Write a short story about a researcher studying AI homogenization.

## Citation

A manuscript describing this work is currently in preparation for submission. Until the paper is published, please cite this repository via its Zenodo DOI:

```bibtex
@software{muto2026groupthink_code,
  title  = {Algorithmic Groupthink: Code and Data for Measuring and
            Mitigating Semantic Convergence in Multi-agent LLM Systems},
  author = {Muto, Hideki and Ogi, Tetsuro},
  year   = {2026},
  doi    = {10.5281/zenodo.19676464},
  url    = {https://doi.org/10.5281/zenodo.19676464}
}
```

The accompanying manuscript can be referenced as:

```bibtex
@misc{muto2026groupthink,
  title  = {Algorithmic Groupthink: Measuring and Mitigating Semantic
            Convergence in Multi-agent LLM Systems},
  author = {Muto, Hideki and Ogi, Tetsuro},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

This section will be updated with the full journal reference once the paper is published.

## License

- Code: MIT License (see `LICENSE`)
- Data: CC-BY 4.0 (see `LICENSE-DATA`)

## Contact

Hideki Muto (muto@keio.jp)
Graduate School of System Design and Management, Keio University

