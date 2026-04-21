# Reproducibility notes

This document explains how to reproduce the numerical results reported in the paper.

## Sources of non-determinism

Even with a fixed random seed, exact reproduction of the text outputs is not possible:

1. **API non-determinism.** The OpenAI, Anthropic, and Google APIs do not guarantee bit-exact reproducibility. Even with `seed` set, responses can vary slightly across runs. The OpenAI API returns a `system_fingerprint` indicating the underlying serving configuration; our experiments were run with whichever fingerprint was active on the access dates listed in `README.md`.
2. **Embedding model updates.** OpenAI's `text-embedding-3-small` is subject to server-side versioning. We cached all embeddings at the time of each experiment (available in the Zenodo archive).
3. **Sampling at temperature 1.0.** At the temperature used throughout the paper, stochastic sampling of the next token is the dominant source of variation.

## What is reproducible

Despite the above, the following are reproducible:

- **Aggregate statistics** (mean diversity, standard deviation, Cohen's $d$, Welch's $t$ values, ANOVA) match within a small range across independent runs.
- **Direction of effects**: individual-history increases diversity; group-history decreases it. This direction is robust across runs, models, and embedding choices.
- **Order of magnitude** of effect sizes is stable.

## Exact reproduction of reported numbers

To reproduce the exact numbers in the paper (not just the direction):

1. Use the CSV files in `data/raw_outputs/`. These contain the exact text outputs generated during our experiments.
2. Use the embeddings archived on Zenodo (see `README.md`).
3. Run the analysis scripts in `analysis/` on these fixed inputs.

In other words, re-running the experiments will give statistically consistent but not bit-identical results. Re-running the *analysis* on our archived outputs will give bit-identical results.

## Random seeds

We used `seed=42` throughout. This is passed to the API call (where supported) and to NumPy. Scripts in `experiments/` specify this explicitly.

## Compute environment

All experiments were run on a single Mac workstation with Python 3.11 and the package versions in `requirements.txt`. No GPU was needed (the API providers handle inference). A single full experiment (e.g., Experiment 1 with 20 tasks) completes within 1--2 hours on a stable network connection.

## API costs

Approximate total API cost for reproducing all experiments in the paper:

| Experiment | Estimated cost (USD) |
| --- | --- |
| Experiment 1 (20 tasks, 2 modes) | $3--5 |
| Experiment 2 (share ratios 0--0.75) | $2--3 |
| Experiment 3 (countermeasures) | $2--3 |
| Control experiments (3) | $2--3 |
| Cross-model (Claude + Gemini) | $5--10 |
| GPT-4o replication | $15--25 |
| Expanded Exp 3 replication | $3--5 |
| Quality evaluation (LLM-as-judge) | $1--2 |
| **Total** | **$30--55** |

Pricing as of early 2026; actual costs will vary.
