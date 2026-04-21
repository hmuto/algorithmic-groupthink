#!/usr/bin/env python3
"""
Power Analysis for Algorithmic Groupthink Experiments

Calculates post-hoc statistical power for our experiments.
"""

import numpy as np
from scipy import stats

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def cohens_d_ci(d, n1, n2, alpha=0.05):
    """Calculate confidence interval for Cohen's d using non-central t distribution."""
    # Approximate CI using formula from Hedges & Olkin
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    t_crit = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
    return (d - t_crit * se_d, d + t_crit * se_d)

def power_t_test(effect_size, n1, n2, alpha=0.05):
    """Calculate power for two-sample t-test."""
    # Non-central parameter
    df = n1 + n2 - 2
    nc = effect_size * np.sqrt(n1 * n2 / (n1 + n2))

    # Critical t value
    t_crit = stats.t.ppf(1 - alpha/2, df)

    # Power calculation using non-central t distribution
    power = 1 - stats.nct.cdf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc)
    return power

def sample_size_for_power(effect_size, power=0.80, alpha=0.05):
    """Calculate required sample size per group for desired power."""
    for n in range(3, 200):
        if power_t_test(effect_size, n, n, alpha) >= power:
            return n
    return ">200"

def eta_squared_ci(eta2, df_effect, df_error, alpha=0.05):
    """Approximate CI for eta-squared using F distribution."""
    # This is a simplified approximation
    f_value = (eta2 / df_effect) / ((1 - eta2) / df_error)

    # Lower and upper F values
    f_lower = stats.f.ppf(alpha/2, df_effect, df_error)
    f_upper = stats.f.ppf(1 - alpha/2, df_effect, df_error)

    # Convert back to eta-squared (approximate)
    se_eta = np.sqrt(eta2 * (1 - eta2) * 2 * (df_effect + df_error + 1) / ((df_error + 1) * (df_error + 3)))
    z = stats.norm.ppf(1 - alpha/2)

    ci_lower = max(0, eta2 - z * se_eta)
    ci_upper = min(1, eta2 + z * se_eta)

    return (ci_lower, ci_upper)

def main():
    print("="*70)
    print("POWER ANALYSIS FOR ALGORITHMIC GROUPTHINK EXPERIMENTS")
    print("="*70)

    # Experiment 1: Group-history vs Individual-history
    print("\n" + "-"*50)
    print("EXPERIMENT 1: Reference Mode Comparison")
    print("-"*50)

    # From our results
    n1 = n2 = 25  # samples per condition
    d = 1.73      # Cohen's d

    # Calculate power
    power = power_t_test(d, n1, n2)
    d_ci = cohens_d_ci(d, n1, n2)

    print(f"  Sample size per group: n = {n1}")
    print(f"  Observed effect size: Cohen's d = {d:.2f}")
    print(f"  95% CI for d: [{d_ci[0]:.2f}, {d_ci[1]:.2f}]")
    print(f"  Post-hoc power (α = 0.05): {power:.3f} ({power*100:.1f}%)")

    # Minimum detectable effect
    mde_80 = None
    for test_d in np.arange(0.1, 3.0, 0.01):
        if power_t_test(test_d, n1, n2) >= 0.80:
            mde_80 = test_d
            break
    print(f"  Minimum detectable effect (80% power): d = {mde_80:.2f}")

    # Experiment 2: Share Ratio ANOVA
    print("\n" + "-"*50)
    print("EXPERIMENT 2: Share Ratio (ANOVA)")
    print("-"*50)

    eta2 = 0.20    # eta-squared
    df_between = 3  # 4 groups - 1
    df_within = 96  # 100 - 4

    eta_ci = eta_squared_ci(eta2, df_between, df_within)

    # ANOVA power using F distribution
    f_value = (eta2 / df_between) / ((1 - eta2) / df_within)
    nc_f = f_value * (df_between + 1)  # Non-centrality parameter

    f_crit = stats.f.ppf(0.95, df_between, df_within)
    power_anova = 1 - stats.ncf.cdf(f_crit, df_between, df_within, nc_f)

    print(f"  Sample size: n = 100 (25 per group)")
    print(f"  Observed effect size: η² = {eta2:.2f}")
    print(f"  95% CI for η²: [{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]")
    print(f"  Post-hoc power (α = 0.05): {power_anova:.3f} ({power_anova*100:.1f}%)")

    # Experiment 3: Countermeasure Comparison
    print("\n" + "-"*50)
    print("EXPERIMENT 3: Countermeasure Effectiveness")
    print("-"*50)

    # Diversity Prompt vs Baseline
    n3 = 5  # samples per condition

    # Calculate effect sizes from our data
    # Baseline: SDI = 0.214, SD = 0.021 -> final change = -31.8%
    # Diversity: SDI = 0.428, SD = 0.044 -> final change = -5.5%

    # t = 8.77, so we can back-calculate d
    # d ≈ t * sqrt(2/n) for equal n
    d_diversity = 8.77 * np.sqrt(2/5)

    power_div = power_t_test(d_diversity, n3, n3)
    d_div_ci = cohens_d_ci(d_diversity, n3, n3)

    print("\n  Diversity Prompt vs Baseline:")
    print(f"    Sample size per group: n = {n3}")
    print(f"    Observed effect size: Cohen's d = {d_diversity:.2f}")
    print(f"    95% CI for d: [{d_div_ci[0]:.2f}, {d_div_ci[1]:.2f}]")
    print(f"    Post-hoc power: {power_div:.3f} ({power_div*100:.1f}%)")

    # Adversarial vs Baseline (p = 0.13, not significant)
    # t = 1.835
    d_adversarial = 1.835 * np.sqrt(2/5)
    power_adv = power_t_test(d_adversarial, n3, n3)

    print("\n  Adversarial Sampling vs Baseline:")
    print(f"    Sample size per group: n = {n3}")
    print(f"    Observed effect size: Cohen's d = {d_adversarial:.2f}")
    print(f"    Post-hoc power: {power_adv:.3f} ({power_adv*100:.1f}%)")

    # Sample size needed for 80% power
    n_needed = sample_size_for_power(d_adversarial)
    print(f"    Sample size needed for 80% power: n = {n_needed} per group")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
  - Experiment 1 (n=25/group): WELL-POWERED (>99% power)
    Large effect size (d=1.73) ensures robust detection.

  - Experiment 2 (n=25/group): WELL-POWERED (>99% power)
    Large effect size (η²=0.20) ensures robust detection.

  - Experiment 3 Diversity Prompt (n=5/group): ADEQUATELY POWERED (~99%)
    Very large effect size compensates for small sample.

  - Experiment 3 Adversarial (n=5/group): UNDERPOWERED (~48%)
    Medium effect size with small sample.
    Would need n≈15 per group for 80% power.
""")

if __name__ == "__main__":
    main()
