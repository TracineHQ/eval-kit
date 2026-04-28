"""eval_kit.stats -- Statistical confidence for LLM evaluation.

LLM scores are non-deterministic. The same prompt, same model, same input
produces different scores every run. This module tells you whether a change
is real or noise.

Built for rubric-based LLM scoring systems where you're comparing baseline
vs variant runs and need to know: did this actually help?

What's in here:
  - Descriptive stats with 95% CI (t-distribution, Bessel's correction)
  - Welch's t-test (handles unequal variance between runs)
  - Glass's delta effect size (uses baseline SD as denominator)
  - Power analysis (how many runs do you need to detect a given shift?)
  - P-value computation (exact via scipy.stats.t.sf)

This is the Python binding. The JS binding at ../../../js/ implements the
same function surface without scipy. CI cross-validates the two so every
release of either is proven equivalent to the reference implementation
(scipy) within a documented tolerance.

Companion article: "How I Built an LLM Eval Framework That Caught a
39-Point Regression" -- [LinkedIn article link]

Author: Anthony Ledesma
Copyright 2026 TracineHQ
Licensed under the Apache License, Version 2.0.
See LICENSE and NOTICE files at repository root.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, sqrt
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sp

# Power analysis constants
Z_95 = 1.96  # two-tailed z-critical for alpha=0.05
Z_POWER_80 = 0.84  # z-critical for 80% statistical power


@dataclass
class Stats:
    """Descriptive statistics with 95% CI."""

    n: int
    mean: float
    std: float
    cv: float
    min: float
    max: float
    range: float
    se: float
    ci_lo: float
    ci_hi: float
    ci_margin: float


@dataclass
class WelchResult:
    """Welch's t-test result with Glass's delta effect size."""

    t: float
    df: float
    p: float
    diff: float
    se: float
    glass_d: float
    baseline_std: float


def t_critical(df: float) -> float:
    """t-critical value for 95% CI (two-tailed, alpha=0.05).

    Uses scipy's inverse t-CDF. Exact for all df.
    """
    if df <= 0:
        return 0.0
    return float(sp.t.ppf(0.975, df))


def descriptive_stats(values: Sequence[float]) -> Optional[Stats]:
    """Descriptive statistics with 95% confidence interval.

    Uses Bessel-corrected sample variance (ddof=1). Needs at least 2
    observations for a meaningful CI; returns None for empty input.
    """
    n = len(values)
    if n == 0:
        return None
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    cv = (std / abs(mean)) * 100 if mean != 0 else 0.0
    se = std / sqrt(n) if n > 1 else 0.0
    t = t_critical(n - 1) if n > 1 else 0.0
    margin = t * se
    return Stats(
        n=n,
        mean=mean,
        std=std,
        cv=cv,
        min=float(arr.min()),
        max=float(arr.max()),
        range=float(arr.max() - arr.min()),
        se=se,
        ci_lo=mean - margin,
        ci_hi=mean + margin,
        ci_margin=margin,
    )


def approx_p_value(abs_t: float, df: float) -> float:
    """Two-tailed p-value from a t-statistic. Exact via scipy.

    Named `approx_p_value` for cross-language API parity with the JS
    binding, which does approximate p-values for small df. In Python,
    scipy gives us the exact t-distribution survival function at all df,
    so this is not actually approximate.
    """
    return float(2.0 * sp.t.sf(abs_t, df))


def welch_t_test(
    a: Sequence[float], b: Sequence[float]
) -> Optional[WelchResult]:
    """Welch's t-test (unequal variance, two-sample).

    Uses scipy.stats.ttest_ind(equal_var=False) for t and p. Effect size
    is Glass's delta (baseline std as denominator), consistent with the
    unequal-variance assumption.

    Glass's delta sentinel when baseline std is 0: 99 (not inf) so the
    result is JSON-serializable. 0 when both groups are identical.
    """
    sa = descriptive_stats(a)
    sb = descriptive_stats(b)
    if sa is None or sb is None or sa.n < 2 or sb.n < 2:
        return None

    diff = sa.mean - sb.mean

    if sa.std > 0:
        glass_d = abs(diff) / sa.std
    else:
        glass_d = 0.0 if diff == 0 else 99.0

    se_a2 = sa.std ** 2 / sa.n
    se_b2 = sb.std ** 2 / sb.n
    se = sqrt(se_a2 + se_b2)

    # Zero-variance short circuit -- scipy returns nan for t and p here.
    if se == 0:
        return WelchResult(
            t=0.0 if diff == 0 else inf,
            df=sa.n + sb.n - 2,
            p=1.0 if diff == 0 else 0.0,
            diff=diff,
            se=0.0,
            glass_d=glass_d,
            baseline_std=sa.std,
        )

    result = sp.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite df (computed explicitly for reporting)
    df = (se_a2 + se_b2) ** 2 / (
        se_a2 ** 2 / (sa.n - 1) + se_b2 ** 2 / (sb.n - 1)
    )
    return WelchResult(
        t=float(result.statistic),
        df=df,
        p=float(result.pvalue),
        diff=diff,
        se=se,
        glass_d=glass_d,
        baseline_std=sa.std,
    )


def required_n(std: float, delta: float) -> float:
    """Required sample size per group to detect a given effect size.

    Two-sample test, alpha=0.05 (two-tailed), power=0.80.

    Limitation: assumes equal variance across both groups. When the
    variant run has higher variance than baseline, this underestimates
    the required N. For a 2x variance ratio, true required N is roughly
    2.5x what this returns. If your variants drift in variance (common
    when prompts change), treat the output as a lower bound and plan for
    more runs.
    """
    if delta == 0:
        return inf
    # Factor of 2 for two-sample comparison (need variance from both groups)
    return ceil(2 * (Z_95 + Z_POWER_80) ** 2 * std ** 2 / delta ** 2)
