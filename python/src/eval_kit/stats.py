"""eval_kit.stats -- Statistical confidence for LLM evaluation.

LLM scores are non-deterministic. The same prompt, same model, same input
produces different scores every run. This module tells you whether a change
is real or noise.

Built for rubric-based LLM evals where you're comparing baseline vs variant
runs and need to know: did this actually help?

What's in here:
  - Descriptive stats with 95% CI (t-distribution, Bessel's correction)
  - Welch's t-test (independent runs, unequal variance)
  - Paired t-test (same items scored in both arms)
  - Glass's delta and Cohen's dz effect sizes
  - Power analysis (how many runs or items do you need?)
  - Exact p-values and critical values via scipy.stats.t
  - verdict(): improved / regressed / no_change / cant_tell

This is the Python binding. The JS binding at ../../../js/ implements the
same function surface without scipy. CI cross-validates the two against
the reference implementation (scipy) to 1e-9.

Author: Anthony Ledesma
Copyright 2026 TracineHQ
Licensed under the Apache License, Version 2.0.
See LICENSE and NOTICE files at repository root.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil, hypot, inf, isfinite, sqrt
from typing import Literal

import numpy as np
from scipy import stats as sp

# z-critical values, used only to seed the exact sample-size search.
Z_95 = 1.959963984540054  # two-tailed, alpha = 0.05
Z_POWER_80 = 0.8416212335729143  # one-tailed, power = 0.80
EXACT_N_LIMIT = 1e8
TARGET_POWER = 0.8


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


@dataclass
class VerdictResult:
    """The decision from verdict(). shift is variant minus baseline."""

    verdict: Literal["improved", "regressed", "no_change", "cant_tell"]
    reason: Literal["significant", "below_target", "within_target", "underpowered", "zero_spread"]
    design: Literal["welch", "paired"]
    target: float
    lower_is_better: bool
    shift: float
    ci_lo: float
    ci_hi: float
    p: float
    effect: float
    effect_name: Literal["glassD", "dz"]
    n_baseline: int
    n_variant: int
    needed: float | None
    constant_arm: Literal["baseline", "variant"] | None = None


@dataclass
class PairedResult:
    """Paired t-test result with Cohen's dz effect size."""

    t: float
    df: float
    p: float
    diff: float
    se: float
    dz: float
    sd_diff: float
    n: int


def t_critical(df: float) -> float:
    """t-critical value for 95% CI (two-tailed, alpha=0.05).

    Uses scipy's inverse t-CDF. Exact for all df.
    """
    if df <= 0:
        return 0.0
    return float(sp.t.ppf(0.975, df))


def descriptive_stats(values: Sequence[float]) -> Stats | None:
    """Descriptive statistics with 95% confidence interval.

    Uses Bessel-corrected sample variance (ddof=1). At n = 1, std, se and
    ci_margin are 0 by construction (no spread estimate), not a claim of
    precision; the t-tests refuse n < 2 for this reason. Returns None for
    empty input.
    """
    n = len(values)
    if n == 0:
        return None
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    # Identical values have zero spread, up to floating-point dust (0.1 + 0.2
    # vs 0.3); don't let rounding say otherwise.
    lo, hi = float(arr.min()), float(arr.max())
    flat = hi - lo <= 8 * np.finfo(float).eps * max(abs(lo), abs(hi))
    std = float(arr.std(ddof=1)) if n > 1 and not flat else 0.0
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


def p_value(abs_t: float, df: float) -> float:
    """Two-tailed p-value from a t-statistic. Exact via scipy."""
    return float(2.0 * sp.t.sf(abs_t, df))


def approx_p_value(abs_t: float, df: float) -> float:
    """Alias of p_value, kept for compatibility with 0.0.x."""
    return p_value(abs_t, df)


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> WelchResult | None:
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

    # Squared standard errors in units of the larger std, so spreads near
    # 1e-100 or 1e+100 don't underflow or overflow when squared again for df.
    scale = max(sa.std, sb.std)
    u_a = (sa.std / scale) ** 2 / sa.n if scale > 0 else 0.0
    u_b = (sb.std / scale) ** 2 / sb.n if scale > 0 else 0.0
    se = scale * sqrt(u_a + u_b)

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

    # t and p from the scaled standard errors and scipy's t distribution.
    # scipy.stats.ttest_ind squares variances directly, and near 1e-100 that
    # underflows to a silent df = 1; the cross-validation suite still holds
    # this to ttest_ind on ordinary data.
    df = (u_a + u_b) ** 2 / (u_a**2 / (sa.n - 1) + u_b**2 / (sb.n - 1))
    t = diff / se
    p = p_value(abs(t), df)
    return WelchResult(
        t=t,
        df=df,
        p=p,
        diff=diff,
        se=se,
        glass_d=glass_d,
        baseline_std=sa.std,
    )


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> PairedResult | None:
    """Paired t-test on per-item differences.

    Use when the same eval items are scored in both arms. Pairing cancels
    item difficulty, so it detects a shift with far fewer items than an
    unpaired test. For independent full runs, use welch_t_test.

    Uses scipy.stats.ttest_rel for t and p. Effect size is Cohen's dz
    (|mean difference| / std of differences), with sentinel 99 when every
    difference is identical and nonzero. diff is mean(a - b), so positive
    means the variant scored lower. Returns None when the sequences differ
    in length or hold fewer than 2 items.
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    d = descriptive_stats([x - y for x, y in zip(a, b, strict=True)])
    assert d is not None
    df = d.n - 1

    # Every item moved by the same amount, up to floating-point rounding in
    # a - b. scipy returns nan (or ~1e15 with a cancellation warning) here.
    scale = max(max(abs(x) for x in a), max(abs(y) for y in b))
    if d.range <= 8 * np.finfo(float).eps * scale:
        return PairedResult(
            t=0.0 if d.mean == 0 else inf,
            df=df,
            p=1.0 if d.mean == 0 else 0.0,
            diff=d.mean,
            se=0.0,
            dz=0.0 if d.mean == 0 else 99.0,
            sd_diff=0.0,
            n=d.n,
        )

    result = sp.ttest_rel(a, b)
    return PairedResult(
        t=float(result.statistic),
        df=df,
        p=float(result.pvalue),
        diff=d.mean,
        se=d.se,
        dz=abs(d.mean) / d.std,
        sd_diff=d.std,
        n=d.n,
    )


def _t_test_power(df: float, nc: float) -> float:
    """Power of a two-sided t-test at alpha = 0.05 with noncentrality nc.

    The lower tail uses nct.sf(tc, df, -nc), which equals nct.cdf(-tc, df, nc)
    but stays finite where scipy's direct cdf returns nan (large nc).
    """
    tc = t_critical(df)
    if abs(nc) > 37:
        # scipy's nct returns nan out here; use the normal approximation
        # (Abramowitz & Stegun 26.7.10), as the JS binding does. Power is ~1.
        def cdf(t: float) -> float:
            return float(sp.norm.cdf((t * (1 - 1 / (4 * df)) - nc) / sqrt(1 + t * t / (2 * df))))

        return 1 - cdf(tc) + cdf(-tc)
    return float(sp.nct.sf(tc, df, nc) + sp.nct.sf(tc, df, -nc))


def _smallest_n(guess_raw: float, test: Callable[[int], bool]) -> float:
    """Smallest n >= 2 that passes a test that stays passed as n grows (power
    reaching 0.80, a CI narrowing). Start from a z-based guess, gallop to
    bracket the exact boundary, then bisect: a few
    dozen power evaluations. Past EXACT_N_LIMIT the z guess is returned as is:
    its relative error there is under 1e-5, and the t distribution's precision
    at df ~ 1e9 and up is not worth searching on."""
    if not guess_raw <= EXACT_N_LIMIT:
        return guess_raw
    n = max(2, ceil(guess_raw))

    def passes(m: int) -> bool:
        return m >= 2 and test(m)

    step = 1
    if passes(n):
        hi = n
        while True:
            m = n - step
            if m < 2:
                lo = 1
                break
            if not passes(m):
                lo = m
                break
            hi = m
            step *= 2
    else:
        lo = n
        while True:
            m = n + step
            if m > EXACT_N_LIMIT:
                return inf  # never passes in range (nan input)
            if passes(m):
                hi = m
                break
            lo = m
            step *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes(mid):
            hi = mid
        else:
            lo = mid
    return hi


def required_n(std: float, delta: float) -> float:
    """Required runs per arm to detect a given difference in means.

    Two-sample t-test, alpha=0.05 (two-tailed), power=0.80, exact: the
    smallest n whose power under the noncentral t-distribution reaches 0.80.
    Matches statsmodels TTestIndPower and Cohen's tables (d=0.5 needs 64).

    Assumes both arms share the baseline's spread. If the variant's variance
    is r times the baseline's, the true answer is roughly (1 + r) / 2 times
    this: 1.5x for double the variance, 2.5x for double the standard
    deviation. Treat it as a floor when variants drift. Returns at least 2,
    and inf when delta is 0.
    """
    if delta == 0:
        return inf
    if std == 0:
        return 2
    d = abs(delta)
    # Ratios and plain products, not **, so tiny or huge inputs give inf
    # instead of raising.
    guess = 2 * (Z_95 + Z_POWER_80) ** 2 * (std / d) * (std / d)
    return _smallest_n(
        float(ceil(guess)) if isfinite(guess) else guess,
        lambda n: _t_test_power(2 * n - 2, d / (std * sqrt(2 / n))) >= TARGET_POWER,
    )


def required_n_paired(sd_diff: float, delta: float) -> float:
    """Required number of paired items to detect a given mean difference.

    Paired t-test, alpha=0.05 (two-tailed), power=0.80, exact under the
    noncentral t-distribution. Matches statsmodels TTestPower. sd_diff is
    the standard deviation of per-item differences from a pilot. No factor
    of 2: each item already contributes both arms. Returns at least 2, and
    inf when delta is 0.
    """
    if delta == 0:
        return inf
    if sd_diff == 0:
        return 2
    d = abs(delta)
    guess = (Z_95 + Z_POWER_80) ** 2 * (sd_diff / d) * (sd_diff / d)
    return _smallest_n(
        float(ceil(guess)) if isfinite(guess) else guess,
        lambda n: _t_test_power(n - 1, d / (sd_diff / sqrt(n))) >= TARGET_POWER,
    )


def _resolving_n(se1: float, df_of: Callable[[int], float], target: float) -> float:
    """Smallest n (runs per arm, or items) at which the 95% CI half-width, at
    the spread seen so far, is under half the target. A CI that narrow cannot
    hold both 0 and a shift of target, so the verdict has to be a call. se1 is
    the standard error at n = 1; df_of gives the degrees of freedom at n."""
    q = 2 * Z_95 * se1 / target
    guess = q * q
    return _smallest_n(
        float(ceil(guess)) if isfinite(guess) else guess,
        lambda n: t_critical(df_of(n)) * (se1 / sqrt(n)) < target / 2,
    )


def verdict(
    baseline: Sequence[float],
    variant: Sequence[float],
    target: float,
    paired: bool = False,
    lower_is_better: bool = False,
) -> VerdictResult | None:
    """The decision: did the variant move the score, and can this data say so?

    target is the smallest shift worth acting on, in score units, picked
    before looking at results. Rules, in order:
      - no spread in either arm (or in the paired differences): cant_tell,
        check the harness
      - 95% CI of the shift inside (-target, +target): no_change, even when
        p < 0.05 (reason below_target: real, but smaller than worth acting on)
      - p < 0.05: improved or regressed, by the sign of the shift
    "p < 0.05" is read off the 95% CI (it excludes 0), so the rules and the
    reported CI can never disagree at the rounding edge.
      - otherwise: cant_tell

    shift is variant minus baseline (the opposite sign of welch_t_test's
    diff). needed is the runs per arm (Welch) or items (paired) at which, at
    the spread seen so far, the 95% CI is narrower than half the target on
    each side, so it cannot span both no change and the target: every verdict
    at that n is a call. It sizes from both arms' spread and is None when
    there is no spread to size from. Past 1e8 it is the z-approximation
    (slightly low, by about 1e-8 relative); inf when the target is too small
    for the spread to size at all. For planning a run before collecting
    data, use required_n (80% power to detect a target-sized shift), which is
    smaller. Returns None for invalid input: fewer than 2 values per arm,
    mismatched lengths when paired, or a target that is not a positive number.
    """
    if not (isfinite(target) and target > 0):
        return None
    r: WelchResult | PairedResult | None = (
        paired_t_test(baseline, variant) if paired else welch_t_test(baseline, variant)
    )
    if r is None:
        return None

    shift = -r.diff
    margin = 0.0 if r.se == 0 else t_critical(r.df) * r.se
    needed: float | None
    constant_arm: Literal["baseline", "variant"] | None = None
    if isinstance(r, PairedResult):
        effect, effect_name, flat = r.dz, "dz", r.sd_diff == 0
        needed = None if flat else _resolving_n(r.sd_diff, lambda n: n - 1, target)
    else:
        variant_stats = descriptive_stats(variant)
        assert variant_stats is not None
        effect, effect_name = r.glass_d, "glassD"
        # No spread anywhere to test against: almost always a broken harness.
        # One constant arm is fine for Welch (a ceiling, or a harness to
        # check), so the verdict stands and constant_arm names it.
        flat = r.baseline_std == 0 and variant_stats.std == 0
        if not flat and r.baseline_std == 0:
            constant_arm = "baseline"
        elif not flat and variant_stats.std == 0:
            constant_arm = "variant"
        # Welch at equal n per arm: se^2 = (vA + vB) / n, Welch-Satterthwaite df.
        # (1 + r)^2 / (1 + r^2) with r the variance ratio, written as
        # 1 + 2 / (r + 1/r) so no spread, however tiny or huge, gives 0/0.
        needed = None
        if not flat:
            se1 = hypot(r.baseline_std, variant_stats.std)
            if constant_arm:
                # One constant arm: the ratio is 0 or inf, and df is n - 1.
                needed = _resolving_n(se1, lambda n: n - 1, target)
            else:
                sd_ratio = variant_stats.std / r.baseline_std
                ratio = sd_ratio * sd_ratio
                needed = _resolving_n(se1, lambda n: (n - 1) * (1 + 2 / (ratio + 1 / ratio)), target)

    significant = shift - margin > 0 or shift + margin < 0
    if flat:
        result, reason = "cant_tell", "zero_spread"
    elif shift - margin > -target and shift + margin < target:
        result = "no_change"
        reason = "below_target" if significant else "within_target"
    elif significant:
        result = "improved" if (shift > 0) != lower_is_better else "regressed"
        reason = "significant"
    else:
        result, reason = "cant_tell", "underpowered"

    return VerdictResult(
        verdict=result,  # type: ignore[arg-type]
        reason=reason,  # type: ignore[arg-type]
        design="paired" if paired else "welch",
        target=target,
        lower_is_better=lower_is_better,
        shift=shift,
        ci_lo=shift - margin,
        ci_hi=shift + margin,
        p=r.p,
        effect=effect,
        effect_name=effect_name,  # type: ignore[arg-type]
        n_baseline=len(baseline),
        n_variant=len(variant),
        needed=needed,
        constant_arm=constant_arm,
    )
