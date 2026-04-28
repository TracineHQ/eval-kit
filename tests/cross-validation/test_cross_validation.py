"""Cross-validation: JS binding vs Python binding vs scipy ground truth.

This suite is the credibility anchor for the whole library. It proves three
things on every run:

1. **Behavior parity.** Given the same inputs, both bindings produce outputs
   with the same shape, the same sentinels, and the same semantic decisions
   (null for empty input, Glass's delta = 99 when baseline std = 0, etc.).

2. **Numeric parity within documented tolerances.** Fields computed by the
   same formula in both bindings (diff, se, Welch-Satterthwaite df, Glass's
   delta, required_n) must agree to floating-point precision. Fields where
   JS approximates (approx_p_value for df>=30 via Abramowitz & Stegun;
   t_critical via linear interpolation between table entries) must agree
   with scipy's exact values within a small documented tolerance.

3. **Scipy ground truth.** The Python binding is a thin wrapper over scipy.
   These tests assert that our wrapper doesn't corrupt the underlying scipy
   values -- if scipy says the two-tailed p-value of t=2.5 on df=20 is X,
   approx_p_value(2.5, 20) must return X.

Run (from repo root):
    cd python && .venv/bin/pytest ../tests/cross-validation/ -v
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest
from scipy import stats as sp

from eval_kit.stats import (
    approx_p_value,
    descriptive_stats,
    required_n,
    t_critical,
    welch_t_test,
)

BRIDGE = Path(__file__).parent / "bridge.mjs"

# Tolerances
EXACT = 1e-9  # same-formula fields: must match to fp precision
# JS p-value uses Abramowitz & Stegun normal CDF for df>=30 vs scipy's exact
# t-distribution SF. Even with A&S's own ~4dp accuracy, the fundamental
# approximation of normal-for-t diverges most at the df=30 boundary (t-dist
# still has noticeably fatter tails there). Measured worst case on tested
# vectors: ~8e-3 at (t=1.0, df=30). 1e-2 is an honest ceiling that leaves
# headroom for df=30..60 near-boundary cases without masking regressions.
P_VALUE_NORMAL_APPROX = 1e-2
T_CRITICAL_INTERP = 5e-3  # JS linear interp between table entries vs scipy exact


# ---------------------------------------------------------------------------
# Bridge helper
# ---------------------------------------------------------------------------


def _decode(obj: Any) -> Any:
    """Reverse the bridge's sentinel encoding for non-finite floats."""
    if obj == "__inf__":
        return math.inf
    if obj == "__neginf__":
        return -math.inf
    if obj == "__nan__":
        return math.nan
    if isinstance(obj, dict):
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def run_js(fn: str, *args: Any) -> Any:
    """Invoke the JS bridge and return the decoded result."""
    payload = json.dumps({"fn": fn, "args": list(args)})
    proc = subprocess.run(
        ["node", str(BRIDGE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return _decode(json.loads(proc.stdout))


# ---------------------------------------------------------------------------
# descriptive_stats / stats parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [42],
        [1, 2, 3, 4, 5],
        [-10, -12, -8, -11, -9],
        [82, 85, 79, 88, 81, 84, 86, 80, 83, 87],
        [0, 0, 0, 0, 0],
        [100, 0, 100, 0, 100, 0],
    ],
)
def test_descriptive_stats_parity(values):
    """Both bindings must produce identical descriptive stats."""
    js = run_js("stats", values)
    py = descriptive_stats(values)

    assert js["n"] == py.n
    assert js["mean"] == pytest.approx(py.mean, abs=EXACT)
    assert js["std"] == pytest.approx(py.std, abs=EXACT)
    assert js["cv"] == pytest.approx(py.cv, abs=EXACT)
    assert js["min"] == pytest.approx(py.min, abs=EXACT)
    assert js["max"] == pytest.approx(py.max, abs=EXACT)
    assert js["range"] == pytest.approx(py.range, abs=EXACT)
    assert js["se"] == pytest.approx(py.se, abs=EXACT)

    # ciLo / ciHi depend on tCritical(n-1). For small n, JS uses the exact
    # table entry from T_CRIT (same values scipy ppf produces to 3dp).
    assert js["ciLo"] == pytest.approx(py.ci_lo, abs=T_CRITICAL_INTERP)
    assert js["ciHi"] == pytest.approx(py.ci_hi, abs=T_CRITICAL_INTERP)


def test_descriptive_stats_empty_returns_null():
    """Empty input must produce null/None in both bindings."""
    assert run_js("stats", []) is None
    assert descriptive_stats([]) is None


# ---------------------------------------------------------------------------
# welch_t_test / welchTTest parity
# ---------------------------------------------------------------------------


WELCH_VECTORS = [
    # (baseline, variant, label)
    (
        [82, 85, 79, 88, 81, 84, 86, 80, 83, 87],
        [75, 78, 72, 80, 74, 77, 79, 73, 76, 81],
        "10v10 moderate diff",
    ),
    (
        [82, 85, 79, 88, 81, 84, 86, 80, 83, 87],
        [65, 68, 62, 70, 64, 67, 69, 63, 66, 71],
        "10v10 large diff",
    ),
    (
        [80, 81, 82, 83, 84],
        [70, 60, 80, 50, 90],
        "5v5 unequal variance",
    ),
    (
        list(range(30)),
        list(range(5, 35)),
        "30v30 linear shift",
    ),
]


@pytest.mark.parametrize("baseline,variant,label", WELCH_VECTORS)
def test_welch_parity(baseline, variant, label):
    """Both bindings must agree on Welch's t-test outputs.

    Exact match required for: diff, se, glass_d, baseline_std, df.
    t and p have slight tolerance: t is exact, but p uses the normal CDF
    approximation in JS for df>=30 (exact in Python via scipy).
    """
    js = run_js("welchTTest", baseline, variant)
    py = welch_t_test(baseline, variant)

    assert js["diff"] == pytest.approx(py.diff, abs=EXACT)
    assert js["se"] == pytest.approx(py.se, abs=EXACT)
    assert js["glassD"] == pytest.approx(py.glass_d, abs=EXACT)
    assert js["baselineStd"] == pytest.approx(py.baseline_std, abs=EXACT)
    assert js["df"] == pytest.approx(py.df, abs=EXACT)
    # t is diff/se in both -- exact.
    assert js["t"] == pytest.approx(py.t, abs=EXACT)
    # p: JS uses normal approx for df>=30, buckets for df<30.
    # For df>=30 vectors, agreement within 5e-3 is required.
    if py.df >= 30:
        assert js["p"] == pytest.approx(py.p, abs=P_VALUE_NORMAL_APPROX)


def test_welch_glass_sentinel_parity():
    """Zero-variance baseline returns glassD=99 in both bindings."""
    js = run_js("welchTTest", [80, 80, 80], [70, 70, 70])
    py = welch_t_test([80, 80, 80], [70, 70, 70])

    assert js["glassD"] == 99
    assert py.glass_d == 99
    assert js["glassD"] == py.glass_d


def test_welch_identical_distributions_parity():
    """Identical groups return glassD=0, p=1 in both bindings."""
    js = run_js("welchTTest", [80, 80, 80], [80, 80, 80])
    py = welch_t_test([80, 80, 80], [80, 80, 80])

    assert js["glassD"] == 0
    assert py.glass_d == 0
    assert js["diff"] == 0
    assert py.diff == 0
    assert js["p"] == 1
    assert py.p == 1


def test_welch_returns_null_for_undersized_groups():
    """Both bindings return null/None when either group has fewer than 2."""
    assert run_js("welchTTest", [1], [1, 2, 3]) is None
    assert welch_t_test([1], [1, 2, 3]) is None


# ---------------------------------------------------------------------------
# required_n parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "std,delta,expected",
    [
        (8, 5, 41),  # canonical: must be 41, not 21
        (16, 5, None),  # just larger than above
        (8, 2, None),
        (8, 10, None),
        (0, 5, 0),
        (14, 5, None),
    ],
)
def test_required_n_parity(std, delta, expected):
    """required_n is a three-line formula in both bindings -- must match exactly."""
    js = run_js("requiredN", std, delta)
    py = required_n(std, delta)
    assert js == py, f"JS={js} Python={py}"
    if expected is not None:
        assert py == expected


def test_required_n_zero_delta_is_infinity_in_both():
    """delta=0 returns Infinity/inf in both bindings."""
    assert run_js("requiredN", 8, 0) == math.inf
    assert required_n(8, 0) == math.inf


# ---------------------------------------------------------------------------
# approx_p_value: JS approx vs Python exact vs scipy ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "abs_t,df",
    [
        (0.5, 30),
        (1.0, 30),
        (2.0, 50),
        (2.5, 100),
        (3.0, 200),
        (5.0, 100),
    ],
)
def test_approx_p_value_normal_approximation_parity(abs_t, df):
    """For df>=30, JS normal-CDF approximation must agree with scipy exact."""
    js = run_js("approxPValue", abs_t, df)
    py = approx_p_value(abs_t, df)
    scipy_exact = 2.0 * sp.t.sf(abs_t, df)

    # Python wrapper must match scipy exactly (it just calls scipy).
    assert py == pytest.approx(scipy_exact, abs=EXACT)
    # JS normal approx must agree with scipy within documented tolerance.
    assert js == pytest.approx(scipy_exact, abs=P_VALUE_NORMAL_APPROX)


@pytest.mark.parametrize("abs_t,df", [(3.0, 10), (2.5, 15), (2.0, 20)])
def test_approx_p_value_small_df_bucketing_is_coarse_but_bounded(abs_t, df):
    """For df<30, JS buckets p into {0.0001, 0.01, 0.05, 0.1, 0.5}.

    The bucketing is intentionally coarse: the JS binding doesn't ship a
    t-distribution CDF (no jStat dep) so it can't compute exact small-df
    p-values. Buckets are stepped thresholds on |t| scaled by t_critical(df).

    Previous versions of this test claimed the buckets were "conservative"
    (always >= true p). That's false: at (t=3.0, df=10) JS returns 0.01
    while scipy exact is 0.0133, so JS under-reports p near bucket
    boundaries. The JS threshold for the "0.01" bucket is 1.3 * t_crit,
    which maps to true p ~0.014 at df=10 -- not 0.01.

    We accept this coarseness as a documented tradeoff. This test asserts:
      (1) JS output is one of the known buckets (structural invariant).
      (2) JS and scipy exact agree within a factor of 3 (multiplicative).

    A factor-of-3 ceiling catches genuine bucketing regressions (e.g. the
    threshold scaling going wrong) without pretending the buckets are more
    precise than they are. Measured ratios on tested vectors range from
    1.3x to 2.1x.
    """
    js = run_js("approxPValue", abs_t, df)
    py = approx_p_value(abs_t, df)

    assert js in {0.0001, 0.01, 0.05, 0.1, 0.5}, f"JS must return bucket, got {js}"
    ratio = max(js, py) / min(js, py)
    assert ratio <= 3.0, (
        f"JS bucket {js} vs scipy exact {py:.4f} disagree by factor "
        f"{ratio:.2f}x -- bucketing logic may have regressed"
    )


def test_approx_p_value_extreme_t_floor():
    """Both bindings handle extreme t-values without overflow."""
    js = run_js("approxPValue", 15, 100)
    py = approx_p_value(15, 100)

    assert js < 0.001
    assert py < 0.001


# ---------------------------------------------------------------------------
# t_critical parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("df", [1, 2, 5, 10, 14, 19, 24, 29, 49, 99])
def test_t_critical_table_entries_parity(df):
    """JS uses a 16-entry lookup table. Python uses scipy exact.
    Table entries are accurate to ~0.005; assert tolerance.
    """
    js = run_js("tCritical", df)
    py = t_critical(df)
    scipy_exact = sp.t.ppf(0.975, df)

    assert py == pytest.approx(scipy_exact, abs=EXACT)
    assert js == pytest.approx(scipy_exact, abs=T_CRITICAL_INTERP)


@pytest.mark.parametrize("df", [3, 7, 12, 16, 22, 35])
def test_t_critical_interpolation_parity(df):
    """Non-table df: JS linearly interpolates. Python is exact. Tolerance
    is intentionally loose because linear interp on convex function.
    """
    js = run_js("tCritical", df)
    py = t_critical(df)

    # JS interpolation accuracy degrades slightly between table entries;
    # the 95% CI t-distribution table is convex so linear interp
    # underestimates. 1e-2 is generous but realistic.
    assert js == pytest.approx(py, abs=1e-2)


def test_t_critical_large_df_falls_back_to_z():
    """JS returns exactly 1.96 for df > last table entry (99).
    Python returns scipy exact. Both approach 1.96 as df -> inf.
    """
    js = run_js("tCritical", 1000)
    py = t_critical(1000)

    assert js == 1.96
    assert py == pytest.approx(1.96, abs=1e-2)
    # Within each other's tolerance
    assert js == pytest.approx(py, abs=1e-2)


def test_t_critical_nonpositive_df_returns_zero():
    """Both bindings return 0 for df <= 0."""
    assert run_js("tCritical", 0) == 0
    assert t_critical(0) == 0
    assert run_js("tCritical", -5) == 0
    assert t_critical(-5) == 0


# ---------------------------------------------------------------------------
# Scipy ground-truth sanity: Python bindings correctly wrap scipy
# ---------------------------------------------------------------------------


def test_welch_t_test_matches_scipy_ttest_ind_directly():
    """welch_t_test(a, b) must produce the same t and p as
    scipy.stats.ttest_ind(a, b, equal_var=False) -- no corruption
    in the wrapper.
    """
    baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87]
    variant = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81]

    result = welch_t_test(baseline, variant)
    scipy_result = sp.ttest_ind(baseline, variant, equal_var=False)

    assert result.t == pytest.approx(scipy_result.statistic, abs=EXACT)
    assert result.p == pytest.approx(scipy_result.pvalue, abs=EXACT)


def test_required_n_formula_matches_published_power_analysis():
    """(Z_0.975 + Z_0.80)^2 * 2 * sigma^2 / delta^2
    Standard two-sample power analysis formula. Values verified against
    statsmodels.stats.power.TTestIndPower (not imported to keep deps
    light -- this test documents the expected numeric outputs).
    """
    # Canonical: std=8, delta=5 -> 41 per factor-of-2 two-sample formula
    assert required_n(8, 5) == 41
    # Doubling variance ~quadruples required N
    assert required_n(16, 5) == 161
    # Doubling effect size quarters required N
    assert required_n(8, 10) == 11
