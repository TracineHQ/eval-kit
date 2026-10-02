"""Cross-validation: JS binding vs Python binding vs scipy ground truth.

This suite is the credibility anchor for the whole library. It proves three
things on every run:

1. **Behavior parity.** Given the same inputs, both bindings produce outputs
   with the same shape, the same sentinels, and the same semantic decisions
   (null for empty input, Glass's delta = 99 when baseline std = 0, etc.).

2. **Numeric parity.** Every numeric field must agree to 1e-9: descriptive
   stats, Welch and paired t-test outputs, required-N, and the p-values and
   t-critical values that JS computes itself (incomplete beta function,
   zero dependencies) and Python takes from scipy.

3. **Scipy ground truth.** The Python binding is a thin wrapper over scipy.
   These tests assert that our wrapper doesn't corrupt the underlying scipy
   values -- if scipy says the two-tailed p-value of t=2.5 on df=20 is X,
   p_value(2.5, 20) must return X.

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
    p_value,
    paired_t_test,
    required_n,
    required_n_paired,
    t_critical,
    welch_t_test,
)

BRIDGE = Path(__file__).parent / "bridge.mjs"

# Tolerance: every field must match to floating-point precision. The JS
# incomplete-beta implementation measures ~1e-12 or better against scipy.
EXACT = 1e-9


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

    assert js["ciLo"] == pytest.approx(py.ci_lo, abs=EXACT)
    assert js["ciMargin"] == pytest.approx(py.ci_margin, abs=EXACT)
    assert js["ciHi"] == pytest.approx(py.ci_hi, abs=EXACT)


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
    (
        [80.1, 82.4, 79.0, 83.5, 81.2, 80.6, 82.9, 79.8, 81.7, 80.9, 82.2, 81.4],
        [70, 60, 80, 50],
        "12v4 unequal n and variance",
    ),
]


@pytest.mark.parametrize("baseline,variant,label", WELCH_VECTORS)
def test_welch_parity(baseline, variant, label):
    """Both bindings must agree on Welch's t-test outputs.

    Exact match required on every field.
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
    assert js["p"] == pytest.approx(py.p, abs=EXACT)


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
        (8, 5, 42),  # canonical: z-approximation says 41, exact is 42
        (16, 5, 162),
        (8, 2, 253),
        (8, 10, 12),
        (0, 5, 2),
        (14, 5, None),
        (1, 0.2, 394),  # Cohen's tables
        (1, 0.5, 64),
        (1, 0.8, 26),
        (1, 100, 2),
        (8, -5, 42),
        (10, 0.3, None),  # n in the tens of thousands
        (2.5, 1.7, None),
        (16, 0.25, 64299),  # z seed overshoots, so the search steps down
        (0.03, 0.01, None),
    ],
)
def test_required_n_parity(std, delta, expected):
    """JS computes power with its own noncentral t (AS 243); Python uses
    scipy.stats.nct. Two independent implementations must return the same n."""
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
# paired t-test parity
# ---------------------------------------------------------------------------


PAIRED_VECTORS = [
    # (baseline, variant, label)
    ([80, 72, 91, 65, 77], [78, 70, 90, 62, 76], "5 items small shift"),
    (
        [40, 55, 62, 70, 78, 85, 91, 48, 66, 73],
        [42, 57, 63, 72, 80, 88, 92, 50, 69, 75],
        "10 items varied difficulty",
    ),
    ([0.2, 0.9, 0.5, 0.7], [0.4, 0.8, 0.6, 0.9], "4 items fractional scores"),
    (list(range(40)), [x + (x % 3) for x in range(40)], "40 items df>=30"),
]


@pytest.mark.parametrize("baseline,variant,label", PAIRED_VECTORS)
def test_paired_parity(baseline, variant, label):
    """Both bindings must agree on every paired t-test field."""
    js = run_js("pairedTTest", baseline, variant)
    py = paired_t_test(baseline, variant)

    for js_key, py_key in [
        ("t", "t"),
        ("df", "df"),
        ("p", "p"),
        ("diff", "diff"),
        ("se", "se"),
        ("dz", "dz"),
        ("sdDiff", "sd_diff"),
        ("n", "n"),
    ]:
        assert js[js_key] == pytest.approx(getattr(py, py_key), abs=EXACT), js_key


def test_paired_sentinels_parity():
    """Identical nonzero differences give dz=99; identical arms give dz=0."""
    js = run_js("pairedTTest", [5, 6, 7], [4, 5, 6])
    py = paired_t_test([5, 6, 7], [4, 5, 6])
    assert js["dz"] == py.dz == 99
    assert js["t"] == py.t == math.inf

    js = run_js("pairedTTest", [5, 6, 7], [5, 6, 7])
    py = paired_t_test([5, 6, 7], [5, 6, 7])
    assert js["dz"] == py.dz == 0
    assert js["p"] == py.p == 1


@pytest.mark.parametrize(
    "baseline,variant",
    [
        ([0.85, 0.65, 0.95, 0.75, 0.55], [0.8, 0.6, 0.9, 0.7, 0.5]),
        ([0.3, 0.7, 0.1], [0.4, 0.8, 0.2]),
        ([812.4, 790.1, 805.7], [812.5, 790.2, 805.8]),
    ],
)
def test_paired_sentinel_survives_floating_point_rounding(baseline, variant):
    """A constant shift on decimal scores leaves differences like 0.05 and
    0.04999999999999993. That is still every item moving by the same amount,
    so both bindings must return the sentinel, not t ~ 1e15."""
    js = run_js("pairedTTest", baseline, variant)
    py = paired_t_test(baseline, variant)
    assert js["dz"] == py.dz == 99
    assert js["sdDiff"] == py.sd_diff == 0


def test_tiny_but_real_spread_is_not_flattened():
    """The zero-spread tolerance is floating-point dust (8 ulps of scale),
    not a loose threshold that would swallow a real 1e-7 difference."""
    js = run_js("pairedTTest", [1, 2, 3, 4], [1.0000001, 2, 3, 4])
    py = paired_t_test([1, 2, 3, 4], [1.0000001, 2, 3, 4])
    assert js["sdDiff"] > 0 and py.sd_diff > 0
    assert js["dz"] != 99 and py.dz != 99
    assert run_js("stats", [80, 80, 80.00000001])["std"] > 0
    assert descriptive_stats([80, 80, 80.00000001]).std > 0


def test_constant_decimal_scores_have_zero_std():
    """mean([0.1, 0.1, 0.1]) is not exactly 0.1 in floating point; std must
    still be 0 so glassD hits its sentinel."""
    assert run_js("stats", [0.1, 0.1, 0.1])["std"] == 0
    assert descriptive_stats([0.1, 0.1, 0.1]).std == 0
    js = run_js("welchTTest", [0.1, 0.1, 0.1], [0.2, 0.2, 0.2])
    py = welch_t_test([0.1, 0.1, 0.1], [0.2, 0.2, 0.2])
    assert js["glassD"] == py.glass_d == 99


def test_paired_returns_null_for_mismatched_or_undersized_input():
    assert run_js("pairedTTest", [1, 2], [1]) is None
    assert paired_t_test([1, 2], [1]) is None
    assert run_js("pairedTTest", [1], [2]) is None
    assert paired_t_test([1], [2]) is None


@pytest.mark.parametrize(
    "sd_diff,delta,expected",
    [
        (8, 5, 23),
        (8, 8, 10),
        (8, 15, 5),
        (3, 1, 73),
        (10, 2, 199),
        (1, 0.5, 34),  # Cohen's tables
        (7.07, 2, None),
        (8, 0, math.inf),
        (0, 5, 2),
        (8, -5, 23),
        (10, 0.3, None),
    ],
)
def test_required_n_paired_parity(sd_diff, delta, expected):
    js = run_js("requiredNPaired", sd_diff, delta)
    py = required_n_paired(sd_diff, delta)
    assert js == py, f"JS={js} Python={py}"
    if expected is not None:
        assert py == expected


# ---------------------------------------------------------------------------
# p-value: JS incomplete beta vs Python vs scipy ground truth
# ---------------------------------------------------------------------------


P_VALUE_DFS = [1, 1.5, 2, 2.7, 3, 5, 7.3, 10, 15, 29, 30, 45, 100, 1000, 1e4, 1e6, 1e7]
P_VALUE_TS = [0, 1e-4, 0.1, 0.5, 1, 1.5, 1.96, 2.5, 3, 5, 10, 30, 100]


@pytest.mark.parametrize("df", P_VALUE_DFS)
def test_p_value_parity_across_t_grid(df):
    """JS, Python, and scipy agree on the two-tailed p-value across small,
    fractional, and large df. Relative tolerance with no absolute floor, so
    tail p-values far below 1e-9 are held to their own digits."""
    for abs_t in P_VALUE_TS:
        scipy_exact = 2.0 * sp.t.sf(abs_t, df)
        assert p_value(abs_t, df) == pytest.approx(scipy_exact, rel=EXACT, abs=0)
        assert run_js("pValue", abs_t, df) == pytest.approx(scipy_exact, rel=EXACT, abs=0), f"t={abs_t} df={df}"


@pytest.mark.parametrize("df", [3e7, 1e8])
def test_p_value_at_huge_df_holds_absolute_precision(df):
    """Past df = 1e7 the incomplete beta's continued fraction loses relative
    digits roughly in proportion to df; the absolute error stays under 1e-9."""
    for abs_t in P_VALUE_TS:
        scipy_exact = 2.0 * sp.t.sf(abs_t, df)
        assert run_js("pValue", abs_t, df) == pytest.approx(scipy_exact, abs=EXACT)


@pytest.mark.parametrize("abs_t", [1e-8, 1e-12, 0.3, 4.0])
def test_js_p_value_matches_closed_forms_at_df_1_and_2(abs_t):
    """df = 1 (Cauchy) and df = 2 have closed forms, a ground truth that does
    not depend on scipy. At t = 1e-8, df = 1, scipy's t.sf is itself off by
    3e-9; the JS incomplete beta matches the closed form."""
    cauchy = 1 - 2 * math.atan(abs_t) / math.pi
    df2 = 1 - abs_t / math.sqrt(2 + abs_t * abs_t)
    assert run_js("pValue", abs_t, 1) == pytest.approx(cauchy, rel=1e-12)
    assert run_js("pValue", abs_t, 2) == pytest.approx(df2, rel=1e-12)


def test_approx_p_value_alias_parity():
    assert run_js("approxPValue", 2.5, 7.3) == pytest.approx(approx_p_value(2.5, 7.3), abs=EXACT)


# ---------------------------------------------------------------------------
# t_critical parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("df", [0.5, 1, 1.5, 2, 3, 5, 7, 12, 22, 29.4, 30, 35, 99, 100, 1000, 1e4, 1e6])
def test_t_critical_parity(df):
    """JS inverts its own t CDF; Python uses scipy ppf. Both must match."""
    scipy_exact = sp.t.ppf(0.975, df)
    assert t_critical(df) == pytest.approx(scipy_exact, abs=EXACT)
    assert run_js("tCritical", df) == pytest.approx(scipy_exact, abs=EXACT)


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


@pytest.mark.parametrize(
    "std,delta,paired",
    [
        (8, 5, False),
        (16, 5, False),
        (8, 2, False),
        (1, 0.2, False),
        (8, 5, True),
        (3, 1, True),
        (10, 2, True),
        (1, 0.5, True),
        (16, 0.25, False),
    ],
)
def test_required_n_is_the_power_boundary(std, delta, paired):
    """At the returned n, power under the noncentral t (scipy) reaches 0.80;
    at n - 1 it does not. Checks the search, not just the sample values."""

    def power(n):
        df = n - 1 if paired else 2 * n - 2
        nc = delta / (std / math.sqrt(n)) if paired else delta / (std * math.sqrt(2 / n))
        tc = sp.t.ppf(0.975, df)
        return sp.nct.sf(tc, df, nc) + sp.nct.sf(tc, df, -nc)

    n = required_n_paired(std, delta) if paired else required_n(std, delta)
    assert power(n) >= 0.80
    assert power(n - 1) < 0.80


def test_paired_t_test_matches_scipy_ttest_rel_directly():
    """paired_t_test(a, b) must produce the same t and p as
    scipy.stats.ttest_rel(a, b)."""
    baseline = [40, 55, 62, 70, 78, 85, 91, 48, 66, 73]
    variant = [42, 57, 63, 72, 80, 88, 92, 50, 69, 75]

    result = paired_t_test(baseline, variant)
    scipy_result = sp.ttest_rel(baseline, variant)

    assert result.t == pytest.approx(scipy_result.statistic, abs=EXACT)
    assert result.p == pytest.approx(scipy_result.pvalue, abs=EXACT)


# statsmodels 0.15 TTestIndPower / TTestPower .solve_power(effect_size=d,
# alpha=0.05, power=0.8), rounded up. Pinned here instead of installed:
# statsmodels is a heavy dependency for ten numbers.
STATSMODELS_N = [
    (0.1, 1571, 787),
    (0.2, 394, 199),
    (0.3, 176, 90),
    (0.5, 64, 34),
    (0.625, 42, 23),
    (0.8, 26, 15),
    (1.0, 17, 10),
    (1.5, 9, 6),
    (2.0, 6, 5),
    (3.0, 4, 4),
]


@pytest.mark.parametrize(("d", "welch", "paired"), STATSMODELS_N)
def test_required_n_matches_statsmodels(d, welch, paired):
    assert required_n(1, d) == welch == run_js("requiredN", 1, d)
    assert required_n_paired(1, d) == paired == run_js("requiredNPaired", 1, d)
