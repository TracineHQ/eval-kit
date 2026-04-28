"""Tests for eval_kit.stats.

Mirrors js/test/stats.test.mjs one-for-one. Same test cases, same regression
coverage, same edge-case guards. The pair exists so cross-validation CI can
run both bindings against identical vectors and assert agreement.
"""

from __future__ import annotations

import math

import pytest

from eval_kit.stats import (
    approx_p_value,
    descriptive_stats,
    required_n,
    t_critical,
    welch_t_test,
)


# ---------------------------------------------------------------------------
# descriptive_stats()
# ---------------------------------------------------------------------------


def test_descriptive_stats_returns_none_for_empty_input():
    assert descriptive_stats([]) is None


def test_descriptive_stats_handles_single_value_without_divide_by_zero():
    s = descriptive_stats([42])
    assert s.n == 1
    assert s.mean == 42
    assert s.std == 0
    assert s.se == 0
    assert s.ci_margin == 0


def test_descriptive_stats_uses_bessel_corrected_sample_variance():
    # For [1, 2, 3, 4, 5]: mean=3, sum((x-mean)^2)=10, var=10/4=2.5
    s = descriptive_stats([1, 2, 3, 4, 5])
    assert s.mean == 3
    assert s.std == pytest.approx(math.sqrt(2.5))


def test_descriptive_stats_cv_is_positive_for_negative_mean():
    # Regression from JS: CV must use abs(mean) to avoid sign errors.
    s = descriptive_stats([-10, -12, -8, -11, -9])
    assert s.cv > 0, f"CV should be positive for negative-mean data, got {s.cv}"


def test_descriptive_stats_handles_large_arrays():
    # JS regression: spread-operator min/max blows the call stack around
    # 100k elements. Python doesn't have that problem, but we keep the
    # test for parity with the JS suite.
    large = list(range(1_000_000))
    s = descriptive_stats(large)
    assert s.min == 0
    assert s.max == 999_999


# ---------------------------------------------------------------------------
# welch_t_test()
# ---------------------------------------------------------------------------


def test_welch_returns_none_when_either_group_has_fewer_than_2_samples():
    assert welch_t_test([1], [1, 2, 3]) is None
    assert welch_t_test([1, 2, 3], [1]) is None
    assert welch_t_test([], [1, 2, 3]) is None


def test_welch_detects_meaningful_difference():
    baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87]
    variant = [65, 68, 62, 70, 64, 67, 69, 63, 66, 71]
    result = welch_t_test(baseline, variant)
    assert result.diff > 0, "baseline mean should exceed variant mean"
    assert result.p < 0.05, f"p-value should flag real shift, got {result.p}"
    assert (
        result.glass_d > 0.8
    ), f"Glass's delta should indicate large effect, got {result.glass_d}"


def test_welch_uses_glass_delta_not_pooled_cohen():
    # Regression: Cohen's d pools std across groups. Glass's delta divides
    # by baseline std only, consistent with Welch's unequal-variance
    # assumption. Pin the denominator.
    baseline = [80, 81, 82, 83, 84]  # std ~1.58
    variant = [70, 60, 80, 50, 90]  # std much larger
    result = welch_t_test(baseline, variant)
    expected_glass_d = abs(result.diff) / result.baseline_std
    assert (
        abs(result.glass_d - expected_glass_d) < 1e-9
    ), "Glass's delta must equal |diff| / baseline_std"


def test_welch_handles_zero_variance_groups_without_nan():
    # Regression: se==0 previously produced NaN for t and p.
    result = welch_t_test([50, 50, 50, 50], [60, 60, 60, 60])
    assert math.isfinite(result.diff)
    assert result.p == 0


def test_welch_returns_near_zero_t_for_identical_distributions():
    a = [80, 82, 84, 86, 88, 90, 78, 81, 83, 85]
    b = [80, 82, 84, 86, 88, 90, 78, 81, 83, 85]
    result = welch_t_test(a, b)
    assert result.diff == 0
    assert result.t == 0


def test_welch_returns_glass_d_99_when_baseline_std_is_zero_and_diff_nonzero():
    # inf breaks JSON serialization. 99 signals "off-scale because baseline
    # has no variance." Cross-binding sentinel shared with JS.
    result = welch_t_test([80, 80, 80], [70, 70, 70])
    assert result.glass_d == 99
    assert math.isfinite(result.glass_d), "glass_d must be JSON-serializable"


def test_welch_returns_glass_d_0_when_both_groups_have_zero_variance_and_identical_means():
    result = welch_t_test([80, 80, 80], [80, 80, 80])
    assert result.glass_d == 0
    assert result.diff == 0


# ---------------------------------------------------------------------------
# required_n()
# ---------------------------------------------------------------------------


def test_required_n_includes_factor_of_2_for_two_sample_comparison():
    # Regression: original formula omitted the factor of 2, underestimating
    # sample size by half. At std=8, delta=5, correct answer is 41.
    # Formula: ceil(2 * (1.96 + 0.84)^2 * 8^2 / 5^2) = ceil(40.14) = 41
    assert required_n(8, 5) == 41


def test_required_n_scales_with_variance():
    assert required_n(16, 5) > required_n(8, 5)


def test_required_n_scales_inversely_with_effect_size():
    assert required_n(8, 2) > required_n(8, 10)


def test_required_n_returns_inf_for_zero_delta():
    assert required_n(8, 0) == math.inf


# ---------------------------------------------------------------------------
# approx_p_value()
# ---------------------------------------------------------------------------


def test_approx_p_value_returns_p_close_to_zero_for_extreme_t_values():
    p = approx_p_value(15, 100)
    assert p < 0.001, f"expected p<0.001 for t=15, got {p}"


def test_approx_p_value_returns_p_near_1_for_near_zero_t_values():
    p = approx_p_value(0.05, 100)
    assert p > 0.9, f"expected p>0.9 for t=0.05, got {p}"


def test_approx_p_value_is_exact_via_scipy_for_small_df():
    # Unlike the JS binding (which buckets small-df p-values into
    # {0.01, 0.05, 0.1, 0.5}), Python uses scipy's exact t-distribution
    # survival function for all df. So this test asserts scipy behavior,
    # not bucketed behavior.
    p = approx_p_value(3.0, 10)
    # scipy.stats.t.sf(3.0, 10) * 2 is ~0.0133 (exact)
    assert 0.01 < p < 0.02, f"expected ~0.0133, got {p}"


# ---------------------------------------------------------------------------
# t_critical()
# ---------------------------------------------------------------------------


def test_t_critical_returns_exact_values_for_known_df():
    # scipy exact: df=1 -> 12.7062, df=10 -> 2.2281
    assert t_critical(1) == pytest.approx(12.7062, abs=0.001)
    assert t_critical(10) == pytest.approx(2.2281, abs=0.001)


def test_t_critical_interpolates_between_table_entries():
    # df=12 should fall between df=10 (2.228) and df=14 (2.145)
    t = t_critical(12)
    assert 2.145 < t < 2.228, f"expected interpolated value, got {t}"


def test_t_critical_approaches_z_for_large_df():
    # Exact via scipy. Large df converges to z=1.96 but not exactly equal
    # (unlike the JS version which falls back to exactly 1.96).
    assert t_critical(1000) == pytest.approx(1.96, abs=0.01)
