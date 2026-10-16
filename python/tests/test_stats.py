"""Tests for eval_kit.stats.

Mirrors js/test/stats.test.mjs one-for-one. Same test cases, same regression
coverage, same edge-case guards. The pair exists so cross-validation CI can
run both bindings against identical vectors and assert agreement.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest

from eval_kit.cli import JSON_KEYS, main
from eval_kit.stats import (
    approx_p_value,
    descriptive_stats,
    p_value,
    paired_t_test,
    required_n,
    required_n_paired,
    t_critical,
    verdict,
    welch_t_test,
)

# ---------------------------------------------------------------------------
# descriptive_stats()
# ---------------------------------------------------------------------------


def test_descriptive_stats_returns_none_for_empty_input():
    assert descriptive_stats([]) is None


def test_descriptive_stats_handles_single_value_input_without_divide_by_zero():
    s = descriptive_stats([42])
    assert s.n == 1
    assert s.mean == 42
    assert s.std == 0
    assert s.se == 0
    assert s.ci_margin == 0


def test_descriptive_stats_uses_bessel_corrected_sample_variance_n_minus_1():
    # For [1, 2, 3, 4, 5]: mean=3, sum((x-mean)^2)=10, var=10/4=2.5
    s = descriptive_stats([1, 2, 3, 4, 5])
    assert s.mean == 3
    assert s.std == math.sqrt(2.5)


def test_descriptive_stats_cv_uses_abs_mean_to_avoid_sign_errors_near_zero():
    # Regression: pre-fix, a negative mean produced a negative CV.
    s = descriptive_stats([-10, -12, -8, -11, -9])
    assert s.cv > 0, f"CV should be positive for negative-mean data, got {s.cv}"


def test_descriptive_stats_iterative_min_max_handles_large_arrays_without_stack_overflow():
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


def test_welch_t_test_returns_none_when_either_group_has_fewer_than_2_samples():
    assert welch_t_test([1], [1, 2, 3]) is None
    assert welch_t_test([1, 2, 3], [1]) is None
    assert welch_t_test([], [1, 2, 3]) is None


def test_welch_t_test_detects_a_meaningful_difference_between_groups():
    baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87]
    variant = [65, 68, 62, 70, 64, 67, 69, 63, 66, 71]
    result = welch_t_test(baseline, variant)
    assert result.diff > 0, "baseline mean should exceed variant mean"
    assert result.p < 0.05, f"p-value should flag real shift, got {result.p}"
    assert result.glass_d > 0.8, f"Glass's delta should indicate large effect, got {result.glass_d}"


def test_welch_t_test_uses_glass_delta_baseline_std_not_pooled_cohen_d():
    # Regression: Cohen's d pools std across groups, assuming equal variance.
    # Glass's delta divides by baseline std only, consistent with Welch's
    # unequal-variance assumption. This test pins the denominator.
    baseline = [80, 81, 82, 83, 84]  # std ~1.58
    variant = [70, 60, 80, 50, 90]  # std much larger
    result = welch_t_test(baseline, variant)
    expected_glass_d = abs(result.diff) / result.baseline_std
    assert abs(result.glass_d - expected_glass_d) < 1e-9, "Glass's delta must equal |diff| / baseline_std"


def test_welch_t_test_handles_zero_variance_groups_without_nan():
    # Regression: se==0 previously produced NaN for t and p.
    result = welch_t_test([50, 50, 50, 50], [60, 60, 60, 60])
    assert math.isfinite(result.diff)
    assert result.p == 0


def test_welch_t_test_returns_glass_d_99_sentinel_when_baseline_std_is_0_and_diff_nonzero():
    # inf breaks JSON serialization and comparison thresholds. 99 signals
    # "off-scale because baseline has no variance."
    result = welch_t_test([80, 80, 80], [70, 70, 70])
    assert result.glass_d == 99
    assert math.isfinite(result.glass_d), "glass_d must be JSON-serializable"


def test_welch_t_test_returns_glass_d_0_when_both_groups_have_zero_variance_and_identical_means():
    result = welch_t_test([80, 80, 80], [80, 80, 80])
    assert result.glass_d == 0
    assert result.diff == 0


def test_welch_t_test_returns_near_zero_t_when_groups_have_identical_distribution():
    a = [80, 82, 84, 86, 88, 90, 78, 81, 83, 85]
    b = [80, 82, 84, 86, 88, 90, 78, 81, 83, 85]
    result = welch_t_test(a, b)
    assert result.diff == 0
    assert result.t == 0


def test_welch_t_test_with_two_constant_arms_and_different_means():
    r = welch_t_test([50, 50, 50], [60, 60, 60])
    assert r is not None
    # t is inf whatever the sign of diff (-10 here); se == 0 is the check.
    assert r.t == math.inf
    assert r.df == 4
    assert r.se == 0
    assert r.glass_d == 99


def test_welch_t_test_pins_t_df_and_p_at_unequal_variances():
    # scipy.stats.ttest_ind([80,81,82,83,84], [70,60,80,50,90], equal_var=False)
    r = welch_t_test([80, 81, 82, 83, 84], [70, 60, 80, 50, 90])
    assert r is not None
    assert r.t == pytest.approx(1.6886341073527005, abs=1e-12)
    assert r.df == pytest.approx(4.07999200079992, abs=1e-12)
    assert r.p == pytest.approx(0.16515649717911493, abs=1e-12)


# ---------------------------------------------------------------------------
# required_n()
# ---------------------------------------------------------------------------


def test_required_n_is_the_exact_noncentral_t_answer_per_arm():
    # Regression: an early formula omitted the factor of 2 and returned 21.
    # The z-approximation gives 41; the exact answer (statsmodels TTestIndPower) is 42.
    assert required_n(8, 5) == 42
    assert required_n(16, 5) == 162
    assert required_n(8, 10) == 12


def test_required_n_matches_cohen_power_tables():
    # alpha 0.05 two-sided, power 0.80: d = 0.2 / 0.5 / 0.8 -> 394 / 64 / 26 per group
    assert required_n(1, 0.2) == 394
    assert required_n(1, 0.5) == 64
    assert required_n(1, 0.8) == 26


def test_required_n_ignores_the_sign_of_delta_and_never_returns_below_2():
    assert required_n(8, -5) == required_n(8, 5)
    assert required_n(0, 5) == 2
    assert required_n(1, 100) == 2


def test_required_n_scales_with_variance():
    assert required_n(16, 5) > required_n(8, 5)


def test_required_n_scales_inversely_with_effect_size():
    assert required_n(8, 2) > required_n(8, 10)


def test_required_n_returns_inf_for_zero_delta():
    assert required_n(8, 0) == math.inf


def test_required_n_at_std_1_sits_on_the_exact_boundary_for_mid_size_d():
    # required_n's power at these n clears 0.80 by under 0.004, so a df off by
    # one (2n - 3 for 2n - 2) moves the answer up by one. Paired counts ride along.
    for d, welch, paired in [(1.06, 15, 10), (1.41, 9, 7), (1.64, 7, 6), (1.8, 6, 5)]:
        assert required_n(1, d) == welch, f"d={d}"
        assert required_n_paired(1, d) == paired, f"d={d}"


def test_required_n_takes_the_bisection_branch_at_large_n():
    # scipy power is 0.80000002 at 826596 and 0.79999954 at 826595. Only an n
    # this large sends the search through the bisection step that raises lo.
    assert required_n(6.491892124788057, 0.028290668449373238) == 826596


def test_required_n_returns_the_z_value_past_the_exact_limit():
    # Past EXACT_N_LIMIT (1e8) the z guess comes back unsearched. Just past the
    # limit, a search would still move it, so a higher limit fails the second pin.
    assert required_n(2.1, 1e-5) == 692271192570
    assert required_n(1, 3.2e-4) == 153298433


# ---------------------------------------------------------------------------
# paired_t_test()
# ---------------------------------------------------------------------------


def test_paired_t_test_returns_none_for_mismatched_lengths_or_fewer_than_2_items():
    assert paired_t_test([1, 2], [1]) is None
    assert paired_t_test([1], [2]) is None


def test_paired_t_test_matches_scipy_ttest_rel():
    # scipy.stats.ttest_rel([80,72,91,65,77], [78,70,90,62,76])
    r = paired_t_test([80, 72, 91, 65, 77], [78, 70, 90, 62, 76])
    assert r is not None
    assert r.t == pytest.approx(4.810702354423639, abs=1e-9)
    assert r.p == pytest.approx(0.008580918721924785, abs=1e-9)
    assert r.df == 4
    assert r.n == 5
    assert r.diff == pytest.approx(1.8, abs=1e-12)


def test_paired_t_test_finds_a_shift_that_item_difficulty_hides_from_welch_t_test():
    # Items vary widely in difficulty; the variant adds ~2 points to each.
    base = [40, 55, 62, 70, 78, 85, 91, 48, 66, 73]
    vari = [42, 57, 63, 72, 80, 88, 92, 50, 69, 75]
    welch = welch_t_test(base, vari)
    paired = paired_t_test(base, vari)
    assert welch is not None and paired is not None
    assert welch.p > 0.5
    assert paired.p < 0.001


def test_paired_t_test_returns_dz_99_sentinel_when_every_difference_is_identical_and_nonzero():
    r = paired_t_test([5, 6, 7], [4, 5, 6])
    assert r is not None
    assert r.dz == 99
    assert r.p == 0
    assert r.se == 0


def test_paired_t_test_returns_dz_0_and_p_1_when_arms_are_identical():
    r = paired_t_test([5, 6, 7], [5, 6, 7])
    assert r is not None
    assert r.dz == 0
    assert r.p == 1


# ---------------------------------------------------------------------------
# required_n_paired()
# ---------------------------------------------------------------------------


def test_required_n_paired_is_the_exact_noncentral_t_answer():
    # No factor of 2, but the paired test has half the degrees of freedom, so it is
    # not exactly half of required_n. statsmodels TTestPower gives 23 and 34.
    assert required_n_paired(8, 5) == 23
    assert required_n_paired(1, 0.5) == 34
    assert required_n_paired(8, 15) == 5


def test_required_n_paired_ignores_the_sign_of_delta_and_never_returns_below_2():
    assert required_n_paired(8, -5) == required_n_paired(8, 5)
    assert required_n_paired(0, 5) == 2


def test_required_n_paired_returns_inf_for_zero_delta():
    assert required_n_paired(8, 0) == math.inf


# ---------------------------------------------------------------------------
# p_value()
# ---------------------------------------------------------------------------


# References are 2 * scipy.stats.t.sf(t, df).
def test_p_value_matches_scipy_at_small_and_fractional_df():
    assert p_value(11, 1) == pytest.approx(0.057715876752608954, abs=1e-9)
    assert p_value(20, 1) == pytest.approx(0.03180450251235275, abs=1e-9)
    assert p_value(5, 1.5) == pytest.approx(0.06537576762115618, abs=1e-9)
    assert p_value(3.0, 10) == pytest.approx(0.013343655022569572, abs=1e-9)


def test_p_value_uses_the_t_distribution_not_the_normal_at_df_30():
    # The normal approximation would report ~0.050 here.
    assert p_value(1.96, 30) == pytest.approx(0.05934231289605049, abs=1e-9)


def test_p_value_handles_t_0_and_t_inf():
    assert p_value(0, 10) == 1
    assert p_value(math.inf, 10) == 0


def test_p_value_and_t_critical_at_infinite_df():
    # The normal limit: 2 * scipy.stats.norm.sf(1) and scipy.stats.norm.ppf(0.975).
    assert p_value(1, math.inf) == pytest.approx(0.31731050786291415, abs=1e-12)
    assert t_critical(math.inf) == pytest.approx(1.959963984540054, abs=1e-12)


def test_approx_p_value_is_an_alias_of_p_value():
    # approx_p_value is a wrapper function here, not the same object as in JS,
    # so there is no identity check to mirror; the value check is shared.
    assert approx_p_value(2.5, 7.3) == p_value(2.5, 7.3)


# ---------------------------------------------------------------------------
# t_critical()
# ---------------------------------------------------------------------------


# References are scipy.stats.t.ppf(0.975, df).
def test_t_critical_matches_scipy_at_integer_fractional_and_large_df():
    assert t_critical(1) == pytest.approx(12.706204736174694, abs=1e-9)
    assert t_critical(1.5) == pytest.approx(6.016663104427927, abs=1e-9)
    assert t_critical(12) == pytest.approx(2.1788128296672284, abs=1e-9)
    assert t_critical(100) == pytest.approx(1.983971518523552, abs=1e-9)
    assert t_critical(1000) == pytest.approx(1.9623390808264078, abs=1e-9)


def test_t_critical_returns_0_for_df_le_0():
    assert t_critical(0) == 0


# ---------------------------------------------------------------------------
# verdict()
# ---------------------------------------------------------------------------

BASE = [80.1, 82.4, 79.0, 83.5, 81.2, 80.6, 82.9, 79.8]


def test_verdict_says_cant_tell_when_the_interval_spans_no_change_and_the_target():
    v = verdict(BASE[:4], [83.9, 81.2, 86.0, 84.7], 3)
    assert v.verdict == "cant_tell"
    assert v.reason == "underpowered"
    assert v.needed == 16


def test_verdict_calls_a_significant_shift_by_direction_and_flips_it_for_lower_is_better():
    down = [77.3, 78.9, 76.1, 79.4, 77.8, 76.7, 79.1, 78.2]
    assert verdict(BASE, down, 3).verdict == "regressed"
    assert verdict(BASE, down, 3, lower_is_better=True).verdict == "improved"


LONG = [80 + ((i * 7) % 11) / 5 for i in range(60)]


def test_verdict_says_no_change_only_when_the_ci_rules_out_the_target():
    v = verdict(LONG, LONG[::-1], 1)
    assert v.verdict == "no_change"
    assert v.reason == "within_target"


def test_verdict_calls_a_significant_shift_inside_the_target_no_change():
    v = verdict(LONG, [x - 0.3 for x in LONG], 3)
    assert v.p < 0.05
    assert v.verdict == "no_change"
    assert v.reason == "below_target"


def test_verdict_refuses_when_both_arms_have_zero_spread():
    for a, b in [
        ([0.1, 0.1, 0.1], [0.2, 0.2, 0.2]),
        ([0.3, 0.3, 0.1 + 0.2], [0.4, 0.4, 0.4]),
    ]:
        v = verdict(a, b, 0.05)
        assert (v.verdict, v.reason, v.needed, v.constant_arm) == ("cant_tell", "zero_spread", None, None)


def test_verdict_tests_one_constant_arm_and_names_it():
    up = verdict([0.1, 0.1, 0.1], [0.2, 0.25, 0.3], 0.05)
    assert (up.verdict, up.constant_arm) == ("improved", "baseline")
    down = verdict([0.2, 0.25, 0.3], [0.1, 0.1, 0.1], 0.05)
    assert (down.verdict, down.constant_arm) == ("regressed", "variant")
    assert verdict(BASE, [x + 1 for x in BASE], 3).constant_arm is None


def test_verdict_reports_the_effect_as_none_when_there_is_no_spread_to_scale_it_by():
    # Regression: 0.1.x reported Glass's delta sentinel, 99 (or 0), as the effect.
    base = verdict([0.8, 0.8, 0.8, 0.8, 0.8], [0.84, 0.79, 0.86, 0.83, 0.81], 0.03)
    assert base.constant_arm == "baseline"
    assert base.effect is None
    assert all(isinstance(x, float) and math.isfinite(x) for x in [base.ci_lo, base.ci_hi, base.p])
    # Equal means gave the 0 sentinel, which is no more an effect size than 99.
    assert verdict([5, 5, 5], [4, 5, 6], 1).effect is None
    # A constant variant leaves the baseline std as the scale, so the effect stays a number.
    vari = verdict([0.84, 0.79, 0.86, 0.83, 0.81], [0.8, 0.8, 0.8, 0.8, 0.8], 0.03)
    assert vari.constant_arm == "variant"
    assert vari.effect is not None and math.isfinite(vari.effect) and vari.effect > 0


def test_verdict_reports_the_ci_p_and_effect_as_none_with_no_spread_at_all():
    # Regression: 0.1.x reported placeholders: ci_lo = ci_hi = shift, p 0 or 1, effect 99 or 0.
    for a, b, paired in [
        ([80, 80, 80], [85, 85, 85], False),
        ([5, 5, 5], [5, 5, 5], False),
        ([0.85, 0.65, 0.95, 0.75, 0.55], [0.8, 0.6, 0.9, 0.7, 0.5], True),
        ([1, 2, 3], [1, 2, 3], True),
    ]:
        v = verdict(a, b, 1, paired=paired)
        assert v.reason == "zero_spread"
        assert [v.ci_lo, v.ci_hi, v.p, v.effect, v.needed] == [None, None, None, None, None]
        assert math.isfinite(v.shift)


def test_verdict_refuses_on_zero_spread_instead_of_calling_it():
    v = verdict([0.85, 0.65, 0.95, 0.75, 0.55], [0.8, 0.6, 0.9, 0.7, 0.5], 0.1, paired=True)
    assert v.verdict == "cant_tell"
    assert v.reason == "zero_spread"


def test_verdict_reports_shift_as_variant_minus_baseline():
    v = verdict(BASE[:4], [83.9, 81.2, 86.0, 84.7], 3)
    assert abs(v.shift - 2.7) < 1e-9
    assert v.ci_lo < v.shift < v.ci_hi


def test_verdict_returns_none_for_invalid_input():
    assert verdict([1], [1, 2], 1) is None
    assert verdict([1, 2, 3], [1, 2], 1, paired=True) is None
    assert verdict([1, 2, 3], [4, 5, 6], 0) is None
    assert verdict([1, 2, 3], [4, 5, 6], float("nan")) is None
    # No missing-target case: target is a required positional argument here,
    # so omitting it is a TypeError, not None. The JS options bag can omit it.


def test_verdict_calls_a_paired_improvement_significant():
    v = verdict([40, 55, 62, 70, 78, 85, 91, 48, 66, 73], [42, 57, 63, 72, 80, 88, 92, 50, 69, 75], 1, paired=True)
    assert (v.verdict, v.reason, v.needed) == ("improved", "significant", 10)
    assert v.effect == pytest.approx(3, abs=1e-12)


# ---------------------------------------------------------------------------
# CLI (eval_kit.cli). Mirrors the CLI tests in js/test/stats.test.mjs;
# tests/cross-validation/ holds the two CLIs to the same bytes.
# ---------------------------------------------------------------------------


@pytest.fixture
def score_file(tmp_path):
    count = iter(range(1000))

    def write(data) -> str:
        path = tmp_path / f"scores-{next(count)}.json"
        path.write_text(json.dumps(data))
        return str(path)

    return write


@pytest.fixture
def cli(capsys):
    def run(*args: str) -> tuple[int, str, str]:
        code = main(list(args))
        out, err = capsys.readouterr()
        return code, out, err

    return run


@pytest.fixture
def cli_json(cli):
    def run(*args: str) -> dict:
        code, out, err = cli(*args, "--json")
        assert code == 0, err
        return json.loads(out)

    return run


PILOT_BASE = [80.1, 82.4, 79.0, 83.5]
PILOT_VARI = [83.9, 81.2, 86.0, 84.7]


def test_cli_power_sizes_two_arrays_as_independent_runs_from_the_pooled_std(cli, cli_json, score_file):
    # Regression: 0.1.0 paired two arrays by index, answered in items, and
    # refused unequal lengths with a --paired error nobody had asked for.
    base, vari = score_file(PILOT_BASE), score_file(PILOT_VARI)
    pooled = math.sqrt((descriptive_stats(PILOT_BASE).std ** 2 + descriptive_stats(PILOT_VARI).std ** 2) / 2)
    r = cli_json("power", "--baseline", base, "--variant", vari, "--target", "3")
    assert r["design"] == "welch"
    assert r["spread"] == pytest.approx(pooled, abs=1e-12)
    assert r["needed"] == required_n(pooled, 3)
    _, out, _ = cli("power", "--baseline", base, "--variant", vari, "--target", "3")
    assert out.startswith("9 runs per arm ") and "(pooled std 2.05," in out

    code, out, err = cli("power", "--baseline", base, "--variant", score_file([*PILOT_VARI, 85.1]), "--target", "3")
    assert code == 0, err
    assert "runs per arm" in out
    # The baseline-only path is unchanged.
    assert cli("power", "--baseline", base, "--target", "3")[1] == (
        "9 runs per arm to detect a 3-point shift (baseline std 2.06, 80% power, alpha 0.05)\n"
    )


def test_cli_power_pairs_a_pilot_only_when_compare_would(cli, cli_json, score_file):
    base, vari = score_file(PILOT_BASE), score_file(PILOT_VARI)
    by_index = cli_json("power", "--baseline", base, "--variant", vari, "--target", "3", "--paired")
    assert by_index["design"] == "paired"
    assert by_index["spread"] == paired_t_test(PILOT_BASE, PILOT_VARI).sd_diff
    assert by_index["needed"] == required_n_paired(by_index["spread"], 3)

    obj_base, obj_vari = score_file({"a": 40, "b": 55, "c": 62}), score_file({"c": 65, "a": 42, "b": 56})
    for args in [
        ["--baseline", base, "--variant", vari, "--target", "3"],
        ["--baseline", base, "--variant", vari, "--target", "3", "--paired"],
        ["--baseline", obj_base, "--variant", obj_vari, "--target", "1"],
    ]:
        assert cli_json("power", *args)["design"] == cli_json("compare", *args)["design"], args

    code, _, err = cli("power", "--baseline", base, "--variant", score_file([1, 2, 3]), "--target", "3", "--paired")
    assert code == 2
    assert err == "eval-kit: --paired needs equal lengths (baseline 4, variant 3)\n"


def test_cli_power_warns_when_there_is_no_spread_to_size_from(cli, cli_json, score_file):
    minimum = "With no spread to size from, 2 is the minimum, not a plan."
    flat = score_file([0.8, 0.8, 0.8, 0.8, 0.8])
    cases = [
        (["--std", "0"], "A std of 0 means every run scores the same. That is a ceiling or a broken harness;"),
        (["--baseline", flat], "Every baseline run scored the same (0.80). That is a ceiling or a broken harness;"),
        (
            ["--baseline", flat, "--variant", flat],
            "Every run in both arms scored the same. That is a ceiling or a broken harness;",
        ),
        (["--sd-diff", "0"], "An sd of 0 means every item moves by the same amount. That is usually a broken harness;"),
        (
            ["--baseline", score_file({"a": 1, "b": 2}), "--variant", score_file({"a": 2, "b": 3})],
            "Every item moved by exactly the same amount. That is usually a broken harness;",
        ),
    ]
    for args, cause in cases:
        code, out, err = cli("power", *args, "--target", "3")
        assert code == 0, err
        lines = out.rstrip().split("\n")
        assert lines[0].startswith(
            ("2 runs per arm to detect a 3-point shift (", "2 items to detect a 3-point shift (")
        )
        assert lines[0].endswith(" 0.00, 80% power, alpha 0.05)")
        assert lines[1] == cause
        assert lines[2].endswith(minimum), lines[2]

    # One constant arm in a two-arm pilot: sized from the other arm, with compare's warning.
    _, out, _ = cli(
        "power", "--baseline", flat, "--variant", score_file([0.84, 0.79, 0.86, 0.83, 0.81]), "--target", "0.03"
    )
    assert out.rstrip().split("\n")[1:] == [
        "Every baseline run scored the same (0.80). That is a ceiling or a broken harness;",
        "check which before trusting this number.",
    ]
    # --json keeps its shape; spread 0 is the signal there.
    assert cli_json("power", "--std", "0", "--target", "3") == {
        "design": "welch",
        "target": 3,
        "spread": 0,
        "needed": 2,
    }


def test_cli_compare_prints_the_ci_p_and_effect_as_undefined_when_neither_arm_has_spread(cli, cli_json, score_file):
    # Regression: 0.1.0 printed glassD 99.00, p < 0.001 and a zero-width CI here.
    at80, at85 = score_file([80, 80, 80, 80, 80]), score_file([85, 85, 85, 85, 85])
    for vari, shift in [(at85, "+5.00"), (at80, "+0.00")]:
        code, out, _ = cli("compare", "--baseline", at80, "--variant", vari, "--target", "3", "--gate")
        assert code == 3
        assert f"  shift     {shift}  95% CI undefined\n  p         undefined\n" in out, out
        assert "  glassD    undefined (no spread in either arm)\n" in out, out
        assert "CAN'T TELL. Every run in both arms scored the same." in out, out
    _, out, _ = cli("compare", "--baseline", at80, "--variant", at85, "--target", "3", "--paired")
    assert "  dz        undefined (no spread in the item differences)\n" in out, out

    # --json keeps its shape and reports what the data cannot define as null.
    v = cli_json("compare", "--baseline", at80, "--variant", at85, "--target", "3")
    assert list(v) == [JSON_KEYS.get(k, k) for k in asdict(verdict([1, 2], [3, 4], 1))]
    fields = ["verdict", "reason", "shift", "ciLo", "ciHi", "p", "effect", "needed", "constantArm"]
    assert [v[k] for k in fields] == ["cant_tell", "zero_spread", 5, None, None, None, None, None, None]
    paired_json = cli_json("compare", "--baseline", at80, "--variant", at85, "--target", "3", "--paired")
    assert [paired_json[k] for k in ["ciLo", "ciHi", "p", "effect"]] == [None, None, None, None]
    # One constant baseline: a real CI and p; the effect is off-scale in text and null in --json.
    one_args = [
        "compare",
        "--baseline",
        score_file([0.8, 0.8, 0.8, 0.8, 0.8]),
        "--variant",
        score_file([0.84, 0.79, 0.86, 0.83, 0.81]),
        "--target",
        "0.03",
    ]
    _, out, _ = cli(*one_args)
    assert "95% CI [-0.01, +0.06]\n  p         0.098\n  glassD    off-scale (the baseline has no spread)\n" in out, out
    one_json = cli_json(*one_args)
    assert one_json["effect"] is None
    assert all(isinstance(one_json[k], float) for k in ["ciLo", "ciHi", "p"])


def test_cli_gate_holds_on_a_constant_arm_unless_the_verdict_is_regressed(cli, score_file):
    # Regression: 0.1.x exited 0 on an improved verdict over a constant baseline,
    # while the text said to check the harness before trusting it.
    flat = score_file([0.8, 0.8, 0.8, 0.8, 0.8])
    up = score_file([0.86, 0.84, 0.88, 0.85, 0.87])
    down = score_file([0.74, 0.76, 0.72, 0.75, 0.73])
    near = score_file([0.8, 0.81, 0.79, 0.8, 0.8])
    cases = [
        ("constant baseline, improved", flat, up, "IMPROVED.", 3),
        ("constant baseline, no change", flat, near, "NO CHANGE.", 3),
        ("constant baseline, regressed", flat, down, "REGRESSED.", 1),
        ("constant variant, improved", down, flat, "IMPROVED.", 3),
        ("constant variant, regressed", up, flat, "REGRESSED.", 1),
    ]
    for label, base, vari, call, code in cases:
        args = ["compare", "--baseline", base, "--variant", vari, "--target", "0.05"]
        gated_code, gated_out, _ = cli(*args, "--gate")
        assert gated_code == code, label
        assert f"\n{call} " in gated_out, f"{label}: {gated_out}"
        # Only the exit code changes: the verdict and the text are the same as without --gate.
        plain_code, plain_out, _ = cli(*args)
        assert plain_code == 0, label
        assert plain_out == gated_out, label
