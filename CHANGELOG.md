# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Python 3.14 in the supported versions. CI now runs it, Node 26, and the cross-validation
  suite on Python 3.14 and macOS.

### Changed
- The Python package requires Python 3.12 or later, and numpy 1.26 or later, the first
  numpy release with Python 3.12 wheels. Python 3.10 and 3.11 are no longer supported.
- The CLI and JS library require Node 22 or later. Node 20 reached end of life on 2026-04-30.

### Fixed
- The Python package allowed scipy 1.11 and 1.12, whose t quantile at small df differs from
  scipy 1.13 and later, and from the JS binding, by more than the 1e-9 the bindings are held
  to: 4e-8 at df 0.5, 4e-9 at df 1.5. The 95% CI bounds at small n carried the gap. It now
  requires `scipy>=1.13`, and CI runs the unit and cross-validation suites at the lowest
  allowed numpy and scipy.

## [0.2.0] - 2026-10-15

### Changed
- `compare --gate` exits 3 when one arm is constant (`constantArm` set), unless the verdict is
  `regressed`, which still exits 1. The text output says a constant arm is a ceiling or a
  broken harness and to check which before trusting the verdict. A gate cannot check, so it
  holds the merge. 0.1.0 exited 0 on an `improved` or `no_change` verdict over a constant
  arm. The verdict and the text output are unchanged.
- `verdict()` in both bindings, and so `compare --json`, reports what the data cannot define
  as null (`None` in Python) instead of a placeholder. `effect` is null when there is no spread
  to scale it by: reason `zero_spread` (either design), or a constant baseline under Welch's
  test. A constant variant keeps a numeric effect, since Glass's delta divides by the baseline
  std. `ciLo`, `ciHi` and `p` are null for reason `zero_spread`. 0.1.0 reported Glass's delta
  or dz sentinel, 99 (0 when the means match), as the effect, and with no spread at all a
  zero-width CI at `shift` and `p` 0 (1 when the arms match). `welchTTest` / `welch_t_test`
  and `pairedTTest` / `paired_t_test` keep their sentinels. The text output is unchanged.

### Fixed
- `power` with `--baseline` and `--variant` reads the design the way `compare` does. Two
  arrays are independent runs, sized from the pooled std of both arms,
  `sqrt((sd_baseline^2 + sd_variant^2) / 2)`, the common std at which the per-arm math gives
  the standard error the two spreads give. Objects keyed by item id, or arrays with `--paired`,
  are paired items, sized from the std of their differences. 0.1.0 paired any two files by
  index, answered in items, and refused arrays of unequal length with a `--paired` error
  nobody had asked for.
- `compare` with no spread to test against (both arms constant, or every paired difference
  identical) printed a zero-width CI, `p < 0.001`, and an effect size of 99.00. The text output
  now prints them as undefined. The verdict is still `cant_tell`, reason `zero_spread`, and
  `--gate` still exits 3. `--json` keeps its shape and reports the CI bounds, p and effect as
  null (see Changed).
- `power` with no spread to size from (`--std 0`, `--sd-diff 0`, or a constant pilot) printed
  "2 runs per arm" and nothing else. It now names what had no spread and warns that 2 is the
  minimum, not a plan. A two-arm pilot with one constant arm gets `compare`'s constant-arm
  warning. `--json` is unchanged; `spread: 0` is the signal there.

## [0.1.0] - 2026-10-02

First tagged release. 0.0.x was never tagged; the Changed, Removed, and Fixed
entries are for anyone who copied `stats.mjs` from main before this release.

### Added
- `verdict` / `verdict()` in both bindings: the decision as data. Given a
  target set in advance, returns improved, regressed, no_change, or
  cant_tell, with the shift, its 95% CI, p, effect size, and the n at which,
  at the spread seen so far, the result has to be a call (the CI is too
  narrow to span both no change and the target). A significant shift whose
  whole CI sits inside the target is no_change (reason `below_target`), so
  `--gate` fails a build only when the 95% CI reaches the target on the bad
  side. One constant arm still gets a verdict, with `constantArm` naming it
  and a warning in the text output. Both arms constant, or identical paired
  differences, is cant_tell (reason `zero_spread`): check the harness.
- CLI in both bindings (`node js/cli.mjs`, `eval-kit` / `python -m eval_kit`):
  `compare` and `power`, JSON input, `--json` output, and `--gate` exit codes
  for CI (0 improved or no_change, 1 regressed, 2 usage error, 3 cant_tell).
  Cross-validation holds them to identical text, errors, and exit codes,
  and `--json` to the same tokens with computed floats within 1e-9.
- `pairedTTest` / `paired_t_test` with Cohen's dz, and `requiredNPaired` /
  `required_n_paired`, for per-item scores where the same items appear in
  both arms.
- Python binding (`eval-kit` package): same API surface, scipy-backed.
- `pValue` / `p_value` as the primary p-value names.
- `SKILL.md`: the Claude Code skill. It gets the target first, picks the
  design, sizes runs or items before testing, runs the CLI for every number,
  and says when the data can't support a call. `.claude-plugin/plugin.json`
  for the marketplace.
- `claude plugin eval` suite under `evals/`, and `tests/skill/`, which checks
  it in CI with no model calls: each answer key is the CLI's real output, and
  each grader accepts the right answer and rejects the wrong ones.
- Cross-validation suite (`tests/cross-validation/`) holding both bindings to
  scipy within 1e-9, and to each other byte for byte on CLI output.
- CI workflow running all four test suites on a matrix of Node and Python
  versions, plus a `claude plugin validate` manifest check.
- `AGENTS.md`, `CLAUDE.md`, and `CONTRIBUTING.md` with the parity rule.

### Changed
- JS p-values and t-critical values are exact, computed from a regularized
  incomplete beta function with no dependencies. This replaces the lookup
  table, bucketed small-df p-values, and normal approximation at df >= 30,
  which could overstate significance near the 0.05 bar.
- `requiredN` and `requiredNPaired` are exact: the smallest n whose power
  under the noncentral t-distribution reaches 0.80, matching statsmodels and
  Cohen's tables. JS computes the noncentral t itself; Python uses scipy. The
  z-approximation they replace ran one to two short at small n (41 instead of
  42 at std 8, delta 5).
- `approxPValue` / `approx_p_value` are now aliases of `pValue` / `p_value`.
- `glassD` uses sentinel 99 (not Infinity) when baseline std is 0, preserving
  JSON serializability.
- Switched from MIT to Apache 2.0 license.

### Removed
- The stale duplicate `lib/stats.mjs` at the repo root.

### Fixed
- Zero spread is detected on decimal scores. Floating-point dust (`0.1 + 0.2`
  next to `0.3`, or a constant per-item shift) had produced a spread of
  ~1e-17 and t ~ 1e15 instead of the sentinel.
- `requiredN` docs gave the wrong multiplier for unequal variance. Double the
  variance means ~1.5x the runs; double the standard deviation means ~2.5x.

[Unreleased]: https://github.com/TracineHQ/eval-kit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/TracineHQ/eval-kit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/TracineHQ/eval-kit/releases/tag/v0.1.0
