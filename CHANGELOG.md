# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/TracineHQ/eval-kit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TracineHQ/eval-kit/releases/tag/v0.1.0
