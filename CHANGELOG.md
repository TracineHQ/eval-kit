# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- JS binding (`js/lib/stats.mjs`): descriptive stats with 95% CI, Welch's
  t-test, Glass's delta effect size, required-N power analysis, approx
  p-value, t-critical lookup.
- Python binding (`eval-kit` package): same API surface, scipy-backed
  exact p-values and t-critical.
- Cross-validation test suite (`tests/cross-validation/`) asserting both
  bindings agree within documented tolerances, with scipy as ground truth.
- `AGENTS.md` and `CLAUDE.md` for agent-friendly contribution.
- `CONTRIBUTING.md` with parity rule and test commands.
- `.claude-plugin/plugin.json` scaffolding for Claude Code plugin marketplace.
- CI workflow (`.github/workflows/ci.yml`) running all three test suites on
  a matrix of Node and Python versions.

### Changed
- Switched from MIT to Apache 2.0 license.
- Upgraded JS `T_CRIT` lookup table from 3dp to 4dp precision (matches scipy
  to ~1e-5; improves CI bound accuracy on high-variance inputs).
- `glassD` uses sentinel 99 (not Infinity) when baseline std is 0, preserving
  JSON serializability.

[Unreleased]: https://github.com/TracineHQ/eval-kit/commits/main
