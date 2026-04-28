# eval-kit

Statistical confidence library for LLM evaluation. Two language bindings,
same API surface, cross-validated against scipy on every commit.

## What this repo is

A zero-dependency JS library (`js/lib/stats.mjs`) and a scipy-backed Python
package (`python/src/eval_kit/`) that implement the same statistical
functions for comparing LLM eval runs: descriptive stats with CI, Welch's
t-test, Glass's delta effect size, and power analysis.

**Python is the reference implementation** (exact, via scipy).
**JS approximates** p-values for df < 30 using bucketed thresholds and
uses a 16-entry t-critical lookup table with linear interpolation. This
is intentional -- zero runtime dependencies in the browser/Node path.
Do not "fix" the JS approximation without updating cross-validation
tolerances and adding a regression note.

## Architecture

    js/
      lib/stats.mjs              JS binding (Node 20+, zero deps)
      test/stats.test.mjs        JS unit tests (node:test)
    python/
      pyproject.toml             Python package manifest (hatchling)
      src/eval_kit/stats.py  Python binding (numpy + scipy)
      tests/test_stats.py        Python unit tests (pytest, mirrors JS 1:1)
    tests/
      cross-validation/
        bridge.mjs               Spawned by Python to call JS functions
        test_cross_validation.py Parity + scipy ground truth suite

## Running tests

**JS unit tests** (22 tests):

    node --test js/test/stats.test.mjs

**Python unit tests** (22 tests, from `python/`):

    pip install -e ".[dev]"
    pytest

**Cross-validation** (51 tests, from `python/`, requires Node in PATH):

    pip install -e ".[dev]"
    pytest ../tests/cross-validation/ -v

All three suites run clean as of 2026-04-21.

## Parity rule

Every exported function must exist in both bindings with:

1. Mirrored unit tests in `js/test/stats.test.mjs` and
   `python/tests/test_stats.py` (same test name pattern, same edge cases,
   cross-linked regression comments).
2. Cross-validation coverage in `tests/cross-validation/test_cross_validation.py`
   asserting the two bindings agree within documented tolerances.

Do not add a function to one binding without adding it to the other.

## Where not to invent math

All statistical formulas are standard. Do not change:

- Welch-Satterthwaite df approximation
- Glass's delta denominator (baseline std only, not pooled)
- The factor of 2 in `requiredN` / `required_n` (correct for two-sample)
- p-value tolerances in cross-validation without re-measuring against
  scipy exact values

If you believe a formula is wrong, open an issue with a citation.

## Sentinels

- `glassD = 99` when baseline std is 0 and diff != 0 (Infinity breaks
  JSON.stringify and comparison thresholds; 99 means "off-scale").
- `glassD = 0` when both groups are identical (std=0, diff=0).
- `requiredN(std, 0) = Infinity` (no detectable effect).
- Empty input returns `null` / `None` from `stats` / `descriptive_stats`.

These are shared across bindings and tested in cross-validation.

## Code style

- JS: ESM modules, no transpilation, JSDoc on all exports, Node built-ins only.
- Python: dataclasses for return types, docstrings on all public functions,
  type annotations on new code. scipy/numpy for math.

## License

Apache 2.0. New source files must include the license attribution in the
module docstring:

    Copyright 2026 TracineHQ
    Licensed under the Apache License, Version 2.0.
    See LICENSE and NOTICE files at repository root.

**Do not add runtime dependencies to `js/lib/stats.mjs`.** The zero-dependency
constraint is a feature. Python deps (numpy, scipy) are acceptable in the
Python binding.

## Scope

This repo is `eval-kit`. It currently ships the stats module plus a Claude
Code plugin scaffold. The plugin's skill(s) are under active design. Do
not invent new plugin infrastructure without confirmation.
