# eval-kit

[![CI](https://github.com/TracineHQ/eval-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/TracineHQ/eval-kit/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Statistical confidence for LLM evaluation. Welch's t-test, Glass's delta, power
analysis. Two language bindings, cross-validated against scipy.

## What it is

A small library for the math you need when comparing LLM outputs:

- Are these two runs actually different, or just sampled differently from the same distribution?
- How many runs do you need to reliably detect a 5-point shift?
- Is my baseline precise enough to trust the comparison at all?

Two bindings, same API surface:

- **JS** (`js/lib/stats.mjs`) -- Node.js built-ins only, zero dependencies, ~260 lines with JSDoc.
- **Python** (`python/src/eval_kit/`) -- scipy-backed, exact. Install from source.

Cross-validated against scipy on every commit (`tests/cross-validation/`).

## Install

**JavaScript (Node 20+, no install needed -- copy the single file):**

```bash
curl -O https://raw.githubusercontent.com/TracineHQ/eval-kit/main/js/lib/stats.mjs
```

**Python (install from source):**

```bash
git clone https://github.com/TracineHQ/eval-kit
cd eval-kit/python && pip install -e .
```

## Quick start

**JavaScript:**

```javascript
import { stats, welchTTest, requiredN } from './stats.mjs';

const baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87];
const variant  = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81];

console.log(welchTTest(baseline, variant));
// { t, df, p, diff, se, glassD, baselineStd }

console.log(requiredN(stats(baseline).std, 5));
// runs per group needed to detect a 5-point shift
```

**Python:**

```python
from eval_kit import descriptive_stats, welch_t_test, required_n

baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87]
variant  = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81]

print(welch_t_test(baseline, variant))
# WelchResult(t=..., df=..., p=..., diff=..., se=..., glass_d=..., baseline_std=...)

print(required_n(descriptive_stats(baseline).std, 5))
# runs per group needed to detect a 5-point shift
```

## API parity

Both bindings expose the same functions. JS uses camelCase; Python uses snake_case.
Return field names follow the same convention.

| Purpose | JS | Python | Returns |
|---|---|---|---|
| Descriptive stats + 95% CI | `stats(arr)` | `descriptive_stats(arr)` | `{n, mean, std, cv, min, max, range, se, ciLo, ciHi, ciMargin}` |
| Welch's t-test + Glass's delta | `welchTTest(a, b)` | `welch_t_test(a, b)` | `{t, df, p, diff, se, glassD, baselineStd}` |
| Required sample size (power analysis) | `requiredN(std, delta)` | `required_n(std, delta)` | `number` |
| Two-tailed p-value from t-statistic | `approxPValue(absT, df)` | `approx_p_value(abs_t, df)` | `number` |
| t-critical for 95% CI | `tCritical(df)` | `t_critical(df)` | `number` |

**Precision difference:** Python is exact via scipy. JS approximates p-values for
`df < 30` using bucketed thresholds (one of `{0.0001, 0.01, 0.05, 0.1, 0.5}`) and
uses the Abramowitz & Stegun normal CDF for `df >= 30`. Cross-validation tests
pin the maximum divergence at factor 3x on bucketed values and 1e-2 on A&S-vs-exact
p-values.

## Notable design choices

- **Glass's delta, not Cohen's d.** Cohen's d pools both groups' standard deviations, assuming equal variance. That is inconsistent with using Welch's in the first place. Glass's delta uses only the baseline group's standard deviation.
- **Power analysis includes a factor of 2** for two-sample comparison. Without it, the required sample size is underestimated by half.
- **CV uses `Math.abs(mean)`** to avoid sign errors near zero.
- **Iterative min and max** rather than the spread operator, so large arrays do not blow the call stack.
- **Sentinels over Infinity.** `glassD = 99` when baseline std is 0 (Infinity breaks JSON serialization).
- **p-value approximation** documents its small-sample limitations. Use a real t-distribution CDF (jStat, scipy, or the Python binding) when you need precision below n=30.

## Monorepo layout

```
js/
  lib/stats.mjs                JS binding (zero deps, Node 20+)
  test/stats.test.mjs          JS unit tests (node:test)
python/
  pyproject.toml               Python package manifest
  src/eval_kit/stats.py    Python binding (numpy + scipy)
  tests/test_stats.py          Python unit tests (pytest, mirrors JS 1:1)
tests/
  cross-validation/            Parity + scipy ground truth
```

The Python binding is the reference implementation. The JS binding is
cross-validated against it. See `AGENTS.md` for the parity rule that governs
contributions.

## Tests

```bash
# JS (22 tests)
node --test js/test/stats.test.mjs

# Python (22 tests, from python/)
cd python && pip install -e ".[dev]" && pytest

# Cross-validation (51 tests, from python/, requires node in PATH)
cd python && pytest ../tests/cross-validation/ -v
```

All three suites are wired into CI on every push and pull request to main.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The parity rule: every function must
exist in both bindings with mirrored tests and cross-validation coverage.

## License

Apache 2.0 -- see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Author

Anthony Ledesma. I build infrastructure for safe, observable LLM development.
