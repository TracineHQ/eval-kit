# eval-kit

[![CI](https://github.com/TracineHQ/eval-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/TracineHQ/eval-kit/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Did the eval get better, or is that noise? eval-kit is a Claude Code skill for the
decision procedure: size the runs before testing, pick Welch's or a paired t-test, read
p-value and effect size together, and say "can't tell yet" when the data cannot support
a call. It runs on a zero-dependency JS library and a scipy-backed Python one,
cross-validated in CI.

Beta. The math is tested against scipy; the API surface and the skill text may
still change before 1.0.

Three comparisons against the same baseline, chasing a 3-point shift:

```
target: 3 points  shift  95% CI         p      glassD  verdict
4 runs per arm    +2.7   [-0.8, +6.2]   0.11   1.31    can't tell yet: need 16 per arm
8 runs per arm    +3.8   [+2.2, +5.3]   1e-4   2.34    improved
12 runs per arm   +0.0   [-1.0, +1.0]   0.96   0.02    no shift of 3 or more
```

The first row is the one that matters. The variant is up 2.7 points with a large effect
size, and the honest answer is still "not yet." Four runs cannot resolve a 3-point shift at
this spread. With `--gate`, the CLI turns that row into exit code 3, so CI holds the merge
instead of passing or failing it. Reproduce the table with `node js/examples/verdicts.mjs`.

## Install

**Claude Code skill:**

```
/plugin marketplace add TracineHQ/plugins
/plugin install eval-kit@tracine
```

**CLI and JS library (Node 20+, no dependencies):** run in place from a clone.

```bash
git clone https://github.com/TracineHQ/eval-kit && cd eval-kit
node js/cli.mjs --help
```

For the library alone, `js/lib/stats.mjs` is self-contained: copy it into your project.

**Python (not on PyPI):** `cd python && pip install -e .` from the clone. Same functions in
snake_case, and an `eval-kit` command.

## Two designs, two tests

- **Independent runs** (one score per full pass over the suite, n runs per arm): Welch's
  t-test.
- **Same items in both arms** (per-item scores, paired by item): the paired t-test. Pairing
  cancels item difficulty, so it usually needs far fewer items.

## Quick start

**JavaScript:**

```javascript
import { stats, welchTTest, pairedTTest, requiredN } from './js/lib/stats.mjs';

const baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87];
const variant  = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81];

console.log(welchTTest(baseline, variant));
// { t, df, p, diff, se, glassD, baselineStd }
// diff is baseline minus variant: +7 here, the variant scored lower.

console.log(requiredN(stats(baseline).std, 5));
// runs per group needed to detect a 5-point shift

// Same items scored in both arms? Pair them: index i is the same item in both arrays.
const baselineItems = [40, 55, 62, 70, 78, 85, 91, 48, 66, 73];
const variantItems  = [42, 57, 63, 72, 80, 88, 92, 50, 69, 75];
console.log(pairedTTest(baselineItems, variantItems));
// { t, df, p, diff, se, dz, sdDiff, n }
// p = 0.0000055 here. welchTTest on the same arrays gives p = 0.78: pairing is the whole story.
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

## Command line

`node js/cli.mjs`, or `eval-kit` from the Python package. Both print identical text, errors, and
exit codes; `--json` output matches in every key and token except the last digits of computed
floats (they agree to 1e-9). The Claude Code skill runs it for every number.

With `base.json` as `[80.1, 82.4, 79.0, 83.5]` and `new.json` as `[83.9, 81.2, 86.0, 84.7]`:

```bash
node js/cli.mjs compare --baseline base.json --variant new.json --target 3
```

```
eval-kit compare: Welch's t-test, independent runs
  baseline  n=4  mean 81.25
  variant   n=4  mean 83.95
  shift     +2.70  95% CI [-0.84, +6.24]
  p         0.111
  glassD    1.31
  target    3 (higher is better)

CAN'T TELL YET. The 95% CI includes both no change and a 3-point shift.
Resolving it takes 16 runs per arm at the spread seen so far; you have 4.
```

- **Input** is JSON. An array is one score per run (Welch's t-test). An object of
  `{"item id": score}` pairs items across the two files by id (paired t-test). `-` reads stdin,
  so `jq '[.runs[].score]' results.json | node js/cli.mjs compare --baseline base.json --variant - --target 3` works.
  Scores must be under 1e15 in magnitude; rescale anything larger.
- **`--target` is required**: the smallest shift worth acting on, set before looking at results.
  `no_change` means the 95% CI rules out a shift that big, not that nothing moved. A shift that
  is significant but wholly inside the target is `no_change` too (reason `below_target`), so
  `--gate` fails a build only when the 95% CI reaches the target on the bad side. The 95% CI
  makes this stricter than the usual 90% equivalence test.
- **One constant arm** (every baseline run scored exactly the same) still gets a verdict, since
  Welch's test only needs spread in one arm. `constantArm` names it and the text output warns:
  a constant arm is a ceiling or a broken harness, and only you can tell which. Both arms
  constant, or every paired difference identical, is `cant_tell` (reason `zero_spread`).
- **`power`** sizes the comparison first: `node js/cli.mjs power --baseline pilot.json --target 3`
  gives the n with 80% power to detect a target-sized shift.
- **"Resolving it takes N"** in a `cant_tell` is a different, larger n: at the spread seen so
  far, the 95% CI at N is under half the target on each side, so it cannot span both no change
  and the target. Every verdict at N is a call.
- **`--json`** prints the `verdict()` result. **`--gate`** turns it into an exit code for CI:

| Exit | Meaning |
|---|---|
| 0 | `improved` or `no_change` (and every run without `--gate`) |
| 1 | `regressed` |
| 2 | Usage or input error |
| 3 | `cant_tell`: not enough data to call it either way |

A separate code for "can't tell yet" lets a pipeline hold a merge for more runs instead of
reading an underpowered result as a pass or a fail.

## API parity

Both bindings expose the same functions. JS uses camelCase; Python uses snake_case.
Return field names follow the same convention.

| Purpose | JS | Python | Returns |
|---|---|---|---|
| The decision: improved / regressed / no_change / cant_tell | `verdict(a, b, {target, paired, lowerIsBetter})` | `verdict(a, b, target, paired, lower_is_better)` | `{verdict, reason, shift, ciLo, ciHi, p, effect, needed, ...}` |
| Descriptive stats + 95% CI | `stats(arr)` | `descriptive_stats(arr)` | `{n, mean, std, cv, min, max, range, se, ciLo, ciHi, ciMargin}` |
| Welch's t-test + Glass's delta (`diff` = a minus b) | `welchTTest(a, b)` | `welch_t_test(a, b)` | `{t, df, p, diff, se, glassD, baselineStd}` |
| Paired t-test + Cohen's dz (`diff` = a minus b) | `pairedTTest(a, b)` | `paired_t_test(a, b)` | `{t, df, p, diff, se, dz, sdDiff, n}` |
| Runs per arm (power analysis) | `requiredN(std, delta)` | `required_n(std, delta)` | `number` |
| Paired items (power analysis) | `requiredNPaired(sdDiff, delta)` | `required_n_paired(sd_diff, delta)` | `number` |
| Two-tailed p-value from t-statistic | `pValue(absT, df)` | `p_value(abs_t, df)` | `number` |
| t-critical for 95% CI | `tCritical(df)` | `t_critical(df)` | `number` |

`approxPValue` / `approx_p_value` remain as aliases of `pValue` / `p_value` from
0.0.x, when the JS side approximated. Cross-validation holds every numeric output in both
implementations to scipy within 1e-9, including fractional Welch df. P-values are held
to 1e-9 relative (with a 1e-15 absolute floor), so small tail p-values keep their digits.

## Notable design choices

- **Glass's delta, not Cohen's d.** Cohen's d divides by a pooled standard deviation, a well-defined yardstick only when both groups share one spread. Welch's test is for when they may not. Glass's delta divides by the baseline's standard deviation alone, so the effect is measured in units of the thing you compare against. The trade: it is noisier at small n and not symmetric in the two groups.
- **`requiredN` is exact.** The smallest n per arm whose power under the noncentral t reaches 0.80 (alpha 0.05 two-tailed), matching statsmodels and Cohen's tables (d = 0.5 needs 64). The textbook z formula runs one to two short at small n. Past 10^8 runs per arm it returns the z value, which is within 1e-5 of exact there. It assumes the variant's spread matches the baseline's, so it underestimates when the variant is noisier.
- **`requiredNPaired` sizes items, not runs.** Each item already contributes both arms, so there is no factor of 2. The spread that matters is that of the per-item differences, which is small when items are correlated.
- **Sentinel over Infinity for effect sizes.** `glassD = 99` (or `dz = 99`) when the relevant std is 0, so threshold checks still fire and JSON stays intact. In that branch `t` is still `Infinity` (serializes to `null`); check `se === 0` first. At n = 1, `std`, `se` and `ciMargin` are 0 by construction (no spread estimate), not a claim of precision; the t-tests refuse n < 2 for this reason.
- **Full t-distribution without dependencies.** The JS p-value is a regularized incomplete beta function (Lanczos log-gamma, continued fraction); `tCritical` inverts it by bisection, and power analysis uses a noncentral t (Lenth's AS 243). No lookup tables, no normal approximation.

## Scope and related work

eval-kit compares two conditions with t-based tests. It is the right tool for mean scores that
are roughly continuous. It does not do:

- **Clustered standard errors** for items that come in related groups (several questions per
  document, say). See Miller 2024 below.
- **Small-n pass/fail accuracy.** For binary per-item scores with fewer than a few hundred items,
  t- and z-based intervals are unreliable; use exact binomial or Bayesian methods.
- **Bootstrap or rank-based tests.**

For those, [evalstats](https://github.com/ianarawjo/evalstats) and
[statsforevals.com](https://statsforevals.com/) are good places to start.

Reading that shaped this library:

- Evan Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (Anthropic, 2024).
  Standard errors, pairing, clustering, and power analysis for model evals.
- Bowyer, Aitchison, Ivanova, [Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred
  Datapoints](https://arxiv.org/abs/2503.01747) (2025).
- Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
  Multiple trials, and why larger evals are needed to detect smaller effects.

## Claude Code plugin

`SKILL.md` runs the bundled CLI rather than having Claude write statistics code. `evals/`
holds its `claude plugin eval` suite, and `tests/skill/` checks that suite in CI with no model
calls: every case's answer key is what the CLI really outputs, and every grader accepts the right
answer and rejects the wrong ones.

## Monorepo layout

```
SKILL.md                       Claude Code skill: the decision procedure
.claude-plugin/plugin.json     Plugin manifest
js/
  lib/stats.mjs                JS binding (zero deps, Node 20+)
  cli.mjs                      CLI: compare, power
  examples/verdicts.mjs        The README's three comparisons, reproducible
  test/stats.test.mjs          JS unit tests (node:test)
python/
  pyproject.toml               Python package manifest
  src/eval_kit/stats.py        Python binding (numpy + scipy)
  src/eval_kit/cli.py          Same CLI, same output (eval-kit, python -m eval_kit)
  tests/test_stats.py          Python unit tests (pytest, mirrors the JS unit tests)
tests/
  cross-validation/            Parity + scipy ground truth, verdict and CLI byte parity
  skill/                       Eval answer keys, graders, SKILL.md limits (no model calls)
evals/                         claude plugin eval suite for the skill
```

The Python binding is the reference implementation. The JS binding is
cross-validated against it. See `AGENTS.md` for the parity rule that governs
contributions.

## Tests

```bash
# JS
node --test js/test/stats.test.mjs

# Python (from python/)
cd python && pip install -e ".[dev]" && pytest

# Cross-validation (from python/, requires node in PATH)
cd python && pytest ../tests/cross-validation/ -v

# Skill contract: eval answer keys, graders, SKILL.md limits (from python/, no model calls)
cd python && pytest ../tests/skill/ -v

# The skill, with Claude Code (makes real model calls)
claude plugin eval . --allow-tools Write "Bash(node *)"
```

All four suites run in CI on every push and pull request to main, along with a `claude plugin validate` manifest check.
The live `claude plugin eval` run is a separate, manual workflow (`skill-evals.yml`) with a cost ceiling, since it
makes real model calls.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The parity rule: every function must
exist in both bindings with mirrored tests and cross-validation coverage.

## License

Apache 2.0 -- see [LICENSE](LICENSE) and [NOTICE](NOTICE).

> eval-kit is independently maintained and is not affiliated with, endorsed by,
> or sponsored by Anthropic. "Claude" and "Claude Code" are trademarks of
> Anthropic.

## Author

Anthony Ledesma ([@AnthonyLedesma](https://github.com/AnthonyLedesma)).
