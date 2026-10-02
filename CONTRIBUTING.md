# Contributing to eval-kit

This repo is small but has strict parity and test discipline. Read this
before opening a PR.

## Parity rule

Every exported function must exist in both bindings:

- `js/lib/stats.mjs` (camelCase, zero deps)
- `python/src/eval_kit/stats.py` (snake_case, scipy-backed)

When you add, modify, or remove a function, you must update **both** bindings
plus **both** test files plus the cross-validation suite.

1. Mirrored unit tests in `js/test/stats.test.mjs` and
   `python/tests/test_stats.py` -- same test name pattern, same edge cases,
   cross-linked regression comments.
2. A parametrize entry (or dedicated test) in
   `tests/cross-validation/test_cross_validation.py` asserting both bindings
   agree on the function's output to 1e-9.

PRs that add a function to only one binding will not be merged.

The same rule covers the CLI: `js/cli.mjs` and `python/src/eval_kit/cli.py`
must print identical text, errors, and exit codes, and `--json` output that differs only in
the last digits of computed floats (held to 1e-9).
`tests/cross-validation/test_verdict_and_cli.py` checks them.

Every eval case has an `answer-key.json`: the data in its prompt and the answer the
skill should reach. `tests/skill/` checks, with no model calls, that the CLI really
produces that answer, that the graders accept it, and that SKILL.md stays within the
published skill limits. Change a prompt, and update its answer key in the same commit.

The skill has its own suite under `evals/`, run with `claude plugin eval`.
Cases need Bash for the CLI, so grant it at run time:

    claude plugin eval . --allow-tools Write "Bash(node *)"

A run makes real model calls. Add `--runs 1 --ablation none` while iterating.

## Development setup

**JavaScript:**

Requires Node.js 20+. No install step -- the binding uses only Node built-ins.

```bash
node --test js/test/stats.test.mjs
```

**Python:**

Requires Python 3.10+.

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

**Cross-validation** (runs both bindings via a subprocess bridge, requires Node in PATH):

```bash
cd python
pytest ../tests/cross-validation/ -v
```

**Skill contract** (the eval suite's answer keys and graders, no model calls):

```bash
cd python
pytest ../tests/skill/ -v
```

All four suites, plus the lint and typecheck below, must pass before submitting a PR.

## Code style

**JS:**
- ESM modules only. No CommonJS.
- No transpilation. Ship the `.mjs` as-is.
- JSDoc on every exported symbol.
- Node built-ins only in `js/lib/`. No runtime dependencies.

**Python:**
- `@dataclass` for return types.
- Docstrings on every public function (what + why, not how).
- Type annotations everywhere in `src/`; `mypy --strict` must pass.
- scipy / numpy acceptable and expected in Python binding.
- CI runs, from `python/`: `ruff check . ../tests`, `ruff format --check . ../tests`, and
  `mypy`. Run `ruff check --fix` and `ruff format` before pushing.

## Where not to invent math

All statistical formulas are standard. Do not change:

- Welch-Satterthwaite df approximation
- Glass's delta denominator (baseline std only)
- Exact power in `requiredN` / `required_n` and the paired versions: noncentral t, alpha 0.05
  two-sided, power 0.80. Matches statsmodels; `test_required_n_matches_statsmodels` pins it.
- The 1e-9 cross-validation tolerance. A failure there is a bug, not a tolerance to widen.

If you believe a formula is wrong, open an issue with a citation before writing code.

## Sentinels

- `glassD = 99` when baseline std is 0 and diff != 0 (Infinity breaks
  JSON.stringify and comparison thresholds; 99 means "off-scale").
- `glassD = 0` when both groups are identical (std=0, diff=0).
- `dz = 99` when every paired difference is identical and nonzero; `dz = 0` when the arms are identical.
- `requiredN(std, 0) = Infinity`, same for `requiredNPaired`.
- `pairedTTest` returns `null` / `None` for mismatched lengths or n < 2.
- Empty input returns `null` / `None` from `stats` / `descriptive_stats`.

These are shared across bindings. Changes require a matching change in the
other binding + cross-validation test update.

## License headers for new source files

New `.mjs` or `.py` files in `js/lib/` or `python/src/` must include the
Apache 2.0 attribution in the module docstring:

```
Copyright 2026 TracineHQ
Licensed under the Apache License, Version 2.0.
See LICENSE and NOTICE files at repository root.
```

## Commit messages

One line. First person. No type prefix. Describe the change, not the files.

Good: `Add paired t-test to both bindings`
Good: `Fix CV sign for negative-mean inputs`
Bad: `feat(stats): add new p-value function`

## Reporting vulnerabilities

Do not open a public issue for security problems. See [SECURITY.md](SECURITY.md).

## Code of conduct

Be constructive and assume good faith. Issues and PRs are welcome.
