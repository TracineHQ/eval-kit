# Contributing to eval-kit

Thanks for considering a contribution. This repo is small but has strict
parity and test discipline. Read this before opening a PR.

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
   agree on the function's output within documented tolerances.

PRs that add a function to only one binding will not be merged.

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

All three suites must pass before submitting a PR.

## Code style

**JS:**
- ESM modules only. No CommonJS.
- No transpilation. Ship the `.mjs` as-is.
- JSDoc on every exported symbol.
- Node built-ins only in `js/lib/`. No runtime dependencies.

**Python:**
- `@dataclass` for return types.
- Docstrings on every public function (what + why, not how).
- Type annotations on new code.
- scipy / numpy acceptable and expected in Python binding.

## Where not to invent math

All statistical formulas are standard. Do not change:

- Welch-Satterthwaite df approximation
- Glass's delta denominator (baseline std only)
- The factor of 2 in `requiredN` / `required_n`
- p-value tolerances in cross-validation without measuring against scipy exact

If you believe a formula is wrong, open an issue with a citation before writing code.

## Sentinels

- `glassD = 99` when baseline std is 0 and diff != 0.
- `glassD = 0` when both groups are identical.
- `requiredN(std, 0) = Infinity`.
- Empty input returns `null` / `None`.

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

Good: `Add approx_p_value buckets for small df`
Good: `Fix CV sign for negative-mean inputs`
Bad: `feat(stats): add new p-value function`

## Reporting vulnerabilities

Do not open a public issue for security problems. See [SECURITY.md](SECURITY.md).

## Code of conduct

Be constructive. Assume good faith. Attack ideas, not people. This is an
early-stage project; issues and PRs that help it mature are welcome.
