# eval-kit

Statistical confidence library for LLM evaluation. Two language bindings,
same API surface, cross-validated against scipy in CI.

## What this repo is

A zero-dependency JS library (`js/lib/stats.mjs`) and a scipy-backed Python
package (`python/src/eval_kit/`) that implement the same statistical
functions for comparing LLM eval runs: descriptive stats with CI, Welch's
t-test for independent runs, a paired t-test for items scored in both arms,
effect sizes, and power analysis.

**Python is the reference implementation** (exact, via scipy).
**JS is exact too**, with zero dependencies: p-values come from a
regularized incomplete beta function, t-critical values from inverting
it, and power from a noncentral t. Cross-validation holds every field to
scipy within 1e-9. Do not
reintroduce approximations to save lines.

## Architecture

    SKILL.md                     Claude Code skill (root SKILL.md loads as the plugin's skill)
    .claude-plugin/plugin.json   Plugin manifest
    js/
      lib/stats.mjs              JS binding (Node 20+, zero deps)
      cli.mjs                    CLI (compare, power); the skill runs this
      test/stats.test.mjs        JS unit tests (node:test)
    python/
      pyproject.toml             Python package manifest (hatchling)
      src/eval_kit/stats.py      Python binding (numpy + scipy)
      src/eval_kit/cli.py        Same CLI, identical output
      tests/test_stats.py        Python unit tests (pytest, mirrors the JS unit tests)
    tests/
      cross-validation/
        bridge.mjs               Spawned by Python to call JS functions
        test_cross_validation.py Parity + scipy ground truth suite
        test_verdict_and_cli.py  verdict() parity, CLI byte parity
      skill/
        test_skill_contract.py   Eval answer keys, graders, SKILL.md limits (no model calls)
    evals/                       claude plugin eval suite for the skill

## Rules

[CONTRIBUTING.md](CONTRIBUTING.md) is the single copy of the parity rule, test and lint
commands, formulas not to change, shared sentinels, code style, and license headers. Read it
before changing code. The ones agents break most:

- Change both bindings, both unit test files, and the cross-validation suite together.
- `js/lib/stats.mjs` stays zero-dependency.
- A cross-validation failure at 1e-9 is a bug. Do not widen the tolerance.

## Scope

This repo is `eval-kit`: the stats module plus one Claude Code skill
(`SKILL.md` at the repo root, which the plugin system loads as the single
skill; `name:` fixes its invocation name, keep it). Do not add plugin
components (commands, hooks, more skills) without confirmation.
