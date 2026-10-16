## Summary

<!-- What changes, and why. Link the issue if there is one. -->

## Test plan

<!-- The suites you ran (CONTRIBUTING.md lists them) and anything checked by hand. -->

## Checklist

<!-- Tick what applies; strike through or delete what doesn't. -->

- [ ] Both bindings changed together (`js/lib/stats.mjs`, `python/src/eval_kit/stats.py`)
- [ ] Both unit test files updated (`js/test/stats.test.mjs`, `python/tests/test_stats.py`)
- [ ] Cross-validation entry added or updated (`tests/cross-validation/`)
- [ ] CLI output byte-identical across `js/cli.mjs` and `python/src/eval_kit/cli.py`
- [ ] CHANGELOG.md entry under `[Unreleased]`
- [ ] 1e-9 cross-validation tolerance not widened
- [ ] Eval prompts and their answer keys updated together (`evals/`)
