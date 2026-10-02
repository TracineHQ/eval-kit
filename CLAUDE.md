@AGENTS.md

## Claude Code extensions

### Plan mode

Use plan mode before any change that touches mathematical formulas in
`js/lib/stats.mjs` or `python/src/eval_kit/stats.py`. Formula changes
require cross-validation evidence and tolerance justifications.

### Preferred workflow

1. Edit JS and Python bindings together in the same change.
2. Update the mirrored unit tests in both test files.
3. Run all three test suites locally before committing:
   - `node --test js/test/stats.test.mjs`
   - `cd python && pytest`
   - `cd python && pytest ../tests/cross-validation/ -v`
4. Commit with a one-line message in the project's style (no type prefix,
   first person, concise).
