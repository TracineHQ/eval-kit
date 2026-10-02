# eval-kit (Python)

Did the eval get better, or is that noise? The Python binding.

Uses `scipy.stats` as the trusted math layer. Same functions as the JS binding
at `../js/`, in snake_case. The CLI prints the same output as the JS one.

## Install

Not on PyPI. Install from a clone:

```bash
git clone https://github.com/TracineHQ/eval-kit
cd eval-kit/python && pip install -e .
```

## Quick start

```python
from eval_kit import descriptive_stats, welch_t_test, required_n

baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87]
variant = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81]

print(welch_t_test(baseline, variant))
print(required_n(descriptive_stats(baseline).std, 5))
```

`verdict(baseline, variant, target)` gives the decision (improved, regressed,
no_change, or cant_tell) as a dataclass. The same decision is on the command
line, with the same text output as the JS CLI:

```bash
eval-kit compare --baseline base.json --variant new.json --target 3
eval-kit power --baseline pilot.json --target 3
```

See the top-level [README](../README.md) for input formats, exit codes, and
methodology.

## License

Apache 2.0 -- see [LICENSE](../LICENSE) at repo root.
