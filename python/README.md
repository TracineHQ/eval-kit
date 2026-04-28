# eval-kit (Python)

Statistical confidence for LLM evaluation -- Python binding.

Uses `scipy.stats` as the trusted math layer. Same function surface as the JS
binding at `../js/`, so result files, cassettes, and documentation are
interchangeable across language bindings.

## Install

```bash
pip install eval-kit
```

## Quick start

```python
from eval_kit.stats import descriptive_stats, welch_t_test, required_n

baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87]
variant  = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81]

print(welch_t_test(baseline, variant))
print(required_n(descriptive_stats(baseline).std, 5))
```

See the top-level [README](../README.md) for methodology.

## License

Apache 2.0 -- see [LICENSE](../LICENSE) at repo root.
