---
type: llm
---

PASS if the response flags that five identical baseline scores are suspicious (a likely harness, caching, scoring, or determinism problem) and advises checking the eval setup before trusting any conclusion.
FAIL if the response declares the new model better (or not better) without questioning the identical baseline scores.
