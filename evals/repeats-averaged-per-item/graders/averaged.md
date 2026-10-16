---
type: llm
---

PASS if the response averages each item's 3 requests into one score per item per config and compares the configs paired by item.
FAIL if the response treats each request as its own item or as independent samples, uses only one request per item, or compares the two configs' overall averages without pairing by item.
