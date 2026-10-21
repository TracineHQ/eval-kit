---
type: llm
---

PASS if the response says the pilot calls for about 50 items, more than the 20 the suite has, and recommends sending each item several times per config, scoring each item by the mean of its requests, and sizing again from a pilot at that number of repeats.
FAIL if the response recommends running the 20 items once each and testing anyway, treats each repeated request as a separate item, tells the user to find 50 items without addressing that only 20 exist, or never runs a power or sample-size calculation.
