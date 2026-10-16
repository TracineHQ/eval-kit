---
description: A 6-item paired pilot asks for 50 items; the suite has 20. The answer is repeats per item, averaged, and a fresh size at that k, with a warning that a 6-item pilot is a rough size.
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill, Write]
---

I'm A/B testing a config change on our assistant endpoint. The metric is latency in seconds per request, lower is better, and we only care about a change of 1 second or more. Our test suite has 20 items. I piloted the first 6 items, one request per item per config:

| item | old config (s) | new config (s) |
|---|---|---|
| r1 | 3.532 | 5.737 |
| r2 | 6.429 | 3.576 |
| r3 | 5.765 | 4.315 |
| r4 | 5.895 | 9.065 |
| r5 | 4.434 | 4.623 |
| r6 | 6.092 | 3.739 |

How should I run the full comparison so it can actually answer the question?
