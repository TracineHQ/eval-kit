---
description: Per-item scores for the same 12 questions. Paired, the gain is clear (p < 0.001); treated as independent runs it looks like noise (p 0.60).
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill, Write]
---

I scored the same 12 eval questions with the old and new retrieval setup. Per-question accuracy:

| question | old | new |
|---|---|---|
| q1 | 0.62 | 0.66 |
| q2 | 0.81 | 0.83 |
| q3 | 0.45 | 0.50 |
| q4 | 0.90 | 0.91 |
| q5 | 0.58 | 0.63 |
| q6 | 0.73 | 0.75 |
| q7 | 0.39 | 0.44 |
| q8 | 0.85 | 0.88 |
| q9 | 0.67 | 0.70 |
| q10 | 0.51 | 0.55 |
| q11 | 0.77 | 0.80 |
| q12 | 0.60 | 0.63 |

A 0.02 gain in accuracy matters to us. The averages are close and the spread across questions is huge. Is the new setup actually better, or is this noise?
