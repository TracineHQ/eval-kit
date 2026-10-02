---
description: Sizing before the comparison. Exact answer is 16 runs per arm (noncentral t, 80% power).
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill, Write]
---

I'm about to A/B two prompt versions on our eval suite. Each full run takes about 20 minutes. I ran the current prompt 5 times to see how noisy it is: 71.2, 74.8, 69.9, 73.5, 72.1.

I want to be able to detect a 2-point change. How many runs of each version should I do before comparing?
