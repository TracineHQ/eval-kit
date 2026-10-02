---
description: Binary pass/fail on 40 items. Out of scope for t-tests; the skill should say so and point to exact binomial or McNemar-style methods.
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill, Write]
---

We ran 40 test cases against both versions of our agent. Each case is pass or fail.

old version: 29 of 40 passed
new version: 33 of 40 passed

Same 40 cases for both. A 5-point gain in pass rate would matter to us. Is the new version better? Run a t-test on it if that's the right call.
