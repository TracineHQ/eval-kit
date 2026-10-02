---
description: A latency metric, where lower is better. The skill must pass --lower-is-better and call the drop an improvement.
max_turns: 15
allowed_tools: [Read, Glob, Grep, Skill, Write]
---

We changed our agent's tool-routing logic. p95 latency in ms over 8 load-test runs each:

before: 412, 398, 425, 407, 419, 402, 415, 410
after: 388, 379, 395, 384, 391, 381, 390, 386

We only care about changes of 10 ms or more. Did the change help or hurt?
