---
description: Every request, 3 per item per arm. The skill must average each item's requests, pair by item, and report a real 0.38 s speedup that is below the 1 s target (no_change, below_target).
max_turns: 20
allowed_tools: [Read, Glob, Grep, Skill, Write]
---

Latency A/B on our assistant endpoint, old config vs new config. Same 20 items in both, each item sent 3 times per config, with the configs alternating request by request. Seconds per request, lower is better:

| item | old config (s) | new config (s) |
|---|---|---|
| r1 | 3.652, 3.98, 3.455 | 3.466, 3.504, 3.595 |
| r2 | 3.181, 4.033, 4.258 | 3.589, 3.671, 3.435 |
| r3 | 5.841, 4.757, 5.808 | 4.995, 4.709, 5.304 |
| r4 | 5.843, 5.505, 6.443 | 4.442, 5.251, 5.509 |
| r5 | 5.776, 5.298, 4.292 | 4.926, 4.822, 3.667 |
| r6 | 4.193, 7.482, 6.391 | 3.597, 6.922, 3.766 |
| r7 | 5.605, 3.167, 3.07 | 2.998, 3.043, 2.879 |
| r8 | 3.694, 3.361, 4.333 | 3.339, 3.05, 3.416 |
| r9 | 4.737, 4.748, 5.23 | 4.262, 4.927, 4.561 |
| r10 | 9.835, 6.765, 6.083 | 6.712, 6.666, 5.594 |
| r11 | 4.964, 4.185, 3.761 | 3.838, 4.273, 5.417 |
| r12 | 5.343, 5.076, 5.922 | 4.89, 4.758, 4.834 |
| r13 | 7.086, 4.112, 4.004 | 4.486, 6.903, 5.293 |
| r14 | 3.132, 3.128, 2.92 | 3.337, 4.657, 2.979 |
| r15 | 4.175, 5.43, 5.516 | 6.69, 4.018, 6.408 |
| r16 | 4.855, 4.925, 3.524 | 4.544, 4.13, 4.47 |
| r17 | 3.736, 4.713, 6.253 | 4.339, 4.065, 4.75 |
| r18 | 8.09, 6.739, 4.778 | 4.636, 5.95, 5.105 |
| r19 | 3.574, 4.196, 4.485 | 4.151, 4.381, 4.69 |
| r20 | 4.1, 4.996, 3.701 | 3.913, 3.704, 3.073 |

We agreed up front that only a change of 1 second or more matters. Did the new config make it faster?
