---
name: eval-kit
description: Decides whether an LLM eval change is real or noise (A/B tests of prompts, models, or agents), using Welch's or paired t-tests, effect sizes, confidence intervals, and power analysis instead of eyeballing means. Covers which test fits the data, how many runs or items to collect before testing (sample size), how to read the verdict, and when the data cannot support a conclusion at all.
when_to_use: When the user compares two sets of eval scores, prompt versions, or model runs and wants to know if a difference is real; when they report a regression or improvement from a small number of runs; when they ask whether a change is statistically significant or how many runs they need; when they ask what a p-value, effect size, or confidence interval means for their eval; or when an eval score, benchmark, or per-run metric has moved and someone is about to ship or revert based on it.
allowed-tools: Bash(node ${CLAUDE_PLUGIN_ROOT}/js/cli.mjs *)
---

# Deciding whether an eval difference is real

Two mean scores differing is not evidence. This skill is the procedure for telling a real shift
from sampling noise, and for saying "we cannot tell yet" when that is the honest answer. Run the
bundled CLI for every number. Do not compute t-tests, p-values, or sample sizes by hand or in
ad hoc code.

```bash
node ${CLAUDE_PLUGIN_ROOT}/js/cli.mjs compare --baseline base.json --variant new.json --target 3
node ${CLAUDE_PLUGIN_ROOT}/js/cli.mjs power --baseline pilot.json --target 3
```

Needs Node 20+, no install. Add `--json` for machine-readable output. `--help` lists every flag.

## Step 1: Get the target before anything else

`--target` is required: the smallest shift worth acting on, in score units. It is a product
decision, not a statistical one. Ask the user for it, or propose one and confirm, before looking
at results. Picking it after seeing the numbers turns the test into a rationalization. If lower
scores are better (latency, error rate), add `--lower-is-better`.

## Step 2: Pick the design from how the data was collected

- **Independent runs.** Each number is one full pass over the suite. Write each arm as a JSON
  array: `[81.2, 79.8, 83.1]`. The CLI uses Welch's t-test.
- **Same items in both arms.** Each number is one item's score. Write each arm as an object keyed
  by item id: `{"q1": 0.8, "q2": 0.6}`. The CLI pairs by id and uses the paired t-test, which
  cancels item difficulty and needs far fewer items.

Never compare per-item scores as independent arrays. If per-item scores are already two arrays in
the same item order, add `--paired`.

Write each arm to its own JSON file with the Write tool, then run the command exactly as shown:
no `cd`, pipes, or heredocs, so it matches the pre-approved command. Convert CSV, JSONL, or tables
into these two shapes first.

Out of scope: items in related groups (use clustered standard errors) and pass/fail scores on
fewer than a few hundred items (use exact binomial or Bayesian methods). Say so instead of forcing
a t-test onto them.

## Step 3: Size it before running it

If the comparison has not been run yet, or has only a few runs, run `power` first on a baseline
(or a small paired pilot with both `--baseline` and `--variant`). Running 3 versus 3 and testing
afterward usually wastes the runs.

`power` is exact (noncentral t, 80% power, alpha 0.05) and already per arm. It assumes the
variant's spread matches the baseline's, so treat it as a floor when the variant may be noisier.
If it asks for more full runs than the user will ever do, that is the answer: the run-level
design cannot see this shift. Recommend per-item scoring and the paired design instead of
running five and hoping. `power` answers "can we detect a target-sized shift?" (80% of the
time). The n in a `cant_tell` answers "when must the result be a call?", so it is larger.

## Step 4: Run compare and report the verdict

| Verdict | Meaning | What to tell the user |
|---|---|---|
| `improved` / `regressed` | p < 0.05 and the 95% CI reaches the target; direction from the shift and `--lower-is-better` | The shift with its 95% CI. If the CLI says the estimate is below the target, say so: the shift is real and the CI reaches the bar, but the best guess sits under it. |
| `no_change` | The 95% CI rules out a shift of target or more | No meaningful change at this target. Not "no change at all": with reason `below_target` the shift is real, just smaller than the bar. |
| `cant_tell` (underpowered) | The CI includes both zero and a shift of target in either direction | Not yet. Give the n the CLI reports: at the spread seen so far, the result at that n has to be a call. Do not call it a null result, and do not ship or revert on it. |
| `cant_tell` (zero spread) | Every run in both arms, or every item difference, is identical | Almost always a broken harness. Stop and inspect it before interpreting anything. |

If the output also warns that every run in one arm scored the same (`constantArm` in JSON), the
verdict stands, but lead with the warning: that arm is at a ceiling or its harness is broken. Ask
which before anyone acts on the verdict.

Get to the reported n with a fresh run at that n, not by adding runs to the batch already
tested and checking again. Topping up until p clears 0.05 inflates false positives. Before
anything has run, set n from `power` up front.

**More than one comparison** (k variants against one baseline, or k metrics): a p < 0.05 turns
up by chance about k times as often. Count an `improved` or `regressed` verdict only if its p is
below 0.05 / k; otherwise report that comparison as can't tell yet. Say k and the adjusted bar in
the report.

## When the CLI fails

- **Exit 2** prints the reason on stderr (bad JSON, fewer than 2 scores in an arm, item ids that
  don't match, a missing `--target`). Fix the input and rerun.
- **No `node`** (Node 20+ required): say so and stop. Offer the Python binding as the alternative
  (`pip install -e python/` from a clone, then `eval-kit` with the same flags). Do not estimate the
  numbers instead.
- **More than two arms**: compare each variant with the baseline separately and apply the
  multiple-comparison rule above.

## CI gating

`--gate` turns the verdict into an exit code: 0 improved or no_change, 1 regressed, 2 usage or
input error, 3 cant_tell. Without `--gate` a successful analysis always exits 0.

## Reporting

Give the shift in real units with its 95% CI, the effect size, p, n per arm, and the target set in
advance. A result without n is not reproducible, and n is what tells the next person whether to
believe a null.

## Using the library directly

For code that needs the numbers, `${CLAUDE_PLUGIN_ROOT}/js/lib/stats.mjs` exports `verdict` (the
same decision as the CLI) and the functions under it: `welchTTest`, `pairedTTest`, `stats`,
`requiredN`, `requiredNPaired`, `pValue`, `tCritical`. Their `diff` is baseline minus variant, the
opposite sign of the CLI's shift.
