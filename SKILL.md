---
name: eval-kit
description: Decides whether an LLM eval change is real or noise (A/B tests of prompts, models, or agents), using Welch's or paired t-tests, effect sizes, confidence intervals, and power analysis instead of eyeballing means. Covers which test fits the data, how many runs or items to collect before testing (sample size), how to read the verdict, and when the data cannot support a conclusion at all.
when_to_use: When the user compares two sets of eval scores, prompt versions, or model runs and wants to know if a difference is real; when they report a regression or improvement from a small number of runs; when they ask whether a change is statistically significant or how many runs they need; when they ask what a p-value, effect size, or confidence interval means for their eval; or when an eval score, benchmark, or per-run metric has moved and someone is about to ship or revert based on it; when they ask whether a latency, cost, or pass-rate change is real, or how many times to repeat each item.
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
- **Several calls per item.** If each item was asked k times per arm, average each item's calls
  into one score per arm and write the object above. Never key each call as its own item: calls
  on one item are not independent, and the CLI would count k times the items there are. The CLI
  takes one number per item; an array per item is an input error.

Never compare per-item scores as independent arrays. If per-item scores are already two arrays in
the same item order, add `--paired`.

If scores can drift while the arms run (latency, rate limits, caches), interleave the arms,
alternate which arm goes first per item, drop a warm-up call, and record the call order. Run a
pilot the same way as the real run.

Write each arm to its own JSON file with the Write tool, then run the command exactly as shown:
no `cd`, pipes, or heredocs, so it matches the pre-approved command. Convert CSV, JSONL, or tables
into these two shapes first.

Out of scope: items in related groups (use clustered standard errors) and pass/fail scores on
fewer than a few hundred items (use exact binomial or Bayesian methods). Say so instead of forcing
a t-test onto them. For pass/fail, report each arm's count and how many items passed in one arm
and failed in the other, and put no p-value on them.

## Step 3: Size it before running it

If the comparison has not been run yet, or has only a few runs, run `power` first on a baseline,
or on a small pilot of both arms with `--baseline` and `--variant`. `power` reads the design from
the files the way `compare` does: arrays are independent runs; objects keyed by item id, or arrays
with `--paired`, are paired items. Running 3 versus 3 and testing afterward usually wastes the runs.

`power` is exact (noncentral t, 80% power, alpha 0.05) and already per arm. From a baseline alone
it assumes the variant's spread matches the baseline's, so treat it as a floor when the variant
may be noisier; a two-arm pilot sizes from the pooled std of both arms instead. If `power` warns
that there is no spread to size from, its n is the minimum, not a plan: check the harness first.
If it asks for more full runs than the user will ever do, that is the answer: the run-level
design cannot see this shift. Recommend per-item scoring and the paired design instead of
running five and hoping.

If the paired design asks for more items than the suite has, cut the noise inside each item
instead. Ask each item k times per arm (start at 3), interleave the arms so drift in the
provider hits both, score each item by its mean, and size again from a pilot at that k. Repeats
shrink only the call-to-call part of the spread. If `power` still asks for more items than exist
at a k the user can afford, say this suite cannot see the target, and offer the choices: more
items, or a larger target the user sets on product grounds, not to fit the data.

A pilot's spread is an estimate too, and n moves with its square: call an n from a pilot of under
about 10 items a rough size, not a plan, and pilot more when that n decides whether the comparison
is feasible. To size the next paired run from a finished one, run `power` on the same two files:
it prints the sd of differences.

`power` answers "can we detect a target-sized shift?" (80% of the time). The n in a `cant_tell`
answers "when must the result be a call?", so it is larger.

## Step 4: Run compare and report the verdict

| Verdict | Meaning | What to tell the user |
|---|---|---|
| `improved` / `regressed` | p < 0.05 and the 95% CI reaches the target; direction from the shift and `--lower-is-better` | The shift with its 95% CI. If the CLI says the estimate is below the target, say so: the shift is real and the CI reaches the bar, but the best guess sits under it. |
| `no_change` | The 95% CI rules out a shift of target or more | No meaningful change at this target. Not "no change at all": with reason `below_target` the shift is real, just smaller than the bar. |
| `cant_tell` (underpowered) | The CI includes both zero and a shift of target in either direction | Not yet. Give the n the CLI reports: at the spread seen so far, the result at that n has to be a call. Do not call it a null result, and do not ship or revert on it. |
| `cant_tell` (zero spread) | Every run in both arms, or every item difference, is identical | Almost always a broken harness. Stop and inspect it before interpreting anything. The CI, p, and effect size are undefined (null in `--json`); do not put numbers on them. |

If the output also warns that every run in one arm scored the same (`constantArm` in JSON), the
verdict stands, but lead with the warning: that arm is at a ceiling or its harness is broken. Ask
which before anyone acts on the verdict.

Get to the reported n with a fresh run at that n, not by adding runs, or repeats per item, to the
batch already tested and checking again. Topping up until p clears 0.05 inflates false positives.
Before anything has run, set n from `power` up front.

**More than one comparison** (k variants against one baseline, or k metrics): a p < 0.05 turns
up by chance about k times as often. Count an `improved` or `regressed` verdict only if its p is
below 0.05 / k; otherwise report that comparison as can't tell yet. Say k and the adjusted bar in
the report. With several metrics on one run (latency and pass rate, say), name the one the
decision rides on before the run, give each continuous metric its own target, and report every
metric side by side, one block each.

## Latency and other skewed metrics

Per-call latency is right-skewed: a few slow calls (cold starts, retries, timeouts) set the
spread. Average repeats per item before testing; the paired test then reads per-item mean
differences, which sit much closer to symmetric. Give the target and shift in the metric's unit,
and the direction in words: with `--lower-is-better` a negative shift is the improvement, so
report "0.38 s faster", not "-0.38". The CLI tests a difference in units, not a ratio.

Do not drop a slow call or an odd item after seeing the results. An exclusion rule (a warm-up
call, a timeout cap) is set before the run, like the target. If one item or call sits far outside
the rest, name it and check the harness for it before reporting.

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
input error, 3 cant_tell. A constant arm (`constantArm` set) also exits 3 unless the verdict is
regressed: the gate cannot check whether that arm is a ceiling or a broken harness, so it holds.
Without `--gate` a successful analysis always exits 0.

## Reporting

Lead with the verdict (with a constant arm, its warning first), then fill in this block from the
CLI output, one per comparison. Do not round p past what the CLI prints, and do not add
statistics it did not print.

    <VERDICT> at a <target> <unit> target (<higher or lower> is better), set before the run.
    Shift <variant minus baseline> <unit> (<direction in words>), 95% CI [<lo>, <hi>], p <p>, <glassD or dz> <effect>.
    <Welch: n runs per arm> or <paired: n items, k calls per item averaged>.
    <cant_tell: the n the CLI gives, as a fresh run.>
    <More than one comparison: k and the bar, 0.05 / k.>

A result without n is not reproducible, and n is what tells the next person whether to believe a
null.

## Using the library directly

For code that needs the numbers, `${CLAUDE_PLUGIN_ROOT}/js/lib/stats.mjs` exports `verdict` (the
same decision as the CLI) and the functions under it: `welchTTest`, `pairedTTest`, `stats`,
`requiredN`, `requiredNPaired`, `pValue`, `tCritical`. Their `diff` is baseline minus variant, the
opposite sign of the CLI's shift.
