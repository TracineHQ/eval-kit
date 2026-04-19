# tracine-eval

Statistical evaluation toolkit for LLM pipelines. Zero dependencies.

## What's in here

`lib/stats.mjs` -- the stats library from [How I Built an LLM Eval Framework That Caught a 39-Point Regression](https://linkedin.com/in/anthonyledesma).

- **Descriptive stats** with 95% CI (t-distribution, Bessel's correction)
- **Welch's t-test** for comparing baseline vs variant runs (handles unequal variance)
- **Glass's delta** effect size (uses baseline SD -- consistent with Welch's)
- **Power analysis** to right-size your eval budget
- **P-value approximation** (A&S normal CDF for large samples, conservative buckets for small)

## Quick start

```javascript
import { stats, welchTTest, requiredN } from './lib/stats.mjs';

const baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87];
const variant  = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81];

console.log(welchTTest(baseline, variant));
// => { t, df, p, diff, se, glassD, baselineStd }

console.log(requiredN(stats(baseline).std, 5));
// => runs per group to detect a 5-point shift
```

## Why Glass's delta instead of Cohen's d?

Cohen's d pools standard deviations from both groups, assuming equal variance. But the whole reason we use Welch's t-test is that variances aren't equal. Glass's delta uses only the baseline group's SD. Same interpretation scale (small ~0.2, medium ~0.5, large ~0.8), consistent assumptions.

## License

MIT
