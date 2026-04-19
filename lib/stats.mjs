/**
 * llm-eval-stats -- Statistical confidence for LLM evaluation
 *
 * LLM scores are non-deterministic. The same prompt, same model, same input
 * produces different scores every run. This library tells you whether a
 * change is real or noise.
 *
 * Built for rubric-based LLM scoring systems where you're comparing
 * baseline vs variant runs and need to know: did this actually help?
 *
 * What's in here:
 *   - Descriptive stats with 95% CI (t-distribution, Bessel's correction)
 *   - Welch's t-test (handles unequal variance between runs)
 *   - Glass's delta effect size (uses baseline SD as denominator)
 *   - Power analysis (how many runs do you need to detect a given shift?)
 *   - P-value approximation (A&S normal CDF for large samples,
 *     conservative buckets for small -- see approxPValue docs for limits)
 *
 * Quick start:
 *   import { stats, welchTTest, requiredN } from './llm-eval-stats.mjs';
 *
 *   const baselineScores = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87];
 *   const variantScores  = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81];
 *
 *   console.log(welchTTest(baselineScores, variantScores));
 *   // => { t, df, p, diff, se, glassD, baselineStd }
 *
 *   console.log(requiredN(8, 5));
 *   // => 41 (runs per group to detect a 5-point shift)
 *
 * Companion article: "How I Built an LLM Eval Framework That Caught
 * a 39-Point Regression" -- [LinkedIn article link]
 *
 * Author: Anthony Ledesma
 * License: MIT
 */

// t-critical values for 95% CI (two-tailed, alpha=0.05)
const T_CRIT = {
  1: 12.706,
  2: 4.303,
  3: 3.182,
  4: 2.776,
  5: 2.571,
  6: 2.447,
  7: 2.365,
  8: 2.306,
  9: 2.262,
  10: 2.228,
  14: 2.145,
  19: 2.093,
  24: 2.064,
  29: 2.045,
  49: 2.01,
  99: 1.984,
};

/**
 * Look up or interpolate t-critical value for given degrees of freedom.
 * Falls back to z=1.96 for large df.
 */
export function tCritical(df) {
  if (df <= 0) return 0;
  if (T_CRIT[df]) return T_CRIT[df];
  const keys = Object.keys(T_CRIT)
    .map(Number)
    .sort((a, b) => a - b);
  for (let i = 0; i < keys.length - 1; i++) {
    if (df >= keys[i] && df <= keys[i + 1]) {
      const frac = (df - keys[i]) / (keys[i + 1] - keys[i]);
      return T_CRIT[keys[i]] * (1 - frac) + T_CRIT[keys[i + 1]] * frac;
    }
  }
  return 1.96; // fallback to z for large df
}

/**
 * Descriptive statistics with 95% confidence interval.
 * Uses Bessel's correction (n-1) for sample variance.
 * Needs at least 2 observations for meaningful CI; returns null for empty input.
 *
 * @param {number[]} values - Array of numeric observations
 * @returns {{ n, mean, std, cv, min, max, range, se, ciLo, ciHi, ciMargin } | null}
 */
export function stats(values) {
  const n = values.length;
  if (n === 0) return null;
  const mean = values.reduce((s, v) => s + v, 0) / n;
  const variance =
    n > 1 ? values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
  const std = Math.sqrt(variance);
  const cv = mean !== 0 ? (std / Math.abs(mean)) * 100 : 0;
  let min = values[0], max = values[0];
  for (let i = 1; i < n; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  const se = n > 1 ? std / Math.sqrt(n) : 0;
  const df = n - 1;
  const t = df > 0 ? tCritical(df) : 0;
  const ciLo = mean - t * se;
  const ciHi = mean + t * se;
  const ciMargin = t * se;
  return { n, mean, std, cv, min, max, range: max - min, se, ciLo, ciHi, ciMargin };
}

/**
 * Normal CDF complement via Abramowitz & Stegun approximation.
 * Used internally for p-value computation.
 */
function normalCdfComplement(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804 * Math.exp((-x * x) / 2);
  const p =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x >= 0 ? p : 1 - p;
}

/**
 * Approximate two-tailed p-value from t-statistic.
 *
 * For df >= 30, uses Abramowitz & Stegun normal CDF approximation --
 * accurate to ~4 decimal places, good enough for eval decisions.
 *
 * For df < 30, returns conservative bucketed estimates (0.01, 0.05, 0.1,
 * or 0.5) based on t-critical thresholds. This is intentionally coarse --
 * if you need precise small-sample p-values, use a proper t-distribution
 * CDF (jStat, scipy, etc). For eval work with 30+ runs per group, the
 * normal approximation path handles it.
 *
 * @param {number} absT - Absolute value of the t-statistic
 * @param {number} df - Degrees of freedom
 * @returns {number} Two-tailed p-value (approximate)
 */
export function approxPValue(absT, df) {
  if (absT > 10) return 0.0001;
  if (df >= 30) {
    return 2 * normalCdfComplement(absT);
  }
  // Conservative buckets for small df -- see JSDoc for limitations
  const tReq = tCritical(Math.max(1, Math.round(df)));
  if (absT >= tReq * 1.3) return 0.01;
  if (absT >= tReq) return 0.05;
  if (absT >= tReq * 0.8) return 0.1;
  return 0.5;
}

/**
 * Welch's t-test (unequal variance, two-sample).
 *
 * Use this instead of Student's t-test when comparing LLM eval runs --
 * different models/prompts produce different variance, not just different means.
 *
 * Effect size is Glass's delta (uses baseline SD as denominator), which is
 * consistent with the unequal-variance assumption. Interpretation is the
 * same as Cohen's d: small ~0.2, medium ~0.5, large ~0.8.
 *
 * @param {number[]} a - Baseline scores (the reference group)
 * @param {number[]} b - Variant scores (the thing you're testing)
 * @returns {{ t, df, p, diff, se, glassD, baselineStd } | null}
 */
export function welchTTest(a, b) {
  const sA = stats(a);
  const sB = stats(b);
  if (!sA || !sB || sA.n < 2 || sB.n < 2) return null;

  const diff = sA.mean - sB.mean;
  const seA2 = sA.std ** 2 / sA.n;
  const seB2 = sB.std ** 2 / sB.n;
  const se = Math.sqrt(seA2 + seB2);

  if (se === 0) {
    return { t: Infinity, df: sA.n + sB.n - 2, p: 0, diff, se: 0, glassD: Infinity, baselineStd: 0 };
  }

  const t = diff / se;
  // Welch-Satterthwaite degrees of freedom approximation
  const df =
    (seA2 + seB2) ** 2 / (seA2 ** 2 / (sA.n - 1) + seB2 ** 2 / (sB.n - 1));

  const p = approxPValue(Math.abs(t), df);

  // Glass's delta -- uses baseline (a) std as denominator, consistent
  // with Welch's unequal-variance assumption
  const glassD = sA.std > 0 ? Math.abs(diff) / sA.std : Infinity;

  return { t, df, p, diff, se, glassD, baselineStd: sA.std };
}

/**
 * Required sample size per group to detect a given effect size.
 * Two-sample test, alpha=0.05 (two-tailed), power=0.80.
 *
 * Use this to right-size your eval budget:
 *   requiredN(observed_std, 5)   // detect a 5-point shift
 *   requiredN(observed_std, 10)  // detect a 10-point shift
 *   requiredN(observed_std, 15)  // detect a 15-point shift
 *
 * @param {number} std - Observed standard deviation from baseline runs
 * @param {number} delta - Minimum effect size you want to detect (must be > 0)
 * @returns {number} Required sample size per group
 */
export function requiredN(std, delta) {
  if (delta === 0) return Infinity;
  const zAlpha = 1.96; // two-tailed alpha=0.05
  const zBeta = 0.84; // power=0.80
  // Factor of 2 for two-sample comparison (need variance from both groups)
  return Math.ceil((2 * (zAlpha + zBeta) ** 2 * std ** 2) / (delta ** 2));
}

// ---------------------------------------------------------------------------
// Demo -- uncomment and run: node llm-eval-stats.mjs
// ---------------------------------------------------------------------------
// const baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87];
// const variant  = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81];
//
// console.log('Baseline stats:', stats(baseline));
// console.log('Comparison:', welchTTest(baseline, variant));
// console.log('Runs needed to detect 5-pt shift:', requiredN(stats(baseline).std, 5));
// console.log('Runs needed to detect 10-pt shift:', requiredN(stats(baseline).std, 10));
