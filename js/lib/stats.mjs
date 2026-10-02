/**
 * eval-kit -- Statistical confidence for LLM evaluation
 *
 * LLM scores are non-deterministic. The same prompt, same model, same input
 * produces different scores every run. This library tells you whether a
 * change is real or noise.
 *
 * Built for rubric-based LLM evals where you're comparing baseline vs
 * variant runs and need to know: did this actually help?
 *
 * What's in here:
 *   - Descriptive stats with 95% CI (t-distribution, Bessel's correction)
 *   - Welch's t-test (independent runs, unequal variance)
 *   - Paired t-test (same items scored in both arms)
 *   - Glass's delta and Cohen's dz effect sizes
 *   - Power analysis (how many runs or items do you need?)
 *   - Exact t-distribution p-values and critical values, zero dependencies
 *
 * Quick start:
 *   import { stats, welchTTest, requiredN } from './stats.mjs';
 *
 *   const baselineScores = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87];
 *   const variantScores  = [75, 78, 72, 80, 74, 77, 79, 73, 76, 81];
 *
 *   console.log(welchTTest(baselineScores, variantScores));
 *   // => { t, df, p, diff, se, glassD, baselineStd }
 *
 *   console.log(requiredN(8, 5));
 *   // => 42 (runs per group to detect a 5-point shift)
 *
 * Author: Anthony Ledesma
 * Copyright 2026 TracineHQ
 * Licensed under the Apache License, Version 2.0.
 * See LICENSE and NOTICE files in the repository root.
 */

// z-critical values, used only to seed the exact sample-size search.
const Z_95 = 1.959963984540054;        // two-tailed, alpha = 0.05
const Z_POWER_80 = 0.8416212335729143; // one-tailed, power = 0.80
const TARGET_POWER = 0.8;
const EXACT_N_LIMIT = 1e8;

// Lanczos coefficients (g=7, n=9) for log-gamma, accurate to ~1e-15.
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

function logGamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// Continued fraction for the incomplete beta function (modified Lentz).
function betaContinuedFraction(a, b, x, y = 1 - x) {
  const EPS = 1e-15;
  const TINY = 1e-300;
  let c = 1;
  // 1 - (a + b) x / (a + 1), rewritten in y = 1 - x so that x near 1 with
  // large a (huge df) does not cancel away the leading digits.
  let d = x < 0.5 ? 1 - ((a + b) * x) / (a + 1) : (1 - b + (a + b) * y) / (a + 1);
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < EPS) break;
  }
  return h;
}

// log B(a, b). With b = 1/2 and large a (every t-distribution call), the
// difference logGamma(a + 1/2) - logGamma(a) cancels most of its digits, so
// use its asymptotic series instead: error below 1e-14 from a = 20 up.
function logBeta(a, b) {
  const big = Math.max(a, b);
  if (Math.min(a, b) === 0.5 && big >= 20) {
    const ratio = 0.5 * Math.log(big) - 1 / (8 * big) + 1 / (192 * big ** 3)
      - 1 / (640 * big ** 5) + 17 / (14336 * big ** 7);
    return 0.5 * Math.log(Math.PI) - ratio;
  }
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

// Regularized incomplete beta function I_x(a, b). Pass y = 1 - x when it
// is known more precisely than 1 - x can be computed (x close to 1).
function regIncompleteBeta(a, b, x, y = 1 - x) {
  if (x <= 0) return 0;
  if (y <= 0) return 1;
  const logX = x < 0.5 ? Math.log(x) : Math.log1p(-y);
  const logY = y < 0.5 ? Math.log(y) : Math.log1p(-x);
  const front = Math.exp(a * logX + b * logY - logBeta(a, b));
  return x < (a + 1) / (a + b + 2)
    ? (front * betaContinuedFraction(a, b, x, y)) / a
    : 1 - (front * betaContinuedFraction(b, a, y, x)) / b;
}

/**
 * Exact two-tailed p-value from a t-statistic, for any df > 0 including
 * Welch's fractional df. Computed from the regularized incomplete beta
 * function; cross-validation holds it to scipy.stats.t.sf within 1e-9
 * relative through df = 1e7 (tail p-values keep their digits) and 1e-9
 * absolute beyond. df = Infinity gives the normal limit; df <= 0 gives NaN.
 *
 * @param {number} absT - Absolute value of the t-statistic
 * @param {number} df - Degrees of freedom
 * @returns {number} Two-tailed p-value
 */
export function pValue(absT, df) {
  if (!(df > 0) || Number.isNaN(absT)) return NaN;
  if (df === Infinity) return 2 * normalUpperTail(Math.abs(absT));
  const t2 = absT * absT;
  const denom = df + t2;
  return regIncompleteBeta(df / 2, 0.5, df / denom, t2 / denom);
}

/**
 * Alias of pValue, kept for compatibility with 0.0.x. Earlier versions
 * approximated; this one is exact.
 */
export const approxPValue = pValue;

/**
 * t-critical value for a 95% CI (two-tailed, alpha=0.05), for any df > 0.
 * Inverts pValue by bisection; matches scipy.stats.t.ppf(0.975, df) within
 * 1e-9. Returns 0 for df <= 0 and NaN for NaN df.
 *
 * @param {number} df - Degrees of freedom
 * @returns {number} t-critical value
 */
export function tCritical(df) {
  if (Number.isNaN(df)) return NaN;
  if (df === Infinity) return Z_95;
  if (df <= 0) return 0;
  let lo = 0;
  let hi = 1;
  while (pValue(hi, df) > 0.05) hi *= 2;
  for (let i = 0; i < 100 && hi - lo > 1e-12; i++) {
    const mid = (lo + hi) / 2;
    if (pValue(mid, df) > 0.05) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// Regularized upper incomplete gamma Q(a, x): series below a + 1,
// continued fraction above.
function gammaQ(a, x) {
  if (x <= 0) return 1;
  const logFront = -x + a * Math.log(x) - logGamma(a);
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let n = 1; n < 1000; n++) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-16) break;
    }
    return 1 - sum * Math.exp(logFront);
  }
  const TINY = 1e-300;
  let b = x + 1 - a;
  let c = 1 / TINY;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return Math.exp(logFront) * h;
}

// P(Z > x) for a standard normal, via erfc(x / sqrt 2) = Q(1/2, x^2 / 2).
function normalUpperTail(x) {
  const q = 0.5 * gammaQ(0.5, (x * x) / 2);
  return x >= 0 ? q : 1 - q;
}

// Noncentral t CDF, P(T <= t), after Lenth (1989), Applied Statistics AS 243.
// Poisson weights are carried in log space so large noncentrality does not
// underflow. Absolute error against scipy is ~1e-12 for |nc| <= 30; used for
// power only. Past |nc| ~ 38 the recurrences lose everything, so switch to the
// normal approximation (Abramowitz & Stegun 26.7.10), where power is ~1 anyway.
function noncentralTCdf(t, df, nc) {
  if (Math.abs(nc) > 37) {
    const z = (t * (1 - 1 / (4 * df)) - nc) / Math.sqrt(1 + (t * t) / (2 * df));
    return 1 - normalUpperTail(z);
  }
  const negate = t < 0;
  const tt = negate ? -t : t;
  const del = negate ? -nc : nc;
  let cdf = 0;
  const x = (tt * tt) / (tt * tt + df);
  if (x > 0) {
    const lambda = del * del;
    const sign = del < 0 ? -1 : 1;
    let logP = Math.log(0.5) - 0.5 * lambda;
    let logQ = Math.log(0.5 * Math.sqrt(2 / Math.PI)) - 0.5 * lambda + Math.log(Math.abs(del));
    let pSum = Math.exp(logP);
    let a = 0.5;
    const b = 0.5 * df;
    const rxb = Math.pow(1 - x, b);
    const logB = logBeta(0.5, b);
    let xOdd = regIncompleteBeta(a, b, x);
    let gOdd = 2 * rxb * Math.exp(a * Math.log(x) - logB);
    let xEven = 1 - rxb;
    let gEven = b * x * rxb;
    cdf = pSum * xOdd + sign * Math.exp(logQ) * xEven;
    for (let n = 1; n <= 100000; n++) {
      a += 1;
      xOdd -= gOdd;
      xEven -= gEven;
      gOdd *= (x * (a + b - 1)) / a;
      gEven *= (x * (a + b - 0.5)) / (a + 0.5);
      logP += Math.log(lambda / (2 * n));
      logQ += Math.log(lambda / (2 * n + 1));
      const p = Math.exp(logP);
      pSum += p;
      cdf += p * xOdd + sign * Math.exp(logQ) * xEven;
      const errorBound = 2 * (0.5 - pSum) * (xOdd - gOdd);
      if (Math.abs(errorBound) < 1e-14 && n > lambda / 2) break;
    }
  }
  cdf += normalUpperTail(del);
  if (negate) cdf = 1 - cdf;
  return Math.min(1, Math.max(0, cdf));
}

// Power of a two-sided t-test at alpha = 0.05 with noncentrality nc.
function tTestPower(df, nc) {
  const tc = tCritical(df);
  return 1 - noncentralTCdf(tc, df, nc) + noncentralTCdf(-tc, df, nc);
}

// Smallest n >= 2 that passes a test that stays passed as n grows (power
// reaching 0.80, a CI narrowing). Start from a z-based guess, gallop to
// bracket the exact boundary, then bisect: a few
// dozen power evaluations. Past EXACT_N_LIMIT the z guess is returned as is:
// its relative error there is under 1e-5, and the t distribution's precision
// at df ~ 1e9 and up is not worth searching on.
function smallestN(guess, test) {
  if (!(guess <= EXACT_N_LIMIT)) return guess;
  const n = Math.max(2, guess);
  const passes = (m) => m >= 2 && test(m);
  let lo; // fails (or is below 2)
  let hi; // passes
  let step = 1;
  if (passes(n)) {
    hi = n;
    for (;;) {
      const m = n - step;
      if (m < 2) { lo = 1; break; }
      if (!passes(m)) { lo = m; break; }
      hi = m;
      step *= 2;
    }
  } else {
    lo = n;
    for (;;) {
      const m = n + step;
      if (m > EXACT_N_LIMIT) return Infinity; // never passes in range (NaN input)
      if (passes(m)) { hi = m; break; }
      lo = m;
      step *= 2;
    }
  }
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (passes(mid)) hi = mid;
    else lo = mid;
  }
  return hi;
}

/**
 * Descriptive statistics with 95% confidence interval.
 * Uses Bessel's correction (n-1) for sample variance.
 * At n = 1, std, se and ciMargin are 0 by construction (no spread estimate),
 * not a claim of precision; the t-tests refuse n < 2 for this reason.
 * Returns null for empty input.
 *
 * @param {number[]} values - Array of numeric observations
 * @returns {{ n, mean, std, cv, min, max, range, se, ciLo, ciHi, ciMargin } | null}
 */
export function stats(values) {
  const n = values.length;
  if (n === 0) return null;
  const mean = values.reduce((s, v) => s + v, 0) / n;
  let min = values[0], max = values[0];
  for (let i = 1; i < n; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  // Identical values have zero spread, up to floating-point dust (0.1 + 0.2
  // vs 0.3); don't let rounding say otherwise.
  const flat = max - min <= 8 * Number.EPSILON * Math.max(Math.abs(min), Math.abs(max));
  const variance = n > 1 && !flat ? values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
  const std = Math.sqrt(variance);
  const cv = mean !== 0 ? (std / Math.abs(mean)) * 100 : 0;
  const se = n > 1 ? std / Math.sqrt(n) : 0;
  const df = n - 1;
  const t = df > 0 ? tCritical(df) : 0;
  const ciLo = mean - t * se;
  const ciHi = mean + t * se;
  const ciMargin = t * se;
  return { n, mean, std, cv, min, max, range: max - min, se, ciLo, ciHi, ciMargin };
}

/**
 * Welch's t-test (unequal variance, two-sample).
 *
 * Use this instead of Student's t-test when comparing LLM eval runs --
 * different models/prompts produce different variance, not just different means.
 *
 * Effect size is Glass's delta (uses baseline SD as denominator), which is
 * consistent with the unequal-variance assumption. The usual anchors are
 * Cohen's: small ~0.2, medium ~0.5, large ~0.8. Read them loosely; delta
 * runs larger than pooled d whenever the variant is noisier than baseline.
 *
 * diff is a.mean - b.mean, so positive means the variant scored lower.
 * glassD is unsigned (|diff| / std(a)); take direction from diff.
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
  // Squared standard errors in units of the larger std, so spreads near
  // 1e-100 or 1e+100 don't underflow or overflow when squared again for df.
  const scale = Math.max(sA.std, sB.std);
  const uA = scale > 0 ? (sA.std / scale) ** 2 / sA.n : 0;
  const uB = scale > 0 ? (sB.std / scale) ** 2 / sB.n : 0;
  const se = scale * Math.sqrt(uA + uB);

  // Glass's delta when baseline std is 0: sentinel 99 instead of Infinity.
  // Infinity breaks JSON.stringify (becomes null) and comparison thresholds.
  // 99 means "effect size is off-scale because baseline has zero variance";
  // 0 means "no difference" (both groups identical).
  const glassDZeroStd = diff === 0 ? 0 : 99;

  if (se === 0) {
    return {
      t: diff === 0 ? 0 : Infinity,
      df: sA.n + sB.n - 2, // nominal; Welch df is undefined at zero variance
      p: diff === 0 ? 1 : 0,
      diff,
      se: 0,
      glassD: glassDZeroStd,
      baselineStd: 0,
    };
  }

  const t = diff / se;
  // Welch-Satterthwaite degrees of freedom approximation
  const df = (uA + uB) ** 2 / (uA ** 2 / (sA.n - 1) + uB ** 2 / (sB.n - 1));

  const p = pValue(Math.abs(t), df);

  // Glass's delta -- uses baseline (a) std as denominator, consistent
  // with Welch's unequal-variance assumption
  const glassD = sA.std > 0 ? Math.abs(diff) / sA.std : glassDZeroStd;

  return { t, df, p, diff, se, glassD, baselineStd: sA.std };
}

/**
 * Paired t-test on per-item differences.
 *
 * Use this when the same eval items are scored in both arms (one run per
 * arm, or per-item means over several runs). Pairing cancels item
 * difficulty, which usually dwarfs run-to-run noise, so it detects a shift
 * with far fewer items than an unpaired test would. For independent full
 * runs, use welchTTest.
 *
 * Effect size is Cohen's dz: |mean difference| / std of the differences.
 * The usual ~0.2 / ~0.5 / ~0.8 anchors apply loosely; dz runs larger than a
 * between-group d when items are strongly correlated across arms.
 *
 * diff is mean(a - b), so positive means the variant scored lower.
 * dz uses sentinel 99 when every difference is identical and nonzero.
 *
 * @param {number[]} a - Baseline scores, one per item
 * @param {number[]} b - Variant scores for the same items, same order
 * @returns {{ t, df, p, diff, se, dz, sdDiff, n } | null} null when the
 *   arrays differ in length or hold fewer than 2 items
 */
export function pairedTTest(a, b) {
  if (a.length !== b.length || a.length < 2) return null;
  const d = stats(a.map((v, i) => v - b[i]));
  const diff = d.mean;
  const df = d.n - 1;

  // Every item moved by the same amount, up to floating-point rounding in a - b.
  let scale = 0;
  for (let i = 0; i < a.length; i++) scale = Math.max(scale, Math.abs(a[i]), Math.abs(b[i]));
  if (d.range <= 8 * Number.EPSILON * scale) {
    return {
      t: diff === 0 ? 0 : Infinity,
      df,
      p: diff === 0 ? 1 : 0,
      diff,
      se: 0,
      dz: diff === 0 ? 0 : 99,
      sdDiff: 0,
      n: d.n,
    };
  }

  const t = diff / d.se;
  return {
    t,
    df,
    p: pValue(Math.abs(t), df),
    diff,
    se: d.se,
    dz: Math.abs(diff) / d.std,
    sdDiff: d.std,
    n: d.n,
  };
}

/**
 * Required runs per arm to detect a given difference in means.
 * Two-sample t-test, alpha=0.05 (two-tailed), power=0.80, exact: the
 * smallest n whose power under the noncentral t-distribution reaches 0.80.
 * Matches statsmodels TTestIndPower and Cohen's tables (d=0.5 needs 64).
 *
 * Use this to right-size your eval budget:
 *   requiredN(observed_std, 5)   // detect a 5-point shift
 *   requiredN(observed_std, 10)  // detect a 10-point shift
 *
 * Assumes both arms share the baseline's spread. If the variant's variance
 * is r times the baseline's, the true answer is roughly (1 + r) / 2 times
 * this: 1.5x for double the variance, 2.5x for double the standard
 * deviation. Treat it as a floor when variants drift.
 *
 * @param {number} std - Observed standard deviation from baseline runs
 * @param {number} delta - Minimum difference in means to detect
 * @returns {number} Runs per arm (at least 2); Infinity when delta is 0
 */
export function requiredN(std, delta) {
  if (delta === 0) return Infinity;
  if (std === 0) return 2; // no spread: any real difference is visible at the minimum n
  const d = Math.abs(delta);
  // Ratios, not squares, so tiny or huge inputs overflow to Infinity cleanly.
  const guess = Math.ceil(2 * (Z_95 + Z_POWER_80) ** 2 * (std / d) * (std / d));
  return smallestN(guess, (n) => tTestPower(2 * n - 2, d / (std * Math.sqrt(2 / n))) >= TARGET_POWER);
}

/**
 * Required number of paired items to detect a given mean difference.
 * Paired t-test, alpha=0.05 (two-tailed), power=0.80, exact under the
 * noncentral t-distribution. Matches statsmodels TTestPower.
 *
 * sdDiff is the standard deviation of per-item differences, from a pilot
 * (pairedTTest(...).sdDiff). There is no factor of 2: each item already
 * contributes both arms.
 *
 * @param {number} sdDiff - Standard deviation of per-item differences
 * @param {number} delta - Minimum mean difference to detect
 * @returns {number} Items (at least 2); Infinity when delta is 0
 */
export function requiredNPaired(sdDiff, delta) {
  if (delta === 0) return Infinity;
  if (sdDiff === 0) return 2;
  const d = Math.abs(delta);
  const guess = Math.ceil((Z_95 + Z_POWER_80) ** 2 * (sdDiff / d) * (sdDiff / d));
  return smallestN(guess, (n) => tTestPower(n - 1, d / (sdDiff / Math.sqrt(n))) >= TARGET_POWER);
}

// Smallest n (runs per arm, or items) at which the 95% CI half-width, at the
// spread seen so far, is under half the target. A CI that narrow cannot hold
// both 0 and a shift of target, so the verdict has to be a call. se1 is the
// standard error at n = 1; dfOf gives the degrees of freedom at n.
function resolvingN(se1, dfOf, target) {
  const q = (2 * Z_95 * se1) / target;
  const guess = Math.ceil(q * q);
  return smallestN(guess, (n) => tCritical(dfOf(n)) * (se1 / Math.sqrt(n)) < target / 2);
}

/**
 * The decision: did the variant move the score, and can this data say so?
 *
 * target is the smallest shift worth acting on, in score units, picked
 * before looking at results. Rules, in order:
 *   - no spread in either arm (or in the paired differences): cant_tell,
 *     check the harness
 *   - 95% CI of the shift inside (-target, +target): no_change, even when
 *     p < 0.05 (reason below_target: real, but smaller than worth acting on)
 *   - p < 0.05: improved or regressed, by the sign of the shift
 * "p < 0.05" is read off the 95% CI (it excludes 0), so the rules and the
 * reported CI can never disagree at the rounding edge.
 *   - otherwise: cant_tell
 *
 * shift is variant minus baseline (the opposite sign of welchTTest's diff).
 * needed is the runs per arm (Welch) or items (paired) at which, at the
 * spread seen so far, the 95% CI is narrower than half the target on each
 * side, so it cannot span both no change and the target: every verdict at
 * that n is a call. It sizes from both arms' spread and is null when there
 * is no spread to size from. Past 1e8 it is the z-approximation (slightly
 * low, by about 1e-8 relative); Infinity when the target is too small for
 * the spread to size at all. For planning a run before collecting data, use
 * requiredN (80% power to detect a target-sized shift), which is smaller.
 * Returns null for invalid input: fewer than 2 values per arm, mismatched
 * lengths when paired, or a target that is not a positive number.
 *
 * @param {number[]} baseline
 * @param {number[]} variant
 * @param {{ target: number, paired?: boolean, lowerIsBetter?: boolean }} options
 * @returns {{ verdict, reason, design, target, lowerIsBetter, shift, ciLo, ciHi,
 *   p, effect, effectName, nBaseline, nVariant, needed } | null}
 */
export function verdict(baseline, variant, { target, paired = false, lowerIsBetter = false } = {}) {
  if (!(target > 0) || !Number.isFinite(target)) return null;
  const r = paired ? pairedTTest(baseline, variant) : welchTTest(baseline, variant);
  if (!r) return null;

  const shift = -r.diff;
  const margin = r.se === 0 ? 0 : tCritical(r.df) * r.se;
  const effect = paired ? r.dz : r.glassD;
  const variantStd = paired ? 0 : stats(variant).std;
  // No spread anywhere to test against: almost always a broken harness. One
  // constant arm is fine for Welch (a ceiling, or a harness to check), so the
  // verdict stands and constantArm names it.
  const flat = paired ? r.sdDiff === 0 : r.baselineStd === 0 && variantStd === 0;
  let constantArm = null;
  if (!paired && !flat) constantArm = r.baselineStd === 0 ? 'baseline' : variantStd === 0 ? 'variant' : null;
  let needed = null;
  if (!flat && paired) {
    needed = resolvingN(r.sdDiff, (n) => n - 1, target);
  } else if (!flat) {
    // Welch at equal n per arm: se^2 = (vA + vB) / n, Welch-Satterthwaite df.
    // (1 + r)^2 / (1 + r^2) with r the variance ratio, written as
    // 1 + 2 / (r + 1/r) so no spread, however tiny or huge, gives 0/0. With
    // one constant arm r is 0 or Infinity and df is n - 1, as it should be.
    const sdRatio = variantStd / r.baselineStd;
    const ratio = sdRatio * sdRatio;
    const se1 = Math.hypot(r.baselineStd, variantStd);
    needed = resolvingN(se1, (n) => (n - 1) * (1 + 2 / (ratio + 1 / ratio)), target);
  }

  const significant = shift - margin > 0 || shift + margin < 0;
  let result;
  let reason;
  if (flat) {
    result = 'cant_tell';
    reason = 'zero_spread';
  } else if (shift - margin > -target && shift + margin < target) {
    result = 'no_change';
    reason = significant ? 'below_target' : 'within_target';
  } else if (significant) {
    result = (shift > 0) !== lowerIsBetter ? 'improved' : 'regressed';
    reason = 'significant';
  } else {
    result = 'cant_tell';
    reason = 'underpowered';
  }

  return {
    verdict: result,
    reason,
    design: paired ? 'paired' : 'welch',
    target,
    lowerIsBetter,
    shift,
    ciLo: shift - margin,
    ciHi: shift + margin,
    p: r.p,
    effect,
    effectName: paired ? 'dz' : 'glassD',
    nBaseline: baseline.length,
    nVariant: variant.length,
    needed,
    constantArm,
  };
}
