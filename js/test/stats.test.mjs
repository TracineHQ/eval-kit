import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  stats,
  welchTTest,
  requiredN,
  approxPValue,
  tCritical,
} from '../lib/stats.mjs';

// ---------------------------------------------------------------------------
// stats()
// ---------------------------------------------------------------------------

test('stats() returns null for empty input', () => {
  assert.equal(stats([]), null);
});

test('stats() handles single-value input without divide-by-zero', () => {
  const s = stats([42]);
  assert.equal(s.n, 1);
  assert.equal(s.mean, 42);
  assert.equal(s.std, 0);
  assert.equal(s.se, 0);
  assert.equal(s.ciMargin, 0);
});

test('stats() uses Bessel-corrected sample variance (n-1)', () => {
  // For [1, 2, 3, 4, 5]: mean=3, sum((x-mean)^2) = 10, var = 10/4 = 2.5
  const s = stats([1, 2, 3, 4, 5]);
  assert.equal(s.mean, 3);
  assert.equal(s.std, Math.sqrt(2.5));
});

test('stats() CV uses Math.abs(mean) to avoid sign errors near zero', () => {
  // Regression: pre-fix, a negative mean produced a negative CV.
  const s = stats([-10, -12, -8, -11, -9]);
  assert.ok(s.cv > 0, `CV should be positive for negative-mean data, got ${s.cv}`);
});

test('stats() iterative min/max handles large arrays without stack overflow', () => {
  // Regression: spread operator (Math.min(...arr)) blows the call stack
  // around 100k elements. Iterative scan must handle a million.
  const large = new Array(1_000_000).fill(0).map((_, i) => i);
  const s = stats(large);
  assert.equal(s.min, 0);
  assert.equal(s.max, 999_999);
});

// ---------------------------------------------------------------------------
// welchTTest()
// ---------------------------------------------------------------------------

test('welchTTest() returns null when either group has fewer than 2 samples', () => {
  assert.equal(welchTTest([1], [1, 2, 3]), null);
  assert.equal(welchTTest([1, 2, 3], [1]), null);
  assert.equal(welchTTest([], [1, 2, 3]), null);
});

test('welchTTest() detects a meaningful difference between groups', () => {
  const baseline = [82, 85, 79, 88, 81, 84, 86, 80, 83, 87];
  const variant  = [65, 68, 62, 70, 64, 67, 69, 63, 66, 71];
  const result = welchTTest(baseline, variant);
  assert.ok(result.diff > 0, 'baseline mean should exceed variant mean');
  assert.ok(result.p < 0.05, `p-value should flag real shift, got ${result.p}`);
  assert.ok(result.glassD > 0.8, `Glass's delta should indicate large effect, got ${result.glassD}`);
});

test('welchTTest() uses Glass\'s delta (baseline std), not pooled Cohen\'s d', () => {
  // Regression: Cohen's d pools std across groups, assuming equal variance.
  // Glass's delta divides by baseline std only, consistent with Welch's
  // unequal-variance assumption. This test pins the denominator.
  const baseline = [80, 81, 82, 83, 84]; // std ~1.58
  const variant  = [70, 60, 80, 50, 90]; // std much larger
  const result = welchTTest(baseline, variant);
  const expectedGlassD = Math.abs(result.diff) / result.baselineStd;
  assert.ok(
    Math.abs(result.glassD - expectedGlassD) < 1e-9,
    `Glass's delta must equal |diff| / baselineStd`
  );
});

test('welchTTest() handles zero-variance groups without NaN', () => {
  // Regression: se === 0 previously produced NaN for t and p.
  const result = welchTTest([50, 50, 50, 50], [60, 60, 60, 60]);
  assert.ok(Number.isFinite(result.diff));
  assert.equal(result.p, 0);
});

test('welchTTest() returns glassD=99 sentinel when baseline std is 0 and diff != 0', () => {
  // Infinity breaks JSON.stringify (serializes as null) and comparison
  // thresholds. 99 signals "off-scale because baseline has no variance."
  const result = welchTTest([80, 80, 80], [70, 70, 70]);
  assert.equal(result.glassD, 99);
  assert.ok(Number.isFinite(result.glassD), 'glassD must be JSON-serializable');
});

test('welchTTest() returns glassD=0 when both groups have zero variance and identical means', () => {
  const result = welchTTest([80, 80, 80], [80, 80, 80]);
  assert.equal(result.glassD, 0);
  assert.equal(result.diff, 0);
});

test('welchTTest() returns near-zero t when groups have identical distribution', () => {
  const a = [80, 82, 84, 86, 88, 90, 78, 81, 83, 85];
  const b = [80, 82, 84, 86, 88, 90, 78, 81, 83, 85];
  const result = welchTTest(a, b);
  assert.equal(result.diff, 0);
  assert.equal(result.t, 0);
});

// ---------------------------------------------------------------------------
// requiredN()
// ---------------------------------------------------------------------------

test('requiredN() includes factor of 2 for two-sample comparison', () => {
  // Regression: original formula omitted the factor of 2 and underestimated
  // sample size by half. At std=8, delta=5, the correct answer is 41, not 21.
  // Formula: ceil(2 * (1.96 + 0.84)^2 * 8^2 / 5^2) = ceil(40.14) = 41
  assert.equal(requiredN(8, 5), 41);
});

test('requiredN() scales with variance', () => {
  assert.ok(requiredN(16, 5) > requiredN(8, 5));
});

test('requiredN() scales inversely with effect size', () => {
  assert.ok(requiredN(8, 2) > requiredN(8, 10));
});

test('requiredN() returns Infinity for zero delta', () => {
  assert.equal(requiredN(8, 0), Infinity);
});

// ---------------------------------------------------------------------------
// approxPValue()
// ---------------------------------------------------------------------------

test('approxPValue() returns p close to zero for extreme t-values', () => {
  const p = approxPValue(15, 100);
  assert.ok(p < 0.001, `expected p<0.001 for t=15, got ${p}`);
});

test('approxPValue() returns p near 1 for near-zero t-values', () => {
  const p = approxPValue(0.05, 100);
  assert.ok(p > 0.9, `expected p>0.9 for t=0.05, got ${p}`);
});

test('approxPValue() uses conservative buckets for df < 30', () => {
  // Bucketed: {0.01, 0.05, 0.1, 0.5}. Any output must be one of these
  // (or 0.0001 for absT > 10) when df < 30.
  const validBuckets = new Set([0.0001, 0.01, 0.05, 0.1, 0.5]);
  const p = approxPValue(3.0, 10);
  assert.ok(validBuckets.has(p), `expected bucketed value, got ${p}`);
});

// ---------------------------------------------------------------------------
// tCritical()
// ---------------------------------------------------------------------------

test('tCritical() returns table values matching scipy to 4dp for known df', () => {
  // Table entries are 4dp rounding of scipy.stats.t.ppf(0.975, df).
  assert.equal(tCritical(1), 12.7062);
  assert.equal(tCritical(10), 2.2281);
});

test('tCritical() interpolates between table entries', () => {
  // df=12 is between df=10 (2.2281) and df=14 (2.1448)
  const t = tCritical(12);
  assert.ok(t > 2.1448 && t < 2.2281, `expected interpolated value, got ${t}`);
});

test('tCritical() falls back to z=1.96 for large df', () => {
  assert.equal(tCritical(1000), 1.96);
});
