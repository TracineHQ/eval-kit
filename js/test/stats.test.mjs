import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  stats,
  welchTTest,
  requiredN,
  pairedTTest,
  requiredNPaired,
  pValue,
  approxPValue,
  tCritical,
  verdict,
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

test('requiredN() is the exact noncentral-t answer, per arm', () => {
  // Regression: an early formula omitted the factor of 2 and returned 21.
  // The z-approximation gives 41; the exact answer (statsmodels TTestIndPower) is 42.
  assert.equal(requiredN(8, 5), 42);
  assert.equal(requiredN(16, 5), 162);
  assert.equal(requiredN(8, 10), 12);
});

test("requiredN() matches Cohen's power tables", () => {
  // alpha 0.05 two-sided, power 0.80: d = 0.2 / 0.5 / 0.8 -> 394 / 64 / 26 per group
  assert.equal(requiredN(1, 0.2), 394);
  assert.equal(requiredN(1, 0.5), 64);
  assert.equal(requiredN(1, 0.8), 26);
});

test('requiredN() ignores the sign of delta and never returns below 2', () => {
  assert.equal(requiredN(8, -5), requiredN(8, 5));
  assert.equal(requiredN(0, 5), 2);
  assert.equal(requiredN(1, 100), 2);
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
// pairedTTest()
// ---------------------------------------------------------------------------

test('pairedTTest() returns null for mismatched lengths or fewer than 2 items', () => {
  assert.equal(pairedTTest([1, 2], [1]), null);
  assert.equal(pairedTTest([1], [2]), null);
});

test('pairedTTest() matches scipy ttest_rel', () => {
  // scipy.stats.ttest_rel([80,72,91,65,77], [78,70,90,62,76])
  const r = pairedTTest([80, 72, 91, 65, 77], [78, 70, 90, 62, 76]);
  assert.ok(Math.abs(r.t - 4.810702354423639) < 1e-9);
  assert.ok(Math.abs(r.p - 0.008580918721924785) < 1e-9);
  assert.equal(r.df, 4);
  assert.equal(r.n, 5);
  assert.ok(Math.abs(r.diff - 1.8) < 1e-12);
});

test('pairedTTest() finds a shift that item difficulty hides from welchTTest', () => {
  // Items vary widely in difficulty; the variant adds ~2 points to each.
  const base = [40, 55, 62, 70, 78, 85, 91, 48, 66, 73];
  const vari = [42, 57, 63, 72, 80, 88, 92, 50, 69, 75];
  assert.ok(welchTTest(base, vari).p > 0.5);
  assert.ok(pairedTTest(base, vari).p < 0.001);
});

test('pairedTTest() returns dz=99 sentinel when every difference is identical and nonzero', () => {
  const r = pairedTTest([5, 6, 7], [4, 5, 6]);
  assert.equal(r.dz, 99);
  assert.equal(r.p, 0);
  assert.equal(r.se, 0);
});

test('pairedTTest() returns dz=0 and p=1 when arms are identical', () => {
  const r = pairedTTest([5, 6, 7], [5, 6, 7]);
  assert.equal(r.dz, 0);
  assert.equal(r.p, 1);
});

// ---------------------------------------------------------------------------
// requiredNPaired()
// ---------------------------------------------------------------------------

test('requiredNPaired() is the exact noncentral-t answer', () => {
  // No factor of 2, but the paired test has half the degrees of freedom, so it is
  // not exactly half of requiredN. statsmodels TTestPower gives 23 and 34.
  assert.equal(requiredNPaired(8, 5), 23);
  assert.equal(requiredNPaired(1, 0.5), 34);
  assert.equal(requiredNPaired(8, 15), 5);
});

test('requiredNPaired() ignores the sign of delta and never returns below 2', () => {
  assert.equal(requiredNPaired(8, -5), requiredNPaired(8, 5));
  assert.equal(requiredNPaired(0, 5), 2);
});

test('requiredNPaired() returns Infinity for zero delta', () => {
  assert.equal(requiredNPaired(8, 0), Infinity);
});

// ---------------------------------------------------------------------------
// pValue()
// ---------------------------------------------------------------------------

// References are 2 * scipy.stats.t.sf(t, df).
test('pValue() matches scipy at small and fractional df', () => {
  assert.ok(Math.abs(pValue(11, 1) - 0.057715876752608954) < 1e-9);
  assert.ok(Math.abs(pValue(20, 1) - 0.03180450251235275) < 1e-9);
  assert.ok(Math.abs(pValue(5, 1.5) - 0.06537576762115618) < 1e-9);
  assert.ok(Math.abs(pValue(3.0, 10) - 0.013343655022569572) < 1e-9);
});

test('pValue() uses the t-distribution, not the normal, at df=30', () => {
  // The normal approximation would report ~0.050 here.
  assert.ok(Math.abs(pValue(1.96, 30) - 0.05934231289605049) < 1e-9);
});

test('pValue() handles t=0 and t=Infinity', () => {
  assert.equal(pValue(0, 10), 1);
  assert.equal(pValue(Infinity, 10), 0);
});

test('approxPValue() is an alias of pValue()', () => {
  assert.equal(approxPValue(2.5, 7.3), pValue(2.5, 7.3));
  // JS-only: here the alias is the same function object. Python's is a wrapper.
  assert.equal(approxPValue, pValue);
});

// ---------------------------------------------------------------------------
// tCritical()
// ---------------------------------------------------------------------------

// References are scipy.stats.t.ppf(0.975, df).
test('tCritical() matches scipy at integer, fractional, and large df', () => {
  assert.ok(Math.abs(tCritical(1) - 12.706204736174694) < 1e-9);
  assert.ok(Math.abs(tCritical(1.5) - 6.016663104427927) < 1e-9);
  assert.ok(Math.abs(tCritical(12) - 2.1788128296672284) < 1e-9);
  assert.ok(Math.abs(tCritical(100) - 1.983971518523552) < 1e-9);
  assert.ok(Math.abs(tCritical(1000) - 1.9623390808264078) < 1e-9);
});

test('tCritical() returns 0 for df <= 0', () => {
  assert.equal(tCritical(0), 0);
});

// ---------------------------------------------------------------------------
// README example
// ---------------------------------------------------------------------------

// JS-only, no Python mirror: js/examples/verdicts.mjs has no Python counterpart.
test('examples/verdicts.mjs still prints the README verdicts', () => {
  const out = execFileSync(process.execPath, [fileURLToPath(new URL('../examples/verdicts.mjs', import.meta.url))], {
    encoding: 'utf8',
  });
  assert.match(out, /4 runs per arm {4}\+2\.7 .*can't tell yet: need 16 per arm/);
  assert.match(out, /8 runs per arm {4}\+3\.8 .*improved/);
  assert.match(out, /12 runs per arm {3}\+0\.0 .*no shift of 3 or more/);
});

// ---------------------------------------------------------------------------
// verdict()
// ---------------------------------------------------------------------------

const BASE = [80.1, 82.4, 79.0, 83.5, 81.2, 80.6, 82.9, 79.8];

test('verdict() says cant_tell when the interval spans no change and the target', () => {
  const v = verdict(BASE.slice(0, 4), [83.9, 81.2, 86.0, 84.7], { target: 3 });
  assert.equal(v.verdict, 'cant_tell');
  assert.equal(v.reason, 'underpowered');
  assert.equal(v.needed, 16);
});

test('verdict() calls a significant shift by direction, and flips it for lowerIsBetter', () => {
  const down = [77.3, 78.9, 76.1, 79.4, 77.8, 76.7, 79.1, 78.2];
  assert.equal(verdict(BASE, down, { target: 3 }).verdict, 'regressed');
  assert.equal(verdict(BASE, down, { target: 3, lowerIsBetter: true }).verdict, 'improved');
});

const LONG = Array.from({ length: 60 }, (_, i) => 80 + ((i * 7) % 11) / 5);

test('verdict() says no_change only when the CI rules out the target', () => {
  const v = verdict(LONG, [...LONG].reverse(), { target: 1 });
  assert.equal(v.verdict, 'no_change');
  assert.equal(v.reason, 'within_target');
});

test('verdict() calls a significant shift inside the target no_change', () => {
  const v = verdict(LONG, LONG.map((x) => x - 0.3), { target: 3 });
  assert.ok(v.p < 0.05);
  assert.equal(v.verdict, 'no_change');
  assert.equal(v.reason, 'below_target');
});

test('verdict() refuses when both arms have zero spread', () => {
  for (const [a, b] of [
    [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
    [[0.3, 0.3, 0.1 + 0.2], [0.4, 0.4, 0.4]],
  ]) {
    const v = verdict(a, b, { target: 0.05 });
    assert.deepEqual([v.verdict, v.reason, v.needed, v.constantArm], ['cant_tell', 'zero_spread', null, null]);
  }
});

test('verdict() tests one constant arm and names it', () => {
  const up = verdict([0.1, 0.1, 0.1], [0.2, 0.25, 0.3], { target: 0.05 });
  assert.deepEqual([up.verdict, up.constantArm], ['improved', 'baseline']);
  const down = verdict([0.2, 0.25, 0.3], [0.1, 0.1, 0.1], { target: 0.05 });
  assert.deepEqual([down.verdict, down.constantArm], ['regressed', 'variant']);
  assert.equal(verdict(BASE, BASE.map((x) => x + 1), { target: 3 }).constantArm, null);
});

test('verdict() refuses on zero spread instead of calling it', () => {
  const v = verdict([0.85, 0.65, 0.95, 0.75, 0.55], [0.8, 0.6, 0.9, 0.7, 0.5], { target: 0.1, paired: true });
  assert.equal(v.verdict, 'cant_tell');
  assert.equal(v.reason, 'zero_spread');
});

test('verdict() reports shift as variant minus baseline', () => {
  const v = verdict(BASE.slice(0, 4), [83.9, 81.2, 86.0, 84.7], { target: 3 });
  assert.ok(Math.abs(v.shift - 2.7) < 1e-9);
  assert.ok(v.ciLo < v.shift && v.shift < v.ciHi);
});

test('verdict() returns null for invalid input', () => {
  assert.equal(verdict([1], [1, 2], { target: 1 }), null);
  assert.equal(verdict([1, 2, 3], [1, 2], { target: 1, paired: true }), null);
  assert.equal(verdict([1, 2, 3], [4, 5, 6], { target: 0 }), null);
  assert.equal(verdict([1, 2, 3], [4, 5, 6], { target: NaN }), null);
  // JS-only: Python's target is a required positional argument, so it cannot be omitted.
  assert.equal(verdict([1, 2, 3], [4, 5, 6]), null);
});
