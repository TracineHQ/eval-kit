import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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

test('verdict() reports the effect as null when there is no spread to scale it by', () => {
  // Regression: 0.1.x reported Glass's delta sentinel, 99 (or 0), as the effect.
  const base = verdict([0.8, 0.8, 0.8, 0.8, 0.8], [0.84, 0.79, 0.86, 0.83, 0.81], { target: 0.03 });
  assert.equal(base.constantArm, 'baseline');
  assert.equal(base.effect, null);
  assert.ok([base.ciLo, base.ciHi, base.p].every(Number.isFinite));
  // Equal means gave the 0 sentinel, which is no more an effect size than 99.
  assert.equal(verdict([5, 5, 5], [4, 5, 6], { target: 1 }).effect, null);
  // A constant variant leaves the baseline std as the scale, so the effect stays a number.
  const vari = verdict([0.84, 0.79, 0.86, 0.83, 0.81], [0.8, 0.8, 0.8, 0.8, 0.8], { target: 0.03 });
  assert.equal(vari.constantArm, 'variant');
  assert.ok(Number.isFinite(vari.effect) && vari.effect > 0);
});

test('verdict() reports the CI, p and effect as null with no spread at all', () => {
  // Regression: 0.1.x reported placeholders: ciLo = ciHi = shift, p 0 or 1, effect 99 or 0.
  for (const [a, b, paired] of [
    [[80, 80, 80], [85, 85, 85], false],
    [[5, 5, 5], [5, 5, 5], false],
    [[0.85, 0.65, 0.95, 0.75, 0.55], [0.8, 0.6, 0.9, 0.7, 0.5], true],
    [[1, 2, 3], [1, 2, 3], true],
  ]) {
    const v = verdict(a, b, { target: 1, paired });
    assert.equal(v.reason, 'zero_spread');
    assert.deepEqual([v.ciLo, v.ciHi, v.p, v.effect, v.needed], [null, null, null, null, null]);
    assert.ok(Number.isFinite(v.shift));
  }
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

// ---------------------------------------------------------------------------
// CLI (cli.mjs). Mirrored in python/tests/test_stats.py against eval_kit.cli;
// tests/cross-validation/ holds the two CLIs to the same bytes.
// ---------------------------------------------------------------------------

const CLI = fileURLToPath(new URL('../cli.mjs', import.meta.url));
const TMP = mkdtempSync(join(tmpdir(), 'eval-kit-test-'));
let fileCount = 0;

function scoreFile(data) {
  const path = join(TMP, `scores-${fileCount++}.json`);
  writeFileSync(path, JSON.stringify(data));
  return path;
}

function cli(...args) {
  const r = spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8' });
  return { code: r.status, out: r.stdout, err: r.stderr };
}

function cliJson(...args) {
  const r = cli(...args, '--json');
  assert.equal(r.code, 0, r.err);
  return JSON.parse(r.out);
}

const PILOT_BASE = [80.1, 82.4, 79.0, 83.5];
const PILOT_VARI = [83.9, 81.2, 86.0, 84.7];

test('cli power sizes two arrays as independent runs, from the pooled std', () => {
  // Regression: 0.1.0 paired two arrays by index, answered in items, and
  // refused unequal lengths with a --paired error nobody had asked for.
  const base = scoreFile(PILOT_BASE);
  const vari = scoreFile(PILOT_VARI);
  const pooled = Math.sqrt((stats(PILOT_BASE).std ** 2 + stats(PILOT_VARI).std ** 2) / 2);
  const r = cliJson('power', '--baseline', base, '--variant', vari, '--target', '3');
  assert.equal(r.design, 'welch');
  assert.ok(Math.abs(r.spread - pooled) < 1e-12);
  assert.equal(r.needed, requiredN(pooled, 3));
  assert.match(cli('power', '--baseline', base, '--variant', vari, '--target', '3').out, /^9 runs per arm .*\(pooled std 2\.05,/);

  const unequal = cli('power', '--baseline', base, '--variant', scoreFile([...PILOT_VARI, 85.1]), '--target', '3');
  assert.equal(unequal.code, 0, unequal.err);
  assert.match(unequal.out, /runs per arm/);
  // The baseline-only path is unchanged.
  assert.equal(cli('power', '--baseline', base, '--target', '3').out,
    '9 runs per arm to detect a 3-point shift (baseline std 2.06, 80% power, alpha 0.05)\n');
});

test('cli power pairs a pilot only when compare would', () => {
  const base = scoreFile(PILOT_BASE);
  const vari = scoreFile(PILOT_VARI);
  const byIndex = cliJson('power', '--baseline', base, '--variant', vari, '--target', '3', '--paired');
  assert.equal(byIndex.design, 'paired');
  assert.equal(byIndex.spread, pairedTTest(PILOT_BASE, PILOT_VARI).sdDiff);
  assert.equal(byIndex.needed, requiredNPaired(byIndex.spread, 3));

  const objBase = scoreFile({ a: 40, b: 55, c: 62 });
  const objVari = scoreFile({ c: 65, a: 42, b: 56 });
  for (const args of [
    ['--baseline', base, '--variant', vari, '--target', '3'],
    ['--baseline', base, '--variant', vari, '--target', '3', '--paired'],
    ['--baseline', objBase, '--variant', objVari, '--target', '1'],
  ]) {
    assert.equal(cliJson('power', ...args).design, cliJson('compare', ...args).design, args.join(' '));
  }

  const unequal = cli('power', '--baseline', base, '--variant', scoreFile([1, 2, 3]), '--target', '3', '--paired');
  assert.equal(unequal.code, 2);
  assert.equal(unequal.err, 'eval-kit: --paired needs equal lengths (baseline 4, variant 3)\n');
});

test('cli power warns when there is no spread to size from', () => {
  const minimum = 'With no spread to size from, 2 is the minimum, not a plan.';
  const flat = scoreFile([0.8, 0.8, 0.8, 0.8, 0.8]);
  const cases = [
    [['--std', '0'], 'A std of 0 means every run scores the same. That is a ceiling or a broken harness;'],
    [['--baseline', flat], 'Every baseline run scored the same (0.80). That is a ceiling or a broken harness;'],
    [['--baseline', flat, '--variant', flat], 'Every run in both arms scored the same. That is a ceiling or a broken harness;'],
    [['--sd-diff', '0'], 'An sd of 0 means every item moves by the same amount. That is usually a broken harness;'],
    [
      ['--baseline', scoreFile({ a: 1, b: 2 }), '--variant', scoreFile({ a: 2, b: 3 })],
      'Every item moved by exactly the same amount. That is usually a broken harness;',
    ],
  ];
  for (const [args, cause] of cases) {
    const r = cli('power', ...args, '--target', '3');
    assert.equal(r.code, 0, r.err);
    const lines = r.out.trimEnd().split('\n');
    assert.match(lines[0], /^2 (runs per arm|items) to detect a 3-point shift \(.* 0\.00, 80% power, alpha 0\.05\)$/);
    assert.equal(lines[1], cause);
    assert.ok(lines[2].endsWith(minimum), lines[2]);
  }

  // One constant arm in a two-arm pilot: sized from the other arm, with compare's warning.
  const oneFlat = cli('power', '--baseline', flat, '--variant', scoreFile([0.84, 0.79, 0.86, 0.83, 0.81]), '--target', '0.03');
  assert.deepEqual(oneFlat.out.trimEnd().split('\n').slice(1), [
    'Every baseline run scored the same (0.80). That is a ceiling or a broken harness;',
    'check which before trusting this number.',
  ]);
  // --json keeps its shape; spread 0 is the signal there.
  assert.deepEqual(cliJson('power', '--std', '0', '--target', '3'), { design: 'welch', target: 3, spread: 0, needed: 2 });
});

test('cli compare prints the CI, p and effect as undefined when neither arm has spread', () => {
  // Regression: 0.1.0 printed glassD 99.00, p < 0.001 and a zero-width CI here.
  const at80 = scoreFile([80, 80, 80, 80, 80]);
  const at85 = scoreFile([85, 85, 85, 85, 85]);
  for (const [vari, shift] of [[at85, '+5.00'], [at80, '+0.00']]) {
    const r = cli('compare', '--baseline', at80, '--variant', vari, '--target', '3', '--gate');
    assert.equal(r.code, 3);
    assert.ok(r.out.includes(`  shift     ${shift}  95% CI undefined\n  p         undefined\n`), r.out);
    assert.ok(r.out.includes('  glassD    undefined (no spread in either arm)\n'), r.out);
    assert.ok(r.out.includes("CAN'T TELL. Every run in both arms scored the same."), r.out);
  }
  const paired = cli('compare', '--baseline', at80, '--variant', at85, '--target', '3', '--paired');
  assert.ok(paired.out.includes('  dz        undefined (no spread in the item differences)\n'), paired.out);

  // --json keeps its shape and reports what the data cannot define as null.
  const v = cliJson('compare', '--baseline', at80, '--variant', at85, '--target', '3');
  assert.deepEqual(Object.keys(v), Object.keys(verdict([1, 2], [3, 4], { target: 1 })));
  assert.deepEqual(
    [v.verdict, v.reason, v.shift, v.ciLo, v.ciHi, v.p, v.effect, v.needed, v.constantArm],
    ['cant_tell', 'zero_spread', 5, null, null, null, null, null, null],
  );
  const pairedJson = cliJson('compare', '--baseline', at80, '--variant', at85, '--target', '3', '--paired');
  assert.deepEqual([pairedJson.ciLo, pairedJson.ciHi, pairedJson.p, pairedJson.effect], [null, null, null, null]);
  // One constant baseline: a real CI and p; the effect is off-scale in text and null in --json.
  const oneArgs = ['compare', '--baseline', scoreFile([0.8, 0.8, 0.8, 0.8, 0.8]),
    '--variant', scoreFile([0.84, 0.79, 0.86, 0.83, 0.81]), '--target', '0.03'];
  const one = cli(...oneArgs);
  assert.ok(one.out.includes('95% CI [-0.01, +0.06]\n  p         0.098\n  glassD    off-scale (the baseline has no spread)\n'), one.out);
  const oneJson = cliJson(...oneArgs);
  assert.equal(oneJson.effect, null);
  assert.ok([oneJson.ciLo, oneJson.ciHi, oneJson.p].every(Number.isFinite));
});

test('cli --gate holds on a constant arm unless the verdict is regressed', () => {
  // Regression: 0.1.x exited 0 on an improved verdict over a constant baseline,
  // while the text said to check the harness before trusting it.
  const flat = scoreFile([0.8, 0.8, 0.8, 0.8, 0.8]);
  const up = scoreFile([0.86, 0.84, 0.88, 0.85, 0.87]);
  const down = scoreFile([0.74, 0.76, 0.72, 0.75, 0.73]);
  const near = scoreFile([0.8, 0.81, 0.79, 0.8, 0.8]);
  const cases = [
    ['constant baseline, improved', flat, up, 'IMPROVED.', 3],
    ['constant baseline, no change', flat, near, 'NO CHANGE.', 3],
    ['constant baseline, regressed', flat, down, 'REGRESSED.', 1],
    ['constant variant, improved', down, flat, 'IMPROVED.', 3],
    ['constant variant, regressed', up, flat, 'REGRESSED.', 1],
  ];
  for (const [label, base, vari, call, code] of cases) {
    const args = ['compare', '--baseline', base, '--variant', vari, '--target', '0.05'];
    const gated = cli(...args, '--gate');
    assert.equal(gated.code, code, label);
    assert.ok(gated.out.includes(`\n${call} `), `${label}: ${gated.out}`);
    // Only the exit code changes: the verdict and the text are the same as without --gate.
    const plain = cli(...args);
    assert.equal(plain.code, 0, label);
    assert.equal(plain.out, gated.out, label);
  }
});

test('cli refuses a repeated item id', () => {
  // Regression: 0.2.0 kept the last score for a repeated id, so this baseline
  // compared as n=3, mean 2.20, and 3 of its 6 scores vanished. JSON.stringify
  // cannot write a repeated key, so the files are raw text.
  const raw = (text) => {
    const path = join(TMP, `scores-${fileCount++}.json`);
    writeFileSync(path, text);
    return path;
  };
  const base = raw('{"r1":1.9,"r1":2.4,"r1":2.1,"r2":3.0,"r2":3.3,"r3":1.2}');
  const vari = raw('{"r1":1.6,"r2":2.9,"r3":1.0}');
  const hint = 'give each item one score (average repeated calls first)';
  const r = cli('compare', '--baseline', base, '--variant', vari, '--target', '1', '--lower-is-better');
  assert.deepEqual(r, { code: 2, out: '', err: `eval-kit: ${base} repeats item id "r1"; ${hint}\n` });
  // The first id whose second occurrence comes first, decoded and quoted.
  const order = raw('{"a\\n": 1, "b": 2, "b": 3, "a\\u000a": 4}');
  assert.equal(cli('power', '--baseline', order, '--target', '1').err, `eval-kit: ${order} repeats item id "b"; ${hint}\n`);
  // Only the top-level object holds item ids; a nested repeat is the wrong shape.
  const nested = raw('{"a": {"x": 1, "x": 2}, "b": 1}');
  assert.equal(
    cli('compare', '--baseline', nested, '--variant', vari, '--target', '1').err,
    `eval-kit: ${nested} must be a JSON array of numbers or an object of {"item id": number}\n`,
  );
});
