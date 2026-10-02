// The three calls from the README, reproducible: node js/examples/verdicts.mjs
// Baseline vs variant, independent runs, chasing a 3-point shift.
import { verdict } from '../lib/stats.mjs';

const TARGET = 3;
const baseline = [80.1, 82.4, 79.0, 83.5, 81.2, 80.6, 82.9, 79.8, 81.7, 80.9, 82.2, 81.4];
const cases = [
  ['4 runs per arm', baseline.slice(0, 4), [83.9, 81.2, 86.0, 84.7]],
  ['8 runs per arm', baseline.slice(0, 8), [84.3, 85.9, 83.1, 86.4, 84.8, 83.7, 86.1, 85.2]],
  ['12 runs per arm', baseline, [80.6, 81.9, 79.7, 82.8, 81.5, 80.2, 83.1, 80.4, 81.1, 82.0, 81.8, 80.9]],
];

const fmt = (x) => (x >= 0 ? '+' : '') + x.toFixed(1);
const row = (cols) => cols.map((c, i) => c.padEnd([18, 7, 15, 7, 8, 0][i])).join('');
console.log(row([`target: ${TARGET} points`, 'shift', '95% CI', 'p', 'glassD', 'verdict']));
const say = {
  improved: () => 'improved',
  regressed: () => 'regressed',
  no_change: () => `no shift of ${TARGET} or more`,
  cant_tell: (v) => (v.reason === 'zero_spread' ? 'check the harness' : `can't tell yet: need ${v.needed} per arm`),
};
for (const [label, b, variant] of cases) {
  const v = verdict(b, variant, { target: TARGET });
  const p = v.p < 0.001 ? v.p.toExponential(0) : v.p.toFixed(2);
  console.log(row([label, fmt(v.shift), `[${fmt(v.ciLo)}, ${fmt(v.ciHi)}]`, p, v.effect.toFixed(2), say[v.verdict](v)]));
}
