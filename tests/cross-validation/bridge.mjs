#!/usr/bin/env node
/**
 * Cross-validation bridge: reads {fn, args} JSON from stdin, runs the
 * corresponding JS stats function, writes JSON result to stdout.
 *
 * Non-finite floats are encoded as string sentinels so they survive
 * JSON round-trip (JSON.stringify normally converts Infinity -> null):
 *   Infinity    -> "__inf__"
 *   -Infinity   -> "__neginf__"
 *   NaN         -> "__nan__"
 *
 * The Python side decodes these back to math.inf, etc. This lets us
 * cross-validate sentinel behavior (Glass's delta = 99 vs Infinity,
 * etc.) without losing fidelity.
 */

import {
  stats,
  welchTTest,
  requiredN,
  approxPValue,
  tCritical,
} from '../../js/lib/stats.mjs';

const FNS = { stats, welchTTest, requiredN, approxPValue, tCritical };

function replacer(_key, value) {
  if (typeof value === 'number') {
    if (value === Infinity) return '__inf__';
    if (value === -Infinity) return '__neginf__';
    if (Number.isNaN(value)) return '__nan__';
  }
  return value;
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  input += chunk;
});
process.stdin.on('end', () => {
  const { fn, args } = JSON.parse(input);
  if (!(fn in FNS)) {
    process.stderr.write(`Unknown fn: ${fn}\n`);
    process.exit(1);
  }
  const result = FNS[fn](...args);
  process.stdout.write(JSON.stringify(result, replacer));
});
