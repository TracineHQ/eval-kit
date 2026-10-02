#!/usr/bin/env node
/**
 * eval-kit CLI. Thin wrapper over lib/stats.mjs: read scores, call
 * verdict() or the power functions, print. The Python CLI
 * (python -m eval_kit) prints the same text and JSON; cross-validation
 * holds them to it.
 *
 * Copyright 2026 TracineHQ
 * Licensed under the Apache License, Version 2.0.
 */

import { readFileSync, realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { stats, pairedTTest, requiredN, requiredNPaired, verdict } from './lib/stats.mjs';

const VERSION = '0.1.0';

// Exit codes. Outcome codes apply only with --gate, so plain analysis
// never fails a build.
const EXIT_OK = 0;
const EXIT_REGRESSED = 1;
const EXIT_USAGE = 2;
const EXIT_CANT_TELL = 3;

const HELP = `eval-kit ${VERSION}: did the eval get better, or is that noise?

Examples:
  eval-kit compare --baseline base.json --variant new.json --target 3
  eval-kit compare --baseline base.json --variant new.json --target 3 --gate
  jq '[.runs[].score]' results.json | eval-kit compare --baseline base.json --variant - --target 3
  eval-kit power --baseline pilot.json --target 3
  eval-kit power --std 2.1 --target 3

compare   Test a variant against a baseline and give a verdict:
          improved, regressed, no_change, or cant_tell.
power     How many runs per arm (or items, with --paired) to detect --target.

Input files are JSON. An array of numbers is one score per run, compared
with Welch's t-test. An object of {"item id": score} pairs items across the
two files by id and uses the paired t-test. --paired pairs two arrays by index.
Use - to read one of the files from stdin.

Options:
  --baseline FILE     Baseline scores
  --variant FILE      Variant scores (compare; paired power pilot)
  --target N          Smallest shift worth acting on, in score units. Required.
                      Pick it before looking at results.
  --paired            Pair arrays by index (objects always pair by id)
  --lower-is-better   Lower scores are better (latency, error rate)
  --std N             power: baseline standard deviation, instead of --baseline
  --sd-diff N         power --paired: std of per-item differences
  --json              Machine-readable output
  --gate              compare: exit 1 on regressed, 3 on cant_tell
  -h, --help          Show this help
  --version           Show the version

Exit codes: 0 ok, 1 regressed (--gate), 2 usage or input error,
3 cant_tell (--gate).
`;

class UsageError extends Error {}

// OS wording for the common read failures, so both CLIs say the same thing.
const READ_ERRORS = {
  ENOENT: 'No such file or directory',
  EACCES: 'Permission denied',
  EISDIR: 'Is a directory',
  ENOTDIR: 'Not a directory',
  ELOOP: 'Too many levels of symbolic links',
  ENAMETOOLONG: 'File name too long',
};

// Score files are flat. Refuse deep nesting before parsing, identically in
// both CLIs (Python's json recurses and would crash where Node's parses).
const MAX_DEPTH = 64;
function checkDepth(text, name) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (const ch of text) {
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
    } else if (ch === '[' || ch === '{') {
      if (++depth > MAX_DEPTH) throw new UsageError(`${name} is nested too deeply to be a score file`);
    } else if (ch === ']' || ch === '}') {
      depth--;
    }
  }
}

function readScores(path) {
  const name = path === '-' ? 'stdin' : path;
  let text;
  try {
    text = readFileSync(path === '-' ? 0 : path, 'utf8');
  } catch (err) {
    throw new UsageError(`cannot read ${name}: ${READ_ERRORS[err.code] ?? err.code ?? err.message}`);
  }
  checkDepth(text, name);
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new UsageError(`${name} is not valid JSON`);
  }
  const values = Array.isArray(data) ? data : data && typeof data === 'object' ? Object.values(data) : null;
  if (!values || !values.every((v) => typeof v === 'number' && Number.isFinite(v))) {
    throw new UsageError(`${name} must be a JSON array of numbers or an object of {"item id": number}`);
  }
  // Past 1e15, sums and toFixed lose the digits the report prints.
  if (values.some((v) => Math.abs(v) >= 1e15)) throw new UsageError(`${name} has a score of 1e15 or more; rescale the scores`);
  return Array.isArray(data) ? { values: data } : { values, ids: Object.keys(data), byId: data };
}

// Align two score sets. Objects pair by id; arrays pass through.
function align(base, vari, pairedFlag) {
  if (base.ids || vari.ids) {
    if (!base.ids || !vari.ids) throw new UsageError('both files must be arrays, or both objects keyed by item id');
    const missing = base.ids.filter((id) => !Object.hasOwn(vari.byId, id));
    const extra = vari.ids.filter((id) => !Object.hasOwn(base.byId, id));
    if (missing.length || extra.length) {
      const show = (ids) => ids.slice(0, 5).join(', ') + (ids.length > 5 ? ', ...' : '');
      throw new UsageError(
        [missing.length && `missing from variant: ${show(missing)}`, extra.length && `missing from baseline: ${show(extra)}`]
          .filter(Boolean)
          .join('; '),
      );
    }
    return { a: base.values, b: base.ids.map((id) => vari.byId[id]), paired: true };
  }
  if (pairedFlag && base.values.length !== vari.values.length) {
    throw new UsageError(`--paired needs equal lengths (baseline ${base.values.length}, variant ${vari.values.length})`);
  }
  return { a: base.values, b: vari.values, paired: pairedFlag };
}

// Plain decimal or exponent notation only, the same grammar as the Python CLI
// (no hex, underscores, or Infinity).
const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
const toNumber = (raw) => (NUMBER.test(raw) ? Number(raw) : NaN);

function parseTarget(raw) {
  if (raw === undefined) throw new UsageError('--target is required: the smallest shift worth acting on');
  const target = toNumber(raw);
  if (!(target > 0) || !Number.isFinite(target)) throw new UsageError('--target must be a positive number');
  return target;
}

function parseNonNegative(raw, name) {
  const value = toNumber(raw);
  if (!(value >= 0) || !Number.isFinite(value)) throw new UsageError(`${name} must be a non-negative number`);
  return value;
}

// Fixed-point formatting shared with the Python CLI. The bindings agree to
// ~1e-15, which can straddle a display rounding boundary, so settle the value
// at 9 decimals first. Ties away from zero, no negative zero.
const fixed = (x, digits) => (Number(x.toFixed(9)) + 0).toFixed(digits);
const signed = (x, digits) => ((x + 0 >= 0 ? '+' : '') + fixed(x, digits));
const pText = (p) => (p < 0.001 ? '< 0.001' : fixed(p, 3));

function compareText(v, sa, sb) {
  const unit = v.design === 'paired' ? 'items' : 'runs per arm';
  const design = v.design === 'paired' ? 'paired t-test, same items in both arms' : "Welch's t-test, independent runs";
  const lines = [
    `eval-kit compare: ${design}`,
    `  baseline  n=${v.nBaseline}  mean ${fixed(sa.mean, 2)}`,
    `  variant   n=${v.nVariant}  mean ${fixed(sb.mean, 2)}`,
    `  shift     ${signed(v.shift, 2)}  95% CI [${signed(v.ciLo, 2)}, ${signed(v.ciHi, 2)}]`,
    `  p         ${pText(v.p)}`,
    `  ${v.effectName.padEnd(8)}  ${v.constantArm === 'baseline' ? 'off-scale (the baseline has no spread)' : fixed(v.effect, 2)}`,
    `  target    ${v.target} (${v.lowerIsBetter ? 'lower' : 'higher'} is better)`,
    '',
  ];
  const size = fixed(Math.abs(v.shift), 2);
  if (v.verdict === 'improved' || v.verdict === 'regressed') {
    const dir = v.shift > 0 ? 'higher' : 'lower';
    lines.push(`${v.verdict.toUpperCase()}. The variant scores ${dir} by ${size}, p ${pText(v.p)}.`);
    if (Math.abs(v.shift) < v.target) {
      lines.push(`The estimate is below the ${v.target}-point target, but the 95% CI reaches it.`);
    }
  } else if (v.reason === 'below_target') {
    lines.push(`NO CHANGE at this target. The shift is real (p ${pText(v.p)}), but the 95% CI rules out a shift`);
    lines.push(`of ${v.target} or more in either direction.`);
  } else if (v.verdict === 'no_change') {
    lines.push(`NO CHANGE. The 95% CI rules out a shift of ${v.target} or more in either direction.`);
  } else if (v.reason === 'zero_spread') {
    const what =
      v.design === 'paired'
        ? 'Every item moved by exactly the same amount'
        : 'Every run in both arms scored the same';
    lines.push(`CAN'T TELL. ${what}. That is usually a broken harness; check it before reading anything.`);
  } else {
    lines.push(`CAN'T TELL YET. The 95% CI includes both no change and a ${v.target}-point shift.`);
    const have = v.design === 'paired' ? v.nBaseline : Math.min(v.nBaseline, v.nVariant);
    lines.push(
      v.needed > have
        ? `Resolving it takes ${v.needed} ${unit} at the spread seen so far; you have ${have}.`
        : `The CI is still too wide at this spread; add ${unit === 'items' ? 'items' : 'runs to both arms'}.`,
    );
  }
  if (v.constantArm) {
    const mean = v.constantArm === 'baseline' ? sa.mean : sb.mean;
    lines.push(`Every ${v.constantArm} run scored the same (${fixed(mean, 2)}). That is a ceiling or a broken harness;`);
    lines.push('check which before trusting this verdict.');
  }
  return lines.join('\n');
}

function runCompare(opts) {
  if (!opts.baseline || !opts.variant) throw new UsageError('compare needs --baseline and --variant');
  if (opts.baseline === '-' && opts.variant === '-') throw new UsageError('only one of --baseline and --variant can be -');
  const target = parseTarget(opts.target);
  const { a, b, paired } = align(readScores(opts.baseline), readScores(opts.variant), opts.paired);
  if (a.length < 2 || b.length < 2) throw new UsageError('each arm needs at least 2 scores');

  const v = verdict(a, b, { target, paired, lowerIsBetter: opts['lower-is-better'] });
  const sa = stats(a);
  const sb = stats(b);
  if (![sa.mean, sb.mean, v.shift, v.ciLo, v.ciHi, v.p, v.effect].every(Number.isFinite)) {
    throw new UsageError('the scores are out of range for double precision; rescale them');
  }
  if (v.needed !== null && !(v.needed <= Number.MAX_SAFE_INTEGER)) throw new UsageError('--target is too small for this spread: the run count is too large to compute');
  if (opts.json) console.log(JSON.stringify(v, null, 2));
  else console.log(compareText(v, sa, sb));

  if (!opts.gate) return EXIT_OK;
  if (v.verdict === 'regressed') return EXIT_REGRESSED;
  if (v.verdict === 'cant_tell') return EXIT_CANT_TELL;
  return EXIT_OK;
}

function runPower(opts) {
  const target = parseTarget(opts.target);
  let spread;
  let paired = opts.paired;
  if (opts.std !== undefined) {
    if (paired) throw new UsageError('--std sizes independent runs; for --paired use --sd-diff');
    spread = parseNonNegative(opts.std, '--std');
  } else if (opts['sd-diff'] !== undefined) {
    spread = parseNonNegative(opts['sd-diff'], '--sd-diff');
    paired = true;
  } else if (opts.baseline && opts.variant) {
    const { a, b } = align(readScores(opts.baseline), readScores(opts.variant), true);
    const pilot = pairedTTest(a, b);
    if (!pilot) throw new UsageError('the paired pilot needs at least 2 items');
    spread = pilot.sdDiff;
    paired = true;
  } else if (opts.baseline) {
    if (paired) throw new UsageError('--paired power needs --sd-diff, or --baseline and --variant from a pilot');
    const s = stats(readScores(opts.baseline).values);
    if (!s || s.n < 2) throw new UsageError('the baseline needs at least 2 scores');
    spread = s.std;
  } else {
    throw new UsageError('power needs --std, --sd-diff, or --baseline (and --variant for a paired pilot)');
  }

  const needed = paired ? requiredNPaired(spread, target) : requiredN(spread, target);
  if (!(needed <= Number.MAX_SAFE_INTEGER)) throw new UsageError('--target is too small for this spread: the run count is too large to compute');
  const design = paired ? 'paired' : 'welch';
  if (opts.json) {
    console.log(JSON.stringify({ design, target, spread, needed }, null, 2));
  } else {
    const unit = paired ? 'items' : 'runs per arm';
    const what = paired ? 'sd of differences' : 'baseline std';
    console.log(`${needed} ${unit} to detect a ${target}-point shift (${what} ${fixed(spread, 2)}, 80% power, alpha 0.05)`);
  }
  return EXIT_OK;
}

// parseArgs messages, reworded to match the Python CLI's.
function parseErrorText(message) {
  const unknown = message.match(/Unknown option '([^']+)'/);
  if (unknown) return `unknown option: ${unknown[1]}`;
  const missing = message.match(/Option '(--[\w-]+)(?: <value>)?' argument (missing|is ambiguous)/);
  if (missing) return `${missing[1]} needs a value`;
  const extra = message.match(/Option '(--[\w-]+)' does not take an argument/);
  if (extra) return `${extra[1]} takes no value`;
  return message;
}

const VALUE_FLAGS = new Set(['--baseline', '--variant', '--target', '--std', '--sd-diff']);

// parseArgs refuses `--target -1` as ambiguous; argparse reads -1 as the value.
// Attach negative numbers to their flag so both CLIs reach the same check.
function attachNegativeValues(argv) {
  const out = [];
  for (let i = 0; i < argv.length; i++) {
    if (VALUE_FLAGS.has(argv[i]) && /^-(\d|\.\d)/.test(argv[i + 1] ?? '')) {
      out.push(`${argv[i]}=${argv[i + 1]}`);
      i++;
    } else {
      out.push(argv[i]);
    }
  }
  return out;
}

export function main(argv) {
  let parsed;
  try {
    parsed = parseArgs({
      args: attachNegativeValues(argv),
      allowPositionals: true,
      options: {
        baseline: { type: 'string' },
        variant: { type: 'string' },
        target: { type: 'string' },
        paired: { type: 'boolean', default: false },
        'lower-is-better': { type: 'boolean', default: false },
        std: { type: 'string' },
        'sd-diff': { type: 'string' },
        json: { type: 'boolean', default: false },
        gate: { type: 'boolean', default: false },
        help: { type: 'boolean', short: 'h', default: false },
        version: { type: 'boolean', default: false },
      },
    });
  } catch (err) {
    console.error(`eval-kit: ${parseErrorText(err.message)}\nRun eval-kit --help for usage.`);
    return EXIT_USAGE;
  }
  const { values: opts, positionals } = parsed;
  if (opts.version) {
    console.log(VERSION);
    return EXIT_OK;
  }
  if (opts.help || positionals.length === 0) {
    console.log(HELP);
    return opts.help ? EXIT_OK : EXIT_USAGE;
  }
  try {
    if (positionals.length > 1) throw new UsageError(`unexpected argument: ${positionals[1]}`);
    if (positionals[0] === 'compare') return runCompare(opts);
    if (positionals[0] === 'power') return runPower(opts);
    throw new UsageError(`unknown command: ${positionals[0]} (use compare or power)`);
  } catch (err) {
    if (!(err instanceof UsageError)) throw err;
    console.error(`eval-kit: ${err.message}`);
    return EXIT_USAGE;
  }
}

// Run only as a program, not when imported (npm's bin shim is a symlink).
if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}
