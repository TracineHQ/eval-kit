"""eval-kit CLI. Thin wrapper over eval_kit.stats: read scores, call
verdict() or the power functions, print. The JS CLI (js/cli.mjs) prints
the same text and JSON; cross-validation holds them to it.

Copyright 2026 TracineHQ
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn

from eval_kit import __version__
from eval_kit.stats import (
    VerdictResult,
    descriptive_stats,
    paired_t_test,
    required_n,
    required_n_paired,
    verdict,
)

# Exit codes. Outcome codes apply only with --gate, so plain analysis
# never fails a build.
EXIT_OK = 0
EXIT_REGRESSED = 1
EXIT_USAGE = 2
EXIT_CANT_TELL = 3

HELP = f"""eval-kit {__version__}: did the eval get better, or is that noise?

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
with Welch's t-test. An object of {{"item id": score}} pairs items across the
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
"""

# verdict() field names in the JSON contract shared with the JS CLI.
JSON_KEYS = {
    "lower_is_better": "lowerIsBetter",
    "ci_lo": "ciLo",
    "ci_hi": "ciHi",
    "effect_name": "effectName",
    "n_baseline": "nBaseline",
    "n_variant": "nVariant",
    "constant_arm": "constantArm",
}


class UsageError(Exception):
    pass


MAX_SAFE_INTEGER = 2**53 - 1  # JS Number.MAX_SAFE_INTEGER
MAX_DEPTH = 64


def _say(text: str, stream: Any = None) -> None:
    """print(), but lone surrogates (from JSON like "\\ud800") come out as
    U+FFFD, as Node writes them, instead of raising or escaping."""
    clean = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    print(clean, file=stream or sys.stdout)


def _check_depth(text: str, name: str) -> None:
    """Refuse deep nesting before parsing, identically in both CLIs (Python's
    json recurses and would crash; Node's would parse). Score files are flat."""
    depth, in_string, escaped = 0, False, False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise UsageError(f"{name} is nested too deeply to be a score file")
        elif ch in "]}":
            depth -= 1


def _js_key_order(data: dict[str, Any]) -> dict[str, Any]:
    """Keys in the order JS objects keep them: array-index keys ("0", "12")
    ascending first, then the rest in file order. Pairing and messages then
    see items in the same order in both CLIs."""

    def is_index(key: str) -> bool:
        return key.isdigit() and key.isascii() and (key == "0" or key[0] != "0") and int(key) < 2**32 - 1

    indexed = sorted((k for k in data if is_index(k)), key=int)
    return {**{k: data[k] for k in indexed}, **{k: v for k, v in data.items() if not is_index(k)}}


def _reject_constant(name: str) -> NoReturn:
    # NaN and Infinity are not JSON; json.loads accepts them by default.
    raise ValueError(name)


def _as_float(v: Any) -> float | None:
    """A JSON number as JS reads it: finite float, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if isfinite(f) else None


def _read_scores(path: str) -> dict[str, Any]:
    name = "stdin" if path == "-" else path
    try:
        raw = sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()
    except OSError as err:
        raise UsageError(f"cannot read {name}: {err.strerror}") from None
    # Invalid UTF-8 decodes to U+FFFD, as Node's readFileSync does.
    text = raw.decode("utf-8", errors="replace")
    _check_depth(text, name)
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except ValueError:
        raise UsageError(f"{name} is not valid JSON") from None
    if isinstance(data, dict):
        data = _js_key_order(data)
    raw_values = list(data.values()) if isinstance(data, dict) else data
    parsed = [_as_float(v) for v in raw_values] if isinstance(raw_values, list) else [None]
    values = [v for v in parsed if v is not None]
    if len(values) != len(parsed):
        raise UsageError(f'{name} must be a JSON array of numbers or an object of {{"item id": number}}')
    # Past 1e15, sums and toFixed lose the digits the report prints.
    if any(abs(v) >= 1e15 for v in values):
        raise UsageError(f"{name} has a score of 1e15 or more; rescale the scores")
    if isinstance(data, dict):
        return {"values": values, "ids": list(data.keys()), "by_id": dict(zip(data.keys(), values, strict=True))}
    return {"values": values}


def _align(base: dict[str, Any], vari: dict[str, Any], paired_flag: bool) -> tuple[list[float], list[float], bool]:
    """Objects pair by id; arrays pass through."""
    if "ids" in base or "ids" in vari:
        if "ids" not in base or "ids" not in vari:
            raise UsageError("both files must be arrays, or both objects keyed by item id")
        missing = [i for i in base["ids"] if i not in vari["by_id"]]
        extra = [i for i in vari["ids"] if i not in base["by_id"]]
        if missing or extra:

            def show(ids: list[str]) -> str:
                return ", ".join(ids[:5]) + (", ..." if len(ids) > 5 else "")

            parts = []
            if missing:
                parts.append(f"missing from variant: {show(missing)}")
            if extra:
                parts.append(f"missing from baseline: {show(extra)}")
            raise UsageError("; ".join(parts))
        return base["values"], [vari["by_id"][i] for i in base["ids"]], True
    if paired_flag and len(base["values"]) != len(vari["values"]):
        raise UsageError(
            f"--paired needs equal lengths (baseline {len(base['values'])}, variant {len(vari['values'])})"
        )
    return base["values"], vari["values"], paired_flag


# Plain decimal or exponent notation only, the same grammar as the JS CLI
# (no underscores, non-ASCII digits, or inf).
NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)


def _number(raw: str | None) -> float:
    if raw is None or not NUMBER.fullmatch(raw):
        return float("nan")
    return float(raw)


def _parse_target(raw: str | None) -> float:
    if raw is None:
        raise UsageError("--target is required: the smallest shift worth acting on")
    target = _number(raw)
    if not (isfinite(target) and target > 0):
        raise UsageError("--target must be a positive number")
    return target


def _parse_non_negative(raw: str, name: str) -> float:
    value = _number(raw)
    if not (isfinite(value) and value >= 0):
        raise UsageError(f"{name} must be a non-negative number")
    return value


def _to_fixed(x: float, digits: int) -> str:
    """Number.prototype.toFixed: exact binary value, ties away from zero."""
    q = Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{q:f}"


# Fixed-point formatting shared with the JS CLI. The bindings agree to
# ~1e-15, which can straddle a display rounding boundary, so settle the value
# at 9 decimals first. Ties away from zero, no negative zero.
def _fixed(x: float, digits: int) -> str:
    return _to_fixed(float(_to_fixed(x, 9)) + 0.0, digits)


def _signed(x: float, digits: int) -> str:
    return ("+" if x + 0.0 >= 0 else "") + _fixed(x, digits)


def _p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else _fixed(p, 3)


def _num(x: float) -> str:
    """A number as JS's Number#toString prints it: shortest round-trip digits,
    plain decimal for 1e-6 <= |x| < 1e21, exponent form (1e-7, 1e+21) outside."""
    x = float(x)
    if x == 0:
        return "0"
    if 1e-6 <= abs(x) < 1e21:
        text = f"{Decimal(repr(x)):f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    mantissa, exponent = repr(x).split("e")
    mantissa = mantissa[:-2] if mantissa.endswith(".0") else mantissa
    return f"{mantissa}e{int(exponent):+d}"


def _compare_text(v: VerdictResult, mean_a: float, mean_b: float) -> str:
    unit = "items" if v.design == "paired" else "runs per arm"
    design = "paired t-test, same items in both arms" if v.design == "paired" else "Welch's t-test, independent runs"
    target = _num(v.target)
    effect = "off-scale (the baseline has no spread)" if v.constant_arm == "baseline" else _fixed(v.effect, 2)
    lines = [
        f"eval-kit compare: {design}",
        f"  baseline  n={v.n_baseline}  mean {_fixed(mean_a, 2)}",
        f"  variant   n={v.n_variant}  mean {_fixed(mean_b, 2)}",
        f"  shift     {_signed(v.shift, 2)}  95% CI [{_signed(v.ci_lo, 2)}, {_signed(v.ci_hi, 2)}]",
        f"  p         {_p_text(v.p)}",
        f"  {v.effect_name.ljust(8)}  {effect}",
        f"  target    {target} ({'lower' if v.lower_is_better else 'higher'} is better)",
        "",
    ]
    size = _fixed(abs(v.shift), 2)
    if v.verdict in ("improved", "regressed"):
        direction = "higher" if v.shift > 0 else "lower"
        lines.append(f"{v.verdict.upper()}. The variant scores {direction} by {size}, p {_p_text(v.p)}.")
        if abs(v.shift) < v.target:
            lines.append(f"The estimate is below the {target}-point target, but the 95% CI reaches it.")
    elif v.reason == "below_target":
        lines.append(
            f"NO CHANGE at this target. The shift is real (p {_p_text(v.p)}), but the 95% CI rules out a shift"
        )
        lines.append(f"of {target} or more in either direction.")
    elif v.verdict == "no_change":
        lines.append(f"NO CHANGE. The 95% CI rules out a shift of {target} or more in either direction.")
    elif v.reason == "zero_spread":
        what = (
            "Every item moved by exactly the same amount"
            if v.design == "paired"
            else "Every run in both arms scored the same"
        )
        lines.append(f"CAN'T TELL. {what}. That is usually a broken harness; check it before reading anything.")
    else:
        lines.append(f"CAN'T TELL YET. The 95% CI includes both no change and a {target}-point shift.")
        have = v.n_baseline if v.design == "paired" else min(v.n_baseline, v.n_variant)
        assert v.needed is not None
        needed = _num(float(v.needed))
        lines.append(
            f"Resolving it takes {needed} {unit} at the spread seen so far; you have {have}."
            if v.needed > have
            else f"The CI is still too wide at this spread; add {'items' if unit == 'items' else 'runs to both arms'}."
        )
    if v.constant_arm is not None:
        mean = mean_a if v.constant_arm == "baseline" else mean_b
        lines.append(
            f"Every {v.constant_arm} run scored the same ({_fixed(mean, 2)}). That is a ceiling or a broken harness;"
        )
        lines.append("check which before trusting this verdict.")
    return "\n".join(lines)


def _json(obj: dict[str, Any]) -> str:
    """JSON.stringify(obj, null, 2) for a flat object: numbers as JS prints
    them (3, not 3.0; 1e-7, not 1e-07). Only computed floats can differ from
    the JS output, in their last digits."""

    def value(x: Any) -> str:
        if x is None or isinstance(x, (bool, str)):
            return json.dumps(x)
        return _num(x) if isfinite(x) else "null"

    body = ",\n".join(f"  {json.dumps(k)}: {value(x)}" for k, x in obj.items())
    return "{\n" + body + "\n}"


def _verdict_json(v: VerdictResult) -> dict[str, Any]:
    return {JSON_KEYS.get(k, k): val for k, val in asdict(v).items()}


def _run_compare(args: Args) -> int:
    if not args.baseline or not args.variant:
        raise UsageError("compare needs --baseline and --variant")
    if args.baseline == "-" and args.variant == "-":
        raise UsageError("only one of --baseline and --variant can be -")
    target = _parse_target(args.target)
    a, b, paired = _align(_read_scores(args.baseline), _read_scores(args.variant), args.paired)
    if len(a) < 2 or len(b) < 2:
        raise UsageError("each arm needs at least 2 scores")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = verdict(a, b, target, paired=paired, lower_is_better=args.lower_is_better)
        sa, sb = descriptive_stats(a), descriptive_stats(b)
    if v is None or sa is None or sb is None:
        raise UsageError("each arm needs at least 2 scores")
    if not all(isfinite(x) for x in [sa.mean, sb.mean, v.shift, v.ci_lo, v.ci_hi, v.p, v.effect]):
        raise UsageError("the scores are out of range for double precision; rescale them")
    if v.needed is not None and not v.needed <= MAX_SAFE_INTEGER:
        raise UsageError("--target is too small for this spread: the run count is too large to compute")
    if args.json:
        _say(_json(_verdict_json(v)))
    else:
        _say(_compare_text(v, sa.mean, sb.mean))

    if not args.gate:
        return EXIT_OK
    if v.verdict == "regressed":
        return EXIT_REGRESSED
    if v.verdict == "cant_tell":
        return EXIT_CANT_TELL
    return EXIT_OK


def _run_power(args: Args) -> int:
    target = _parse_target(args.target)
    paired = args.paired
    if args.std is not None:
        if paired:
            raise UsageError("--std sizes independent runs; for --paired use --sd-diff")
        spread = _parse_non_negative(args.std, "--std")
    elif args.sd_diff is not None:
        spread = _parse_non_negative(args.sd_diff, "--sd-diff")
        paired = True
    elif args.baseline and args.variant:
        a, b, _ = _align(_read_scores(args.baseline), _read_scores(args.variant), True)
        pilot = paired_t_test(a, b)
        if pilot is None:
            raise UsageError("the paired pilot needs at least 2 items")
        spread = pilot.sd_diff
        paired = True
    elif args.baseline:
        if paired:
            raise UsageError("--paired power needs --sd-diff, or --baseline and --variant from a pilot")
        s = descriptive_stats(_read_scores(args.baseline)["values"])
        if s is None or s.n < 2:
            raise UsageError("the baseline needs at least 2 scores")
        spread = s.std
    else:
        raise UsageError("power needs --std, --sd-diff, or --baseline (and --variant for a paired pilot)")

    needed = required_n_paired(spread, target) if paired else required_n(spread, target)
    if not needed <= MAX_SAFE_INTEGER:
        raise UsageError("--target is too small for this spread: the run count is too large to compute")
    design = "paired" if paired else "welch"
    if args.json:
        _say(_json({"design": design, "target": target, "spread": spread, "needed": needed}))
    else:
        unit = "items" if paired else "runs per arm"
        what = "sd of differences" if paired else "baseline std"
        _say(
            f"{_num(float(needed))} {unit} to detect a {_num(target)}-point shift "
            f"({what} {_fixed(spread, 2)}, 80% power, alpha 0.05)"
        )
    return EXIT_OK


VALUE_FLAGS = ("baseline", "variant", "target", "std", "sd-diff")
BOOLEAN_FLAGS = ("paired", "lower-is-better", "json", "gate", "version", "help")
NEGATIVE_NUMBER = re.compile(r"-(\d|\.\d)")


@dataclass
class Args:
    command: list[str] = field(default_factory=list)
    baseline: str | None = None
    variant: str | None = None
    target: str | None = None
    std: str | None = None
    sd_diff: str | None = None
    paired: bool = False
    lower_is_better: bool = False
    json: bool = False
    gate: bool = False
    version: bool = False
    help: bool = False


def _parse_args(argv: Sequence[str]) -> Args:
    """The JS CLI's node:util parseArgs rules, token by token, so both CLIs
    accept the same command lines and report the same first error: no
    abbreviations, --flag=value or --flag value, -h (bundles like -hj check
    each letter), -- ends options. A value that starts with - is refused,
    except - itself (stdin) and a negative number."""
    args = Args()
    tokens = list(argv)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if token == "--":
            args.command += tokens[i:]
            break
        if token.startswith("--"):
            name, eq, value = token[2:].partition("=")
            attr = name.replace("-", "_")
            if name in VALUE_FLAGS:
                if not eq:
                    nxt = tokens[i] if i < len(tokens) else None
                    if nxt is None or (nxt.startswith("-") and nxt != "-" and not NEGATIVE_NUMBER.match(nxt)):
                        raise UsageError(f"--{name} needs a value")
                    value = nxt
                    i += 1
                setattr(args, attr, value)
            elif name in BOOLEAN_FLAGS:
                if eq:
                    raise UsageError(f"--{name} takes no value")
                setattr(args, attr, True)
            else:
                raise UsageError(f"unknown option: --{name}")
        elif token.startswith("-") and token != "-":
            for letter in token[1:]:
                if letter != "h":
                    raise UsageError(f"unknown option: -{letter}")
                args.help = True
        else:
            args.command.append(token)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(sys.argv[1:] if argv is None else argv)
    except UsageError as err:
        _say(f"eval-kit: {err}\nRun eval-kit --help for usage.", sys.stderr)
        return EXIT_USAGE
    if args.version:
        _say(__version__)
        return EXIT_OK
    if args.help or not args.command:
        _say(HELP)
        return EXIT_OK if args.help else EXIT_USAGE
    try:
        if len(args.command) > 1:
            raise UsageError(f"unexpected argument: {args.command[1]}")
        if args.command[0] == "compare":
            return _run_compare(args)
        if args.command[0] == "power":
            return _run_power(args)
        raise UsageError(f"unknown command: {args.command[0]} (use compare or power)")
    except UsageError as err:
        _say(f"eval-kit: {err}", sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
