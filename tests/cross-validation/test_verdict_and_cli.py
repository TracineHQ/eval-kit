"""Cross-validation for the decision layer: verdict() in both bindings, and
the two CLIs (js/cli.mjs, python -m eval_kit): text, errors, and exit codes
byte for byte; --json token for token, computed floats to 1e-9.

The CLIs are the interface the Claude Code skill and CI pipelines use, so
their text, JSON, and exit codes are part of the contract.

Run (from repo root):
    cd python && .venv/bin/pytest ../tests/cross-validation/ -v
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from eval_kit.stats import paired_t_test, verdict

sys.path.insert(0, str(Path(__file__).parent))
from test_cross_validation import EXACT, run_js  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
JS_CLI = REPO / "js" / "cli.mjs"

WELCH_BASE = [80.1, 82.4, 79.0, 83.5, 81.2, 80.6, 82.9, 79.8, 81.7, 80.9, 82.2, 81.4]

# 60 runs with a small spread, so a 0.3-point shift is significant.
LONG_BASE = [80 + ((i * 7) % 11) / 5 for i in range(60)]

# (label, baseline, variant, options) -- options use the JS names.
VERDICT_CASES = [
    ("cant_tell underpowered", WELCH_BASE[:4], [83.9, 81.2, 86.0, 84.7], {"target": 3}),
    ("improved", WELCH_BASE[:8], [84.3, 85.9, 83.1, 86.4, 84.8, 83.7, 86.1, 85.2], {"target": 3}),
    (
        "no_change",
        WELCH_BASE,
        [80.6, 81.9, 79.7, 82.8, 81.5, 80.2, 83.1, 80.4, 81.1, 82.0, 81.8, 80.9],
        {"target": 3},
    ),
    ("regressed", WELCH_BASE[:8], [77.3, 78.9, 76.1, 79.4, 77.8, 76.7, 79.1, 78.2], {"target": 3}),
    (
        "lower is better flips direction",
        WELCH_BASE[:8],
        [77.3, 78.9, 76.1, 79.4, 77.8, 76.7, 79.1, 78.2],
        {"target": 3, "lowerIsBetter": True},
    ),
    ("real but below target", WELCH_BASE, [x + 2.2 for x in WELCH_BASE], {"target": 3}),
    (
        "paired improved",
        [40, 55, 62, 70, 78, 85, 91, 48, 66, 73],
        [42, 57, 63, 72, 80, 88, 92, 50, 69, 75],
        {"target": 1, "paired": True},
    ),
    (
        "paired zero spread",
        [0.85, 0.65, 0.95, 0.75, 0.55],
        [0.8, 0.6, 0.9, 0.7, 0.5],
        {"target": 0.1, "paired": True},
    ),
    ("welch constant baseline", [0.1, 0.1, 0.1], [0.2, 0.25, 0.3], {"target": 0.05}),
    ("welch constant variant", [0.2, 0.25, 0.3], [0.1, 0.1, 0.1], {"target": 0.05}),
    ("float dust is zero spread", [0.3, 0.3, 0.1 + 0.2], [0.4, 0.4, 0.4], {"target": 1}),
    ("significant but below target", LONG_BASE, [x - 0.3 for x in LONG_BASE], {"target": 3}),
    ("not significant, inside target", LONG_BASE, [x - 0.2 for x in LONG_BASE], {"target": 3}),
    ("marginal p", WELCH_BASE[:8], [round(x + 1.7, 1) for x in WELCH_BASE[:8]], {"target": 3}),
    ("unequal n", WELCH_BASE[:6], [83.9, 81.2, 86.0, 84.7], {"target": 3}),
    ("unequal n and variance", WELCH_BASE, [70, 60, 80, 50], {"target": 3}),
    ("identical arms", [5, 5, 5], [5, 5, 5], {"target": 1}),
    ("rounding tie", [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 9], {"target": 2}),
]

EXPECTED = {
    "cant_tell underpowered": ("cant_tell", "underpowered"),
    "improved": ("improved", "significant"),
    "no_change": ("no_change", "within_target"),
    "regressed": ("regressed", "significant"),
    "lower is better flips direction": ("improved", "significant"),
    "real but below target": ("improved", "significant"),
    "paired improved": ("improved", "significant"),
    "paired zero spread": ("cant_tell", "zero_spread"),
    "welch constant baseline": ("improved", "significant"),
    "welch constant variant": ("regressed", "significant"),
    "float dust is zero spread": ("cant_tell", "zero_spread"),
    "significant but below target": ("no_change", "below_target"),
    "not significant, inside target": ("no_change", "within_target"),
    "marginal p": ("cant_tell", "underpowered"),
    "unequal n": ("cant_tell", "underpowered"),
    "unequal n and variance": ("cant_tell", "underpowered"),
    "identical arms": ("cant_tell", "zero_spread"),
    "rounding tie": ("cant_tell", "underpowered"),
}

PY_FIELDS = {
    "verdict": "verdict",
    "reason": "reason",
    "design": "design",
    "target": "target",
    "lowerIsBetter": "lower_is_better",
    "shift": "shift",
    "ciLo": "ci_lo",
    "ciHi": "ci_hi",
    "p": "p",
    "effect": "effect",
    "effectName": "effect_name",
    "nBaseline": "n_baseline",
    "nVariant": "n_variant",
    "needed": "needed",
    "constantArm": "constant_arm",
}


def _py_verdict(base: list, var: list, opts: dict[str, Any]):
    return verdict(
        base,
        var,
        opts["target"],
        paired=opts.get("paired", False),
        lower_is_better=opts.get("lowerIsBetter", False),
    )


@pytest.mark.parametrize("label,base,var,opts", VERDICT_CASES, ids=[c[0] for c in VERDICT_CASES])
def test_verdict_parity(label, base, var, opts):
    js = run_js("verdict", base, var, opts)
    py = _py_verdict(base, var, opts)
    assert (py.verdict, py.reason) == EXPECTED[label]
    for js_key, py_key in PY_FIELDS.items():
        expected = getattr(py, py_key)
        if isinstance(expected, float):
            assert js[js_key] == pytest.approx(expected, rel=EXACT, abs=EXACT), js_key
        else:
            assert js[js_key] == expected, js_key


def _with_spread(n: int, mean: float, sd: float) -> list[float]:
    """n evenly spaced values with exactly this mean and sample std."""
    raw = [i - (n - 1) / 2 for i in range(n)]
    scale = sd / statistics.stdev(raw)
    return [mean + x * scale for x in raw]


@pytest.mark.parametrize(
    "pilot_base,pilot_var,opts",
    [
        (WELCH_BASE[:4], [83.9, 81.2, 86.0, 84.7], {"target": 3}),
        (WELCH_BASE[:6], [70, 60, 80, 50, 75, 65], {"target": 3}),
        ([40, 55, 62, 70, 78], [43, 56, 66, 71, 81], {"target": 1, "paired": True}),
    ],
    ids=["welch equal spread", "welch noisier variant", "paired"],
)
def test_needed_is_the_n_where_cant_tell_becomes_impossible(pilot_base, pilot_var, opts):
    """At needed, with the spread the pilot showed, the CI is under half the
    target wide on each side, so no true shift can leave it spanning both 0
    and the target. One fewer and a shift of half the target still can."""
    target, paired = opts["target"], opts.get("paired", False)
    pilot = _py_verdict(pilot_base, pilot_var, opts)
    assert run_js("verdict", pilot_base, pilot_var, opts)["needed"] == pilot.needed
    n = int(pilot.needed)
    if paired:
        sd_diff = paired_t_test(pilot_base, pilot_var).sd_diff
    else:
        sd_a, sd_b = statistics.stdev(pilot_base), statistics.stdev(pilot_var)

    def arms(size: int, shift: float) -> tuple[list[float], list[float]]:
        if paired:
            base = [50.0 + 3 * i for i in range(size)]
            diffs = _with_spread(size, shift, sd_diff)
            return base, [b + d for b, d in zip(base, diffs, strict=True)]
        return _with_spread(size, 80, sd_a), _with_spread(size, 80 + shift, sd_b)

    # target / 2 is the worst case: the shift a CI spanning both is centred on.
    for shift in (0, target / 4, target / 2, 3 * target / 4, target, -target / 2):
        a, b = arms(n, shift)
        assert _py_verdict(a, b, opts).verdict != "cant_tell", shift
    a, b = arms(n, target / 2)
    assert run_js("verdict", a, b, opts)["verdict"] != "cant_tell"
    a, b = arms(n - 1, target / 2)
    assert _py_verdict(a, b, opts).verdict == "cant_tell"
    assert run_js("verdict", a, b, opts)["verdict"] == "cant_tell"


@pytest.mark.parametrize(
    "base,var,opts",
    [
        ([1], [1, 2], {"target": 1}),
        ([1, 2, 3], [1, 2], {"target": 1, "paired": True}),
        ([1, 2, 3], [4, 5, 6], {"target": 0}),
        ([1, 2, 3], [4, 5, 6], {"target": -1}),
    ],
)
def test_verdict_invalid_input_is_null_in_both(base, var, opts):
    assert run_js("verdict", base, var, opts) is None
    assert _py_verdict(base, var, opts) is None


# ---------------------------------------------------------------------------
# CLI parity
# ---------------------------------------------------------------------------


def _run(cmd: list[str], args: list[str], stdin: str | None) -> subprocess.CompletedProcess:
    # The timeout turns a hung search into a failure instead of a stalled CI job.
    return subprocess.run(cmd + args, input=stdin, capture_output=True, text=True, cwd=REPO, timeout=60)


def run_both(args: list[str], stdin: str | None = None):
    js = _run(["node", str(JS_CLI)], args, stdin)
    py = _run([sys.executable, "-m", "eval_kit"], args, stdin)
    return js, py


@pytest.fixture
def scores(tmp_path):
    def write(name: str, data: Any) -> str:
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return str(path)

    return write


@pytest.mark.parametrize("label,base,var,opts", VERDICT_CASES, ids=[c[0] for c in VERDICT_CASES])
@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_cli_compare_parity(scores, label, base, var, opts, as_json):
    args = ["compare", "--baseline", scores("b.json", base), "--variant", scores("v.json", var)]
    args += ["--target", str(opts["target"]), "--gate"]
    if opts.get("paired"):
        args.append("--paired")
    if opts.get("lowerIsBetter"):
        args.append("--lower-is-better")
    if as_json:
        args.append("--json")

    js, py = run_both(args)
    assert js.returncode == py.returncode, (js.stderr, py.stderr)
    expected_exit = {"improved": 0, "no_change": 0, "regressed": 1, "cant_tell": 3}[EXPECTED[label][0]]
    assert py.returncode == expected_exit
    if as_json:
        # Same keys in the same order, same layout. Computed floats may differ
        # in the last digits, so they are compared as values; every other
        # token (strings, booleans, null, integers like n and needed) must
        # match byte for byte.
        def tokens(text: str) -> list[tuple[str, Any]]:
            return json.loads(text, object_pairs_hook=list, parse_float=lambda f: ("float", f))

        js_pairs, py_pairs = tokens(js.stdout), tokens(py.stdout)
        assert [k for k, _ in js_pairs] == [k for k, _ in py_pairs]
        for (key, js_val), (_, py_val) in zip(js_pairs, py_pairs, strict=True):
            if isinstance(js_val, tuple) or isinstance(py_val, tuple):
                assert isinstance(js_val, tuple) and isinstance(py_val, tuple), key
                assert float(js_val[1]) == pytest.approx(float(py_val[1]), rel=EXACT, abs=EXACT), key
            else:
                assert js_val == py_val and type(js_val) is type(py_val), key
        assert len(js.stdout.splitlines()) == len(py.stdout.splitlines())
    else:
        assert js.stdout == py.stdout


def test_cli_rounding_tie_matches_to_fixed(scores):
    """shift = 0.125 exactly: both CLIs round ties away from zero, as
    Number.prototype.toFixed does, so the text is identical."""
    base = scores("b.json", [1, 2, 3, 4, 5, 6, 7, 8])
    var = scores("v.json", [1, 2, 3, 4, 5, 6, 7, 9])
    js, py = run_both(["compare", "--baseline", base, "--variant", var, "--target", "2"])
    assert "shift     +0.13" in py.stdout
    assert js.stdout == py.stdout


def test_cli_objects_pair_by_item_id(scores):
    base = scores("b.json", {"q1": 40, "q2": 55, "q3": 62, "q4": 70, "q5": 78})
    var = scores("v.json", {"q5": 80, "q1": 42, "q2": 57, "q3": 63, "q4": 72})
    js, py = run_both(["compare", "--baseline", base, "--variant", var, "--target", "1", "--json"])
    assert json.loads(py.stdout)["design"] == "paired"
    assert json.loads(py.stdout)["shift"] == pytest.approx(1.8)
    assert json.loads(js.stdout) == pytest.approx(json.loads(py.stdout))


def test_cli_without_gate_exits_zero_on_regression(scores):
    base = scores("b.json", WELCH_BASE[:8])
    var = scores("v.json", [77.3, 78.9, 76.1, 79.4, 77.8, 76.7, 79.1, 78.2])
    js, py = run_both(["compare", "--baseline", base, "--variant", var, "--target", "3"])
    assert js.returncode == py.returncode == 0
    assert "REGRESSED" in py.stdout


def test_cli_reads_stdin(scores):
    base = scores("b.json", WELCH_BASE[:4])
    js, py = run_both(
        ["compare", "--baseline", base, "--variant", "-", "--target", "3"],
        stdin=json.dumps([83.9, 81.2, 86.0, 84.7]),
    )
    assert js.returncode == py.returncode == 0
    assert js.stdout == py.stdout
    assert "Resolving it takes 16 runs per arm at the spread seen so far; you have 4." in py.stdout


@pytest.mark.parametrize(
    "args_fn",
    [
        lambda s: ["power", "--std", "2.06", "--target", "3"],
        lambda s: ["power", "--sd-diff", "8", "--target", "5"],
        lambda s: ["power", "--baseline", s("b.json", WELCH_BASE[:4]), "--target", "3"],
        lambda s: [
            "power",
            "--baseline",
            s("b.json", {"a": 40, "b": 55, "c": 62}),
            "--variant",
            s("v.json", {"a": 42, "b": 56, "c": 65}),
            "--target",
            "1",
        ],
        lambda s: ["power", "--std", "1", "--target", "0.5", "--json"],
    ],
    ids=["std", "sd-diff", "baseline file", "paired pilot", "json"],
)
def test_cli_power_parity(scores, args_fn):
    args = args_fn(scores)
    js, py = run_both(args)
    assert js.returncode == py.returncode == 0, (js.stderr, py.stderr)
    assert js.stdout == py.stdout
    if "--json" in args:
        assert json.loads(py.stdout)["needed"] == 64


@pytest.mark.parametrize(
    "args_fn,stdin",
    [
        (lambda s: ["compare", "--baseline", s("b.json", [1, 2, 3]), "--variant", s("v.json", [4, 5, 6])], None),
        (
            lambda s: [
                "compare",
                "--baseline",
                s("b.json", [1, 2, 3]),
                "--variant",
                s("v.json", [4, 5, 6]),
                "--target",
                "zero",
            ],
            None,
        ),
        (
            lambda s: ["compare", "--baseline", s("b.json", [1]), "--variant", s("v.json", [4, 5]), "--target", "1"],
            None,
        ),
        (
            lambda s: [
                "compare",
                "--baseline",
                s("b.json", [1, "x"]),
                "--variant",
                s("v.json", [4, 5]),
                "--target",
                "1",
            ],
            None,
        ),
        (
            lambda s: [
                "compare",
                "--baseline",
                s("b.json", {"a": 1, "b": 2}),
                "--variant",
                s("v.json", {"a": 1, "c": 2}),
                "--target",
                "1",
            ],
            None,
        ),
        (lambda s: ["compare", "--baseline", s("b.json", [1, 2]), "--variant", "-", "--target", "1"], "not json"),
        (
            lambda s: [
                "compare",
                "--baseline",
                s("b.json", [True, 80, 82]),
                "--variant",
                s("v.json", [4, 5]),
                "--target",
                "1",
            ],
            None,
        ),
        (
            lambda s: ["compare", "--baseline", "/nonexistent.json", "--variant", "/nonexistent.json", "--target", "1"],
            None,
        ),
        (lambda s: ["frobnicate"], None),
        (lambda s: ["power", "--target", "1"], None),
        (lambda s: ["compare", "--nope"], None),
    ],
    ids=[
        "missing target",
        "bad target",
        "too few",
        "non-numeric",
        "id mismatch",
        "bad stdin",
        "boolean score",
        "missing file",
        "unknown command",
        "power without spread",
        "unknown flag",
    ],
)
def test_cli_usage_errors_exit_2_in_both(scores, args_fn, stdin):
    js, py = run_both(args_fn(scores), stdin)
    assert js.returncode == py.returncode == 2, (js.stderr, py.stderr)
    assert js.stdout == py.stdout == ""
    assert js.stderr.startswith("eval-kit: ")
    assert py.stderr.startswith("eval-kit: ")
    # Messages are shared wording, except where each arg parser words its own.
    if args_fn(scores)[-1] != "--nope":
        assert js.stderr == py.stderr


def _raw(tmp_path: Path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


EDGE_CASES = {
    # Input files the parsers see differently unless the CLI pins it down.
    "invalid utf-8": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b"[1, 2, \xff]"),
        "--variant",
        _raw(t, "v.json", b"[1, 2]"),
        "--target",
        "1",
    ],
    "integer past float range": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b"[" + b"9" * 400 + b", 1, 2]"),
        "--variant",
        _raw(t, "v.json", b"[1, 2]"),
        "--target",
        "1",
    ],
    "huge scores": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b"[1e308, 1e308, 1e308]"),
        "--variant",
        _raw(t, "v.json", b"[1, 2, 3]"),
        "--target",
        "1",
    ],
    "NaN literal": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b"[NaN, 1, 2]"),
        "--variant",
        _raw(t, "v.json", b"[1, 2]"),
        "--target",
        "1",
    ],
    "prototype-named ids": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b'{"toString": 1, "a": 2, "b": 3}'),
        "--variant",
        _raw(t, "v.json", b'{"a": 2, "b": 4, "c": 5}'),
        "--target",
        "1",
    ],
    # Number grammar and formatting.
    "hex target": lambda t: ["power", "--std", "2", "--target", "0x10"],
    "underscore target": lambda t: ["power", "--std", "2", "--target", "1_000"],
    "full-width digit target": lambda t: ["power", "--std", "2", "--target", "\uff13"],
    "tiny target prints as JS does": lambda t: ["power", "--std", "0", "--target", "1e-8"],
    "small decimal target": lambda t: ["power", "--std", "0", "--target", "0.00001"],
    "exponent boundary": lambda t: ["power", "--std", "0", "--target", "0.000001"],
    "just under the boundary": lambda t: ["power", "--std", "0", "--target", "9.5e-7"],
    "huge target": lambda t: ["power", "--std", "0", "--target", "1e21"],
    "big integer target": lambda t: ["power", "--std", "0", "--target", "123456789012345680000"],
    # The sample-size search stays fast when n runs to the trillions.
    "search into the trillions": lambda t: ["power", "--std", "2.1", "--target", "0.00001"],
    "search past the exact limit": lambda t: ["power", "--std", "2.1", "--target", "1e-7"],
    "spread too large to size": lambda t: ["power", "--std", "1e155", "--target", "1e-155"],
    # Flags.
    "abbreviated flag": lambda t: ["power", "--std", "2", "--tar", "3"],
    "help": lambda t: ["--help"],
    "version": lambda t: ["--version"],
    "version wins over help": lambda t: ["--help", "--version"],
    "unknown flag beats help": lambda t: ["--nope", "--help"],
    "bundled short flags": lambda t: ["-hj"],
    "no command": lambda t: [],
    "flag missing its value": lambda t: ["power", "--target"],
    "value on a boolean flag": lambda t: ["power", "--json=1"],
    "extra positional": lambda t: ["compare", "extra"],
    "--std with --paired": lambda t: ["power", "--std", "2", "--target", "3", "--paired"],
    "negative target": lambda t: ["power", "--std", "2", "--target", "-1"],
    "negative std": lambda t: ["power", "--std", "-2", "--target", "1"],
    "negative fraction": lambda t: ["power", "--std", "2", "--target", "-.5"],
    # Command-line shapes where node:util parseArgs and argparse used to differ.
    "exponent-form negative value": lambda t: ["power", "--std", "2", "--target", "-1e5"],
    "trailing-dot negative value": lambda t: ["power", "--std", "-1.", "--target", "1"],
    "help after a negative value": lambda t: ["compare", "--std", "1e21", "-h", "--sd-diff", "-1e5"],
    "negative number as the command": lambda t: ["-5"],
    "-- ends the options": lambda t: ["compare", "--target", "1", "--", "x"],
    "positional after flags": lambda t: ["compare", "--json", "x"],
    "first error wins": lambda t: ["--foo", "--target"],
    "non-ASCII space in a number": lambda t: ["power", "--std", "2", "--target", "\u00a03"],
    # Inputs that used to hang, crash, or word things differently.
    "deep nesting": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b"[" * 100000),
        "--variant",
        _raw(t, "v.json", b"[1, 2]"),
        "--target",
        "1",
    ],
    "path through a file": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b"[1, 2]") + "/x",
        "--variant",
        _raw(t, "v.json", b"[1, 2]"),
        "--target",
        "1",
    ],
    "ids in JS key order": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b'{"b": 1, "a": 2, "2": 3, "1": 4}'),
        "--variant",
        _raw(t, "v.json", b'{"z": 1}'),
        "--target",
        "1",
    ],
    "lone surrogate id": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b'{"\\ud800": 1, "a": 2}'),
        "--variant",
        _raw(t, "v.json", b'{"a": 2, "b": 3}'),
        "--target",
        "1",
    ],
    "spreads near 1e-100": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b"[0, 1e-100, 3e-100]"),
        "--variant",
        _raw(t, "v.json", b"[0, 2e-100, 5e-100]"),
        "--target",
        "1e-100",
    ],
    "tiny target on compare": lambda t: [
        "compare",
        "--baseline",
        _raw(t, "b.json", b"[1, 2, 3]"),
        "--variant",
        _raw(t, "v.json", b"[2, 3, 5]"),
        "--target",
        "1e-300",
    ],
    "tiny target on power": lambda t: ["power", "--std", "1", "--target", "1e-300"],
    "near-constant baseline": lambda t: [
        "power",
        "--baseline",
        _raw(t, "b.json", b"[80, 80.00000000001, 80]"),
        "--target",
        "5",
    ],
    "near-constant paired pilot": lambda t: [
        "power",
        "--baseline",
        _raw(t, "b.json", b'{"a": 80, "b": 80.00000000001}'),
        "--variant",
        _raw(t, "v.json", b'{"a": 81, "b": 81}'),
        "--target",
        "5",
    ],
    # --json prints numbers as JS does: 3 not 3.0, 1e-7 not 1e-07.
    "power json": lambda t: ["power", "--std", "2.1", "--target", "3", "--json"],
    "power json, exponent target": lambda t: ["power", "--std", "2.1", "--target", "1e-7", "--json"],
    "power json, zero spread": lambda t: ["power", "--std", "0", "--target", "3", "--json"],
}


@pytest.mark.parametrize("args_fn", EDGE_CASES.values(), ids=EDGE_CASES.keys())
def test_cli_edge_cases_match_byte_for_byte(tmp_path, args_fn):
    """Everything a user can type or feed in, both CLIs answer the same:
    same exit code, same stdout, same stderr. No tracebacks."""
    js, py = run_both(args_fn(tmp_path))
    assert (js.returncode, js.stdout, js.stderr) == (py.returncode, py.stdout, py.stderr)
    assert js.returncode in (0, 2)
    assert "Traceback" not in py.stderr


def test_importing_the_js_cli_does_not_run_it():
    """main() is exported for reuse; importing the module must not parse
    process.argv or set an exit code."""
    probe = f"import({json.dumps(JS_CLI.as_uri())}).then((m) => console.log(typeof m.main, process.exitCode))"
    out = subprocess.run(["node", "-e", probe], capture_output=True, text=True, timeout=60)
    assert (out.returncode, out.stdout, out.stderr) == (0, "function undefined\n", "")


def test_cli_random_command_lines_match(tmp_path):
    """Seeded fuzz: random command lines built from every flag, value shape,
    and score file kind. Both CLIs must answer each one identically."""
    files = {
        "A": [80.1, 82.4, 79.0, 83.5, 81.2],
        "B": [83.9, 81.2, 86.0, 84.7, 85.1],
        "O1": {"q1": 0.8, "q2": 0.6, "q3": 0.9, "2": 0.7, "1": 0.5},
        "O2": {"q1": 0.85, "q2": 0.62, "q3": 0.95, "2": 0.71, "1": 0.55},
        "F": [5, 5, 5],
        "BAD": "nope",
        "ONE": [3],
    }
    paths = {
        k: _raw(tmp_path, f"{k}.json", (v if isinstance(v, str) else json.dumps(v)).encode()) for k, v in files.items()
    }
    vocab = [
        "compare",
        "power",
        "--baseline",
        "--variant",
        "--target",
        "--std",
        "--sd-diff",
        "--paired",
        "--lower-is-better",
        "--json",
        "--gate",
        "--help",
        "-h",
        "-hj",
        "--version",
        "--",
        "-",
        "-1",
        "-1e5",
        "3",
        "0",
        "1e-7",
        "0.5",
        "x",
        "--nope",
        "--target=2",
        "--json=1",
        "--tar",
        "-5",
        "1e21",
        ".5",
        "2.",
        "+3",
        *files,
    ]
    rng = random.Random(20260930)
    cases = []
    for _ in range(150):
        tokens = [rng.choice(vocab) for _ in range(rng.randint(1, 8))]
        if rng.random() < 0.6:
            tokens = [
                "compare",
                "--baseline",
                rng.choice(["A", "O1", "F", "ONE", "BAD"]),
                "--variant",
                rng.choice(["B", "O2", "F", "A"]),
                "--target",
                rng.choice(["3", "0.05", "-1", "x", "1e-7"]),
                *tokens[:3],
            ]
        cases.append([paths.get(t, t) for t in tokens])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda argv: (argv, *run_both(argv, "[1, 2, 3]")), cases))
    mismatched = [
        argv
        for argv, js, py in results
        if (js.returncode, js.stdout, js.stderr) != (py.returncode, py.stdout, py.stderr)
    ]
    assert not mismatched, mismatched[:3]
    assert all("Traceback" not in py.stderr for _, _, py in results)
