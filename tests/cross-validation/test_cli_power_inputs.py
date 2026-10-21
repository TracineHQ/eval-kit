"""CLI inputs that power and compare would misread, refused the same way by
both CLIs (js/cli.mjs, python -m eval_kit): same stderr, exit 2, no stdout.
Plus the pilot size on the power line and in power --json.

Run (from repo root):
    cd python && .venv/bin/pytest ../tests/cross-validation/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_cross_validation import EXACT  # noqa: E402
from test_verdict_and_cli import _raw, run_both  # noqa: E402

BASE = [80.1, 82.4, 79.0, 83.5]
VARI = [83.9, 81.2, 86.0, 84.7]
FLAT = [0.8] * 5
NEAR = [0.84, 0.79, 0.86, 0.83, 0.81]
# A synthetic 6-item paired pilot: sd of differences 2.47, 50 items for a 1-point shift.
ITEMS_BASE = {"a": 2.0, "b": 3.0, "c": 2.5, "d": 4.0, "e": 3.5, "f": 3.0}
ITEMS_VARI = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 5, "e": 4.5, "f": 7}

HINT = "Run eval-kit --help for usage.\n"


def _file(tmp_path: Path, name: str, data: Any) -> str:
    return _raw(tmp_path, name, json.dumps(data).encode())


def assert_refused(args: list[str], stderr: str, stdin: str | None = None) -> None:
    js, py = run_both(args, stdin)
    assert (js.returncode, js.stdout, js.stderr) == (2, "", stderr), args
    assert (py.returncode, py.stdout, py.stderr) == (2, "", stderr), args


def assert_same_run(args: list[str], stdin: str | None = None) -> str:
    js, py = run_both(args, stdin)
    assert js.returncode == py.returncode == 0, (js.stderr, py.stderr)
    assert js.stdout == py.stdout
    return py.stdout


def assert_same_json(args: list[str]) -> dict[str, Any]:
    """Same keys in the same order and the same tokens; computed floats to 1e-9."""
    js, py = run_both([*args, "--json"])
    assert js.returncode == py.returncode == 0, (js.stderr, py.stderr)

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
    return dict(json.loads(py.stdout))


# ---------------------------------------------------------------------------
# power: --gate, one spread source, one stdin, a lone object pilot
# ---------------------------------------------------------------------------


def test_power_refuses_gate():
    assert_refused(
        ["power", "--std", "2", "--target", "3", "--gate"],
        "eval-kit: --gate is for compare; power has no verdict to gate on\n",
    )


SOURCES = {
    "--std": lambda t: ["--std", "2"],
    "--sd-diff": lambda t: ["--sd-diff", "0.5"],
    "--baseline": lambda t: ["--baseline", _file(t, "b.json", BASE)],
    "--variant": lambda t: ["--variant", _file(t, "v.json", VARI)],
    "both pilot files": lambda t: ["--baseline", _file(t, "b.json", BASE), "--variant", _file(t, "v.json", VARI)],
}
SOURCE_PAIRS = [
    ("--std", "--sd-diff"),
    ("--std", "--baseline"),
    ("--std", "--variant"),
    ("--std", "both pilot files"),
    ("--sd-diff", "--baseline"),
    ("--sd-diff", "--variant"),
    ("--sd-diff", "both pilot files"),
]


@pytest.mark.parametrize("first,second", SOURCE_PAIRS, ids=[f"{a} and {b}" for a, b in SOURCE_PAIRS])
def test_power_wants_one_spread_source(tmp_path, first, second):
    # 0.2.0 used the first of --std, --sd-diff, pilot files and ignored the rest.
    message = "eval-kit: give one spread source: --std, --sd-diff, or pilot files (--baseline, optionally --variant)\n"
    for order in [(first, second), (second, first)]:
        args = ["power", *SOURCES[order[0]](tmp_path), *SOURCES[order[1]](tmp_path), "--target", "1"]
        assert_refused(args, message)
    all_three = ["power", "--std", "2", "--sd-diff", "0.5", *SOURCES["both pilot files"](tmp_path), "--target", "1"]
    assert_refused(all_three, message)


def test_both_arms_on_stdin_is_refused_by_power_and_compare():
    # 0.2.0 power read stdin twice and blamed the JSON.
    message = "eval-kit: only one of --baseline and --variant can be -\n"
    for command in ("power", "compare"):
        assert_refused([command, "--baseline", "-", "--variant", "-", "--target", "1"], message, stdin="[1, 2, 3]")


def test_power_refuses_a_lone_baseline_keyed_by_item_id(tmp_path):
    # 0.2.0 sized per-item scores as independent runs, the design they never get.
    tail = " is keyed by item id, so it sizes paired items: add --variant from the same items, or give --sd-diff\n"
    path = _file(tmp_path, "b.json", ITEMS_BASE)
    assert_refused(["power", "--baseline", path, "--target", "1"], f"eval-kit: {path}{tail}")
    assert_refused(
        ["power", "--baseline", "-", "--target", "1"], f"eval-kit: stdin{tail}", stdin=json.dumps(ITEMS_BASE)
    )
    # Either fix it names works.
    paired = assert_same_json(
        ["power", "--baseline", path, "--variant", _file(tmp_path, "v.json", ITEMS_VARI), "--target", "1"]
    )
    assert (paired["design"], paired["needed"]) == ("paired", 50)
    assert_same_run(["power", "--sd-diff", "2.47", "--target", "1"])


def test_power_checks_run_in_order(tmp_path):
    """--target first, then --gate, then one spread source, then one stdin."""
    assert_refused(
        ["power", "--gate", "--std", "1", "--sd-diff", "1"],
        "eval-kit: --target is required: the smallest shift worth acting on\n",
    )
    assert_refused(
        ["power", "--target", "1", "--gate", "--std", "1", "--sd-diff", "1"],
        "eval-kit: --gate is for compare; power has no verdict to gate on\n",
    )
    assert_refused(
        ["power", "--target", "1", "--std", "1", "--baseline", "-", "--variant", "-"],
        "eval-kit: give one spread source: --std, --sd-diff, or pilot files (--baseline, optionally --variant)\n",
    )


# ---------------------------------------------------------------------------
# compare: --std and --sd-diff are for power
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--std", "--sd-diff"])
def test_compare_refuses_power_spread_flags(tmp_path, flag):
    message = "eval-kit: compare reads the spread from the scores; --std and --sd-diff are for power\n"
    files = ["--baseline", _file(tmp_path, "b.json", BASE), "--variant", _file(tmp_path, "v.json", VARI)]
    assert_refused(["compare", *files, "--target", "3", flag, "2"], message)
    # It is the first check: it comes before the missing arms and target.
    assert_refused(["compare", flag, "2"], message)


# ---------------------------------------------------------------------------
# Repeated value flags
# ---------------------------------------------------------------------------

VALUE_FLAGS = ["--baseline", "--variant", "--target", "--std", "--sd-diff"]
SPELLINGS = {
    "--flag v --flag v": lambda f, v: [f, v, f, v],
    "--flag=v --flag v": lambda f, v: [f"{f}={v}", f, v],
    "--flag v --flag=v": lambda f, v: [f, v, f"{f}={v}"],
}


@pytest.mark.parametrize("spelling", SPELLINGS, ids=SPELLINGS.keys())
@pytest.mark.parametrize("flag", VALUE_FLAGS)
@pytest.mark.parametrize("command", ["compare", "power"])
def test_a_repeated_value_flag_is_refused(tmp_path, command, flag, spelling):
    # 0.2.0 used the last value: compare --target 100 --target 3 tested at 3.
    value = _file(tmp_path, "b.json", BASE) if flag in ("--baseline", "--variant") else "3"
    assert_refused([command, *SPELLINGS[spelling](flag, value)], f"eval-kit: {flag} given more than once\n{HINT}")


def test_the_flag_repeated_first_is_named(tmp_path):
    args = ["power", "--std", "1", "--target", "1", "--target", "2", "--std", "2"]
    assert_refused(args, f"eval-kit: --target given more than once\n{HINT}")


def test_parse_errors_come_before_a_repeated_flag():
    assert_refused(["power", "--target", "1", "--target", "2", "--nope"], f"eval-kit: unknown option: --nope\n{HINT}")
    assert_refused(["power", "--target", "1", "--target", "2", "--std"], f"eval-kit: --std needs a value\n{HINT}")


def test_a_repeated_boolean_flag_still_runs():
    out = assert_same_run(["power", "--std", "1", "--target", "0.5", "--json", "--json"])
    assert json.loads(out)["needed"] == 64


# ---------------------------------------------------------------------------
# --lower-is-better has no effect on power
# ---------------------------------------------------------------------------

POWER_SOURCES = {
    "--std": lambda t: ["--std", "2.06"],
    "--sd-diff": lambda t: ["--sd-diff", "2.47"],
    "lone baseline": lambda t: ["--baseline", _file(t, "b.json", BASE)],
    "two arrays": lambda t: ["--baseline", _file(t, "b.json", BASE), "--variant", _file(t, "v.json", VARI)],
    "paired pilot": lambda t: [
        "--baseline",
        _file(t, "b.json", ITEMS_BASE),
        "--variant",
        _file(t, "v.json", ITEMS_VARI),
    ],
}


@pytest.mark.parametrize("source", POWER_SOURCES.values(), ids=POWER_SOURCES.keys())
def test_lower_is_better_does_not_change_power(tmp_path, source):
    args = ["power", *source(tmp_path), "--target", "1"]
    plain = assert_same_run(args)
    assert assert_same_run([*args, "--lower-is-better"]) == plain
    assert assert_same_json([*args, "--lower-is-better"]) == assert_same_json(args)


def test_help_says_lower_is_better_has_no_effect_on_power():
    out = assert_same_run(["--help"])
    assert "  --lower-is-better   Lower scores are better (latency, error rate); no effect on power\n" in out


# ---------------------------------------------------------------------------
# The pilot size power sized from
# ---------------------------------------------------------------------------


def test_the_power_line_names_the_pilot_size(tmp_path):
    base, vari = _file(tmp_path, "b.json", BASE), _file(tmp_path, "v.json", VARI)
    items_base, items_vari = _file(tmp_path, "ib.json", ITEMS_BASE), _file(tmp_path, "iv.json", ITEMS_VARI)
    cases = [
        (
            ["--baseline", base, "--target", "3"],
            "9 runs per arm to detect a 3-point shift (baseline std 2.06 from 4 runs, 80% power, alpha 0.05)\n",
        ),
        (
            ["--baseline", base, "--variant", vari, "--target", "3"],
            "9 runs per arm to detect a 3-point shift (pooled std 2.05 from 4 and 4 runs, 80% power, alpha 0.05)\n",
        ),
        (
            ["--baseline", items_base, "--variant", items_vari, "--target", "1"],
            "50 items to detect a 1-point shift (sd of differences 2.47 from 6 items, 80% power, alpha 0.05)\n",
        ),
        (
            ["--baseline", base, "--variant", vari, "--paired", "--target", "3"],
            "13 items to detect a 3-point shift (sd of differences 3.52 from 4 items, 80% power, alpha 0.05)\n",
        ),
        # --std and --sd-diff have no pilot: unchanged.
        (
            ["--std", "2.06", "--target", "3"],
            "9 runs per arm to detect a 3-point shift (baseline std 2.06, 80% power, alpha 0.05)\n",
        ),
        (
            ["--sd-diff", "2.47", "--target", "1"],
            "50 items to detect a 1-point shift (sd of differences 2.47, 80% power, alpha 0.05)\n",
        ),
    ]
    for args, line in cases:
        assert assert_same_run(["power", *args]) == line


def test_unequal_arms_name_both_sizes(tmp_path):
    args = [
        "power",
        "--baseline",
        _file(tmp_path, "b.json", BASE),
        "--variant",
        _file(tmp_path, "v.json", [*VARI, 85.1]),
    ]
    out = assert_same_run([*args, "--target", "3"])
    assert " from 4 and 5 runs, 80% power" in out, out


# label -> (args, design, nBaseline, nVariant, constantArm)
JSON_CASES = {
    "--std": (lambda t: ["--std", "2.06"], "welch", None, None, None),
    "--sd-diff": (lambda t: ["--sd-diff", "2.47"], "paired", None, None, None),
    "lone baseline": (lambda t: ["--baseline", _file(t, "b.json", BASE)], "welch", 4, None, None),
    "lone constant baseline": (lambda t: ["--baseline", _file(t, "b.json", FLAT)], "welch", 5, None, None),
    "two arrays": (POWER_SOURCES["two arrays"], "welch", 4, 4, None),
    "two arrays, unequal n": (
        lambda t: ["--baseline", _file(t, "b.json", BASE), "--variant", _file(t, "v.json", [*VARI, 85.1])],
        "welch",
        4,
        5,
        None,
    ),
    "constant baseline": (
        lambda t: ["--baseline", _file(t, "b.json", FLAT), "--variant", _file(t, "v.json", NEAR)],
        "welch",
        5,
        5,
        "baseline",
    ),
    "constant variant": (
        lambda t: ["--baseline", _file(t, "b.json", NEAR), "--variant", _file(t, "v.json", FLAT)],
        "welch",
        5,
        5,
        "variant",
    ),
    "both arms constant": (
        lambda t: ["--baseline", _file(t, "b.json", FLAT), "--variant", _file(t, "v.json", FLAT)],
        "welch",
        5,
        5,
        None,
    ),
    "paired pilot": (POWER_SOURCES["paired pilot"], "paired", 6, 6, None),
    "arrays with --paired": (
        lambda t: ["--baseline", _file(t, "b.json", BASE), "--variant", _file(t, "v.json", VARI), "--paired"],
        "paired",
        4,
        4,
        None,
    ),
}


@pytest.mark.parametrize("args_fn,design,n_base,n_vari,constant", JSON_CASES.values(), ids=JSON_CASES.keys())
def test_power_json_gives_the_pilot_size_and_constant_arm(tmp_path, args_fn, design, n_base, n_vari, constant):
    target = "0.03" if constant else "1"
    out = assert_same_json(["power", *args_fn(tmp_path), "--target", target])
    assert list(out) == ["design", "target", "spread", "needed", "nBaseline", "nVariant", "constantArm"]
    assert (out["design"], out["nBaseline"], out["nVariant"], out["constantArm"]) == (design, n_base, n_vari, constant)
