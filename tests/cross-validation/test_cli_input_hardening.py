"""CLI input hardening: repeated item ids, printed values past 1e19, integers
too long to parse, ids in messages, and UTF-8 output whatever the locale.
Every case runs both CLIs and expects the same exit code, stdout and stderr,
no traceback, and the exact result written here.

Run (from repo root):
    cd python && .venv/bin/pytest ../tests/cross-validation/ -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_verdict_and_cli import JS_CLI, REPO, _raw, run_both  # noqa: E402

SHAPE = 'must be a JSON array of numbers or an object of {"item id": number}'
REPEAT_HINT = "give each item one score (average repeated calls first)"


def both(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run both CLIs, require identical results, return the Python one."""
    js, py = run_both(args, stdin)
    assert (js.returncode, js.stdout, js.stderr) == (py.returncode, py.stdout, py.stderr)
    assert "Traceback" not in py.stderr
    return py


def refused(r: subprocess.CompletedProcess, message: str) -> None:
    assert (r.returncode, r.stdout, r.stderr) == (2, "", f"eval-kit: {message}\n")


def compare(base: str, var: str, *extra: str) -> list[str]:
    return ["compare", "--baseline", base, "--variant", var, "--target", "1", *extra]


# ---------------------------------------------------------------------------
# Repeated item ids
# ---------------------------------------------------------------------------

R1_BASE = b'{"r1":1.9,"r1":2.4,"r1":2.1,"r2":3.0,"r2":3.3,"r3":1.2}'
R1_VARI = b'{"r1":1.6,"r2":2.9,"r3":1.0}'


def test_cli_refuses_a_repeated_item_id(tmp_path):
    """Regression: 0.2.0 kept the last score for a repeated id, so this
    baseline compared as n=3, mean 2.20, and 3 of its 6 scores vanished."""
    base, var = _raw(tmp_path, "b.json", R1_BASE), _raw(tmp_path, "v.json", R1_VARI)
    for extra in [["--lower-is-better"], ["--lower-is-better", "--gate"], ["--json"]]:
        refused(both(compare(base, var, *extra)), f'{base} repeats item id "r1"; {REPEAT_HINT}')
    refused(
        both(compare("-", var, "--lower-is-better"), stdin=R1_BASE.decode()),
        f'stdin repeats item id "r1"; {REPEAT_HINT}',
    )
    # The id named is the first whose second occurrence comes first in the file.
    order = _raw(tmp_path, "order.json", b'{"a": 1, "b": 2, "b": 3, "a": 4}')
    refused(both(compare(order, var)), f'{order} repeats item id "b"; {REPEAT_HINT}')


def test_cli_repeated_id_written_with_escapes(tmp_path):
    """Ids compare after JSON decoding: "\\u0061" is "a". The message prints
    the decoded id, quoted."""
    var = _raw(tmp_path, "v.json", b'{"a": 1, "b": 2}')
    cases = [
        (b'{"a": 1, "\\u0061": 2, "b": 3}', '"a"'),
        (b'{"q\\"1": 1, "q\\u00221": 2, "b": 3}', '"q\\"1"'),
        (b'{"a\\nb": 1, "a\\u000ab": 2, "b": 3}', '"a\\u000ab"'),
        (b'{"\\ud83d\\ude00": 1, "\xf0\x9f\x98\x80": 2, "b": 3}', '"\U0001f600"'),
        (b'{ "a" : 1 ,\n\t"b"\r\n:\n2, "a"\t: 3 }', '"a"'),
    ]
    for i, (text, quoted) in enumerate(cases):
        base = _raw(tmp_path, f"b{i}.json", text)
        refused(both(compare(base, var)), f"{base} repeats item id {quoted}; {REPEAT_HINT}")


def test_cli_repeated_id_in_the_variant(tmp_path):
    base, var = _raw(tmp_path, "b.json", R1_VARI), _raw(tmp_path, "v.json", R1_BASE)
    refused(both(compare(base, var, "--lower-is-better")), f'{var} repeats item id "r1"; {REPEAT_HINT}')
    # Files are read baseline first, so a repeat in both names the baseline.
    base = _raw(tmp_path, "b2.json", b'{"x": 1, "x": 2}')
    refused(both(compare(base, var)), f'{base} repeats item id "x"; {REPEAT_HINT}')


def test_cli_repeated_id_in_a_power_pilot(tmp_path):
    base, var = _raw(tmp_path, "b.json", R1_BASE), _raw(tmp_path, "v.json", R1_VARI)
    message = f'{base} repeats item id "r1"; {REPEAT_HINT}'
    refused(both(["power", "--baseline", base, "--variant", var, "--target", "1"]), message)
    refused(both(["power", "--baseline", var, "--variant", base, "--target", "1"]), message)
    refused(both(["power", "--baseline", base, "--target", "1"]), message)


def test_cli_nested_repeat_keeps_the_shape_error(tmp_path):
    """Only the top-level object holds item ids. A repeat deeper down is a
    file of the wrong shape, as it was before."""
    var = _raw(tmp_path, "v.json", b'{"a": 1, "b": 2}')
    nested = [b'{"a": {"x": 1, "x": 2}, "b": 1}', b'[{"x": 1, "x": 2}, 3]', b'{"a": [{"x": 1, "x": 2}]}']
    for i, text in enumerate(nested):
        base = _raw(tmp_path, f"b{i}.json", text)
        refused(both(compare(base, var)), f"{base} {SHAPE}")


def test_cli_checks_each_file_in_order(tmp_path):
    """Per file: read, nesting depth, JSON validity, repeated ids, then shape
    and the 1e15 limit."""
    var = _raw(tmp_path, "v.json", b'{"a": 1, "b": 2}')
    cases = [
        (b'{"a": 1, "a": 2,', "is not valid JSON"),
        (b'{"a": 1, "a": 2} {"a": 3}', "is not valid JSON"),
        (b'{"a": 1, "a": ' + b"[" * 100 + b"]" * 100 + b"}", "is nested too deeply to be a score file"),
        (b'{"a": 1, "a": "x"}', f'repeats item id "a"; {REPEAT_HINT}'),
        (b'{"a": 1e16, "a": 2}', f'repeats item id "a"; {REPEAT_HINT}'),
        (b'{"a": "x", "b": 2}', SHAPE),
        (b'{"a": 1e16, "b": 2}', "has a score of 1e15 or more; rescale the scores"),
        # A string value that looks like a key is not an id.
        (b'{"a": "a", "b": ":"}', SHAPE),
    ]
    for i, (text, message) in enumerate(cases):
        base = _raw(tmp_path, f"b{i}.json", text)
        refused(both(compare(base, var)), f"{base} {message}")


# ---------------------------------------------------------------------------
# Number formatting and parsing
# ---------------------------------------------------------------------------


def test_cli_prints_huge_values_without_a_traceback(tmp_path):
    """Regression: the Python CLI raised decimal.InvalidOperation (exit 1,
    which --gate reads as regressed) once a printed value reached 1e19. From
    1e21 Number#toFixed prints exponent form, and so does the Python CLI."""
    power = "runs per arm to detect a {t}-point shift (baseline std {s}, 80% power, alpha 0.05)\n"
    cases = [
        ("1e19", "17 " + power.format(t="10000000000000000000", s="10000000000000000000.00")),
        ("9.99e20", "17 " + power.format(t="999000000000000000000", s="999000000000000000000.00")),
        ("1e21", "17 " + power.format(t="1e+21", s="1e+21")),
        ("1e25", "17 " + power.format(t="1e+25", s="1e+25")),
    ]
    for value, expected in cases:
        r = both(["power", "--std", value, "--target", value])
        assert (r.returncode, r.stdout, r.stderr) == (0, expected, "")

    # glassD past 1e21: a tiny baseline spread under a huge shift.
    base = _raw(tmp_path, "b.json", b"[0, 5e-7, 1e-6]")
    var = _raw(tmp_path, "v.json", b"[990000000000000, 990000000000000.1, 990000000000000.2]")
    r = both(compare(base, var))
    assert r.returncode == 0
    assert "  glassD    1.98e+21\n" in r.stdout, r.stdout


def test_cli_long_integers_get_the_shape_error(tmp_path):
    """Regression: past CPython's 4300-digit int limit the Python CLI said
    "is not valid JSON" where the JS CLI read Infinity and gave the shape
    error. Both now read a long integer as a float."""
    var = _raw(tmp_path, "v.json", b"[1, 2]")
    for digits in [4301, 5000]:
        for sign in [b"", b"-"]:
            base = _raw(tmp_path, f"b{digits}{sign.decode()}.json", b"[" + sign + b"9" * digits + b", 1, 2]")
            refused(both(compare(base, var)), f"{base} {SHAPE}")
    obj = _raw(tmp_path, "obj.json", b'{"a": ' + b"1" * 5000 + b', "b": 2}')
    refused(both(compare(obj, _raw(tmp_path, "v2.json", b'{"a": 1, "b": 2}'))), f"{obj} {SHAPE}")


# ---------------------------------------------------------------------------
# Ids in messages
# ---------------------------------------------------------------------------


def test_cli_quotes_ids_in_messages(tmp_path):
    """Regression: ids printed raw, so an id with a newline forged a second
    line of output and an ESC sequence reached the terminal. Ids are now
    quoted, with backslash, quote and C0/C1 control characters escaped."""
    base = _raw(
        tmp_path,
        "b.json",
        b'{"a\\nmissing from variant: zzz": 1, "\\u001b[2J": 2, "say \\"hi\\" C:\\\\x": 3,'
        b' "\\u007f\\u0085\\u009f\\u00a0": 4, "caf\xc3\xa9 \\t": 5, "\\ud800": 6, "ok": 7}',
    )
    var = _raw(tmp_path, "v.json", b'{"ok": 1, "c\\rd": 2}')
    r = both(compare(base, var))
    refused(
        r,
        'missing from variant: "a\\u000amissing from variant: zzz", "\\u001b[2J", "say \\"hi\\" C:\\\\x",'
        ' "\\u007f\\u0085\\u009f\u00a0", "caf\u00e9 \\u0009", ...; missing from baseline: "c\\u000dd"',
    )
    assert "\x1b" not in r.stderr and r.stderr.count("\n") == 1

    lone = _raw(tmp_path, "lone.json", b'{"\\ud800": 1, "ok": 2}')
    refused(both(compare(lone, var)), 'missing from variant: "\ufffd"; missing from baseline: "c\\u000dd"')


def test_cli_truncates_long_id_lists(tmp_path):
    var = _raw(tmp_path, "v.json", b'{"x": 1, "y": 2}')
    seven = _raw(tmp_path, "seven.json", b'{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}')
    refused(
        both(compare(seven, var)),
        'missing from variant: "a", "b", "c", "d", "e", ...; missing from baseline: "x", "y"',
    )
    five = _raw(tmp_path, "five.json", b'{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}')
    refused(
        both(compare(var, five)),
        'missing from variant: "x", "y"; missing from baseline: "a", "b", "c", "d", "e"',
    )


def test_cli_writes_utf8_whatever_the_locale(tmp_path):
    """Regression: the Python CLI wrote through the stream encoding, so under
    PYTHONIOENCODING=cp1252 an emoji id printed as \\U0001f600 where Node
    writes UTF-8. Both now write UTF-8 bytes with \\n line ends."""
    base = _raw(tmp_path, "b.json", '{"\U0001f600": 1, "caf\u00e9": 2}'.encode())
    var = _raw(tmp_path, "v.json", '{"caf\u00e9": 1, "\U0001f389": 2}'.encode())
    runs = _raw(tmp_path, "runs.json", b"[80.1, 82.4, 79.0, 83.5]")
    mismatch = 'eval-kit: missing from variant: "\U0001f600"; missing from baseline: "\U0001f389"\n'.encode()
    for encoding in ["cp1252", "ascii"]:
        env = {**os.environ, "PYTHONIOENCODING": encoding}
        for args in [compare(base, var), compare(runs, runs), ["--help"]]:
            js = subprocess.run(["node", str(JS_CLI), *args], capture_output=True, cwd=REPO, env=env, timeout=60)
            py = subprocess.run(
                [sys.executable, "-m", "eval_kit", *args], capture_output=True, cwd=REPO, env=env, timeout=60
            )
            assert (js.returncode, js.stdout, js.stderr) == (py.returncode, py.stdout, py.stderr), (encoding, args)
            assert b"\r\n" not in py.stdout + py.stderr
            assert py.stdout or py.stderr
            if args == compare(base, var):
                assert (py.returncode, py.stdout, py.stderr) == (2, b"", mismatch)
