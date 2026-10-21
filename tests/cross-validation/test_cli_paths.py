"""CLI paths the other suites never run: usage errors with their exact wording,
lines a verdict prints only on some data, and the two ways the CLI is installed
(npm's bin symlink, the eval-kit console script). Both CLIs must agree byte for
byte on each.

Run (from repo root):
    cd python && .venv/bin/pytest ../tests/cross-validation/ -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_verdict_and_cli import JS_CLI, REPO, _raw, run_both  # noqa: E402

# The README's CLI example.
README_BASE = [80.1, 82.4, 79.0, 83.5]
README_NEW = [83.9, 81.2, 86.0, 84.7]

Case = Callable[[Path], tuple[list[str], str]]


def _json(tmp: Path, name: str, data: Any) -> str:
    return _raw(tmp, name, json.dumps(data).encode())


def _compare(tmp: Path, baseline: str) -> list[str]:
    """compare this baseline against a valid variant."""
    return ["compare", "--baseline", baseline, "--variant", _json(tmp, "v.json", [1, 2, 3]), "--target", "1"]


def _bad_baseline(content: bytes, problem: str) -> Case:
    """compare a baseline file holding content; stderr names the file, then the problem."""

    def build(tmp: Path) -> tuple[list[str], str]:
        path = _raw(tmp, "b.json", content)
        return _compare(tmp, path), f"{path} {problem}"

    return build


def _unreadable(tmp: Path) -> tuple[list[str], str]:
    path = _raw(tmp, "locked.json", b"[1, 2, 3]")
    os.chmod(path, 0)
    return _compare(tmp, path), f"cannot read {path}: Permission denied"


SHAPE = 'must be a JSON array of numbers or an object of {"item id": number}'

USAGE_ERRORS: dict[str, Case] = {
    "compare without --variant": lambda t: (
        ["compare", "--baseline", _json(t, "b.json", [1, 2, 3]), "--target", "1"],
        "compare needs --baseline and --variant",
    ),
    "both arms from stdin": lambda t: (
        ["compare", "--baseline", "-", "--variant", "-", "--target", "1"],
        "only one of --baseline and --variant can be -",
    ),
    "paired pilot of one item": lambda t: (
        [
            "power",
            "--baseline",
            _json(t, "b.json", {"a": 1}),
            "--variant",
            _json(t, "v.json", {"a": 2}),
            "--target",
            "1",
        ],
        "the paired pilot needs at least 2 items",
    ),
    "--paired power from a baseline alone": lambda t: (
        ["power", "--baseline", _json(t, "b.json", [1, 2, 3]), "--paired", "--target", "1"],
        "--paired power needs --sd-diff, or --baseline and --variant from a pilot",
    ),
    "power baseline of one score": lambda t: (
        ["power", "--baseline", _json(t, "b.json", [3]), "--target", "1"],
        "the baseline needs at least 2 scores",
    ),
    "a directory": lambda t: (_compare(t, str(t)), f"cannot read {t}: Is a directory"),
    "unreadable file": _unreadable,
    "null": _bad_baseline(b"null", SHAPE),
    "bare number": _bad_baseline(b"3", SHAPE),
    "bare string": _bad_baseline(b'"x"', SHAPE),
    "object against an array": lambda t: (
        _compare(t, _json(t, "b.json", {})),
        "both files must be arrays, or both objects keyed by item id",
    ),
    "empty array": lambda t: (_compare(t, _json(t, "b.json", [])), "each arm needs at least 2 scores"),
    "empty file": _bad_baseline(b"", "is not valid JSON"),
    "a score of exactly 1e15": _bad_baseline(b"[1e15, 1, 2]", "has a score of 1e15 or more; rescale the scores"),
}


@pytest.mark.parametrize("case", USAGE_ERRORS.values(), ids=USAGE_ERRORS.keys())
def test_cli_usage_error_wording(tmp_path, case):
    if case is _unreadable and os.geteuid() == 0:
        pytest.skip("root can read a chmod 000 file")
    args, message = case(tmp_path)
    js, py = run_both(args, stdin="")
    assert (js.returncode, js.stdout, js.stderr) == (py.returncode, py.stdout, py.stderr)
    assert (js.returncode, js.stdout, js.stderr) == (2, "", f"eval-kit: {message}\n")


def test_cli_paired_cant_tell_counts_items(tmp_path):
    js, py = run_both(
        [
            "compare",
            "--baseline",
            _json(tmp_path, "b.json", {"a": 40, "b": 55, "c": 62, "d": 70}),
            "--variant",
            _json(tmp_path, "v.json", {"a": 43, "b": 54, "c": 66, "d": 69}),
            "--target",
            "1",
        ]
    )
    assert (js.returncode, js.stdout, js.stderr) == (py.returncode, py.stdout, py.stderr)
    assert js.returncode == 0
    assert "Resolving it takes 109 items at the spread seen so far; you have 4." in js.stdout.splitlines()


def test_cli_estimate_equal_to_target_has_no_below_target_note(tmp_path):
    """The note is for an estimate under the target. A shift of exactly the target is not under it."""
    js, py = run_both(
        [
            "compare",
            "--baseline",
            _json(tmp_path, "b.json", [10, 11, 12, 13, 14]),
            "--variant",
            _json(tmp_path, "v.json", [13, 14, 15, 16, 17]),
            "--target",
            "3",
        ]
    )
    assert (js.returncode, js.stdout, js.stderr) == (py.returncode, py.stdout, py.stderr)
    lines = js.stdout.splitlines()
    assert "IMPROVED. The variant scores higher by 3.00, p 0.017." in lines
    assert not any(line.startswith("The estimate is below") for line in lines)


def _readme_compare(tmp: Path) -> list[str]:
    base = _json(tmp, "base.json", README_BASE)
    new = _json(tmp, "new.json", README_NEW)
    return ["compare", "--baseline", base, "--variant", new, "--target", "3"]


def _run(cmd: list[str]) -> tuple[int, str, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=60)
    return r.returncode, r.stdout, r.stderr


def test_cli_runs_through_a_symlink(tmp_path):
    """npm installs the bin as a symlink. The CLI runs only when it is the program,
    so it has to see through the link, or it exits 0 and prints nothing."""
    link = tmp_path / "eval-kit"
    os.symlink(JS_CLI, link)
    for args in (["--version"], _readme_compare(tmp_path)):
        direct = _run(["node", str(JS_CLI), *args])
        assert direct[1], direct
        assert _run(["node", str(link), *args]) == direct


def test_console_script_matches_python_m(tmp_path):
    script = Path(sys.executable).with_name("eval-kit")
    if not script.exists():
        pytest.skip(f"no eval-kit console script next to {sys.executable}")
    for args in (["--version"], _readme_compare(tmp_path)):
        module = _run([sys.executable, "-m", "eval_kit", *args])
        assert module[1], module
        assert _run([str(script), *args]) == module
