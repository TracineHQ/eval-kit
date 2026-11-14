"""README quick-start snippets run, and what their comments claim is what they return.

test_skill_contract.py pins the README's hero table and CLI block. This covers the rest of
the README's code: the JS and Python quick starts, the API table's return fields, and the
"no dependencies" claim for the JS library.

Run (from repo root):
    cd python && pytest ../tests/skill/ -v
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import eval_kit

REPO = Path(__file__).resolve().parents[2]
README = (REPO / "README.md").read_text()


def fenced(lang: str) -> str:
    match = re.search(rf"```{lang}\n(.*?)```", README, re.S)
    assert match, f"README has no {lang} block"
    return match.group(1)


def run_js(prelude: str, body: str, epilogue: str) -> Any:
    script = prelude + "\n" + body + "\n" + epilogue
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=REPO, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def js_quick_start() -> dict[str, Any]:
    """Run the README's JS block from the repo root. Records what each console.log printed,
    and the Welch p on the paired arrays that a comment quotes but the block does not print."""
    code = fenced("javascript").replace("console.log(", "__log(")
    imports, _, body = code.partition("\n\n")
    log = (
        "const __out = [];\n"
        "const __log = (v) => __out.push(typeof v === 'object' ? { keys: Object.keys(v), v } : { v });"
    )
    epilogue = (
        "const welchItemsP = welchTTest(baselineItems, variantItems).p;\n"
        "process.stdout.write(JSON.stringify({ logged: __out, welchItemsP }));"
    )
    return run_js(imports + "\n" + log, body, epilogue)


def camel(name: str) -> str:
    return re.sub(r"_(\w)", lambda m: m.group(1).upper(), name)


def test_js_quick_start_return_shapes_match_its_comments():
    logged = [entry for entry in js_quick_start()["logged"] if "keys" in entry]
    claimed = re.findall(r"^// \{ (.+?) \}$", fenced("javascript"), re.M)
    assert [entry["keys"] for entry in logged] == [c.split(", ") for c in claimed]


def test_js_quick_start_numbers_match_its_comments():
    run = js_quick_start()
    welch, _, paired = run["logged"]
    assert welch["v"]["diff"] == 7  # "+7 here"
    assert f"{paired['v']['p']:.7f}" == "0.0000055"  # "p = 0.0000055 here"
    assert f"{run['welchItemsP']:.2f}" == "0.78"  # "welchTTest on the same arrays gives p = 0.78"


def test_python_quick_start_runs():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(fenced("python"), {})
    first, second = out.getvalue().splitlines()
    assert first.startswith("WelchResult(t=")  # "WelchResult(t=..., ...)"
    assert int(second) >= 2  # "runs per group needed"


def api_table() -> dict[str, list[str]]:
    """The API parity table's Returns column, keyed by Purpose, for the rows that return an object."""
    section = README.split("## API parity", 1)[1].split("\n## ", 1)[0]
    rows = [line.split("|") for line in section.splitlines() if line.startswith("|")][2:]
    return {cells[1].strip(): cells[4].strip().strip("`{}").split(", ") for cells in rows}


def test_api_table_return_fields_are_real():
    # The verdict row ends in "..." on purpose, so it is not checked here.
    rows = api_table()
    js_keys = run_js(
        "import { stats, welchTTest, pairedTTest } from './js/lib/stats.mjs';",
        "const a = [80, 72, 91, 65, 77]; const b = [78, 70, 90, 62, 76];",
        "process.stdout.write(JSON.stringify("
        "[stats(a), welchTTest(a, b), pairedTTest(a, b)].map((r) => Object.keys(r))));",
    )
    for (purpose, result_type), keys in zip(
        [
            ("Descriptive stats + 95% CI", eval_kit.Stats),
            ("Welch's t-test + Glass's delta (`diff` = a minus b)", eval_kit.WelchResult),
            ("Paired t-test + Cohen's dz (`diff` = a minus b)", eval_kit.PairedResult),
        ],
        js_keys,
        strict=True,
    ):
        py_fields = [camel(f.name) for f in dataclasses.fields(result_type)]
        assert rows[purpose] == py_fields == keys, purpose


def test_js_library_imports_nothing():
    """The README says the JS library has no dependencies and can be copied on its own."""
    source = (REPO / "js" / "lib" / "stats.mjs").read_text()
    assert not re.search(r"^\s*import\b|^\s*export\b.*\bfrom\b|\bimport\s*\(|\brequire\s*\(", source, re.M)
    package = json.loads((REPO / "js" / "package.json").read_text())
    for field in ("dependencies", "peerDependencies", "optionalDependencies"):
        assert not package.get(field), field
