"""Deterministic checks on the Claude Code skill and its eval suite. No model calls.

`claude plugin eval` runs the skill against a live model, which costs money and
varies run to run. This suite pins everything around it that can be pinned:

1. **Answer keys are true.** Each eval case's answer-key.json holds the data in
   its prompt and the answer the skill should reach. The CLI must actually
   produce that answer, from exactly the numbers the prompt shows. The keys
   come from the CLI, so their independent anchor is tests/cross-validation/,
   which holds the math under the CLI to scipy.
2. **Graders accept the right answer.** Regex graders must match the CLI's own
   output, and tool_used graders must match the command SKILL.md tells Claude
   to run. A grader that cannot pass is a broken eval, not a strict one.
3. **SKILL.md follows the published limits** (Agent Skills best practices and
   the Claude Code skills reference) and only mentions flags and functions
   that exist.
4. **README examples are real output.**

Run (from repo root):
    cd python && pytest ../tests/skill/ -v
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
EVALS = REPO / "evals"
JS_CLI = REPO / "js" / "cli.mjs"
SKILL = REPO / "SKILL.md"
CASES = sorted(p.parent.name for p in EVALS.glob("*/prompt.md"))

# prompt.md and grader frontmatter keys from the plugin eval reference.
PROMPT_KEYS = {
    "schema_version", "name", "description", "tags", "plugins", "runs", "expected_outcome",
    "model", "max_turns", "timeout_seconds", "allowed_tools", "append_system_prompt", "env",
}  # fmt: skip
GRADER_TYPES = {"regex", "tool_used", "tool_order", "file_exists", "llm", "baseline"}


def split_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """Flat `key: value` frontmatter (all this repo uses) and the body."""
    text = path.read_text()
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    assert match, f"{path} has no frontmatter"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields, match.group(2)


def yaml_scalar(raw: str) -> str:
    """Unquote a YAML scalar the way this repo writes them."""
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1].replace("''", "'")
    if raw.startswith('"') and raw.endswith('"'):
        return json.loads(raw)
    return raw


def js_regex(pattern: str, text: str, flags: str = "") -> bool:
    """Grader regexes are JavaScript regexes; test them with JavaScript."""
    script = (
        "const [p,f,t]=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "process.stdout.write(String(new RegExp(p,f).test(t)))"
    )
    out = subprocess.run(
        ["node", "-e", script], input=json.dumps([pattern, flags, text]), capture_output=True, text=True, check=True
    )
    return out.stdout == "true"


def run_cli(cmd: list[str], key: dict[str, Any], tmp: Path, *extra: str) -> subprocess.CompletedProcess:
    args = [key["command"], "--target", str(key["target"]), *key.get("flags", []), *extra]
    for arm in ("baseline", "variant"):
        if arm in key:
            path = tmp / f"{arm}.json"
            path.write_text(json.dumps(key[arm]))
            args += [f"--{arm}", str(path)]
    return subprocess.run(cmd + args, capture_output=True, text=True, cwd=REPO)


def js_cli(key: dict[str, Any], tmp: Path, *extra: str) -> subprocess.CompletedProcess:
    return run_cli(["node", str(JS_CLI)], key, tmp, *extra)


def numbers_in(text: str) -> Counter:
    return Counter(float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\.?\d|\w)", text))


def answer_key(case: str) -> dict[str, Any]:
    return json.loads((EVALS / case / "answer-key.json").read_text())


def graders(case: str) -> dict[str, tuple[dict[str, str], str]]:
    return {p.stem: split_frontmatter(p) for p in sorted((EVALS / case / "graders").glob("*.md"))}


# ---------------------------------------------------------------------------
# Eval suite shape
# ---------------------------------------------------------------------------


def test_suite_has_at_least_three_cases():
    """Agent Skills best practices: at least three evaluations."""
    assert len(CASES) >= 3


@pytest.mark.parametrize("case", CASES)
def test_case_files_follow_the_eval_reference(case):
    fields, body = split_frontmatter(EVALS / case / "prompt.md")
    assert set(fields) <= PROMPT_KEYS, f"unknown prompt.md keys: {set(fields) - PROMPT_KEYS}"
    assert body.strip(), "empty prompt"
    assert (EVALS / case / "answer-key.json").exists()
    found = graders(case)
    assert found, "a case needs at least one grader"
    for name, (meta, _) in found.items():
        assert meta.get("type") in GRADER_TYPES, f"{name}: bad type {meta.get('type')}"


# Grader keys this suite uses, per type. claude plugin validate does not check
# grader files, so a misspelled key (patern, input_mach) would silently stop
# grading. Add a key here only after checking it in the plugin eval reference.
GRADER_KEYS = {
    "llm": {"type", "arm"},
    "regex": {"type", "arm", "pattern", "flags"},
    "tool_used": {"type", "arm", "tool", "input_match", "min", "max"},
}


@pytest.mark.parametrize("case", CASES)
def test_grader_keys_and_values_are_known(case):
    for name, (meta, body) in graders(case).items():
        kind = meta["type"]
        assert kind in GRADER_KEYS, f"{name}: no key list for type {kind}"
        assert set(meta) <= GRADER_KEYS[kind], f"{name}: unknown keys {set(meta) - GRADER_KEYS[kind]}"
        assert meta.get("arm", "with-only") in {"with-only", "both"}, name
        for bound in ("min", "max"):
            assert re.fullmatch(r"\d+", meta.get(bound, "0")), f"{name}: {bound} must be a whole number"
        if kind == "llm":
            assert "PASS" in body and "FAIL" in body, f"{name}: rubric needs PASS and FAIL conditions"


NUMERIC = [c for c in CASES if "command" in answer_key(c)]
NEGATIVE = [c for c in CASES if answer_key(c).get("negative")]


def skill_graders(case: str) -> list[dict[str, str]]:
    return [meta for meta, _ in graders(case).values() if meta["type"] == "tool_used" and meta.get("tool") == "Skill"]


@pytest.mark.parametrize("case", CASES)
def test_every_case_checks_the_skill_and_the_result(case):
    """One grader on the steps, one on the result (plugin eval guidance).
    Negative cases assert the skill stayed out, scored in both arms."""
    types = {meta["type"] for meta, _ in graders(case).values()}
    assert {"llm", "regex"} & types, "no grader on the result"
    fired = skill_graders(case)
    assert fired, "no Skill grader"
    if case in NEGATIVE:
        assert all(m.get("min") == "0" and m.get("max") == "0" and m.get("arm") == "both" for m in fired)
    else:
        assert all("max" not in m for m in fired)


@pytest.mark.parametrize("case", CASES)
def test_prompt_grants_the_tools_the_case_needs(case):
    """The Skill grader cannot pass without Skill, and the CLI needs input
    files, which only Write can create (the Bash grant is the CLI alone)."""
    fields, _ = split_frontmatter(EVALS / case / "prompt.md")
    tools = {t.strip() for t in fields["allowed_tools"].strip("[]").split(",")}
    assert "Skill" in tools
    if case not in NEGATIVE:
        assert "Write" in tools


def test_suite_covers_the_skill_rules():
    """Each rule in SKILL.md has a case: underpowered, pairing, sizing,
    zero spread, missing target, lower-is-better, out of scope, and a
    prompt the skill must ignore."""
    expected = {
        "underpowered-improvement", "per-item-pairing", "size-before-running", "zero-spread-harness",
        "asks-for-target", "lower-is-better", "pass-fail-out-of-scope", "ignores-unrelated-request",
    }  # fmt: skip
    assert expected <= set(CASES)


# ---------------------------------------------------------------------------
# Answer keys are true
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [c for c in CASES if "baseline" in answer_key(c) or "target" in answer_key(c)])
def test_prompt_shows_exactly_the_answer_key_data(case):
    key = answer_key(case)
    _, body = split_frontmatter(EVALS / case / "prompt.md")
    in_prompt = numbers_in(body)
    wanted: Counter = Counter()
    for arm in ("baseline", "variant"):
        data = key.get(arm)
        if data is None:
            continue
        values = data.values() if isinstance(data, dict) else data
        wanted.update(float(v) for v in values)
        if isinstance(data, dict):
            for item in data:
                assert re.search(rf"\b{re.escape(item)}\b", body), f"item {item} missing from prompt"
    if "target" in key:
        wanted[float(key["target"])] += 1
    missing = wanted - in_prompt
    assert not missing, f"answer-key numbers not in the prompt: {dict(missing)}"


@pytest.mark.parametrize("case", NUMERIC)
def test_cli_produces_the_answer_key(case, tmp_path):
    key = answer_key(case)
    js = js_cli(key, tmp_path, "--json")
    assert js.returncode == 0, js.stderr
    result = json.loads(js.stdout)
    for field, value in key["expect"].items():
        assert result[field] == value, f"{field}: CLI says {result[field]}, key says {value}"

    py = run_cli([sys.executable, "-m", "eval_kit"], key, tmp_path, "--json")
    assert py.returncode == 0, py.stderr
    for field, value in key["expect"].items():
        assert json.loads(py.stdout)[field] == value


def test_pairing_case_discriminates(tmp_path):
    """The per-item case only tests the skill if the wrong design gives a
    different answer: as independent arrays the same data must not read as
    improved."""
    key = answer_key("per-item-pairing")
    unpaired = {**key, "baseline": list(key["baseline"].values()), "variant": list(key["variant"].values())}
    result = json.loads(js_cli(unpaired, tmp_path, "--json").stdout)
    for field, value in key["unpaired_expect"].items():
        assert result[field] == value


def test_lower_is_better_case_discriminates(tmp_path):
    """Without --lower-is-better the same data reads as a regression."""
    key = answer_key("lower-is-better")
    result = json.loads(js_cli({**key, "flags": []}, tmp_path, "--json").stdout)
    for field, value in key["without_flags_expect"].items():
        assert result[field] == value


@pytest.mark.parametrize("case", [c for c in NUMERIC if "needed" in answer_key(c)["expect"]])
def test_llm_rubric_numbers_match_the_answer_key(case):
    """Rubrics that quote the answer must quote the CLI's answer."""
    needed = answer_key(case)["expect"]["needed"]
    _, prompt = split_frontmatter(EVALS / case / "prompt.md")
    allowed = {needed, 2 * needed} | set(numbers_in(prompt))
    for name, (meta, body) in graders(case).items():
        if meta["type"] == "llm":
            stray = set(numbers_in(body)) - allowed
            assert not stray, f"{name} quotes numbers the answer key does not: {stray}"


# ---------------------------------------------------------------------------
# Graders accept the right answer and reject the wrong ones
# ---------------------------------------------------------------------------


def regex_graders(case: str) -> dict[str, dict[str, str]]:
    found = {name: meta for name, (meta, _) in graders(case).items() if meta["type"] == "regex"}
    for name, meta in found.items():
        assert meta.get("target", "last_message") == "last_message", name
    return found


def grader_matches(meta: dict[str, str], text: str) -> bool:
    return js_regex(yaml_scalar(meta["pattern"]), text, meta.get("flags", ""))


@pytest.mark.parametrize("case", NUMERIC)
def test_regex_graders_match_the_cli_answer(case, tmp_path):
    """A response that relays the CLI's verdict must pass every regex grader."""
    text = js_cli(answer_key(case), tmp_path).stdout
    for name, meta in regex_graders(case).items():
        assert grader_matches(meta, text), f"{name} does not match:\n{text}"


@pytest.mark.parametrize("case", CASES)
def test_regex_graders_on_paraphrases(case):
    """Graders read Claude's paraphrase, not the CLI's wording. Every regex
    grader has phrases it must accept and phrases it must reject."""
    phrases = answer_key(case).get("phrases", {})
    for name, meta in regex_graders(case).items():
        assert name in phrases, f"{name} has no accept/reject phrases in answer-key.json"
        for text in phrases[name]["accept"]:
            assert grader_matches(meta, text), f"{name} rejects a right answer: {text!r}"
        for text in phrases[name]["reject"]:
            assert not grader_matches(meta, text), f"{name} accepts a wrong answer: {text!r}"


@pytest.mark.parametrize("case", [c for c in CASES if c not in NEGATIVE])
def test_regex_graders_do_not_match_the_prompt(case):
    """A reply that only restates the prompt must not pass."""
    _, body = split_frontmatter(EVALS / case / "prompt.md")
    for name, meta in regex_graders(case).items():
        assert not grader_matches(meta, body), f"{name} matches the prompt itself"


def bash_input(command: str, target: Any, *flags: str) -> str:
    """The Bash input Claude sends when it follows SKILL.md."""
    line = f"node /plugins/eval-kit/js/cli.mjs {command} --baseline b.json --variant v.json --target {target}"
    return json.dumps({"command": " ".join([line, *flags])})


@pytest.mark.parametrize("case", CASES)
def test_tool_used_graders_match_what_the_skill_runs(case):
    skill_name = split_frontmatter(SKILL)[0]["name"]
    key = answer_key(case)
    for name, (meta, _) in graders(case).items():
        if meta["type"] != "tool_used":
            continue
        pattern = yaml_scalar(meta.get("input_match", ""))
        if meta["tool"] == "Skill":
            assert js_regex(pattern, json.dumps({"skill": skill_name})), name
            assert js_regex(pattern, json.dumps({"skill": f"{skill_name}:{skill_name}"})), name
        elif meta["tool"] == "Bash":
            command, target, flags = key["command"], key["target"], key.get("flags", [])
            other = "power" if command == "compare" else "compare"
            assert js_regex(pattern, bash_input(command, target, *flags)), f"{name} misses the right command"
            assert js_regex(pattern, bash_input(command, target, *flags).replace("--target ", "--target=")), name
            assert not js_regex(pattern, bash_input(other, target, *flags)), f"{name} accepts {other}"
            assert not js_regex(pattern, bash_input(command, target * 10 + 1, *flags)), f"{name} ignores the target"
            if flags:
                assert not js_regex(pattern, bash_input(command, target)), f"{name} ignores {flags}"
            assert meta.get("arm") == "with-only", f"{name}: the CLI path exists only with the plugin"


# ---------------------------------------------------------------------------
# SKILL.md follows the published limits
# ---------------------------------------------------------------------------


def test_skill_frontmatter_limits():
    fields, body = split_frontmatter(SKILL)
    plugin = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())
    name, description = fields["name"], fields["description"]
    assert name == plugin["name"]
    assert re.fullmatch(r"[a-z0-9-]{1,64}", name)
    assert "claude" not in name and "anthropic" not in name
    assert 0 < len(description) <= 1024
    assert not re.match(r"(I|You|We)\b", description), "descriptions are third person"
    assert len(description) + len(fields.get("when_to_use", "")) <= 1536, "listing truncates at 1,536"
    assert len(body.splitlines()) < 500


def test_allowed_tools_cover_every_command_the_skill_shows():
    fields, body = split_frontmatter(SKILL)
    grant = re.fullmatch(r"Bash\((.*) \*\)", fields["allowed-tools"])
    assert grant, "expected one Bash(<prefix> *) grant"
    prefix = grant.group(1)
    commands = re.findall(r"^node \$\{CLAUDE_PLUGIN_ROOT\}/js/cli\.mjs .*$", body, re.M)
    assert commands
    for command in commands:
        assert command.startswith(prefix + " "), command


def test_skill_mentions_only_flags_the_cli_has():
    _, body = split_frontmatter(SKILL)
    help_text = subprocess.run(["node", str(JS_CLI), "--help"], capture_output=True, text=True).stdout
    for flag in set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", body)):
        assert flag in help_text, f"SKILL.md mentions {flag}, the CLI has no such flag"


def test_skill_names_only_exported_functions():
    _, body = split_frontmatter(SKILL)
    section = body.split("## Using the library directly", 1)[1]
    library = (REPO / "js" / "lib" / "stats.mjs").read_text()
    exported = set(re.findall(r"^export (?:function|const) (\w+)", library, re.M))
    named = set(re.findall(r"`(\w+)`", section)) - {"diff"}
    named = {n for n in named if n[0].islower() and n not in {"verdict().shift"}}
    assert named <= exported, f"not exported: {named - exported}"


# ---------------------------------------------------------------------------
# README examples are real output
# ---------------------------------------------------------------------------


def readme_block_after(marker: str) -> str:
    readme = (REPO / "README.md").read_text()
    after = readme.split(marker, 1)[1]
    return re.search(r"```\n(.*?)```", after, re.S).group(1)


def test_readme_hero_is_the_example_output():
    out = subprocess.run(
        ["node", str(REPO / "js" / "examples" / "verdicts.mjs")], capture_output=True, text=True, check=True
    ).stdout
    assert readme_block_after("chasing a 3-point shift:") == out


def test_readme_cli_example_is_real_output(tmp_path):
    readme = (REPO / "README.md").read_text()
    data = re.search(r"`base.json` as `(\[.*?\])` and `new.json` as `(\[.*?\])`", readme)
    assert data, "README must show the data behind its CLI example"
    key = {
        "command": "compare",
        "baseline": json.loads(data.group(1)),
        "variant": json.loads(data.group(2)),
        "target": 3,
    }
    out = js_cli(key, tmp_path).stdout
    assert readme_block_after("--variant new.json --target 3\n```") == out


def test_every_version_string_agrees():
    """One release, one version: the plugin manifest, both packages, both
    CLIs' --version, and the changelog's latest entry."""
    versions = {
        "plugin.json": json.loads((REPO / ".claude-plugin" / "plugin.json").read_text())["version"],
        "package.json": json.loads((REPO / "js" / "package.json").read_text())["version"],
        "pyproject.toml": re.search(r'^version = "(.+)"', (REPO / "python" / "pyproject.toml").read_text(), re.M)[1],
        "__init__.py": re.search(
            r'__version__ = "(.+)"', (REPO / "python" / "src" / "eval_kit" / "__init__.py").read_text()
        )[1],
        "js --version": subprocess.run(
            ["node", str(JS_CLI), "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "CHANGELOG.md": re.search(r"^## \[(\d[^\]]*)\]", (REPO / "CHANGELOG.md").read_text(), re.M)[1],
    }
    assert len(set(versions.values())) == 1, versions
