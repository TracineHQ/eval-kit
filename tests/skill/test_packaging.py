"""What the npm package and the Python wheel and sdist carry. No build step.

Apache-2.0 asks every redistribution to include LICENSE (section 4(a)) and NOTICE
(section 4(d)). npm and hatchling pack from the package directory, so js/ and
python/ hold byte copies of the root files. The package READMEs become
the npm and PyPI pages, where relative links resolve to nothing.

Run (from repo root):
    cd python && pytest ../tests/skill/ -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ["python", "js"]


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
def test_license_copies_match_the_root(package, name):
    copy = REPO / package / name
    assert not copy.is_symlink(), f"{package}/{name} must be a regular file, not a symlink"
    assert copy.read_bytes() == (REPO / name).read_bytes(), f"{package}/{name} differs from the root {name}"


@pytest.mark.parametrize("package", PACKAGES)
def test_package_readmes_use_absolute_links(package):
    readme = (REPO / package / "README.md").read_text()
    targets = re.findall(r"\]\(([^)\s]+)", readme)
    assert targets, f"{package}/README.md has no links to check"
    relative = [t for t in targets if t.startswith(("../", "./"))]
    assert not relative, f"{package}/README.md links break on the registry page: {relative}"


def test_npm_package_lists_its_files():
    files = json.loads((REPO / "js" / "package.json").read_text())["files"]
    assert {"lib", "cli.mjs", "README.md", "LICENSE", "NOTICE"} <= set(files)


def test_python_package_is_typed():
    assert (REPO / "python" / "src" / "eval_kit" / "py.typed").is_file()
