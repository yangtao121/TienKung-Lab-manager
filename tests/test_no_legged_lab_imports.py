from __future__ import annotations

import re
from pathlib import Path


IMPORT_PATTERN = re.compile(r"^\s*(from|import)\s+legged_lab(?:\.|\b)")


def test_no_legged_lab_imports_in_package_code() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "tienkung_manager_lab"

    violations: list[str] = []
    for py_file in package_root.rglob("*.py"):
        for line_no, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), start=1):
            if IMPORT_PATTERN.search(line):
                violations.append(f"{py_file}:{line_no}:{line.strip()}")

    assert not violations, "Found forbidden legged_lab imports:\n" + "\n".join(violations)
