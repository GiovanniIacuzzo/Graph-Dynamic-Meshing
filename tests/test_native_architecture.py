"""Static architecture guard: no parallel Python mesh/FEM numerical backend."""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_geometry_and_numerics_packages_are_removed() -> None:
    assert not (ROOT / "python" / "geometry").exists()
    assert not (ROOT / "python" / "numerics").exists()


def test_project_python_sources_do_not_import_removed_backends() -> None:
    source_roots = [ROOT / "python", ROOT / "training", ROOT / "experiments", ROOT / "tests"]
    forbidden = ("geometry", "numerics")
    violations: list[str] = []
    for source_root in source_roots:
        for file_path in source_root.rglob("*.py"):
            if any(part in {"__pycache__", "build"} for part in file_path.parts):
                continue
            try:
                tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            except SyntaxError as exc:
                violations.append(f"{file_path}: SyntaxError: {exc}")
                continue
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = next((alias.name for alias in node.names if alias.name), None)
                if module and any(module == name or module.startswith(name + ".") for name in forbidden):
                    violations.append(f"{file_path}:{node.lineno}: import {module}")
    assert not violations, "Riferimenti residui ai vecchi backend Python:\n" + "\n".join(violations)


def test_python_package_discovery_excludes_removed_backends() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"geometry*"' not in pyproject
    assert '"numerics*"' not in pyproject
    assert '"gdm*"' in pyproject
