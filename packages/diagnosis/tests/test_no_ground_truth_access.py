"""Static leak guard over the code (not the prose) of ``iqs_diagnosis``.

- Only the ``evaluation`` package may contain a string literal naming the
  ground-truth directory.
- No module outside ``evaluation`` may import from ``iqs_diagnosis.evaluation``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import iqs_diagnosis

SRC = Path(iqs_diagnosis.__file__).resolve().parent


def _modules() -> list[tuple[str, ast.Module]]:
    return [
        (str(p.relative_to(SRC)), ast.parse(p.read_text(encoding="utf-8")))
        for p in sorted(SRC.rglob("*.py"))
    ]


def _docstrings(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _outside_evaluation(rel: str) -> bool:
    return not rel.startswith("evaluation/")


def test_only_evaluation_names_ground_truth() -> None:
    offenders = []
    for rel, tree in _modules():
        docs = _docstrings(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docs
                and "ground_truth" in node.value
                and _outside_evaluation(rel)
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_agent_path_does_not_import_evaluation() -> None:
    offenders = []
    for rel, tree in _modules():
        if not _outside_evaluation(rel):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            if any(n.startswith("iqs_diagnosis.evaluation") for n in names):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_guard_sees_the_evaluation_modules() -> None:
    """The guard is not vacuous: evaluation code does name the directory."""
    rels = [rel for rel, _ in _modules()]
    assert "evaluation/labels.py" in rels
