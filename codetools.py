"""
AST and similarity helpers for the child validation gate (step 6).

Everything here is intentionally cheap and dependency-free: parsing, a
docstring-stripped AST dump for exact no-op detection, a node-type sequence for
structural similarity that is robust to renames and reformatting, and entrypoint
discovery. The gate composes these in validation.py.
"""

import ast
import difflib
from typing import List, Optional, Set


def parse(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]
    return tree


def normalized_dump(code: str) -> Optional[str]:
    """Docstring-stripped AST dump. Identical dumps means the two programs differ
    only in comments, whitespace, or docstrings, i.e. a cosmetic / no-op edit."""
    tree = parse(code)
    if tree is None:
        return None
    tree = _strip_docstrings(tree)
    return ast.dump(tree, annotate_fields=False)


def node_type_sequence(code: str) -> Optional[List[str]]:
    tree = parse(code)
    if tree is None:
        return None
    return [type(n).__name__ for n in ast.walk(tree)]


def defined_functions(code: str) -> Set[str]:
    tree = parse(code)
    if tree is None:
        return set()
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def is_cosmetic_change(parent_code: str, child_code: str) -> bool:
    a = normalized_dump(parent_code)
    b = normalized_dump(child_code)
    if a is None or b is None:
        return (parent_code or "").strip() == (child_code or "").strip()
    return a == b


def similarity(code_a: str, code_b: str) -> float:
    """Structural similarity in [0, 1]. Uses the AST node-type sequence when both
    parse (rename / reformat robust); falls back to a raw-text ratio otherwise."""
    sa = node_type_sequence(code_a)
    sb = node_type_sequence(code_b)
    if sa is not None and sb is not None:
        return difflib.SequenceMatcher(None, sa, sb).ratio()
    return difflib.SequenceMatcher(None, code_a or "", code_b or "").ratio()
