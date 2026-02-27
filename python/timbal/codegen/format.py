import subprocess
from pathlib import Path

import libcst as cst

RUFF_FORMAT_ARGS = [
    "ruff",
    "format",
    "--isolated",
    "--line-length",
    "79",
    "--config",
    "indent-width=4",
    "--config",
    "format.indent-style='space'",
    "--config",
    "format.quote-style='double'",
    "--config",
    "format.skip-magic-trailing-comma=false",
]

RUFF_FIX_ARGS = [
    "ruff",
    "check",
    "--isolated",
    "--fix",
    "--select",
    "F401,F811",
]


class _NameCounter(cst.CSTTransformer):
    """Count every occurrence of each Name node in the module."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def visit_Name(self, node: cst.Name) -> None:
        self.counts[node.value] = self.counts.get(node.value, 0) + 1


def _remove_unused_code(code: str, protected: set[str]) -> str:
    """Iteratively remove unused top-level variables and functions.

    A definition is considered unused when its name appears only once in the
    entire module (at its own definition site). The loop repeats until no
    more removals are possible, so cascading dead code is handled.
    """
    while True:
        tree = cst.parse_module(code)

        # Collect top-level definition names.
        var_defs: set[str] = set()
        func_defs: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, cst.FunctionDef):
                func_defs.add(stmt.name.value)
            elif isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.Assign):
                        for target in item.targets:
                            if isinstance(target.target, cst.Name):
                                var_defs.add(target.target.value)

        candidates = (var_defs | func_defs) - protected
        if not candidates:
            return code

        # Count every Name occurrence in the tree.
        counter = _NameCounter()
        tree.visit(counter)

        # A name that appears only once is only at its definition site.
        unused = {name for name in candidates if counter.counts.get(name, 0) <= 1}
        if not unused:
            return code

        unused_vars = unused & var_defs
        unused_funcs = unused & func_defs

        class _Remover(cst.CSTTransformer):
            def leave_FunctionDef(
                self,
                original_node: cst.FunctionDef,
                updated_node: cst.FunctionDef,
            ) -> cst.FunctionDef | cst.RemovalSentinel:
                if updated_node.name.value in unused_funcs:
                    return cst.RemovalSentinel.REMOVE
                return updated_node

            def leave_SimpleStatementLine(
                self,
                original_node: cst.SimpleStatementLine,
                updated_node: cst.SimpleStatementLine,
            ) -> cst.SimpleStatementLine | cst.RemovalSentinel:
                for item in updated_node.body:
                    if isinstance(item, cst.Assign):
                        for target in item.targets:
                            if isinstance(target.target, cst.Name) and target.target.value in unused_vars:
                                return cst.RemovalSentinel.REMOVE
                return updated_node

        code = tree.visit(_Remover()).code


def format_code(code: str, source_path: Path, *, entry_point: str | None = None) -> str:
    # 1. Remove unused top-level variables and functions (iterative).
    protected = {entry_point} if entry_point else set()
    code = _remove_unused_code(code, protected)

    # 2. Fix unused imports (F401) and redefinitions (F811).
    result = subprocess.run(
        [*RUFF_FIX_ARGS, "--stdin-filename", str(source_path), "-"],
        input=code,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"ruff check --fix failed:\n{result.stderr}")
    code = result.stdout or code

    # 3. Format.
    result = subprocess.run(
        [*RUFF_FORMAT_ARGS, "--stdin-filename", str(source_path), "-"],
        input=code,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ruff format failed:\n{result.stderr}")
    return result.stdout
