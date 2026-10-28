"""Offline docs checks that go beyond navigation and import existence.

Blockers are returned as errors. Warnings are reported and do not fail CI.
Snippet execution swaps string model ids for TestModel, blocks sockets, and
never needs a provider key.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

MODEL_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w./-]+$")
TIMBAL_CMD = re.compile(r"^`?\$?\s*timbal\s+([a-z][a-z0-9-]*)", re.M)
TIMBAL_LINE = re.compile(r"^`?\$?\s*timbal\s+[a-z][a-z0-9-]*")
PY_MODULE = re.compile(r"python(?:3)?\s+-m\s+(timbal(?:\.[A-Za-z0-9_]+)*)")
HTTP_CALL = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(?:https?://[^/\s]+)?(/[A-Za-z0-9_{}*./~-]*)")
_ZIG_ACTION = re.compile(r'std\.mem\.eql\(u8, action, "([a-z0-9-]+)"\)')
_ZIG_FLAG = re.compile(r"--[a-z][a-z0-9-]*")
INTERNAL_PHRASES = (
    "Composer integration",
    "generate_costs_sql.py",
    "this change must",
)
HOUSE_MODELS = {
    "openai": "openai/gpt-5-mini",
    "anthropic": "anthropic/claude-sonnet-4-6",
    "google": "google/gemini-3.5-flash",
}


def deprecated_model_ids(catalog: dict) -> set[str]:
    banned: set[str] = set()
    for model in catalog.get("models", []):
        notes = str(model.get("notes") or "")
        lowered = notes.lower()
        if "deprecated" in lowered or "access restricted" in lowered:
            banned.add(model["id"])
    return banned


def _until_statement_end(text: str, start: int) -> str:
    """Slice from ``start`` through the line that ends the Zig string statement."""
    lines: list[str] = []
    for line in text[start:].splitlines():
        lines.append(line)
        if line.rstrip().endswith('";') or line.rstrip().endswith("';"):
            break
    return "\n".join(lines)


def _usage_blob(source: str) -> str:
    """Help text only, so tests and comments that mention fake flags are ignored."""
    parts: list[str] = []
    at = source.find("\nconst usage =")
    if at != -1:
        rest = source[at:]
        end = rest.find("\n;")
        if end != -1:
            parts.append(rest[:end])
    at = source.find("\nfn printUsage()")
    if at != -1:
        lines: list[str] = []
        for line in source[at:].splitlines():
            lines.append(line)
            if line == "}":
                break
        parts.append("\n".join(lines))
    return "\n".join(parts)


def cli_surface(cli_dir: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Commands from ``cli/main.zig`` dispatch, and flags from each command's help text.

    The command file is ``cli/commands/<name>.zig`` with hyphens turned into underscores.
    Global flags come from ``global_options_help`` in ``cli/utils.zig`` and apply to every command.
    """
    if not (cli_dir / "main.zig").is_file():
        return set(), {}
    main = (cli_dir / "main.zig").read_text(encoding="utf-8", errors="replace")
    commands = {name for name in _ZIG_ACTION.findall(main) if not name.startswith("-")}
    global_flags: set[str] = set()
    utils = cli_dir / "utils.zig"
    if utils.is_file():
        text = utils.read_text(encoding="utf-8", errors="replace")
        at = text.find("global_options_help")
        global_flags = set(_ZIG_FLAG.findall(_until_statement_end(text, at) if at != -1 else ""))
    flags: dict[str, set[str]] = {}
    for command in commands:
        source = cli_dir / "commands" / f"{command.replace('-', '_')}.zig"
        own: set[str] = set()
        if source.is_file():
            blob = _usage_blob(source.read_text(encoding="utf-8", errors="replace"))
            own = set()
            for line in blob.splitlines():
                if "git " in line:
                    continue
                own.update(_ZIG_FLAG.findall(line))
        flags[command] = own | global_flags
    return commands, flags


def server_paths(server_dir: Path) -> set[str]:
    paths: set[str] = set()
    if not server_dir.is_dir():
        return paths
    prefix_re = re.compile(r'APIRouter\(\s*prefix\s*=\s*"([^"]+)"')
    route_re = re.compile(r'@\w+\.(?:get|post|put|patch|delete|websocket)\(\s*"([^"]+)"')
    for path in server_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        prefix = ""
        found = prefix_re.search(text)
        if found:
            prefix = found.group(1).rstrip("/")
        for route in route_re.findall(text):
            if route == "/":
                full = prefix or "/"
            else:
                full = f"{prefix}{route if route.startswith('/') else '/' + route}"
            paths.add(full)
    return paths


def openapi_paths(schema: dict) -> set[str]:
    return set(schema.get("paths", {}))


def _path_known(path: str, known: set[str]) -> bool:
    if path in known:
        return True
    # Trailing slash and template names differ (`{id}` vs `{run_id}`).
    normalized = re.sub(r"\{[^}]+\}", "{}", path.rstrip("/"))
    if any(re.sub(r"\{[^}]+\}", "{}", item.rstrip("/")) == normalized for item in known):
        return True
    for item in known:
        if "{*" not in item:
            continue
        prefix = item.split("{*", 1)[0].rstrip("/")
        if path.rstrip("/").startswith(prefix + "/") or path.rstrip("/") == prefix:
            return True
    return False


class _SwapModels(ast.NodeTransformer):
    def visit_keyword(self, node: ast.keyword):
        self.generic_visit(node)
        if node.arg in {"model", "sampling_model"} and _is_model(node.value):
            node.value = _test_model_call()
        return node

    def visit_Dict(self, node: ast.Dict):
        self.generic_visit(node)
        for index, (key, value) in enumerate(zip(node.keys, node.values, strict=True)):
            if isinstance(key, ast.Constant) and key.value in {"model", "sampling_model"} and _is_model(value):
                node.values[index] = _test_model_call()
        return node


def _is_model(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(MODEL_RE.match(node.value))


def _test_model_call() -> ast.Call:
    return ast.Call(func=ast.Name(id="__TestModel", ctx=ast.Load()), args=[], keywords=[])


def _top_level_await(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    return _expr_await(stmt)


def _expr_await(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return False
    if isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
        return True
    return any(_expr_await(child) for child in ast.iter_child_nodes(node))


def _ellipsis_arg(tree: ast.AST) -> bool:
    """Flag `...` used as a stand-in argument. Pydantic `Field(...)` is required-field syntax."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name == "Field":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value is Ellipsis:
                return True
    return False


def _agent_collect(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "collect":
            continue
        target = node.func.value
        if isinstance(target, ast.Name):
            return True
        if isinstance(target, ast.Await) and isinstance(target.value, ast.Name):
            return True
    return False


def _missing_max_tokens(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name != "Agent":
            continue
        model = next((kw.value for kw in node.keywords if kw.arg == "model"), None)
        if not _is_model(model) or not isinstance(model, ast.Constant):
            continue
        if not str(model.value).startswith("anthropic/"):
            continue
        if not any(kw.arg == "max_tokens" for kw in node.keywords):
            return True
    return False


class _NetBlock:
    def __enter__(self):
        self._connect = socket.socket.connect
        self._create = socket.create_connection
        self._popen = subprocess.Popen

        def blocked(_sock, addr):  # noqa: ANN001
            raise OSError(f"network blocked: {addr}")

        def _no_connection(*_args, **_kwargs):
            raise OSError("network blocked")

        socket.socket.connect = blocked  # type: ignore[method-assign]
        socket.create_connection = _no_connection  # type: ignore[assignment]

        class _BlockedPopen(self._popen):  # type: ignore[misc, valid-type]
            def __init__(self, *_args, **_kwargs):
                raise OSError("subprocess blocked")

        subprocess.Popen = _BlockedPopen  # type: ignore[misc]
        return self

    def __exit__(self, *_args):
        socket.socket.connect = self._connect  # type: ignore[method-assign]
        socket.create_connection = self._create  # type: ignore[assignment]
        subprocess.Popen = self._popen  # type: ignore[assignment]


def _alarm(_signum, _frame):
    raise TimeoutError("snippet exceeded 4s")


def _run_code(code: str, namespace: dict) -> None:
    tree = ast.parse(code)
    if _ellipsis_arg(tree):
        raise ValueError("ellipsis placeholder in a call")
    if _agent_collect(tree):
        raise ValueError("agent.collect() is not a method; call agent(...).collect()")
    if _missing_max_tokens(tree):
        raise ValueError("anthropic Agent requires max_tokens")
    tree = _SwapModels().visit(tree)
    ast.fix_missing_locations(tree)
    before = set(namespace)
    if any(_top_level_await(stmt) for stmt in tree.body):
        function = ast.AsyncFunctionDef(
            name="__docs_main",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[
                *tree.body,
                ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[])),
            ],
            decorator_list=[],
        )
        ast.fix_missing_locations(function)
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, "<docs>", "exec"), namespace, namespace)
        returned = asyncio.run(namespace["__docs_main"]())
        for key, value in returned.items():
            if key != "__docs_main":
                namespace[key] = value
    else:
        exec(compile(tree, "<docs>", "exec"), namespace, namespace)
    if "status" in code:
        return
    for key, value in namespace.items():
        if key in before:
            continue
        status = getattr(value, "status", None)
        if getattr(status, "code", None) == "error":
            raise RuntimeError(f"collect() status is error: {getattr(value, 'error', None)}")


def _validate_voice(namespace: dict) -> None:
    agent = namespace.get("agent")
    config = getattr(agent, "voice_config", None)
    if not isinstance(config, dict):
        return
    from timbal.voice.config import VoiceConfig

    VoiceConfig.model_validate(config)


def execute_python_snippets(docs: Path, *, split_fences) -> tuple[list[str], int]:
    from timbal.core import TestModel

    errors: list[str] = []
    executed = 0
    cwd = Path.cwd()
    work = Path(os.environ.get("TMPDIR", "/tmp")) / "timbal-docs-snippets"
    work.mkdir(exist_ok=True)
    os.chdir(work)
    try:
        with _NetBlock():
            for path in sorted(docs.rglob("*.mdx")) + sorted(docs.rglob("*.md")):
                if path.name == "docs.json":
                    continue
                source = path.read_text(encoding="utf-8")
                _, blocks, _ = split_fences(source)
                namespace: dict = {"__name__": "docs_snippet", "__TestModel": TestModel}
                rel = path.relative_to(docs).as_posix()
                for number, language, code in blocks:
                    tokens = language.split()
                    if not tokens or tokens[0] not in {"python", "py"}:
                        continue
                    if "fragment" in tokens:
                        try:
                            ast.parse(textwrap.dedent(code))
                        except SyntaxError as exc:
                            errors.append(f"{rel}:{number}: fragment does not parse: {exc.msg}")
                        continue
                    executed += 1
                    try:
                        signal.signal(signal.SIGALRM, _alarm)
                        signal.alarm(4)
                        _run_code(code, namespace)
                        _validate_voice(namespace)
                    except Exception as exc:
                        message = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
                        errors.append(f"{rel}:{number}: snippet failed: {type(exc).__name__}: {message}")
                    finally:
                        signal.alarm(0)
    finally:
        os.chdir(cwd)
    return errors, executed


def static_checks(
    docs: Path,
    *,
    catalog: dict,
    schema: dict,
    cli_dir: Path,
    server_dir: Path,
    split_fences,
) -> tuple[list[str], list[str]]:
    import yaml

    errors: list[str] = []
    warnings: list[str] = []
    description_warnings = 0
    pages = sorted(docs.rglob("*.mdx")) + sorted(p for p in docs.rglob("*.md") if p.name != "docs.json")
    config = __import__("json").loads((docs / "docs.json").read_text(encoding="utf-8"))
    listed_list = _page_list(config.get("navigation"))
    listed = set(listed_list)
    duplicates = sorted({page for page in listed_list if listed_list.count(page) > 1})
    errors.extend(f"navigation: duplicate page {page}" for page in duplicates)
    markdown_ids = {p.relative_to(docs).with_suffix("").as_posix() for p in pages if p.suffix == ".md"}
    errors.extend(f"navigation: unlisted page {page}" for page in sorted(markdown_ids - listed))
    banned = deprecated_model_ids(catalog)
    commands, flags_by_command = cli_surface(cli_dir)
    known_paths = openapi_paths(schema) | server_paths(server_dir)
    titles: dict[str, list[str]] = {}
    model_uses: dict[str, int] = {}
    for path in pages:
        if path.name == "docs.json":
            continue
        rel = path.relative_to(docs).as_posix()
        source = path.read_text(encoding="utf-8")
        prose, blocks, _ = split_fences(source)
        for phrase in INTERNAL_PHRASES:
            if phrase in source:
                errors.append(f"{rel}: internal note {phrase!r}")
        if not rel.startswith("models/"):
            for mid in sorted(set(re.findall(r"[A-Za-z0-9][\w.-]*/[\w./-]+", source)) & banned):
                errors.append(f"{rel}: deprecated or restricted model {mid}")
            for mid in re.findall(r'model\s*=\s*["\']([^"\']+)["\']', source):
                provider = mid.split("/", 1)[0]
                if provider in HOUSE_MODELS:
                    model_uses[mid] = model_uses.get(mid, 0) + 1
        for line in source.splitlines():
            if "no `timbal" in line or "not a `timbal" in line or "not `timbal" in line:
                continue
            candidates: list[str] = []
            stripped = line.strip()
            if stripped.startswith("$"):
                stripped = stripped[1:].strip()
            if stripped.startswith("`"):
                stripped = stripped[1:]
            head = re.match(r"timbal\s+([a-z][a-z0-9-]*)", stripped)
            if head:
                candidates.append(head.group(1))
            candidates.extend(re.findall(r"`timbal\s+([a-z][a-z0-9-]*)", line))
            seen_commands = set(candidates)
            for command in sorted(seen_commands):
                if command not in commands:
                    errors.append(f"{rel}: unknown CLI command timbal {command}")
            if len(seen_commands) != 1 or not seen_commands <= commands:
                continue
            command = next(iter(seen_commands))
            allowed = flags_by_command.get(command, set())
            for flag in sorted(set(re.findall(r"--[a-z0-9-]+", line))):
                if flag not in allowed:
                    errors.append(f"{rel}: unknown flag {flag} for timbal {command}")
        for module_name in sorted(set(PY_MODULE.findall(source))):
            if importlib.util.find_spec(module_name) is None:
                errors.append(f"{rel}: missing module {module_name}")
        http_sources = [prose]
        for _number, language, code in blocks:
            if language.split(" ", 1)[0] in {"http", "bash"} or language.startswith("http"):
                http_sources.append(code)
        for http_source in http_sources:
            for method, endpoint in HTTP_CALL.findall(http_source):
                if endpoint.startswith("//") or endpoint in {"/", ""}:
                    continue
                if not _path_known(endpoint, known_paths):
                    errors.append(f"{rel}: unknown HTTP endpoint {method} {endpoint}")
        frontmatter = re.match(r"\A---\n(.*?)\n---(?:\n|$)", source, re.S)
        if frontmatter and path.suffix == ".mdx":
            metadata = yaml.safe_load(frontmatter.group(1)) or {}
            title = metadata.get("title")
            if isinstance(title, str):
                titles.setdefault(title, []).append(rel)
            description = metadata.get("description")
            if not isinstance(description, str) or not 40 <= len(description) <= 160:
                description_warnings += 1
        for _number, language, code in blocks:
            if language.split(" ", 1)[0] not in {"python", "py"}:
                continue
            if re.search(r"\._[A-Za-z]", code) or re.search(r"\b_[A-Za-z][A-Za-z0-9_]*\s*=", code):
                warnings.append(f"{rel}: private attribute in a Python example")
    for title, paths in sorted(titles.items()):
        if len(paths) > 1:
            warnings.append(f"duplicate title {title!r}: {', '.join(paths)}")
    if description_warnings:
        warnings.append(f"{description_warnings} pages have a description outside 40-160 characters")
    extras = sorted(mid for mid in model_uses if mid not in HOUSE_MODELS.values())
    if len({mid.split("/")[0] for mid in model_uses}) and len(extras) > 4:
        warnings.append(
            "house models are "
            + ", ".join(HOUSE_MODELS.values())
            + f"; other example models in use: {', '.join(extras[:12])}"
        )
    return errors, warnings


def _page_list(node: object) -> list[str]:
    pages: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pages" and isinstance(value, list):
                pages.extend(item for item in value if isinstance(item, str))
            pages.extend(_page_list(value))
    elif isinstance(node, list):
        for item in node:
            pages.extend(_page_list(item))
    return pages


def extend_audit(
    docs: Path,
    catalog_path: Path,
    *,
    execute: bool,
    root: Path,
    split_fences,
) -> tuple[list[str], list[str], dict]:
    import yaml

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    schema_path = docs / "api-reference/openapi.json"
    schema = (
        __import__("json").loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else {"paths": {}}
    )
    errors, warnings = static_checks(
        docs,
        catalog=catalog,
        schema=schema,
        cli_dir=root / "cli",
        server_dir=root / "python/timbal/server",
        split_fences=split_fences,
    )
    executed = 0
    if execute:
        sys.path.insert(0, str(root / "python"))
        snippet_errors, executed = execute_python_snippets(docs, split_fences=split_fences)
        errors.extend(snippet_errors)
    return errors, warnings, {"snippets_executed": executed}
