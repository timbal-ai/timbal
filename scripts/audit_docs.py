#!/usr/bin/env python3
"""Check docs navigation, links, snippets, API endpoints, and explicit catalog prices.

Offline only. --runtime additionally checks Timbal imports in Python examples;
it imports package modules but never executes the examples or calls providers.
This checks the checked-in OpenAPI schema, not the deployed cloud API.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
import textwrap
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})(.*)$")
LINK = re.compile(r"(?:href=[\"']|\]\()([^\"')\s]+)")
MODEL = re.compile(r"(?:model\s*=\s*[\"']|model:\s*[\"']?)([\w]+/[\w./-]+)")
PRICE_PAIR = re.compile(r"\$([\d.]+)\s*(?:input\s*)?/\s*\$([\d.]+)")
CACHE_READ = re.compile(r"(?:cached input|cache reads?)\s*:?\s*\$([\d.]+)", re.I)
PRICE_TRIPLE = re.compile(
    r"input\s*/\s*output\s*/\s*cached input:\s*\$([\d.]+)\s*/\s*\$([\d.]+)\s*/\s*\$([\d.]+)", re.I
)


def navigation_pages(node: object) -> set[str]:
    pages: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pages" and isinstance(value, list):
                pages.update(item for item in value if isinstance(item, str))
            pages.update(navigation_pages(value))
    elif isinstance(node, list):
        for item in node:
            pages.update(navigation_pages(item))
    return pages


def split_fences(source: str) -> tuple[str, list[tuple[int, str, str]], list[str]]:
    """Return prose and (opening line, language, code) without matching links in code."""
    prose: list[str] = []
    blocks: list[tuple[int, str, str]] = []
    errors: list[str] = []
    opening: tuple[int, str, str] | None = None
    code: list[str] = []
    for number, line in enumerate(source.splitlines(), 1):
        fence = FENCE.match(line)
        if opening is None:
            if fence:
                opening = (number, fence[2], fence[3].strip())
                code = []
            else:
                prose.append(line)
        elif fence and fence[2][0] == opening[1][0] and len(fence[2]) >= len(opening[1]) and not fence[3].strip():
            blocks.append((opening[0], opening[2], textwrap.dedent("\n".join(code))))
            opening = None
        else:
            code.append(line)
    if opening:
        errors.append(f"line {opening[0]}: unclosed code fence")
    return "\n".join(prose), blocks, errors


def check_imports(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module != "timbal" and not node.module.startswith("timbal."):
            continue
        try:
            module = importlib.import_module(node.module)
            for name in node.names:
                if name.name != "*" and not hasattr(module, name.name):
                    errors.append(f"line {node.lineno}: missing import {node.module}.{name.name}")
        except Exception as exc:
            errors.append(f"line {node.lineno}: cannot import {node.module}: {exc}")
    return errors


def audit(docs: Path, catalog: Path, *, runtime: bool = False) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    paths = sorted(docs.rglob("*.mdx"))
    available = {p.relative_to(docs).with_suffix("").as_posix() for p in paths}
    config = json.loads((docs / "docs.json").read_text(encoding="utf-8"))
    listed = navigation_pages(config["navigation"])
    errors.extend(f"navigation: missing page {page}" for page in sorted(listed - available))
    errors.extend(f"navigation: unlisted page {page}" for page in sorted(available - listed))
    models = {m["id"]: m for m in yaml.safe_load(catalog.read_text(encoding="utf-8"))["models"]}
    schema = json.loads((docs / "api-reference/openapi.json").read_text(encoding="utf-8"))
    counts = {"pages": len(paths), "python_snippets": 0, "api_operations": 0, "price_pairs": 0, "cache_prices": 0}
    for path in paths:
        label = path.relative_to(docs).as_posix()
        source = path.read_text(encoding="utf-8")
        prose, blocks, fence_errors = split_fences(source)
        errors.extend(f"{label}: {error}" for error in fence_errors)
        frontmatter = re.match(r"\A---\n(.*?)\n---(?:\n|$)", source, re.S)
        if not frontmatter:
            errors.append(f"{label}: missing frontmatter")
        else:
            metadata = yaml.safe_load(frontmatter[1])
            if not isinstance(metadata, dict) or not metadata.get("title"):
                errors.append(f"{label}: missing title")
            elif metadata.get("openapi"):
                method, endpoint = metadata["openapi"].split(" ", 1)
                counts["api_operations"] += 1
                if method.lower() not in schema.get("paths", {}).get(endpoint, {}):
                    errors.append(f"{label}: unknown OpenAPI operation {method} {endpoint}")
        for target in LINK.findall(prose):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            link = unquote(parsed.path)
            candidate = docs / link.lstrip("/") if link.startswith("/") else path.parent / link
            if not any(p.is_file() for p in (candidate, candidate.with_suffix(".mdx"), candidate / "index.mdx")):
                errors.append(f"{label}: broken local link {target}")
        for mid in sorted(set(MODEL.findall(source)) - models.keys()):
            errors.append(f"{label}: model not in catalog: {mid}")
        for number, language, code in blocks:
            if language.split(" ", 1)[0] not in ("python", "py"):
                continue
            counts["python_snippets"] += 1
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                errors.append(f"{label}:{number + (exc.lineno or 1)}: Python syntax: {exc.msg}")
                continue
            if runtime:
                errors.extend(f"{label}:{number}: {error}" for error in check_imports(tree))
        if path.parent.name == "models":
            for card in re.findall(r"<Card\b[^>]*>(.*?)</Card>", source.replace("&#36;", "$"), re.S):
                ids = [mid for mid in re.findall(r"`([a-z]+/[\w./-]+)`", card) if mid in models]
                if not ids:
                    continue
                model = models[ids[0]]
                pair = PRICE_PAIR.search(card)
                if pair and model.get("input_price") is not None and model.get("output_price") is not None:
                    counts["price_pairs"] += 1
                    actual = [float(pair[1]), float(pair[2])]
                    expected = [model["input_price"], model["output_price"]]
                    if actual != expected:
                        errors.append(f"{label}: {ids[0]} card prices {actual} != catalog {expected}")
                cached = CACHE_READ.search(card)
                if cached:
                    counts["cache_prices"] += 1
                    expected_cached = model.get("cached_input_price")
                    triple = PRICE_TRIPLE.search(card)
                    cached_price = float(triple[3] if triple else cached[1])
                    if cached_price != expected_cached:
                        errors.append(f"{label}: {ids[0]} cached input {cached_price} != catalog {expected_cached}")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Check Timbal snippet imports in the installed package")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "python"))
    errors, counts = audit(ROOT / "docs", ROOT / "python/timbal/models.yaml", runtime=args.runtime)
    for error in errors:
        print(error)  # noqa: T201
    print(f"Docs audit: {counts}; {len(errors)} errors")  # noqa: T201
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
