"""Production-time tool result offloading.

Large tool results dominate agent context windows. This module reduces a tool result *the
moment it is produced* — before it ever enters memory, the serialized dump, or the provider
request — so the reduction happens exactly once and history stays append-only (prompt-cache
friendly: the oversized payload never occupies a cached prefix that would later be rewritten).

Three pieces:

- ``ToolResultLimit`` — the config: a size ``threshold`` plus an ``action`` (``Spill`` or
  ``Truncate``). Set globally on ``Agent(tool_result_limit=...)`` or per tool via
  ``Tool(result_limit=...)``.
- ``OffloadStore`` / ``LocalOffloadStore`` — where spilled payloads live. Handles are
  backend-relative keys (never absolute paths), so a different backend can resolve the same
  handle in another process.
- ``read_tool_result`` (via :func:`create_read_tool_result`) — a bounded paging tool the
  model uses to read spilled payloads back on demand.

Distinct from ``memory_compaction``: that layer rewrites history that is already inside the
window; this layer keeps oversized payloads out of the window at production time. Compaction
strategies treat offloaded results as already-compacted (see ``compact_tool_results``).
"""

import json
import re
import threading
import time
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ..types.content import FileContent, TextContent
from ..types.content.tool_result import ToolResultContent

logger = structlog.get_logger("timbal.core.tool_result_offload")

__all__ = [
    "LocalOffloadStore",
    "OffloadStore",
    "Spill",
    "ToolResultLimit",
    "Truncate",
    "apply_tool_result_limit",
    "create_read_tool_result",
]

OFFLOAD_MARKER = "[Tool result offloaded:"
"""Prefix of the inline placeholder text for spilled results. Kept stable for tests and
downstream detection; programmatic detection should use ``ToolResultContent.offload_handle``."""

_SEGMENT_SAFE = re.compile(r"[^A-Za-z0-9._-]")

# read_tool_result hard caps — the read-back tool must never blow the window back up.
_READ_MAX_LINES = 500
_READ_MAX_CHARS = 50_000


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class Truncate(BaseModel):
    """Clamp the result text to a character budget. Lossy, zero-cost.

    ``head`` keeps the first characters (good for headers/schemas), ``tail`` keeps the last
    (good for build/test output where errors land at the end), ``head_tail`` keeps both ends
    and elides the middle (default — degenerate bulk is usually low-entropy repetition).
    """

    strategy: Literal["head", "tail", "head_tail"] = "head_tail"
    max_chars: int = Field(default=2_000, ge=1)


class Spill(BaseModel):
    """Persist the full payload to the offload store and keep a preview + handle inline.

    Lossless: the model reads the payload back on demand through ``read_tool_result``.
    ``fallback`` applies when no store is available or the store write fails — default is a
    bounded truncation so an oversized result is never silently passed through.
    """

    preview_chars: int = Field(default=1_000, ge=0)
    fallback: Truncate | None = Field(default_factory=Truncate)


class ToolResultLimit(BaseModel):
    """Size limit for tool results, applied once when the result is produced.

    Results whose concatenated text content reaches ``threshold`` characters get ``action``
    applied. Smaller results pass through untouched. Error results, pinned results, and
    ``read_tool_result``'s own output are always exempt.

    ``store`` is only honored on the agent-level config (``Agent(tool_result_limit=...)``);
    per-tool configs share the agent's store.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: int = Field(default=20_000, ge=1)
    action: Spill | Truncate = Field(default_factory=Spill)
    store: Any = None
    """Optional OffloadStore for spilled payloads. Defaults to a LocalOffloadStore."""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@runtime_checkable
class OffloadStore(Protocol):
    """Narrow protocol for spilled-payload backends."""

    async def write(self, key: str, data: bytes) -> str:
        """Persist ``data`` under ``key``; return the handle used for later reads."""
        ...

    async def read(self, handle: str) -> bytes:
        """Return the payload for ``handle``. Raise if the handle is unknown."""
        ...


def _sanitize_key(key: str) -> Path:
    """Turn a handle/key into a safe relative path.

    Rejects absolute paths and dot segments; every other unsafe character is replaced. This
    runs on both writes and reads so a crafted handle can never traverse out of the root.
    """
    segments = [s for s in key.split("/") if s]
    if not segments or key.startswith(("/", "\\")):
        raise ValueError(f"Invalid offload key: {key!r}")
    safe = []
    for segment in segments:
        if segment in (".", ".."):
            raise ValueError(f"Invalid offload key segment: {segment!r}")
        safe.append(_SEGMENT_SAFE.sub("_", segment))
    return Path(*safe)


class LocalOffloadStore:
    """Default store: one file per key under a stable local root.

    Keep-forever by default — deleting on run end would break a later run (session chaining,
    resume) that still holds handles. Opt into age-based pruning with ``cleanup_after``;
    pruning runs on a daemon thread off the hot path and never raises into the agent run.
    """

    def __init__(self, root: str | Path | None = None, cleanup_after: timedelta | None = None) -> None:
        self.root = (Path(root) if root else Path.home() / ".timbal" / "offload").expanduser().resolve()
        self.cleanup_after = cleanup_after

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError:  # e.g. exotic filesystems — permissions are best-effort hardening
            pass

    async def write(self, key: str, data: bytes) -> str:
        self._ensure_root()
        rel = _sanitize_key(key)
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never clobber: a retried call or key collision gets a distinct file.
        final = path
        n = 1
        while final.exists():
            final = path.with_name(f"{path.name}-{n}")
            n += 1
        final.write_bytes(data)
        if self.cleanup_after is not None:
            threading.Thread(target=self._prune, daemon=True).start()
        return final.relative_to(self.root).as_posix()

    async def read(self, handle: str) -> bytes:
        rel = _sanitize_key(handle)
        path = (self.root / rel).resolve()
        # Resolve (following symlinks) and re-check containment so neither a crafted handle
        # nor a symlink planted inside the root can escape it.
        if not path.is_relative_to(self.root):
            raise ValueError(f"Handle escapes the offload root: {handle!r}")
        if not path.is_file():
            raise FileNotFoundError(f"No offloaded content found for handle {handle!r}.")
        return path.read_bytes()

    def _prune(self) -> None:
        try:
            cutoff = time.time() - self.cleanup_after.total_seconds()
            for path in self.root.rglob("*"):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001 — cleanup must never fail a run
            warnings.warn(f"Offload store prune failed: {e}", stacklevel=2)


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def _shape_sketch(text: str) -> str | None:
    """One-line sketch of the top-level structure when the text parses as JSON."""

    def _t(v: Any) -> str:
        if isinstance(v, dict):
            return "object"
        if isinstance(v, list):
            return f"list[{len(v)}]"
        if v is None:
            return "null"
        return type(v).__name__

    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(obj, dict):
        items = list(obj.items())
        sketch = "{" + ", ".join(f'"{k}": {_t(v)}' for k, v in items[:8])
        if len(items) > 8:
            sketch += ", ..."
        sketch += "}"
    elif isinstance(obj, list):
        sketch = f"list[{len(obj)}]" + (f" of {_t(obj[0])}" if obj else "")
    else:
        return None
    return sketch[:200]


def _truncate_text(text: str, tool_name: str, action: Truncate) -> str:
    total = len(text)
    max_chars = action.max_chars
    if total <= max_chars:
        return text
    removed = total - max_chars
    marker = f"\n[... truncated {removed:,} of {total:,} chars from '{tool_name}' tool result ...]\n"
    if action.strategy == "head":
        return text[:max_chars] + marker
    if action.strategy == "tail":
        return marker + text[-max_chars:]
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + marker + text[-tail:]


def _spill_placeholder(tool_name: str, total_chars: int, handle: str, preview: str, sketch: str | None) -> str:
    lines = [
        f"{OFFLOAD_MARKER} {total_chars:,} chars from '{tool_name}'. The full content was saved and "
        f'can be read with read_tool_result(handle="{handle}") — page with offset/limit or filter '
        "with pattern.]",
    ]
    if sketch:
        lines.append(f"Shape: {sketch}")
    if preview:
        lines.append(f"Preview (first {len(preview):,} of {total_chars:,} chars):")
        lines.append(preview)
    return "\n".join(lines)


async def apply_tool_result_limit(
    result: ToolResultContent,
    *,
    limit: ToolResultLimit,
    tool_name: str,
    store: OffloadStore | None,
    run_id: str,
) -> dict[str, Any] | None:
    """Reduce ``result`` in place if its text content reaches ``limit.threshold``.

    Returns a metadata record describing what happened, or ``None`` when the result passed
    through untouched. Non-text content items (files) are preserved as-is.
    """
    text_items = [c for c in result.content if isinstance(c, TextContent)]
    other_items = [c for c in result.content if not isinstance(c, TextContent)]
    if any(not isinstance(c, TextContent | FileContent) for c in other_items):
        # Defensive: unknown content types are never safe to reduce around.
        return None
    text = "\n".join(c.text for c in text_items)
    total_chars = len(text)
    if total_chars < limit.threshold:
        return None

    action = limit.action
    record: dict[str, Any] = {
        "tool": tool_name,
        "call_id": result.id,
        "original_chars": total_chars,
    }

    if isinstance(action, Spill):
        if store is not None:
            try:
                handle = await store.write(f"{run_id}/{result.id}", text.encode())
                preview = text[: action.preview_chars]
                placeholder = _spill_placeholder(tool_name, total_chars, handle, preview, _shape_sketch(text))
                result.content = [TextContent(text=placeholder), *other_items]
                result.offload_handle = handle
                record.update(action="spill", handle=handle)
                return record
            except Exception:
                logger.exception(
                    "Offload store write failed; falling back.",
                    tool=tool_name,
                    call_id=result.id,
                )
        else:
            logger.warning(
                "Spill configured but no offload store available; falling back.",
                tool=tool_name,
                call_id=result.id,
            )
        if action.fallback is None:
            return None
        result.content = [TextContent(text=_truncate_text(text, tool_name, action.fallback)), *other_items]
        record.update(action="truncate_fallback", strategy=action.fallback.strategy)
        return record

    result.content = [TextContent(text=_truncate_text(text, tool_name, action)), *other_items]
    record.update(action="truncate", strategy=action.strategy)
    return record


# ---------------------------------------------------------------------------
# read_tool_result
# ---------------------------------------------------------------------------


def create_read_tool_result(store: OffloadStore) -> Any:
    """Build the bounded ``read_tool_result`` tool for a store.

    Output is hard-capped (lines and chars) so a read can never blow the context window back
    up, and ``pattern`` is a literal substring — a model-supplied value cannot trigger regex
    backtracking. The tool's own results are exempt from offloading (``result_limit=None``).
    """
    from .tool import Tool  # Local import: tool.py imports this module for the config types.

    async def _read_tool_result(
        handle: str = Field(..., description="The handle from an offload placeholder or compacted-transcript list."),
        offset: int = Field(0, description="Line offset to start reading from (0-based)."),
        limit: int = Field(200, description=f"Maximum lines to return (capped at {_READ_MAX_LINES})."),
        pattern: str | None = Field(
            None,
            description="Optional literal substring filter: only lines containing it are returned (offset/limit then apply to the matches).",
        ),
    ) -> str:
        """Read part of an offloaded tool result. Results are line-numbered; page with offset/limit."""
        data = await store.read(handle)
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        offset = max(0, offset)
        limit = max(1, min(limit, _READ_MAX_LINES))

        if pattern is not None:
            numbered = [(i, line) for i, line in enumerate(lines, start=1) if pattern in line]
            matched = len(numbered)
            selected = numbered[offset : offset + limit]
            header = f"[{len(selected)} of {matched} matching lines ({total} total) for {pattern!r} in {handle}]"
        else:
            selected = list(enumerate(lines, start=1))[offset : offset + limit]
            header = f"[lines {offset + 1}-{offset + len(selected)} of {total} in {handle}]"

        out_lines = [header]
        used = len(header)
        clipped = False
        for lineno, line in selected:
            entry = f"{lineno}: {line}"
            if used + len(entry) + 1 > _READ_MAX_CHARS:
                clipped = True
                break
            out_lines.append(entry)
            used += len(entry) + 1
        if clipped:
            out_lines.append(f"[output clipped at {_READ_MAX_CHARS:,} chars — continue with a higher offset]")
        return "\n".join(out_lines)

    return Tool(
        name="read_tool_result",
        description=(
            "Read the full content of an offloaded tool result. Use the handle from the "
            "offload placeholder. Page through long content with offset/limit, or pass a "
            "literal substring as pattern to return only matching lines."
        ),
        handler=_read_tool_result,
        result_limit=None,  # its own output is bounded and must never be offloaded again
    )
