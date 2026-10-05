"""Bound oversized file attachments in the request sent to the model.

Providers receive an image as an image and a PDF as a file, but every other attachment
(plain text, JSON, logs, code, and ``.xlsx``/``.docx`` after extraction) is pasted into the
prompt whole, as text. One large upload can overflow the context window, and because the
message stays in memory, every later turn of that conversation fails the same way.

Tool results are reduced once, when they are produced (:mod:`.tool_result_offload`).
Attachments are reduced in the *request* instead: memory and traces keep the original
``FileContent``, and each LLM call sends a bounded stand-in. The stand-in is a pure function
of the file's text and the limit, so repeated calls send identical bytes (prompt-cache
friendly), and a conversation that already holds an oversized attachment is bounded on its
next call without rewriting stored history.

Two layers:

- :func:`apply_attachment_limit` — opt-in through ``Agent(attachment_limit=...)``, using the
  same :class:`~.tool_result_offload.ToolResultLimit` config as tool results. ``Spill`` saves
  the text to the offload store under a content hash; the model pages it back with
  ``read_offloaded``.
- :func:`bound_unfittable_attachments` — always on, in the LLM router: a text attachment that
  is estimated to exceed the model's context window on its own is cut to a preview.
  This is a character-based heuristic, not an exact token budget.
"""

import asyncio
import hashlib
import inspect
from typing import Any

import structlog

from ..types.content import FileContent, TextContent
from ..types.content.file import AVAILABLE_ENCODINGS, _extract_docx_content, _extract_xlsx_content
from ..types.message import Message
from .models import get_context_window
from .tool_result_offload import (
    OffloadStore,
    Spill,
    ToolResultLimit,
    Truncate,
    _shape_sketch,
    _truncate_text,
    read_offloaded_call,
)

logger = structlog.get_logger("timbal.core.attachment_limit")

__all__ = [
    "ATTACHMENT_OFFLOAD_MARKER",
    "ATTACHMENT_TOO_LARGE_MARKER",
    "ATTACHMENT_UNAVAILABLE_MARKER",
    "AttachmentSpills",
    "apply_attachment_limit",
    "attachment_text",
    "bound_unfittable_attachments",
    "load_attachments",
]

ATTACHMENT_OFFLOAD_MARKER = "[Attached file offloaded:"
"""Prefix of the stand-in for a spilled attachment. Kept stable for tests and detection."""

ATTACHMENT_TOO_LARGE_MARKER = "[Attached file too large for the model's context window:"
"""Prefix of the safety-net stand-in (no attachment limit configured)."""

ATTACHMENT_UNAVAILABLE_MARKER = "[Attached file no longer available:"
"""Prefix of the note sent in place of a file whose source is gone (see :func:`load_attachments`)."""

# Approximate token density. Actual tokenization varies with content and model; this
# catches obviously oversized uploads but cannot guarantee that a full request fits.
_ESTIMATED_CHARS_PER_TOKEN = 4
_SAFETY_PREVIEW_CHARS = 20_000


def _decode(raw: bytes) -> str | None:
    for encoding in AVAILABLE_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def attachment_text(content: FileContent, threshold: int = 0) -> str | None:
    """The text the provider converters would paste for ``content``, when it has at least
    ``threshold`` characters.

    ``None`` when the file is sent natively (image, PDF, audio), has its own conversion
    (``.eml``), does not decode, or is smaller than ``threshold``. The file must be loaded.
    """
    file = content.file
    mime = file.__content_type__ or ""
    extension = (file.__source_extension__ or "").lower()
    if extension == ".eml" or mime.startswith(("image/", "audio/")) or mime == "application/pdf":
        return None
    try:
        file.seek(0)
        if extension == ".xlsx":
            text = _extract_xlsx_content(file)
        elif extension == ".docx":
            text = _extract_docx_content(file)
        else:
            raw = file.read()
            # Decoded text never has more characters than the file has bytes.
            if len(raw) < threshold:
                return None
            text = _decode(raw)
    except Exception:  # noqa: BLE001 — leave unreadable files to the converters, as before
        return None
    finally:
        try:
            file.seek(0)
        except Exception:  # noqa: BLE001
            pass
    if text is None or len(text) < threshold:
        return None
    return text


class AttachmentSpills:
    """Content hash → offload handle. Reuse readable payloads and restore missing ones
    from the original attachment before sending another stand-in."""

    def __init__(self) -> None:
        self._handles: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def handle(self, store: OffloadStore, text: str) -> str:
        key = f"attachments/{hashlib.sha256(text.encode()).hexdigest()[:32]}"
        async with self._lock:
            # Cached handles may have been pruned, or the lazily resolved store root may
            # have changed. Validate them before emitting another stand-in; memory still
            # has the original attachment, so a missing payload can be persisted again.
            # On a fresh process, built-in stores' handles are their content-hash keys.
            handle = self._handles.get(key, key)
            if not await _stored(store, handle):
                handle = await store.write(key, text.encode())
            self._handles[key] = handle
            return handle


async def _stored(store: OffloadStore, key: str) -> bool:
    exists = getattr(store, "exists", None)
    try:
        if exists is not None:
            found = exists(key)
            return bool(await found if inspect.isawaitable(found) else found)
        await store.read(key)
        return True
    except Exception:  # noqa: BLE001 — unknown key (or an unreachable store): write it
        return False


def _label(content: FileContent) -> str:
    mime = content.file.__content_type__ or "application/octet-stream"
    return f'"{content.name or "attachment"}" ({mime})'


def _offload_stand_in(content: FileContent, text: str, handle: str, preview_chars: int) -> str:
    total = len(text)
    lines = [
        f"{ATTACHMENT_OFFLOAD_MARKER} {_label(content)}, {total:,} chars. The full text was saved and can "
        f"be read with {read_offloaded_call(handle)} — page with offset/limit or filter with pattern.]",
    ]
    sketch = _shape_sketch(text)
    if sketch:
        lines.append(f"Shape: {sketch}")
    preview = text[:preview_chars]
    if preview:
        lines.append(f"Preview (first {len(preview):,} of {total:,} chars):")
        lines.append(preview)
    return "\n".join(lines)


def _truncate_stand_in(content: FileContent, text: str, action: Truncate) -> str:
    name = content.name or "attachment"
    return f"[Attached file {_label(content)}, {len(text):,} chars:]\n" + _truncate_text(
        text, name, action, kind="attachment"
    )


def _gone_status(error: BaseException) -> int | None:
    """The status to report when ``error`` means the file is gone for good, else ``None``."""
    import httpx

    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if 400 <= status < 500 and status not in (408, 429):
            return status
    if isinstance(error, FileNotFoundError):
        return 404
    return None


async def load_attachments(messages: list[Message]) -> list[Message]:
    """Load every unloaded file in ``messages`` and return the messages to send.

    A file whose source is gone for good (a 4xx, a missing local path) is replaced by a note
    instead of failing the request: the message stays in memory, so failing would fail every
    later turn of the conversation too. Transient errors still raise. ``messages`` is never
    modified; untouched messages are returned as the same objects.
    """
    unloaded = [
        c
        for m in messages
        for c in m.content
        if isinstance(c, FileContent) and object.__getattribute__(c.file, "__fileobj__") is None
    ]
    if not unloaded:
        return messages
    from .llm.clients import _get_file_client

    results = await asyncio.gather(*(c.file.load(client=_get_file_client()) for c in unloaded), return_exceptions=True)
    gone: dict[int, int] = {}
    for content, result in zip(unloaded, results, strict=True):
        if not isinstance(result, BaseException):
            continue
        status = _gone_status(result)
        if status is None:
            raise result
        gone[id(content)] = status
        logger.warning("Attached file is no longer available; sending a note instead.", attachment=content.name, status=status)
    if not gone:
        return messages
    out: list[Message] = []
    for message in messages:
        if not any(id(c) in gone for c in message.content):
            out.append(message)
            continue
        content = [
            TextContent(text=f"{ATTACHMENT_UNAVAILABLE_MARKER} {_label(c)} (status {gone[id(c)]}); it was not sent.]")
            if id(c) in gone
            else c
            for c in message.content
        ]
        out.append(_replaced(message, content))
    return out


def _replaced(message: Message, content: list[Any]) -> Message:
    return Message(role=message.role, content=content, stop_reason=message.stop_reason, metadata=message._metadata)


async def apply_attachment_limit(
    messages: list[Message],
    *,
    limit: ToolResultLimit,
    store: OffloadStore | None,
    spills: AttachmentSpills,
) -> list[Message]:
    """The messages to send: attachments whose text reaches ``limit.threshold`` are replaced
    by a stand-in. ``messages`` and its items are never modified; untouched messages are
    returned as the same objects."""
    messages = await load_attachments(messages)
    out = list(messages)
    for i, message in enumerate(messages):
        content: list[Any] = []
        changed = False
        for item in message.content:
            text = attachment_text(item, limit.threshold) if isinstance(item, FileContent) else None
            if text is None:
                content.append(item)
                continue
            content.append(TextContent(text=await _stand_in(item, text, limit, store, spills)))
            changed = True
        if changed:
            out[i] = _replaced(message, content)
    return out


async def _stand_in(
    content: FileContent, text: str, limit: ToolResultLimit, store: OffloadStore | None, spills: AttachmentSpills
) -> str:
    action = limit.action
    if isinstance(action, Truncate):
        return _truncate_stand_in(content, text, action)
    if isinstance(action, Spill):
        if store is not None:
            try:
                handle = await spills.handle(store, text)
                return _offload_stand_in(content, text, handle, action.preview_chars)
            except Exception:
                logger.exception("Attachment offload failed; falling back.", attachment=content.name)
        if action.fallback is not None:
            return _truncate_stand_in(content, text, action.fallback)
        # No store and no fallback: still never paste the whole file.
        return _truncate_stand_in(content, text, Truncate())
    return _truncate_stand_in(content, text, Truncate())


def bound_unfittable_attachments(messages: list[Message], model: str) -> list[Message]:
    """Safety net for every model call: a text attachment estimated to exceed ``model``'s
    context window on its own is replaced by a preview. Unknown windows leave messages alone.

    Files must already be loaded (the router loads them right before this runs).
    """
    window = get_context_window(model)
    if not window:
        return messages
    ceiling = window * _ESTIMATED_CHARS_PER_TOKEN
    out: list[Message] | None = None
    for i, message in enumerate(messages):
        content: list[Any] = []
        changed = False
        for item in message.content:
            text = attachment_text(item, ceiling) if isinstance(item, FileContent) else None
            if text is None:
                content.append(item)
                continue
            logger.warning(
                "Attachment is estimated to exceed the model's context window; sending a preview.",
                attachment=item.name,
                chars=len(text),
                model=model,
                context_window=window,
            )
            preview = text[:_SAFETY_PREVIEW_CHARS]
            content.append(
                TextContent(
                    text=f"{ATTACHMENT_TOO_LARGE_MARKER} {_label(item)}, {len(text):,} chars; "
                    f"showing the first {len(preview):,}.]\n{preview}"
                )
            )
            changed = True
        if changed:
            out = out if out is not None else list(messages)
            out[i] = _replaced(message, content)
    return out if out is not None else messages
