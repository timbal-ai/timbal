"""Recover tool calls that GPT-5.x leaks into assistant text.

GPT-5.x on the Responses API occasionally fails to emit a structured ``function_call``
item and instead writes the Harmony wire form of the call into a ``message`` item::

     to=functions.wait_for_background_tasks  (json 恒一
    {"task_ids":["o20s53ln5l51"],"timeout_seconds":120}

``to=functions.<name>`` is the recipient header; the junk between it and the JSON
is where the ``<|constrain|>json<|message|>`` control tokens should have been
(decoded as text they surface as high-frequency training tokens — CJK spam, stray
Cyrillic). The name and the JSON object are intact in every sample we have; the
framing is what broke. Known across the ecosystem (livekit/agents-js#1632,
openclaw#30441, hermes-agent ``codex_responses_adapter``); this module is the
recovery every serious runtime ends up shipping.

Two entry points:

- :func:`leak_state` — for the streaming hold-back. While a text block is short and
  still a prefix of the marker (`` to=fun``) it is *undecided* and must not be
  emitted; once it either matches or diverges the collector knows whether to stream
  it or swallow it.
- :func:`parse_leaked_tool_calls` — for the collected block. Returns the clean text
  that preceded the first marker and every ``(name, arguments)`` it could parse, in
  order and deduplicated (the model tends to repeat the same call several times).
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Literal

LEAK_MARKER = "to=functions."
# The model's parallel-call wrapper leaks too:
#   to=multi_tool_use.parallel … {"tool_uses":[{"recipient_name":"functions.X","parameters":{…}}, …]}
PARALLEL_MARKER = "to=multi_tool_use.parallel"
LEAK_MARKERS = (LEAK_MARKER, PARALLEL_MARKER)

# `Message.metadata["kind"]` the collector sets when a leak carried no runnable call —
# the agent loop re-requests instead of ending the turn with nothing.
LEAKED_TOOL_CALL_UNRECOVERED = "leaked_tool_call_unrecovered"

# The recipient header. A tool name is what the Responses API allows for a function
# name plus the dots timbal uses for namespaced tools (``timbal__codegen`` has none,
# but MCP-style ``server.tool`` names exist elsewhere).
# `functions.` is usually present; the bare form (`to=timbal__codegen code: {…}`) shows up
# too. A bare name is accepted only when a JSON object follows it (see the parser) — the
# agent's unknown-tool path answers a wrong name with an error the model can act on.
_HEADER_RE = re.compile(
    r"to=(?:(multi_tool_use\.parallel)|(?:functions\.)?([A-Za-z_][A-Za-z0-9_.\-]*))(?![A-Za-z0-9_.\-])"
)
_BARE_HEAD_RE = re.compile(r"^to=[A-Za-z_][A-Za-z0-9_.\-]*(?:\s|\()")
_BARE_TYPING_RE = re.compile(r"^to=[A-Za-z_]?[A-Za-z0-9_.\-]*$")

LeakState = Literal["undecided", "leak", "clean"]


def leak_state(text: str) -> LeakState:
    """Whether the text so far *starts* a leaked tool call.

    Leading whitespace is ignored (the leak arrives as `` to=functions.…``). Only a
    leak at the very start of the block is decidable while streaming; a marker after
    real prose is found by :func:`parse_leaked_tool_calls` once the block is complete.
    """
    head = text.lstrip()
    if not head:
        return "undecided"
    if any(head.startswith(m) for m in LEAK_MARKERS):
        return "leak"
    if any(m.startswith(head) for m in LEAK_MARKERS):
        return "undecided"
    # Bare recipient (`to=timbal__codegen …`): a leak once the name is complete; while
    # the name is still being typed, undecided. Prose does not start with `to=`.
    if _BARE_HEAD_RE.match(head) and not head.startswith("to=functions.") and not head.startswith("to=multi_tool_use."):
        return "leak"
    if _BARE_TYPING_RE.match(head):
        return "undecided"
    return "clean"


def contains_leak(text: str) -> bool:
    """A leak header that counts: at a line start, outside code fences."""
    return bool(_leak_headers(text or ""))


def _expand_parallel(args: dict) -> list[tuple[str, dict]]:
    """``{"tool_uses": [{"recipient_name": "functions.X", "parameters": {…}}, …]}`` → calls."""
    out: list[tuple[str, dict]] = []
    for use in args.get("tool_uses") or []:
        if not isinstance(use, dict):
            continue
        name = str(use.get("recipient_name") or "")
        name = name.split("functions.", 1)[1] if "functions." in name else name
        params = use.get("parameters")
        if name and isinstance(params, dict):
            out.append((name, params))
    return out


def _extract_json_object(text: str, start: int) -> tuple[str, int] | None:
    """The first balanced ``{…}`` at or after ``start``; ``(json_text, end_index)``.

    Tracks string literals so braces inside argument strings do not end the scan.
    """
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i : j + 1], j + 1
    return None


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    """Spans of ``` fenced code blocks — a header inside one is being *quoted*, not issued."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for m in re.finditer(r"^[ \t]*```", text, re.M):
        if start is None:
            start = m.start()
        else:
            spans.append((start, m.end()))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _at_line_start(text: str, pos: int) -> bool:
    """The header begins a line (after optional indentation). Leaks always do; a header
    embedded mid-sentence is someone describing the syntax."""
    line_start = text.rfind("\n", 0, pos) + 1
    return text[line_start:pos].strip() == ""


def _leak_headers(text: str) -> list[re.Match]:
    fenced = _fenced_ranges(text)
    return [
        m for m in _HEADER_RE.finditer(text or "")
        if _at_line_start(text, m.start()) and not any(a <= m.start() < b for a, b in fenced)
    ]


def _looks_corrupted(raw: str) -> bool:
    """Control-token debris landed *inside* the JSON.

    The debris is what the decoder makes of special tokens: private-use, unassigned and
    replacement code points (``\\U0004e7ff``, ``񟿿``, ``￼``, ``\\ufffd``). Real text — in
    any script — does not contain those, so their presence means the arguments cannot
    be trusted (a `builder` prompt with a garbage line in it would still parse). Such
    a call is not recovered; with nothing else runnable the step is re-requested.
    """
    for ch in raw:
        if ch in ("\ufffc", "\ufffd"):
            return True
        cat = unicodedata.category(ch)
        if cat in ("Co", "Cn", "Cs"):
            return True
    return False


def parse_leaked_tool_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """Split a text block into the prose before the first leak and the calls it carries.

    Each header is paired with the first balanced JSON object that follows it and
    precedes the next header; a header without a parseable object is skipped (there
    is nothing to run). Identical ``(name, arguments)`` pairs collapse into one call.
    Text *between* or *after* the calls is dropped: it is either control-token debris
    or a summary the model wrote believing the calls had run.

    A header only counts when it starts a line and sits outside a ``` fence; an
    object carrying decoder debris (see ``_looks_corrupted``) is not recovered.
    """
    matches = _leak_headers(text or "")
    if not matches:
        return text, []
    prefix = text[: matches[0].start()].rstrip()
    calls: list[tuple[str, dict]] = []
    seen: set[tuple[str, str]] = set()
    for k, m in enumerate(matches):
        limit = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        found = _extract_json_object(text[:limit], m.end())
        if found is None:
            continue
        raw, _ = found
        if _looks_corrupted(raw):
            continue
        try:
            args = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(args, dict):
            continue
        expanded = _expand_parallel(args) if m.group(1) else [(m.group(2), args)]
        for name, a in expanded:
            key = (name, json.dumps(a, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            calls.append((name, a))
    return prefix, calls
