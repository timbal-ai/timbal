"""Notion tools: search, pages (read as Markdown, create, update, append), data sources, comments.

Two ways to connect, both sending ``Authorization: Bearer <token>``:

- **oauth** (public integration): the person who signs in picks the pages the integration may
  see. Token from ``Integration("notion")`` (``token`` / ``access_token``), tool field ``token``,
  or env ``NOTION_ACCESS_TOKEN``; or a ``refresh_token`` (tool field or env ``NOTION_REFRESH_TOKEN``,
  never an integration's) exchanged with the integration's ``client_id`` + ``client_secret``.
  Notion issues a new refresh token on every refresh, so point ``token_file`` / env
  ``NOTION_TOKEN_FILE`` at a file to keep the newest one across restarts
  (``examples/notion/setup_credentials.py`` writes it). A rejected access token is refreshed
  once and the call retried.
- **api_key** (internal integration): the integration's secret (``api_key`` / env
  ``NOTION_API_KEY``) from notion.so/profile/integrations. It sees the pages shared with it
  under a page's ``•••`` > Connections.

``auth_method`` (tool field, integration ``auth_method``, or env ``NOTION_AUTH_METHOD``) pins one
of the two. Unset, OAuth sources are tried before the API key.

Data model: since API version 2025-09-03 a *database* is a container of one or more *data
sources*, and rows are queried per data source. The tools accept a database or a data source ID
(or a Notion URL) wherever one is expected and resolve the other.

Integration credentials (type: oauth or credentials):
- token / access_token: an OAuth access token. The platform refreshes it and keeps Notion's
  rotated refresh token; a ``refresh_token`` alone is refused, never exchanged here, since that
  would spend the token the platform stores. A token Notion rejects is replaced by resolving
  the integration again.
- api_key: an internal integration secret
- auth_method: "oauth" or "api_key". Unset, an integration carrying a user token is OAuth.

With an integration, credentials come from it and the tool's own fields only: ``NOTION_*``
environment variables (and the token file they name) are not read, so a deployer's sign-in
never stands in for a user.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..errors import CredentialNotAvailable
from ..platform.integrations import Integration
from . import _oauth_store

API_URL = "https://api.notion.com/v1"
#: Default ``Notion-Version``; override with the tool field ``notion_version`` or env ``NOTION_VERSION``.
NOTION_VERSION = "2026-03-11"
AUTHORIZE_URL = f"{API_URL}/oauth/authorize"
TOKEN_URL = f"{API_URL}/oauth/token"

_MAX_TEXT = 2000  # characters per rich_text segment
_MAX_RICH_TEXT = 100  # segments per rich_text array
_MAX_CHILDREN = 100  # blocks per create/append request
_MAX_RETRIES = 3

_ENV_VARS = [
    "NOTION_API_KEY",
    "NOTION_ACCESS_TOKEN",
    "NOTION_REFRESH_TOKEN",
    "NOTION_TOKEN_FILE",
    "NOTION_CLIENT_ID",
    "NOTION_CLIENT_SECRET",
]

AuthMethod = Literal["oauth", "api_key"]


class NotionAuth(NamedTuple):
    token: str
    method: AuthMethod
    #: Whether a rejected OAuth token can be replaced: through the refresh grant, or by resolving
    #: the platform integration again (the platform refreshes its tokens itself).
    can_refresh: bool = False


class NotionAPIError(ValueError):
    """Notion answered with an error. ``status`` and ``code`` are Notion's (``object_not_found``, ...)."""

    def __init__(self, message: str, *, status: int, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class NotionPartialWriteError(NotionAPIError):
    """A write spread over several requests failed partway: ``page_id`` already holds the first
    ``blocks_written`` of ``blocks_total`` blocks, so repeating the whole write would duplicate them."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        code: str | None = None,
        page_id: str,
        blocks_written: int,
        blocks_total: int,
    ) -> None:
        super().__init__(message, status=status, code=code)
        self.page_id = page_id
        self.blocks_written = blocks_written
        self.blocks_total = blocks_total


class NotionTokenError(ValueError):
    """Notion's /oauth/token refused a grant. ``code`` is the OAuth error (``invalid_grant``, ...)."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    text = str(value).strip()
    return text or None


def _first(*values: Any) -> str | None:
    for value in values:
        text = _secret_value(value)
        if text:
            return text
    return None


EnvLookup = Callable[[str], str | None]


def _no_env(_name: str) -> None:
    return None


def _auth_method(tool: Any, creds: dict[str, Any], env: EnvLookup = os.getenv) -> AuthMethod | None:
    raw = _first(getattr(tool, "auth_method", None), creds.get("auth_method"), env("NOTION_AUTH_METHOD"))
    if raw is None:
        return None
    method = raw.lower().replace("-", "_")
    if method not in ("oauth", "api_key"):
        raise ValueError(f"Notion auth_method must be 'oauth' or 'api_key', got {raw!r}.")
    return method  # type: ignore[return-value]


def _client_parts(tool: Any, creds: dict[str, Any], env: EnvLookup = os.getenv) -> tuple[str, str] | None:
    client_id = _first(getattr(tool, "client_id", None), creds.get("client_id"), env("NOTION_CLIENT_ID"))
    client_secret = _first(
        getattr(tool, "client_secret", None), creds.get("client_secret"), env("NOTION_CLIENT_SECRET")
    )
    if client_id and client_secret:
        return client_id, client_secret
    return None


async def token_request(client_id: str, client_secret: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST /v1/oauth/token with HTTP Basic client auth and a JSON body; the raw token response."""
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(
            TOKEN_URL,
            json=payload,
            auth=(client_id, client_secret),
            headers={"Accept": "application/json", "Notion-Version": os.getenv("NOTION_VERSION") or NOTION_VERSION},
        )
    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = {}
        code = body.get("error") or body.get("code") if isinstance(body, dict) else None
        detail = (body.get("error_description") or body.get("message")) if isinstance(body, dict) else None
        message = f"Notion token request ({payload.get('grant_type')}) failed: {response.status_code} {code or ''}"
        if detail:
            message += f" {str(detail).rstrip('. ')}"
        if code == "invalid_client":
            message += ". The client_id or client_secret is wrong: copy them again from the integration's settings."
        elif code == "invalid_grant" and payload.get("grant_type") == "refresh_token":
            message += (
                ". The refresh token was refused: it was revoked or already used (Notion issues a new one on "
                "every refresh). Sign in again, and set NOTION_TOKEN_FILE so the newest one survives restarts."
            )
        raise NotionTokenError(message.strip(), code=code)
    body = response.json()
    if not isinstance(body.get("access_token"), str):
        raise NotionTokenError("Notion token response did not include access_token.")
    return body


async def exchange_code(client_id: str, client_secret: str, code: str, redirect_uri: str | None) -> dict[str, Any]:
    """Authorization code -> token response (access_token, refresh_token, bot_id, workspace_name, owner, ...)."""
    payload: dict[str, Any] = {"grant_type": "authorization_code", "code": code}
    if redirect_uri:
        payload["redirect_uri"] = redirect_uri
    return await token_request(client_id, client_secret, payload)


async def get_token_from_refresh_token(
    client_id: str,
    client_secret: str,
    refresh_token: str | None,
    *,
    token_file: str | os.PathLike[str] | None = None,
    rejected: str | None = None,
) -> str:
    """OAuth: an access token from the refresh token, persisting Notion's rotated refresh token."""
    path = os.path.abspath(os.fspath(token_file)) if token_file is not None else None
    key = _oauth_store.cache_key("notion", client_id, client_secret, path or refresh_token or "")

    async def exchange(candidate: str) -> dict[str, Any]:
        return await token_request(
            client_id, client_secret, {"grant_type": "refresh_token", "refresh_token": candidate}
        )

    return await _oauth_store.refreshed_access_token(
        key=key,
        owner={"provider": "notion", "client_id": client_id},
        refresh_token=refresh_token,
        token_file=path,
        exchange=exchange,
        is_refused=lambda exc: isinstance(exc, NotionTokenError) and exc.code == "invalid_grant",
        rejected=rejected,
    )


async def _resolve_auth(tool: Any, *, rejected: str | None = None) -> NotionAuth:
    """Bearer token for this tool (see the module docstring for the order).

    ``rejected`` is an OAuth access token Notion just refused: it is skipped and, when a
    refresh token is available, replaced through the refresh grant. With a platform
    integration it is replaced only by what the integration resolves to now.
    """
    integration = getattr(tool, "integration", None)
    from_integration = isinstance(integration, Integration)
    creds: dict[str, Any] = (await integration.resolve() or {}) if from_integration else {}
    # With a platform integration, credentials come from it and the tool's own fields only. The
    # process environment (and a token file it names) holds the deployer's Notion sign-in, which
    # must never answer for a user whose integration lacks a credential or was rejected.
    env: EnvLookup = _no_env if from_integration else os.getenv
    method = _auth_method(tool, creds, env)
    if method is None and any(_first(creds.get(k)) for k in ("token", "access_token", "refresh_token")):
        method = "oauth"  # an OAuth integration stays OAuth instead of falling back to an API key
    # The platform refreshes an OAuth integration's token itself and stores the newest refresh
    # token. Notion rotates refresh tokens on every use, so exchanging it here would leave the
    # platform holding a dead one: an integration has to supply the access token.
    integration_refresh_only = bool(_first(creds.get("refresh_token"))) and not _first(
        creds.get("token"), creds.get("access_token")
    )
    token_file_missing: str | None = None

    if method != "api_key":
        token = _first(
            getattr(tool, "token", None),
            creds.get("token"),
            creds.get("access_token"),
            env("NOTION_ACCESS_TOKEN"),
        )
        refresh = _first(getattr(tool, "refresh_token", None), env("NOTION_REFRESH_TOKEN"))
        token_file = _first(getattr(tool, "token_file", None), env("NOTION_TOKEN_FILE"))
        if token_file and not refresh and _oauth_store.read_token_file(token_file) is None:
            token_file_missing, token_file = token_file, None
        client = _client_parts(tool, creds, env)
        can_refresh = bool((refresh or token_file) and client)
        if token and token != rejected:
            return NotionAuth(token, "oauth", can_refresh or from_integration)
        if integration_refresh_only:
            raise CredentialNotAvailable(
                "Notion",
                missing=["token"],
                message="The Notion integration supplied a refresh token but no access token. The platform "
                "must refresh it and return `token`: exchanging it here would spend the refresh token it stores.",
            )
        if can_refresh:
            assert client is not None
            access = await get_token_from_refresh_token(*client, refresh, token_file=token_file, rejected=rejected)
            return NotionAuth(access, "oauth", True)

    if method != "oauth":
        api_key = _first(getattr(tool, "api_key", None), creds.get("api_key"), env("NOTION_API_KEY"))
        if api_key:
            return NotionAuth(api_key, "api_key")

    if token_file_missing and method != "api_key":
        raise CredentialNotAvailable(
            "Notion",
            missing=["refresh_token"],
            env_vars=[] if from_integration else ["NOTION_TOKEN_FILE"],
            message=f"Notion token file {token_file_missing!r} is missing or holds no refresh token: "
            "sign in first (examples/notion/setup_credentials.py writes it).",
        )
    if from_integration:
        missing = ["api_key"] if method == "api_key" else ["token"] if method == "oauth" else ["token", "api_key"]
    elif method == "oauth":
        missing = ["token", "refresh_token + client_id + client_secret"]
    elif method == "api_key":
        missing = ["api_key"]
    else:
        missing = ["api_key", "token", "refresh_token + client_id + client_secret"]
    raise CredentialNotAvailable("Notion", missing=missing, env_vars=[] if from_integration else _ENV_VARS)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _error_message(status: int, body: Any, auth: NotionAuth) -> tuple[str | None, str]:
    code = body.get("code") if isinstance(body, dict) else None
    detail = body.get("message") if isinstance(body, dict) else None
    detail = str(detail).rstrip(". ") if detail else None
    message = f"Notion API {status}" + (f" {code}" if code else "") + (f": {detail}" if detail else "")
    if code in ("object_not_found", "restricted_resource"):
        message += (
            ". If the page or database exists, it is not shared with this integration: in Notion open it, "
            "then ••• > Connections and add the integration (OAuth: sign in again and select it)."
        )
    elif status == 401:
        message += (
            ". The API key is wrong or the integration was deleted: check NOTION_API_KEY."
            if auth.method == "api_key"
            else ". The OAuth connection was revoked or expired: sign in again."
        )
    return code, message


def _retry_after(response: Any, attempt: int) -> float:
    raw = response.headers.get("Retry-After")
    try:
        delay = float(raw) if raw is not None else 2.0**attempt
    except ValueError:
        delay = 2.0**attempt
    return min(max(delay, 0.5), 30.0)


async def _notion_request(
    tool: Any,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """One Notion API call: auth, version header, 429/529 back-off, and one refresh on a rejected OAuth token."""
    import httpx

    auth = await _resolve_auth(tool)
    version = _first(getattr(tool, "notion_version", None), os.getenv("NOTION_VERSION")) or NOTION_VERSION
    refreshed = False
    attempt = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        while True:
            response = await client.request(
                method,
                f"{API_URL}{path}",
                json=json_body,
                params={k: v for k, v in (params or {}).items() if v is not None} or None,
                headers={
                    "Authorization": f"Bearer {auth.token}",
                    "Notion-Version": version,
                    "Accept": "application/json",
                },
            )
            if response.status_code == 401 and auth.method == "oauth" and auth.can_refresh and not refreshed:
                refreshed = True
                try:
                    auth = await _resolve_auth(tool, rejected=auth.token)
                except CredentialNotAvailable:
                    break  # nothing to replace it with (an integration still on the same token): report the 401
                continue
            if response.status_code in (429, 529) and attempt < _MAX_RETRIES:
                try:
                    blocked = (response.json().get("additional_data") or {}).get("rate_limit_reason")
                except (ValueError, AttributeError):
                    blocked = None
                if blocked != "public_api_request_blocked":
                    await asyncio.sleep(_retry_after(response, attempt))
                    attempt += 1
                    continue
            break
    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = response.text[:300]
        code, message = _error_message(response.status_code, body, auth)
        raise NotionAPIError(message, status=response.status_code, code=code)
    return response.json() if response.content else {}


# ---------------------------------------------------------------------------
# IDs, rich text and properties
# ---------------------------------------------------------------------------

# A whole 32-hex word: in a slug URL (`Title-Words-<id>`) the title's last word can be hex too
# (`Roadmap-2025-<id>`, `Cafe-<id>`), so a match never runs across the dash before the ID.
_HEX32 = re.compile(r"(?<![0-9a-f])([0-9a-f]{32})(?![0-9a-f])", re.I)
_UUID = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)


def normalize_id(value: str) -> str:
    """A Notion ID from an ID (dashed or not) or a notion.so URL, in dashed UUID form."""
    text = value.strip()
    path = text.split("?")[0].split("#")[0]
    found = _UUID.findall(path) or _UUID.findall(text)
    if found:
        return found[-1].lower()
    compact = _HEX32.findall(path) or _HEX32.findall(text)
    if not compact:
        raise ValueError(f"Not a Notion ID or URL: {value!r}")
    h = compact[-1].lower()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def plain_text(rich_text: Any) -> str:
    if not isinstance(rich_text, list):
        return ""
    return "".join(str(part.get("plain_text") or (part.get("text") or {}).get("content") or "") for part in rich_text)


def _user_name(user: Any) -> Any:
    if not isinstance(user, dict):
        return user
    return user.get("name") or (user.get("person") or {}).get("email") or user.get("id")


def property_value(prop: dict[str, Any]) -> Any:
    """A Notion property value as plain JSON (text, number, names, dates...)."""
    kind = prop.get("type")
    value = prop.get(kind) if kind else None
    if kind in ("title", "rich_text"):
        return plain_text(value)
    if kind in ("select", "status"):
        return (value or {}).get("name")
    if kind == "multi_select":
        return [option.get("name") for option in value or []]
    if kind == "date":
        if not value:
            return None
        return value.get("start") if not value.get("end") else {"start": value.get("start"), "end": value.get("end")}
    if kind == "people":
        return [_user_name(user) for user in value or []]
    if kind in ("created_by", "last_edited_by"):
        return _user_name(value)
    if kind == "relation":
        return [item.get("id") for item in value or []]
    if kind == "files":
        return [
            {"name": f.get("name"), "url": (f.get("file") or f.get("external") or {}).get("url")} for f in value or []
        ]
    if kind == "formula":
        return (value or {}).get((value or {}).get("type")) if value else None
    if kind == "rollup":
        if not value:
            return None
        inner = value.get(value.get("type"))
        if value.get("type") == "array":
            return [property_value(item) for item in inner or []]
        return inner
    if kind == "unique_id":
        if not value:
            return None
        return f"{value.get('prefix')}-{value.get('number')}" if value.get("prefix") else value.get("number")
    if kind == "verification":
        return (value or {}).get("state")
    return value


def title_of(obj: dict[str, Any]) -> str:
    if isinstance(obj.get("title"), list):  # data sources and databases
        return plain_text(obj["title"])
    for prop in (obj.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return plain_text(prop.get("title"))
    return ""


def _parent(obj: dict[str, Any]) -> dict[str, Any] | None:
    parent = obj.get("parent") or {}
    kind = parent.get("type")
    return (
        {"type": kind, "id": parent.get(kind)} if kind and kind != "workspace" else ({"type": kind} if kind else None)
    )


def page_summary(page: dict[str, Any], *, with_properties: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "id": page.get("id"),
        "object": page.get("object"),
        "title": title_of(page),
        "url": page.get("url"),
        "parent": _parent(page),
        "last_edited_time": page.get("last_edited_time"),
    }
    if page.get("in_trash") or page.get("archived"):
        summary["in_trash"] = True
    if with_properties and page.get("object") == "page":
        summary["properties"] = {name: property_value(prop) for name, prop in (page.get("properties") or {}).items()}
    return summary


# ---------------------------------------------------------------------------
# Markdown <-> blocks
# ---------------------------------------------------------------------------

_INLINE = re.compile(
    r"(\*\*(?P<bold>.+?)\*\*|`(?P<code>[^`]+)`|\[(?P<label>[^\]]+)\]\((?P<href>[^)\s]+)\)|(?<![\w*])\*(?P<italic>[^*\s][^*]*?)\*(?![\w*]))"
)

_CODE_LANGUAGES = {
    "bash", "c", "c#", "c++", "css", "diff", "docker", "go", "graphql", "html", "java", "javascript", "json",
    "kotlin", "markdown", "php", "plain text", "powershell", "python", "r", "ruby", "rust", "scala", "shell",
    "sql", "swift", "typescript", "xml", "yaml",
}  # fmt: skip
_LANGUAGE_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "sh": "shell",
    "yml": "yaml",
    "": "plain text",
}


def _text_segments(text: str, annotations: dict[str, bool] | None = None, href: str | None = None) -> list[dict]:
    segments = []
    for start in range(0, len(text), _MAX_TEXT):
        segment: dict[str, Any] = {"type": "text", "text": {"content": text[start : start + _MAX_TEXT]}}
        if href:
            segment["text"]["link"] = {"url": href}
        if annotations:
            segment["annotations"] = annotations
        segments.append(segment)
    return segments


def _inline_segments(text: str) -> list[dict[str, Any]]:
    """Every rich_text segment for a line of Markdown, with no cap on how many."""
    parts: list[dict[str, Any]] = []
    position = 0
    for match in _INLINE.finditer(text):
        if match.start() > position:
            parts += _text_segments(text[position : match.start()])
        if match.group("bold") is not None:
            parts += _text_segments(match.group("bold"), {"bold": True})
        elif match.group("code") is not None:
            parts += _text_segments(match.group("code"), {"code": True})
        elif match.group("label") is not None:
            parts += _text_segments(match.group("label"), href=match.group("href"))
        else:
            parts += _text_segments(match.group("italic"), {"italic": True})
        position = match.end()
    if position < len(text):
        parts += _text_segments(text[position:])
    return parts


def rich_text(text: str) -> list[dict[str, Any]]:
    """Notion rich_text for one value that cannot be split (a title, a comment): **bold**, *italic*,
    `code` and [links](url), within Notion's 100 segments of 2000 characters.

    Past 100 segments the formatting goes before the text does: the trailing segments are merged
    into plain text. Text that would not fit even then raises instead of being cut.
    """
    parts = _inline_segments(text)
    if len(parts) <= _MAX_RICH_TEXT:
        return parts
    for keep in range(_MAX_RICH_TEXT - 1, -1, -1):
        tail = _text_segments("".join(p["text"]["content"] for p in parts[keep:]))
        if keep + len(tail) <= _MAX_RICH_TEXT:
            return parts[:keep] + tail
    raise ValueError(
        f"Text of {len(text):,} characters is too long for one Notion value "
        f"(at most {_MAX_RICH_TEXT * _MAX_TEXT:,}): shorten it."
    )


def _blocks(kind: str, text: str = "", **extra: Any) -> list[dict[str, Any]]:
    """One block, or several when its text needs more than Notion's 100 rich_text segments.

    Nothing is dropped: the overflow goes into continuation blocks, of the same kind for
    paragraphs and quotes, plain paragraphs after a heading or a list item.
    """
    if kind == "divider":
        return [{"object": "block", "type": kind, kind: dict(extra)}]
    parts = _inline_segments(text)
    chunks = [parts[i : i + _MAX_RICH_TEXT] for i in range(0, len(parts), _MAX_RICH_TEXT)] or [[]]
    follow = kind if kind in ("paragraph", "quote") else "paragraph"
    blocks = [{"object": "block", "type": kind, kind: {"rich_text": chunks[0], **extra}}]
    blocks += [{"object": "block", "type": follow, follow: {"rich_text": chunk}} for chunk in chunks[1:]]
    return blocks


def markdown_to_blocks(markdown: str) -> list[dict[str, Any]]:
    """Notion blocks for simple Markdown: headings, lists, to-dos, quotes, code fences, dividers, paragraphs."""
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []
    lines = markdown.replace("\r\n", "\n").split("\n")

    def flush() -> None:
        if paragraph:
            blocks.extend(_blocks("paragraph", "\n".join(paragraph)))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            language = stripped[3:].strip().lower()
            language = _LANGUAGE_ALIASES.get(language, language)
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            segments = _text_segments("\n".join(code)) or _text_segments(" ")
            # A fence past 100 segments of 2000 characters continues in further code blocks.
            for start in range(0, len(segments), _MAX_RICH_TEXT):
                blocks.append(
                    {
                        "object": "block",
                        "type": "code",
                        "code": {
                            "rich_text": segments[start : start + _MAX_RICH_TEXT],
                            "language": language if language in _CODE_LANGUAGES else "plain text",
                        },
                    }
                )
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        todo = re.match(r"^[-*]\s+\[( |x|X)\]\s+(.*)$", stripped)
        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if not stripped:
            flush()
        elif heading:
            flush()
            blocks.extend(_blocks(f"heading_{len(heading.group(1))}", heading.group(2)))
        elif re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush()
            blocks.extend(_blocks("divider"))
        elif todo:
            flush()
            blocks.extend(_blocks("to_do", todo.group(2), checked=todo.group(1).lower() == "x"))
        elif bullet:
            flush()
            blocks.extend(_blocks("bulleted_list_item", bullet.group(1)))
        elif numbered:
            flush()
            blocks.extend(_blocks("numbered_list_item", numbered.group(1)))
        elif stripped.startswith(">"):
            flush()
            blocks.extend(_blocks("quote", stripped.lstrip(">").strip()))
        else:
            paragraph.append(stripped)
        i += 1
    flush()
    return blocks


def _block_line(block: dict[str, Any], numbered_index: int) -> str | None:
    kind = block.get("type", "")
    body = block.get(kind) or {}
    text = plain_text(body.get("rich_text"))
    if kind == "paragraph":
        return text
    if kind.startswith("heading_"):
        return f"{'#' * int(kind[-1])} {text}"
    if kind == "bulleted_list_item":
        return f"- {text}"
    if kind == "numbered_list_item":
        return f"{numbered_index}. {text}"
    if kind == "to_do":
        return f"- [{'x' if body.get('checked') else ' '}] {text}"
    if kind == "toggle":
        return f"- {text}"
    if kind == "quote":
        return f"> {text}"
    if kind == "callout":
        icon = (body.get("icon") or {}).get("emoji") or ""
        return f"> {icon} {text}".replace(">  ", "> ")
    if kind == "code":
        return f"```{body.get('language') or ''}\n{text}\n```"
    if kind == "equation":
        return f"$$ {body.get('expression', '')} $$"
    if kind == "divider":
        return "---"
    if kind == "child_page":
        return f"[page: {body.get('title', '')}] ({block.get('id')})"
    if kind == "child_database":
        return f"[database: {body.get('title', '')}] ({block.get('id')})"
    if kind in ("image", "file", "pdf", "video", "audio"):
        url = (body.get("file") or body.get("external") or {}).get("url")
        caption = plain_text(body.get("caption")) or body.get("name") or kind
        return f"![{caption}]({url})" if kind == "image" else f"[{caption}]({url})"
    if kind in ("bookmark", "embed", "link_preview"):
        return body.get("url")
    if kind == "table_row":
        return "| " + " | ".join(plain_text(cell) for cell in body.get("cells", [])) + " |"
    if kind in ("table", "column_list", "column", "synced_block", "table_of_contents", "breadcrumb"):
        return None  # structure only; children carry the content
    if kind in ("meeting_notes", "transcription"):
        return f"[meeting notes] ({block.get('id')})"
    return f"[{kind}]" if kind and kind != "unsupported" else None


async def _blocks_markdown(
    tool: Any, block_id: str, *, depth: int, max_depth: int, budget: list[int]
) -> tuple[list[str], bool]:
    """Markdown lines for a block's children (recursing into nested blocks); budget[0] = blocks left."""
    lines: list[str] = []
    cursor: str | None = None
    numbered = 0
    indent = "  " * depth
    while True:
        page = await _notion_request(
            tool, "GET", f"/blocks/{block_id}/children", params={"page_size": 100, "start_cursor": cursor}
        )
        for block in page.get("results", []):
            if budget[0] <= 0:
                return lines, True
            budget[0] -= 1
            numbered = numbered + 1 if block.get("type") == "numbered_list_item" else 0
            line = _block_line(block, numbered)
            if line is not None:
                lines.extend(f"{indent}{part}" for part in line.split("\n"))
            if block.get("has_children") and block.get("type") not in ("child_page", "child_database"):
                if depth + 1 > max_depth:
                    lines.append(f"{indent}  [...nested content: {block.get('id')}]")
                    continue
                nested_depth = depth + (
                    0 if block.get("type") in ("column_list", "column", "synced_block", "table") else 1
                )
                child_lines, truncated = await _blocks_markdown(
                    tool, block["id"], depth=nested_depth, max_depth=max_depth, budget=budget
                )
                lines.extend(child_lines)
                if truncated:
                    return lines, True
        if not page.get("has_more"):
            return lines, False
        cursor = page.get("next_cursor")


async def _append_blocks(
    tool: Any, block_id: str, blocks: list[dict[str, Any]], *, written: int = 0, created: bool = False
) -> int:
    """Append ``blocks`` in batches of 100; ``written`` blocks are already there (sent with the create).

    A batch that fails after others landed raises ``NotionPartialWriteError`` saying how much is on
    the page, so the model does not repeat the whole write and duplicate it.
    """
    total = written + len(blocks)
    for start in range(0, len(blocks), _MAX_CHILDREN):
        try:
            await _notion_request(
                tool,
                "PATCH",
                f"/blocks/{block_id}/children",
                json_body={"children": blocks[start : start + _MAX_CHILDREN]},
            )
        except Exception as exc:
            done = written + start
            if done == 0:
                raise  # nothing landed: the error stands as it is
            first = blocks[start]
            resume = plain_text((first.get(first.get("type", "")) or {}).get("rich_text"))[:80]
            resume_hint = f' (the first one not written begins "{resume}")' if resume else ""
            if created:
                message = (
                    f"Notion created page {block_id} but wrote only {done} of {total} content blocks before an "
                    f"error: {exc}. The page exists: do not create it again; append the remaining content"
                    f"{resume_hint} with notion_append_content."
                )
            else:
                message = (
                    f"Appended {done} of {total} blocks to {block_id} before an error: {exc}. The first {done} "
                    f"are on the page: append only the content after them{resume_hint}, or it would be added twice."
                )
            status = getattr(exc, "status", 0)
            if not status or status >= 500:
                message += (
                    " The failed request may still have been applied: read the page with notion_get_page "
                    "before appending again."
                )
            raise NotionPartialWriteError(
                message,
                status=getattr(exc, "status", 0),
                code=getattr(exc, "code", None),
                page_id=block_id,
                blocks_written=done,
                blocks_total=total,
            ) from exc
    return total


async def _resolve_data_source(tool: Any, ref: str) -> dict[str, Any]:
    """The data source for a database or data source ID/URL (a database with several must be disambiguated)."""
    object_id = normalize_id(ref)
    try:
        database = await _notion_request(tool, "GET", f"/databases/{object_id}")
    except NotionAPIError as exc:
        if exc.code not in ("object_not_found", "validation_error"):
            raise
        return await _notion_request(tool, "GET", f"/data_sources/{object_id}")
    sources = database.get("data_sources") or []
    if len(sources) == 1:
        return await _notion_request(tool, "GET", f"/data_sources/{sources[0]['id']}")
    if not sources:
        raise ValueError(f"Database {object_id} has no data sources.")
    options = ", ".join(f"{s.get('name')!r} ({s.get('id')})" for s in sources)
    raise ValueError(f"Database {object_id} has {len(sources)} data sources; pass one of their IDs: {options}.")


def _title_property(schema: dict[str, Any]) -> str:
    for name, prop in (schema.get("properties") or {}).items():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return name
    return "title"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class _NotionTool(Tool):
    """Shared auth fields for Notion tools (OAuth public integration + internal API key)."""

    integration: Annotated[str, Integration("notion")] | None = None
    auth_method: AuthMethod | None = None
    api_key: SecretStr | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    token_file: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    notion_version: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "auth_method": self.auth_method,
                    "api_key": self.api_key,
                    "token": self.token,
                    "refresh_token": self.refresh_token,
                    "token_file": self.token_file,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "notion_version": self.notion_version,
                }
            ),
        }


class NotionWhoAmI(_NotionTool):
    name: str = "notion_whoami"
    description: str | None = (
        "Show which Notion integration and workspace these tools act as, who owns it, and how they "
        "are connected (oauth or api_key)."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _whoami() -> Any:
            auth = await _resolve_auth(self)
            me = await _notion_request(self, "GET", "/users/me")
            bot = me.get("bot") or {}
            owner = bot.get("owner") or {}
            return {
                "id": me.get("id"),
                "name": me.get("name"),
                "type": me.get("type"),
                "workspace_name": bot.get("workspace_name"),
                "workspace_id": bot.get("workspace_id"),
                "owner": "workspace" if owner.get("type") == "workspace" else _user_name(owner.get("user")),
                "auth_method": auth.method,
            }

        super().__init__(handler=_whoami, **kwargs)


class NotionSearch(_NotionTool):
    name: str = "notion_search"
    description: str | None = (
        "Search the Notion pages and databases (data sources) shared with this integration by title. "
        "An empty query lists recently edited ones."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _search(
            query: str = Field("", description="Words of the title to search for. Empty lists recent pages."),
            object_type: Literal["page", "data_source"] | None = Field(
                None, description='Only "page" or only "data_source" (database tables). Empty returns both.'
            ),
            page_size: int = Field(20, description="Results per call (1-100)."),
            start_cursor: str | None = Field(None, description="`next_cursor` from a previous call."),
        ) -> Any:
            body: dict[str, Any] = {"page_size": max(1, min(int(page_size), 100))}
            if query.strip():
                body["query"] = query.strip()
            if object_type:
                body["filter"] = {"property": "object", "value": object_type}
            if start_cursor:
                body["start_cursor"] = start_cursor
            if not query.strip():
                body["sort"] = {"timestamp": "last_edited_time", "direction": "descending"}
            result = await _notion_request(self, "POST", "/search", json_body=body)
            return {
                "results": [page_summary(item, with_properties=False) for item in result.get("results", [])],
                "next_cursor": result.get("next_cursor") if result.get("has_more") else None,
            }

        super().__init__(handler=_search, **kwargs)


class NotionGetPage(_NotionTool):
    name: str = "notion_get_page"
    description: str | None = "Read a Notion page: its properties and, by default, its content as Markdown."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_page(
            page_id: str = Field(..., description="Page ID or notion.so URL."),
            include_content: bool = Field(True, description="Also return the page body as Markdown."),
            max_blocks: int = Field(300, description="Stop reading the body after this many blocks (1-2000)."),
            max_depth: int = Field(3, description="How deep to read nested blocks (toggles, sub-lists)."),
        ) -> Any:
            object_id = normalize_id(page_id)
            page = await _notion_request(self, "GET", f"/pages/{object_id}")
            result = page_summary(page)
            if include_content:
                budget = [max(1, min(int(max_blocks), 2000))]
                lines, truncated = await _blocks_markdown(
                    self, object_id, depth=0, max_depth=max(0, int(max_depth)), budget=budget
                )
                result["content"] = "\n".join(lines).strip()
                result["truncated"] = truncated
            return result

        super().__init__(handler=_get_page, **kwargs)


class NotionGetDataSource(_NotionTool):
    name: str = "notion_get_data_source"
    description: str | None = (
        "Read a Notion database's schema: its properties (columns), their types and select options. "
        "Accepts a database or data source ID or URL. Call before querying or creating rows."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_source(
            database_id: str = Field(..., description="Database or data source ID, or its notion.so URL."),
        ) -> Any:
            source = await _resolve_data_source(self, database_id)
            properties = {}
            for name, prop in (source.get("properties") or {}).items():
                kind = prop.get("type")
                entry: dict[str, Any] = {"type": kind}
                options = (prop.get(kind) or {}).get("options") if isinstance(prop.get(kind), dict) else None
                if options:
                    entry["options"] = [o.get("name") for o in options]
                if kind == "status":
                    entry["groups"] = [g.get("name") for g in (prop.get("status") or {}).get("groups", [])]
                properties[name] = entry
            parent = source.get("parent") or {}
            return {
                "data_source_id": source.get("id"),
                "database_id": parent.get("database_id"),
                "title": title_of(source),
                "url": source.get("url"),
                "properties": properties,
            }

        super().__init__(handler=_get_data_source, **kwargs)


class NotionQueryDataSource(_NotionTool):
    name: str = "notion_query_data_source"
    description: str | None = (
        "List rows of a Notion database (data source), with optional Notion filter and sorts. "
        "Accepts a database or data source ID or URL."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _query(
            database_id: str = Field(..., description="Database or data source ID, or its notion.so URL."),
            filter: dict[str, Any] | None = Field(
                None,
                description=(
                    'Notion filter object, e.g. {"property": "Status", "status": {"equals": "Done"}} or '
                    '{"and": [...]}. Property names must match notion_get_data_source.'
                ),
            ),
            sorts: list[dict[str, Any]] | None = Field(
                None, description='Notion sorts, e.g. [{"property": "Due", "direction": "ascending"}].'
            ),
            page_size: int = Field(50, description="Rows per call (1-100)."),
            start_cursor: str | None = Field(None, description="`next_cursor` from a previous call."),
        ) -> Any:
            source = await _resolve_data_source(self, database_id)
            body: dict[str, Any] = {"page_size": max(1, min(int(page_size), 100))}
            if filter:
                body["filter"] = filter
            if sorts:
                body["sorts"] = sorts
            if start_cursor:
                body["start_cursor"] = start_cursor
            result = await _notion_request(self, "POST", f"/data_sources/{source['id']}/query", json_body=body)
            return {
                "data_source_id": source.get("id"),
                "rows": [page_summary(row) for row in result.get("results", [])],
                "next_cursor": result.get("next_cursor") if result.get("has_more") else None,
            }

        super().__init__(handler=_query, **kwargs)


class NotionCreatePage(_NotionTool):
    name: str = "notion_create_page"
    description: str | None = (
        "Create a Notion page under a page, or a row in a database, with a title, optional properties "
        "and Markdown content."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_page(
            parent: str = Field(..., description="Parent page ID/URL, or a database / data source ID/URL."),
            parent_type: Literal["page", "database"] = Field(
                "page", description='"page" for a sub-page, "database" for a new row (database or data source).'
            ),
            title: str | None = Field(None, description="Page title."),
            content: str | None = Field(
                None, description="Body in Markdown: headings, lists, to-dos, quotes, ``` code, **bold**, links."
            ),
            properties: dict[str, Any] | None = Field(
                None,
                description=(
                    "Database rows only: Notion property values, e.g. "
                    '{"Status": {"status": {"name": "Done"}}, "Due": {"date": {"start": "2026-12-01"}}}.'
                ),
            ),
        ) -> Any:
            props = dict(properties or {})
            if parent_type == "database":
                source = await _resolve_data_source(self, parent)
                parent_ref = {"type": "data_source_id", "data_source_id": source["id"]}
                if title:
                    props.setdefault(_title_property(source), {"title": rich_text(title)})
            else:
                parent_ref = {"type": "page_id", "page_id": normalize_id(parent)}
                if title:
                    props["title"] = {"title": rich_text(title)}
            blocks = markdown_to_blocks(content) if content else []
            # `parent`, not `parent_id`: Runnable reserves `parent_id` for chaining runs.
            body: dict[str, Any] = {"parent": parent_ref, "properties": props}
            if blocks:
                body["children"] = blocks[:_MAX_CHILDREN]
            page = await _notion_request(self, "POST", "/pages", json_body=body)
            if len(blocks) > _MAX_CHILDREN:
                await _append_blocks(self, page["id"], blocks[_MAX_CHILDREN:], written=_MAX_CHILDREN, created=True)
            return {**page_summary(page, with_properties=False), "blocks_written": len(blocks)}

        super().__init__(handler=_create_page, **kwargs)


class NotionUpdatePage(_NotionTool):
    name: str = "notion_update_page"
    description: str | None = (
        "Update a Notion page or database row: rename it, set properties, or move it to / restore it from the trash."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_page(
            page_id: str = Field(..., description="Page ID or notion.so URL."),
            title: str | None = Field(None, description="New title."),
            properties: dict[str, Any] | None = Field(
                None, description='Notion property values to set, e.g. {"Status": {"status": {"name": "Done"}}}.'
            ),
            in_trash: bool | None = Field(None, description="true moves the page to the trash, false restores it."),
        ) -> Any:
            object_id = normalize_id(page_id)
            props = dict(properties or {})
            if title is not None:
                current = await _notion_request(self, "GET", f"/pages/{object_id}")
                title_name = next(
                    (n for n, p in (current.get("properties") or {}).items() if p.get("type") == "title"), "title"
                )
                props[title_name] = {"title": rich_text(title)}
            body: dict[str, Any] = {}
            if props:
                body["properties"] = props
            if in_trash is not None:
                body["in_trash"] = in_trash
            if not body:
                raise ValueError("Nothing to update: pass title, properties or in_trash.")
            page = await _notion_request(self, "PATCH", f"/pages/{object_id}", json_body=body)
            return page_summary(page)

        super().__init__(handler=_update_page, **kwargs)


class NotionAppendContent(_NotionTool):
    name: str = "notion_append_content"
    description: str | None = "Append Markdown content to the end of a Notion page (or inside a block)."

    def __init__(self, **kwargs: Any) -> None:
        async def _append(
            page_id: str = Field(..., description="Page or block ID, or a notion.so URL."),
            content: str = Field(
                ..., description="Markdown: headings, lists, to-dos, quotes, ``` code, **bold**, *italic*, links."
            ),
        ) -> Any:
            blocks = markdown_to_blocks(content)
            if not blocks:
                raise ValueError("Nothing to append: content is empty.")
            object_id = normalize_id(page_id)
            written = await _append_blocks(self, object_id, blocks)
            return {"id": object_id, "blocks_written": written}

        super().__init__(handler=_append, **kwargs)


class NotionAddComment(_NotionTool):
    name: str = "notion_add_comment"
    description: str | None = "Add a comment to a Notion page, or reply in an existing discussion."

    def __init__(self, **kwargs: Any) -> None:
        async def _add_comment(
            text: str = Field(..., description="Comment text (Markdown inline formatting allowed)."),
            page_id: str | None = Field(None, description="Page ID or URL, for a new top-level comment."),
            discussion_id: str | None = Field(None, description="Discussion ID, to reply in that thread."),
        ) -> Any:
            if bool(page_id) == bool(discussion_id):
                raise ValueError("Pass exactly one of page_id or discussion_id.")
            body: dict[str, Any] = {"rich_text": rich_text(text)}
            if discussion_id:
                body["discussion_id"] = discussion_id
            else:
                body["parent"] = {"page_id": normalize_id(page_id or "")}
            comment = await _notion_request(self, "POST", "/comments", json_body=body)
            return {"id": comment.get("id"), "discussion_id": comment.get("discussion_id")}

        super().__init__(handler=_add_comment, **kwargs)


class NotionListComments(_NotionTool):
    name: str = "notion_list_comments"
    description: str | None = "List the open comments on a Notion page or block."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_comments(
            page_id: str = Field(..., description="Page or block ID, or a notion.so URL."),
            page_size: int = Field(50, description="Comments per call (1-100)."),
            start_cursor: str | None = Field(None, description="`next_cursor` from a previous call."),
        ) -> Any:
            result = await _notion_request(
                self,
                "GET",
                "/comments",
                params={
                    "block_id": normalize_id(page_id),
                    "page_size": max(1, min(int(page_size), 100)),
                    "start_cursor": start_cursor,
                },
            )
            return {
                "comments": [
                    {
                        "id": c.get("id"),
                        "discussion_id": c.get("discussion_id"),
                        "author": (c.get("created_by") or {}).get("id"),
                        "created_time": c.get("created_time"),
                        "text": plain_text(c.get("rich_text")),
                    }
                    for c in result.get("results", [])
                ],
                "next_cursor": result.get("next_cursor") if result.get("has_more") else None,
            }

        super().__init__(handler=_list_comments, **kwargs)
