"""Knowledge base SQL query — mirrors ``timbal-sdk`` ``src/lib/functions/query.ts``.

* ``query(sql, params?, org_id=..., kb_id=..., legacy=..., explain=...)``
* K2: ``POST /orgs/{org_id}/k2/{kb_id}/query``
* Legacy: ``POST /orgs/{org_id}/kbs/{kb_id}/query`` (response list is wrapped as ``{"rows": ...}``)

``org_id`` / ``kb_id`` resolution matches the SDK: explicit kwargs, then run-context
``platform_config.subject.org_id`` (org only), then ``TIMBAL_ORG_ID`` / ``TIMBAL_KB_ID``.
"""

from __future__ import annotations

import os
from typing import Any

from ...state import get_or_create_run_context
from ..utils import _request


def _resolve_org_id(explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    ctx = get_or_create_run_context()
    if ctx.platform_config and ctx.platform_config.subject:
        return ctx.platform_config.subject.org_id
    return os.environ.get("TIMBAL_ORG_ID")


def _resolve_kb_id(explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    return os.environ.get("TIMBAL_KB_ID")


async def query(
    sql: str,
    params: list[Any] | None = None,
    *,
    org_id: str | None = None,
    kb_id: str | None = None,
    legacy: bool = False,
    explain: bool | None = None,
) -> Any:
    """Execute SQL against a knowledge base; returns parsed JSON (K2 typically includes ``rows``)."""
    oid = _resolve_org_id(org_id)
    kid = _resolve_kb_id(kb_id)
    if not oid:
        raise ValueError(
            "org_id is required. Pass org_id=, set platform_config.subject.org_id, or set TIMBAL_ORG_ID."
        )
    if not kid:
        raise ValueError("kb_id is required. Pass kb_id= or set TIMBAL_KB_ID.")

    segment = "kbs" if legacy else "k2"
    path = f"orgs/{oid}/{segment}/{kid}/query"
    body: dict[str, Any] = {"sql": sql, "params": [] if params is None else list(params)}
    if not legacy and explain is not None:
        body["explain"] = explain

    res = await _request("POST", path, json=body)
    data = res.json()
    if legacy and isinstance(data, list):
        return {"rows": data}
    return data
