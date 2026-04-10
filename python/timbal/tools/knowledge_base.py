"""Platform knowledge base — SQL query as a configurable Tool (K2 by default; optional legacy API)."""

import re
from typing import Any

from pydantic import Field

from ..core.tool import Tool
from ..platform.knowledge_bases import query as kb_query

_MAX_TOOL_NAME_LEN = 64

_DEFAULT_DESCRIPTION = (
    "Execute SQL against the configured knowledge base. Returns JSON (typically a 'rows' field). "
    "Requires platform authentication or environment-backed defaults."
)


def _slug_kb_id_for_tool_name(kb_id: str, *, max_len: int) -> str:
    """Stable snake_case fragment from ``kb_id`` for use inside tool names."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", kb_id.strip()).strip("_").lower()
    if not slug:
        return "kb"[:max_len]
    if slug[0].isdigit():
        slug = "kb_" + slug
    return slug[:max_len]


def _default_tool_name(*, kb_id: str | None) -> str:
    """``knowledge_base_<slug>_query`` when ``kb_id`` is set; else ``knowledge_base_query`` (env KB)."""
    if not kb_id:
        return "knowledge_base_query"
    prefix = "knowledge_base_"
    suffix = "_query"
    max_frag = max(1, _MAX_TOOL_NAME_LEN - len(prefix) - len(suffix))
    frag = _slug_kb_id_for_tool_name(kb_id, max_len=max_frag)
    return f"{prefix}{frag}{suffix}"


def _auto_description(*, about: str | None) -> str:
    """LLM-facing description: no org/kb identifiers (those are constructor-only)."""
    if about and about.strip():
        return f"{_DEFAULT_DESCRIPTION} When to use: {about.strip()}"
    return _DEFAULT_DESCRIPTION


class KnowledgeBaseQuery(Tool):
    """Run SQL against a Timbal platform knowledge base with constructor-bound org/kb defaults.

    Use ``legacy=True`` for the older ``kbs`` HTTP API; default is K2 (``k2``). The model only
    receives ``sql``, ``params``, and ``explain``; ``explain`` is ignored when ``legacy=True``.

    **Multiple KBs on one agent**

    :class:`Agent` rejects duplicate tool ``name`` values. If you omit ``name`` and set ``kb_id``,
    the tool name is ``knowledge_base_<slug>_query`` (slug derived from ``kb_id``) so different ids
    do not collide. With no ``kb_id``, the name is ``knowledge_base_query`` (single KB from env /
    platform). Two instances with the same ``kb_id`` still collide unless you pass distinct ``name=``.

    Pass ``about`` (or a custom ``description``) so the model can tell which KB covers which topic.
    The model never receives ``org_id`` or ``kb_id`` as tool parameters; bind those at construction.
    """

    name: str = "knowledge_base_query"
    description: str | None = _DEFAULT_DESCRIPTION
    kb_id: str | None = None
    org_id: str | None = None
    legacy: bool = False
    about: str | None = None
    """Short hint for the LLM (e.g. "HR policies and PTO"); merged into the default description."""

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "kb_id": self.kb_id,
                    "org_id": self.org_id,
                    "legacy": self.legacy,
                    "about": self.about,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        kb_id = kwargs.get("kb_id")
        about = kwargs.get("about")

        if "name" not in kwargs:
            kwargs["name"] = _default_tool_name(kb_id=kb_id)

        if "description" not in kwargs:
            kwargs["description"] = _auto_description(about=about)

        async def _kb_query(
            sql: str = Field(..., description="SQL to execute against the knowledge base."),
            params: list[Any] | None = Field(
                None,
                description="Optional list of bound parameters for the SQL query.",
            ),
            explain: bool | None = Field(
                None,
                description=(
                    "K2 only: include EXPLAIN for this query. Omitted when unset. "
                    "Ignored when this tool is configured with legacy mode."
                ),
            ),
        ) -> Any:
            eff_explain = None if self.legacy else explain
            return await kb_query(
                sql,
                params,
                org_id=self.org_id,
                kb_id=self.kb_id,
                legacy=self.legacy,
                explain=eff_explain,
            )

        super().__init__(handler=_kb_query, **kwargs)
