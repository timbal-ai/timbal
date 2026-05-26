from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_DEFAULT_BASE_URL = "https://api.cala.ai/v1"
# Back-compat alias for web_search imports
_BASE_URL = _DEFAULT_BASE_URL


async def _resolve_api_key(*, integration: Any = None, api_key: SecretStr | None = None) -> str:
    """Resolve Cala API key from integration, explicit field, or env var."""
    from ._creds import resolve_api_key

    return await resolve_api_key(
        env_var="CALA_API_KEY",
        provider_name="Cala",
        integration=integration,
        api_key=api_key,
    )


def _normalize_base_url(base_url: str | None) -> str:
    return (base_url or _DEFAULT_BASE_URL).rstrip("/")


async def _post_cala(
    *,
    path: str,
    api_key: str,
    base_url: str,
    body: dict[str, Any],
) -> dict:
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_normalize_base_url(base_url)}{path}",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=body,
            timeout=httpx.Timeout(10.0, read=None),
        )
        response.raise_for_status()
        return response.json()


def _search_request_body(
    *,
    input: str,
    explainability: bool = True,
    return_entities: bool = True,
) -> dict[str, Any]:
    return {
        "input": input,
        "explainability": explainability,
        "return_entities": return_entities,
    }


def _query_request_body(
    *,
    input: str,
    return_entities: bool = True,
) -> dict[str, Any]:
    return {
        "input": input,
        "return_entities": return_entities,
    }


class _CalaKnowledgeTool(Tool):
    integration: Annotated[str, Integration("cala")] | None = None
    api_key: SecretStr | None = None
    base_url: str = _DEFAULT_BASE_URL

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "api_key": self.api_key,
                    "base_url": self.base_url,
                }
            ),
        }


class CalaSearch(_CalaKnowledgeTool):
    name: str = "cala_search"
    description: str | None = (
        "Get a succinct natural-language answer in markdown with explainability, source citations, "
        "and matching entities. Accepts Cala QL or a natural-language question. "
        "Use CalaQuery for structured tabular rows instead."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _cala_search(
            input: str = Field(
                ...,
                description="Cala QL expression or natural-language question (API field: input)",
            ),
            explainability: bool = True,
            return_entities: bool = True,
        ) -> dict:
            api_key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            return await _post_cala(
                path="/knowledge/search",
                api_key=api_key,
                base_url=self.base_url,
                body=_search_request_body(
                    input=input,
                    explainability=explainability,
                    return_entities=return_entities,
                ),
            )

        super().__init__(handler=_cala_search, **kwargs)


class CalaQuery(_CalaKnowledgeTool):
    name: str = "cala_query"
    description: str | None = (
        "Get structured JSON rows plus matching entities from Cala's knowledge base. "
        "Accepts Cala QL dot-notation filters (e.g. startups.location=Spain.funding>10M) "
        "or a natural-language question. Use CalaSearch for markdown answers with sources."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _cala_query(
            input: str = Field(
                ...,
                description="Cala QL expression or natural-language question (API field: input)",
            ),
            return_entities: bool = True,
        ) -> dict:
            api_key = await _resolve_api_key(integration=self.integration, api_key=self.api_key)
            return await _post_cala(
                path="/knowledge/query",
                api_key=api_key,
                base_url=self.base_url,
                body=_query_request_body(input=input, return_entities=return_entities),
            )

        super().__init__(handler=_cala_query, **kwargs)
