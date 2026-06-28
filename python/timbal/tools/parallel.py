from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

_BASE_URL = "https://api.parallel.ai"


class ParallelSearch(Tool):
    name: str = "parallel_search"
    description: str | None = (
        "Search the web with Parallel. Takes a natural-language objective and/or keyword "
        "queries and returns ranked results with LLM-optimized excerpts, titles, and URLs."
    )
    integration: Annotated[str, Integration("parallel")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _parallel_search(
            search_queries: list[str] = Field(..., description="Concise keyword search queries, 3-6 words each"),
            objective: str | None = Field(None, description="Natural-language description of the underlying question or goal"),
            mode: str = Field("advanced", description='Search mode: "turbo", "basic", or "advanced"'),
            max_results: int = Field(10, description="Maximum number of results to return"),
            max_chars_total: int | None = Field(None, description="Upper bound on total excerpt characters across results"),
            include_domains: list[str] | None = Field(None, description="Only include results from these domains"),
            exclude_domains: list[str] | None = Field(None, description="Exclude results from these domains"),
            location: str | None = Field(None, description="ISO 3166-1 alpha-2 country code to bias results (e.g. 'US')"),
        ) -> dict:
            api_key = await resolve_api_key(tool=self, provider_name="Parallel", env_var="PARALLEL_API_KEY")
            import httpx

            advanced_settings: dict[str, Any] = {"max_results": max_results}
            if location:
                advanced_settings["location"] = location
            source_policy: dict[str, Any] = {}
            if include_domains:
                source_policy["include_domains"] = include_domains
            if exclude_domains:
                source_policy["exclude_domains"] = exclude_domains
            if source_policy:
                advanced_settings["source_policy"] = source_policy

            payload: dict[str, Any] = {
                "search_queries": search_queries,
                "mode": mode,
                "advanced_settings": advanced_settings,
            }
            if objective:
                payload["objective"] = objective
            if max_chars_total is not None:
                payload["max_chars_total"] = max_chars_total

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/v1/search",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(60.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_parallel_search, **kwargs)


class ParallelExtract(Tool):
    name: str = "parallel_extract"
    description: str | None = (
        "Extract content from specific web pages with Parallel. Provide up to 20 URLs and get back "
        "LLM-optimized excerpts (and optionally full page content) for each, ranked against an "
        "optional objective and queries."
    )
    integration: Annotated[str, Integration("parallel")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _parallel_extract(
            urls: list[str] = Field(..., description="URLs to extract content from (up to 20)"),
            objective: str | None = Field(None, description="Natural-language description of what to extract from the pages"),
            search_queries: list[str] | None = Field(None, description="Optional keyword queries to rank excerpts against"),
            max_chars_total: int | None = Field(None, description="Upper bound on total characters across all excerpts"),
            max_chars_per_result: int | None = Field(None, description="Upper bound on excerpt characters per URL"),
            full_content: bool = Field(False, description="Return the full cleaned page content in addition to excerpts"),
        ) -> dict:
            api_key = await resolve_api_key(tool=self, provider_name="Parallel", env_var="PARALLEL_API_KEY")
            import httpx

            advanced_settings: dict[str, Any] = {}
            if max_chars_per_result is not None:
                advanced_settings["excerpt_settings"] = {"max_chars_per_result": max_chars_per_result}
            if full_content:
                advanced_settings["full_content"] = True

            payload: dict[str, Any] = {"urls": urls}
            if objective:
                payload["objective"] = objective
            if search_queries:
                payload["search_queries"] = search_queries
            if max_chars_total is not None:
                payload["max_chars_total"] = max_chars_total
            if advanced_settings:
                payload["advanced_settings"] = advanced_settings

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_BASE_URL}/v1/extract",
                    headers={"x-api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=httpx.Timeout(120.0, read=None),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_parallel_extract, **kwargs)


class ParallelTaskRun(Tool):
    name: str = "parallel_task_run"
    description: str | None = (
        "Run a Parallel research task: give a plain-language objective and Parallel performs web "
        "research and returns an answer with citations. Optionally pass a JSON schema to get "
        "structured output. Blocks until the run completes."
    )
    integration: Annotated[str, Integration("parallel")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _parallel_task_run(
            input: str | dict = Field(..., description="The research objective as text, or a structured input object"),
            processor: str = Field("base", description='Depth/latency tier: "lite", "base", "core", "pro", or "ultra" (append "-fast" for low-latency variants, e.g. "ultra-fast")'),
            output_schema: dict | None = Field(None, description="Optional JSON schema for structured output. Omit to get a text answer."),
            output_description: str | None = Field(None, description="Natural-language description of the desired output when not using a JSON schema"),
        ) -> dict:
            api_key = await resolve_api_key(tool=self, provider_name="Parallel", env_var="PARALLEL_API_KEY")
            import httpx

            task_spec: dict[str, Any] = {}
            if output_schema is not None:
                task_spec["output_schema"] = {"type": "json", "json_schema": output_schema}
            elif output_description is not None:
                task_spec["output_schema"] = {"type": "text", "description": output_description}

            payload: dict[str, Any] = {"input": input, "processor": processor}
            if task_spec:
                payload["task_spec"] = task_spec

            headers = {"x-api-key": api_key, "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                create = await client.post(
                    f"{_BASE_URL}/v1/tasks/runs",
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(60.0, read=None),
                )
                create.raise_for_status()
                run_id = create.json()["run_id"]

                # Blocks until the run is completed.
                result = await client.get(
                    f"{_BASE_URL}/v1/tasks/runs/{run_id}/result",
                    headers=headers,
                    timeout=httpx.Timeout(600.0, read=None),
                )
                result.raise_for_status()
                return result.json()

        super().__init__(handler=_parallel_task_run, **kwargs)
