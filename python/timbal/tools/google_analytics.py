import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._google_oauth import auth_headers, config_fields, resolve_google_token

_DATA_BASE = "https://analyticsdata.googleapis.com/v1beta"
_ADMIN_BASE = "https://analyticsadmin.googleapis.com/v1beta"
_REFRESH_TOKEN_ENV = "GOOGLE_ANALYTICS_REFRESH_TOKEN"


def _normalize_property_id(property_id: str) -> str:
    if property_id.startswith("properties/"):
        return property_id
    return f"properties/{property_id}"


def _resolve_property_id(tool: Any, property_id: str | None) -> str:
    resolved = property_id or tool.default_property_id or os.getenv("GOOGLE_ANALYTICS_PROPERTY_ID")
    if not resolved:
        raise ValueError(
            "GA4 property_id is required. Pass property_id, set default_property_id on the tool, "
            "or set GOOGLE_ANALYTICS_PROPERTY_ID."
        )
    return _normalize_property_id(resolved)


async def _resolve_token(tool: Any) -> str:
    return await resolve_google_token(
        provider_name="Google Analytics",
        integration=tool.integration,
        token=tool.token,
        refresh_token=tool.refresh_token,
        refresh_token_env=_REFRESH_TOKEN_ENV,
        access_token_env="GOOGLE_ANALYTICS_ACCESS_TOKEN",
    )


class GoogleAnalyticsListAccountSummaries(Tool):
    name: str = "google_analytics_list_account_summaries"
    description: str | None = "List GA4 account summaries and accessible properties."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_account_summaries(
            page_size: int = Field(200, description="Maximum number of account summaries to return"),
            page_token: str | None = Field(None, description="Pagination token from a previous response"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ADMIN_BASE}/accountSummaries",
                    headers=auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_account_summaries, **kwargs)


class GoogleAnalyticsRunReport(Tool):
    name: str = "google_analytics_run_report"
    description: str | None = "Run a standard GA4 report with dimensions, metrics, and date ranges."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _run_report(
            property_id: str | None = Field(None, description="GA4 property ID, e.g. 123456789"),
            start_date: str = Field(..., description="Start date (YYYY-MM-DD) or relative like '7daysAgo'"),
            end_date: str = Field(..., description="End date (YYYY-MM-DD) or relative like 'today'"),
            dimensions: list[str] = Field(default_factory=list, description="Dimension API names"),
            metrics: list[str] = Field(..., description="Metric API names"),
            limit: int = Field(1000, description="Maximum rows to return"),
            offset: int = Field(0, description="Row offset for pagination"),
            dimension_filter: dict | None = Field(None, description="Dimension filter expression object"),
            metric_filter: dict | None = Field(None, description="Metric filter expression object"),
            order_bys: list[dict] | None = Field(None, description="Order by clauses"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            body: dict[str, Any] = {
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "metrics": [{"name": m} for m in metrics],
                "limit": limit,
                "offset": offset,
            }
            if dimensions:
                body["dimensions"] = [{"name": d} for d in dimensions]
            if dimension_filter:
                body["dimensionFilter"] = dimension_filter
            if metric_filter:
                body["metricFilter"] = metric_filter
            if order_bys:
                body["orderBys"] = order_bys

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DATA_BASE}/{prop}:runReport",
                    headers=auth_headers(token),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_run_report, **kwargs)


class GoogleAnalyticsRunRealtimeReport(Tool):
    name: str = "google_analytics_run_realtime_report"
    description: str | None = "Run a GA4 realtime report for active users and events."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _run_realtime_report(
            property_id: str | None = Field(None, description="GA4 property ID"),
            dimensions: list[str] = Field(default_factory=list, description="Realtime dimension API names"),
            metrics: list[str] = Field(..., description="Realtime metric API names"),
            limit: int = Field(1000, description="Maximum rows to return"),
            dimension_filter: dict | None = Field(None, description="Dimension filter expression object"),
            metric_filter: dict | None = Field(None, description="Metric filter expression object"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            body: dict[str, Any] = {
                "metrics": [{"name": m} for m in metrics],
                "limit": limit,
            }
            if dimensions:
                body["dimensions"] = [{"name": d} for d in dimensions]
            if dimension_filter:
                body["dimensionFilter"] = dimension_filter
            if metric_filter:
                body["metricFilter"] = metric_filter

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DATA_BASE}/{prop}:runRealtimeReport",
                    headers=auth_headers(token),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_run_realtime_report, **kwargs)


class GoogleAnalyticsRunPivotReport(Tool):
    name: str = "google_analytics_run_pivot_report"
    description: str | None = "Run a GA4 pivot report."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _run_pivot_report(
            property_id: str | None = Field(None, description="GA4 property ID"),
            start_date: str = Field(..., description="Start date (YYYY-MM-DD) or relative"),
            end_date: str = Field(..., description="End date (YYYY-MM-DD) or relative"),
            dimensions: list[str] = Field(default_factory=list, description="Dimension API names"),
            metrics: list[str] = Field(..., description="Metric API names"),
            pivots: list[dict] = Field(..., description="Pivot definitions"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            body: dict[str, Any] = {
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "metrics": [{"name": m} for m in metrics],
                "pivots": pivots,
            }
            if dimensions:
                body["dimensions"] = [{"name": d} for d in dimensions]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DATA_BASE}/{prop}:runPivotReport",
                    headers=auth_headers(token),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_run_pivot_report, **kwargs)


class GoogleAnalyticsBatchRunReports(Tool):
    name: str = "google_analytics_batch_run_reports"
    description: str | None = "Run multiple GA4 reports in a single batch request."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _batch_run_reports(
            property_id: str | None = Field(None, description="GA4 property ID"),
            requests: list[dict] = Field(..., description="List of RunReportRequest objects"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_DATA_BASE}/{prop}:batchRunReports",
                    headers=auth_headers(token),
                    json={"requests": requests},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_batch_run_reports, **kwargs)


class GoogleAnalyticsGetMetadata(Tool):
    name: str = "google_analytics_get_metadata"
    description: str | None = "Get available GA4 dimensions and metrics metadata for a property."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_metadata(
            property_id: str | None = Field(None, description="GA4 property ID"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_DATA_BASE}/{prop}/metadata",
                    headers=auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_metadata, **kwargs)


class GoogleAnalyticsListProperties(Tool):
    name: str = "google_analytics_list_properties"
    description: str | None = "List GA4 properties under an account."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **config_fields(self)}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_properties(
            account_id: str = Field(..., description="Account ID, e.g. 123456789 or accounts/123456789"),
            page_size: int = Field(200, description="Maximum properties to return"),
            page_token: str | None = Field(None, description="Pagination token"),
            show_deleted: bool = Field(False, description="Include soft-deleted properties"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            account = account_id if account_id.startswith("accounts/") else f"accounts/{account_id}"
            params: dict[str, Any] = {
                "filter": f"parent:{account}",
                "pageSize": page_size,
                "showDeleted": show_deleted,
            }
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ADMIN_BASE}/properties",
                    headers=auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_properties, **kwargs)


class GoogleAnalyticsGetProperty(Tool):
    name: str = "google_analytics_get_property"
    description: str | None = "Get GA4 property details."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_property(
            property_id: str | None = Field(None, description="GA4 property ID"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ADMIN_BASE}/{prop}",
                    headers=auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_property, **kwargs)


class GoogleAnalyticsListDataStreams(Tool):
    name: str = "google_analytics_list_data_streams"
    description: str | None = "List GA4 data streams for a property."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_streams(
            property_id: str | None = Field(None, description="GA4 property ID"),
            page_size: int = Field(200, description="Maximum data streams to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ADMIN_BASE}/{prop}/dataStreams",
                    headers=auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_data_streams, **kwargs)


class GoogleAnalyticsGetDataStream(Tool):
    name: str = "google_analytics_get_data_stream"
    description: str | None = "Get a GA4 data stream by ID."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_stream(
            property_id: str | None = Field(None, description="GA4 property ID"),
            data_stream_id: str = Field(..., description="Data stream ID"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            stream = data_stream_id if data_stream_id.startswith("dataStreams/") else f"dataStreams/{data_stream_id}"
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ADMIN_BASE}/{prop}/{stream}",
                    headers=auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_data_stream, **kwargs)


class GoogleAnalyticsListCustomDimensions(Tool):
    name: str = "google_analytics_list_custom_dimensions"
    description: str | None = "List custom dimensions for a GA4 property."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_custom_dimensions(
            property_id: str | None = Field(None, description="GA4 property ID"),
            page_size: int = Field(200, description="Maximum items to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ADMIN_BASE}/{prop}/customDimensions",
                    headers=auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_custom_dimensions, **kwargs)


class GoogleAnalyticsListCustomMetrics(Tool):
    name: str = "google_analytics_list_custom_metrics"
    description: str | None = "List custom metrics for a GA4 property."
    integration: Annotated[str, Integration("google_analytics")] | None = None
    token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    default_property_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **config_fields(self, extra={"default_property_id": self.default_property_id}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_custom_metrics(
            property_id: str | None = Field(None, description="GA4 property ID"),
            page_size: int = Field(200, description="Maximum items to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            prop = _resolve_property_id(self, property_id)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_ADMIN_BASE}/{prop}/customMetrics",
                    headers=auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_custom_metrics, **kwargs)
