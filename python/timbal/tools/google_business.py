import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_ACCOUNT_BASE = "https://mybusinessaccountmanagement.googleapis.com/v1"
_INFO_BASE = "https://mybusinessbusinessinformation.googleapis.com/v1"
_PERFORMANCE_BASE = "https://businessprofileperformance.googleapis.com/v1"
_LEGACY_BASE = "https://mybusiness.googleapis.com/v4"


def _normalize_account_id(account_id: str) -> str:
    if account_id.startswith("accounts/"):
        return account_id
    return f"accounts/{account_id}"


def _normalize_location_id(location_id: str) -> str:
    if location_id.startswith("locations/"):
        return location_id
    return f"locations/{location_id}"


def _resolve_account_id(tool: Any, account_id: str | None) -> str:
    resolved = account_id or tool.default_account_id or os.getenv("GOOGLE_BUSINESS_ACCOUNT_ID")
    if not resolved:
        raise ValueError(
            "Business Profile account_id is required. Pass account_id, set default_account_id on the tool, "
            "or set GOOGLE_BUSINESS_ACCOUNT_ID."
        )
    return _normalize_account_id(resolved)


def _resolve_location_id(tool: Any, location_id: str | None) -> str:
    resolved = location_id or tool.default_location_id or os.getenv("GOOGLE_BUSINESS_LOCATION_ID")
    if not resolved:
        raise ValueError(
            "Business Profile location_id is required. Pass location_id, set default_location_id on the tool, "
            "or set GOOGLE_BUSINESS_LOCATION_ID."
        )
    return _normalize_location_id(resolved)


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError("Google Business Profile credentials not found. Configure an integration or pass token.")


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class GoogleBusinessListAccounts(Tool):
    name: str = "google_business_list_accounts"
    description: str | None = "List Google Business Profile accounts accessible to the user."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config({"integration": self.integration, "token": self.token})}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_accounts(
            page_size: int = Field(20, description="Maximum accounts to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_ACCOUNT_BASE}/accounts",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_accounts, **kwargs)


class GoogleBusinessListLocations(Tool):
    name: str = "google_business_list_locations"
    description: str | None = "List Business Profile locations for an account."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_account_id: str | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "token": self.token,
                    "default_account_id": self.default_account_id,
                    "default_location_id": self.default_location_id,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_locations(
            account_id: str | None = Field(None, description="Business Profile account ID"),
            page_size: int = Field(100, description="Maximum locations to return"),
            page_token: str | None = Field(None, description="Pagination token"),
            read_mask: str = Field(
                "name,title,storefrontAddress,phoneNumbers,websiteUri,categories",
                description="Comma-separated fields to return",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            account = _resolve_account_id(self, account_id)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size, "readMask": read_mask}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_INFO_BASE}/{account}/locations",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_locations, **kwargs)


class GoogleBusinessGetLocation(Tool):
    name: str = "google_business_get_location"
    description: str | None = "Get Business Profile location details."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_account_id: str | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "token": self.token,
                    "default_account_id": self.default_account_id,
                    "default_location_id": self.default_location_id,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_location(
            account_id: str | None = Field(None, description="Business Profile account ID"),
            location_id: str | None = Field(None, description="Business Profile location ID"),
            read_mask: str = Field(
                "name,title,storefrontAddress,phoneNumbers,websiteUri,categories,profile,regularHours",
                description="Comma-separated fields to return",
            ),
        ) -> Any:
            token = await _resolve_token(self)
            account = _resolve_account_id(self, account_id)
            location = _resolve_location_id(self, location_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_INFO_BASE}/{account}/{location}",
                    headers=_auth_headers(token),
                    params={"readMask": read_mask},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_location, **kwargs)


class GoogleBusinessUpdateLocation(Tool):
    name: str = "google_business_update_location"
    description: str | None = "Update Business Profile location fields."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_account_id: str | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "token": self.token,
                    "default_account_id": self.default_account_id,
                    "default_location_id": self.default_location_id,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_location(
            location: dict = Field(..., description="Location resource with name and fields to update"),
            update_mask: str = Field(..., description="Comma-separated fields to update, e.g. title,websiteUri"),
            validate_only: bool = Field(False, description="Validate the request without applying changes"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"updateMask": update_mask}
            if validate_only:
                params["validateOnly"] = "true"

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.patch(
                    f"{_INFO_BASE}/{location['name']}",
                    headers=_auth_headers(token),
                    params=params,
                    json=location,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_location, **kwargs)


class GoogleBusinessListReviews(Tool):
    name: str = "google_business_list_reviews"
    description: str | None = "List reviews for a Business Profile location."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_account_id: str | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "token": self.token,
                    "default_account_id": self.default_account_id,
                    "default_location_id": self.default_location_id,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_reviews(
            account_id: str | None = Field(None, description="Business Profile account ID"),
            location_id: str | None = Field(None, description="Business Profile location ID"),
            page_size: int = Field(50, description="Maximum reviews to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            account = _resolve_account_id(self, account_id)
            location = _resolve_location_id(self, location_id)
            import httpx

            params: dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_LEGACY_BASE}/{account}/{location}/reviews",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_reviews, **kwargs)


class GoogleBusinessGetReview(Tool):
    name: str = "google_business_get_review"
    description: str | None = "Get a specific Business Profile review."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_account_id: str | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "token": self.token,
                    "default_account_id": self.default_account_id,
                    "default_location_id": self.default_location_id,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_review(
            review_id: str = Field(..., description="Review ID"),
            account_id: str | None = Field(None, description="Business Profile account ID"),
            location_id: str | None = Field(None, description="Business Profile location ID"),
        ) -> Any:
            token = await _resolve_token(self)
            account = _resolve_account_id(self, account_id)
            location = _resolve_location_id(self, location_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_LEGACY_BASE}/{account}/{location}/reviews/{review_id}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_review, **kwargs)


class GoogleBusinessReplyToReview(Tool):
    name: str = "google_business_reply_to_review"
    description: str | None = "Reply to a Business Profile review."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_account_id: str | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "token": self.token,
                    "default_account_id": self.default_account_id,
                    "default_location_id": self.default_location_id,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _reply_to_review(
            review_id: str = Field(..., description="Review ID"),
            comment: str = Field(..., description="Reply text"),
            account_id: str | None = Field(None, description="Business Profile account ID"),
            location_id: str | None = Field(None, description="Business Profile location ID"),
        ) -> Any:
            token = await _resolve_token(self)
            account = _resolve_account_id(self, account_id)
            location = _resolve_location_id(self, location_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.put(
                    f"{_LEGACY_BASE}/{account}/{location}/reviews/{review_id}/reply",
                    headers=_auth_headers(token),
                    json={"comment": comment},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_reply_to_review, **kwargs)


class GoogleBusinessDeleteReviewReply(Tool):
    name: str = "google_business_delete_review_reply"
    description: str | None = "Delete the reply to a Business Profile review."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_account_id: str | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "token": self.token,
                    "default_account_id": self.default_account_id,
                    "default_location_id": self.default_location_id,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_review_reply(
            review_id: str = Field(..., description="Review ID"),
            account_id: str | None = Field(None, description="Business Profile account ID"),
            location_id: str | None = Field(None, description="Business Profile location ID"),
        ) -> Any:
            token = await _resolve_token(self)
            account = _resolve_account_id(self, account_id)
            location = _resolve_location_id(self, location_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.delete(
                    f"{_LEGACY_BASE}/{account}/{location}/reviews/{review_id}/reply",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return {"deleted": True, "reviewId": review_id}

        super().__init__(handler=_delete_review_reply, **kwargs)


class GoogleBusinessGetDailyMetrics(Tool):
    name: str = "google_business_get_daily_metrics"
    description: str | None = "Get daily performance metrics for a Business Profile location."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_location_id": self.default_location_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_daily_metrics(
            location_id: str | None = Field(None, description="Business Profile location ID"),
            daily_metric: str = Field(
                "BUSINESS_IMPRESSIONS_DESKTOP_MAPS",
                description="Daily metric enum, e.g. BUSINESS_IMPRESSIONS_DESKTOP_MAPS",
            ),
            daily_range_start_year: int = Field(..., description="Start year"),
            daily_range_start_month: int = Field(..., description="Start month (1-12)"),
            daily_range_start_day: int = Field(..., description="Start day (1-31)"),
            daily_range_end_year: int = Field(..., description="End year"),
            daily_range_end_month: int = Field(..., description="End month (1-12)"),
            daily_range_end_day: int = Field(..., description="End day (1-31)"),
        ) -> Any:
            token = await _resolve_token(self)
            location = _resolve_location_id(self, location_id)
            import httpx

            body = {
                "dailyMetric": daily_metric,
                "dailyRange": {
                    "startDate": {
                        "year": daily_range_start_year,
                        "month": daily_range_start_month,
                        "day": daily_range_start_day,
                    },
                    "endDate": {
                        "year": daily_range_end_year,
                        "month": daily_range_end_month,
                        "day": daily_range_end_day,
                    },
                },
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_PERFORMANCE_BASE}/{location}:getDailyMetricsTimeSeries",
                    headers=_auth_headers(token),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_daily_metrics, **kwargs)


class GoogleBusinessGetMultiDailyMetrics(Tool):
    name: str = "google_business_get_multi_daily_metrics"
    description: str | None = "Get multiple daily performance metrics time series for a location."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_location_id": self.default_location_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_multi_daily_metrics(
            location_id: str | None = Field(None, description="Business Profile location ID"),
            daily_metrics: list[str] = Field(..., description="List of daily metric enum values"),
            daily_range_start_year: int = Field(..., description="Start year"),
            daily_range_start_month: int = Field(..., description="Start month (1-12)"),
            daily_range_start_day: int = Field(..., description="Start day (1-31)"),
            daily_range_end_year: int = Field(..., description="End year"),
            daily_range_end_month: int = Field(..., description="End month (1-12)"),
            daily_range_end_day: int = Field(..., description="End day (1-31)"),
        ) -> Any:
            token = await _resolve_token(self)
            location = _resolve_location_id(self, location_id)
            import httpx

            body = {
                "dailyMetrics": daily_metrics,
                "dailyRange": {
                    "startDate": {
                        "year": daily_range_start_year,
                        "month": daily_range_start_month,
                        "day": daily_range_start_day,
                    },
                    "endDate": {
                        "year": daily_range_end_year,
                        "month": daily_range_end_month,
                        "day": daily_range_end_day,
                    },
                },
            }

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_PERFORMANCE_BASE}/{location}:fetchMultiDailyMetricsTimeSeries",
                    headers=_auth_headers(token),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_multi_daily_metrics, **kwargs)


class GoogleBusinessListCategories(Tool):
    name: str = "google_business_list_categories"
    description: str | None = "List available Business Profile categories."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config({"integration": self.integration, "token": self.token})}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_categories(
            region_code: str = Field("US", description="CLDR region code"),
            language_code: str = Field("en", description="BCP-47 language code"),
            filter_query: str | None = Field(None, description="Optional filter, e.g. displayName=restaurant"),
            page_size: int = Field(100, description="Maximum categories to return"),
            page_token: str | None = Field(None, description="Pagination token"),
            view: str = Field("FULL", description="BASIC or FULL category view"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {
                "regionCode": region_code,
                "languageCode": language_code,
                "pageSize": page_size,
                "view": view,
            }
            if filter_query:
                params["filter"] = filter_query
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_INFO_BASE}/categories",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_categories, **kwargs)


class GoogleBusinessCreateLocalPost(Tool):
    name: str = "google_business_create_local_post"
    description: str | None = "Create a local post on a Business Profile location."
    integration: Annotated[str, Integration("google_business")] | None = None
    token: SecretStr | None = None
    default_account_id: str | None = None
    default_location_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "integration": self.integration,
                    "token": self.token,
                    "default_account_id": self.default_account_id,
                    "default_location_id": self.default_location_id,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_local_post(
            local_post: dict = Field(..., description="Local post resource object"),
            account_id: str | None = Field(None, description="Business Profile account ID"),
            location_id: str | None = Field(None, description="Business Profile location ID"),
        ) -> Any:
            token = await _resolve_token(self)
            account = _resolve_account_id(self, account_id)
            location = _resolve_location_id(self, location_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_LEGACY_BASE}/{account}/{location}/localPosts",
                    headers=_auth_headers(token),
                    json=local_post,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_local_post, **kwargs)
