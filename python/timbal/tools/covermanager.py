"""CoverManager API tools.

Auth uses an API key (user token):
- GET requests: api key in the URL path segment.
- POST requests: ``apikey`` HTTP header.

Integration credentials (type: credentials):
- api_key: CoverManager API key
- restaurant_slug: Default restaurant slug (optional)

Docs: https://doc-api.covermanager.com/
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal
from urllib.parse import quote

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_API_ROOT = "https://www.covermanager.com"


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    text = str(value).strip()
    return text or None


async def _resolve_api_key(tool: Any) -> str:
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    api_key = (
        _secret_value(creds.get("api_key"))
        or _secret_value(getattr(tool, "api_key", None))
        or _secret_value(os.getenv("COVERMANAGER_API_KEY"))
    )
    if api_key:
        return api_key
    raise ValueError(
        "CoverManager API key not found. Configure integration with api_key or set COVERMANAGER_API_KEY."
    )


async def _resolve_restaurant_slug(tool: Any, override: str | None = None) -> str | None:
    if override:
        return override
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()
    return (
        _secret_value(creds.get("restaurant_slug"))
        or _secret_value(getattr(tool, "restaurant_slug", None))
        or _secret_value(os.getenv("COVERMANAGER_RESTAURANT_SLUG"))
    )


def _format_path(template: str, api_key: str, segments: dict[str, str]) -> str:
    path = template.replace("{api_key}", quote(api_key, safe=""))
    for key, value in segments.items():
        path = path.replace("{" + key + "}", quote(str(value), safe=""))
    return path


async def _covermanager_request(
    tool: Any,
    *,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    import httpx

    url = path if path.startswith("http") else f"{_API_ROOT}{path}"
    headers: dict[str, str] = {"Accept": "application/json"}
    if method.upper() != "GET":
        headers["apikey"] = await _resolve_api_key(tool)
        headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.request(
            method.upper(),
            url,
            headers=headers,
            params=params or None,
            json=json_body,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        try:
            return response.json()
        except Exception:
            return {"raw": response.text}


def _covermanager_config_fields(tool: Any) -> dict[str, Any]:
    return {
        "integration": tool.integration,
        "api_key": tool.api_key,
        "restaurant_slug": tool.restaurant_slug,
    }


class _CoverManagerTool(Tool):
    integration: Annotated[str, Integration("covermanager")] | None = None
    api_key: SecretStr | None = None
    restaurant_slug: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_covermanager_config_fields(self))}


class CoverManagerRequest(_CoverManagerTool):
    """Call any CoverManager API endpoint."""

    name: str = "covermanager_request"
    description: str | None = (
        "Call any CoverManager API endpoint. POST requests send the apikey header automatically."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _request(
            method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = Field("GET", description="HTTP method."),
            path: str = Field(..., description="Path relative to https://www.covermanager.com."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query string parameters."),
            body: dict[str, Any] | None = Field(None, description="Optional JSON request body."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method=method,
                path=path,
                params=query_params,
                json_body=body,
            )

        super().__init__(handler=_request, **kwargs)



class CoverManagerGetRestaurantList(_CoverManagerTool):
    name: str = "covermanager_get_restaurant_list"
    description: str | None = 'CoverManager Restaurant / Get Restaurant List (GET /api/restaurant/list/{api_key}/{city})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            city: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            segments["city"] = str(city)
            path = _format_path('/api/restaurant/list/{api_key}/{city}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetRestaurantListWithPagination(_CoverManagerTool):
    name: str = "covermanager_get_restaurant_list_with_pagination"
    description: str | None = 'CoverManager Restaurant / Get Restaurant List With Pagination (GET /api/restaurant/list/{api_key}/{city})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            city: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            segments["city"] = str(city)
            path = _format_path('/api/restaurant/list/{api_key}/{city}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetRestaurantFromSlug(_CoverManagerTool):
    name: str = "covermanager_get_restaurant_from_slug"
    description: str | None = 'CoverManager Restaurant / Get Restaurant from Slug (GET /api/restaurant/slug/{api_key}/{slug})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            slug: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            segments["slug"] = str(slug)
            path = _format_path('/api/restaurant/slug/{api_key}/{slug}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetCompanies(_CoverManagerTool):
    name: str = "covermanager_get_companies"
    description: str | None = 'CoverManager Restaurant / Get Companies (POST /api/restaurant/slug/{api_key}/{slug})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/restaurant/slug/UnOeMof4a1MqZ0ioo0J3/:slug',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetRestaurantsSubgroup(_CoverManagerTool):
    name: str = "covermanager_get_restaurants_subgroup"
    description: str | None = 'CoverManager Restaurant / Get Restaurants Subgroup (POST /api/restaurant/subgroups)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/restaurant/subgroups',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetReservsWebhook(_CoverManagerTool):
    name: str = "covermanager_set_reservs_webhook"
    description: str | None = 'CoverManager Restaurant / Set reservs Webhook (POST /api/restaurant/set_webhook_reservs)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/restaurant/set_webhook_reservs',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetRestaurantByName(_CoverManagerTool):
    name: str = "covermanager_get_restaurant_by_name"
    description: str | None = 'CoverManager Restaurant / Get Restaurant By Name (POST /api/restaurant/get_restaurant_by_name/{api_key})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/restaurant/get_restaurant_by_name/UnOeMof4a1MqZ0ioo0J3',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetRestaurantByPlace(_CoverManagerTool):
    name: str = "covermanager_get_restaurant_by_place"
    description: str | None = 'CoverManager Restaurant / Get Restaurant By Place (POST /api/restaurant/get_restaurant_by_place/{api_key})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/restaurant/get_restaurant_by_place/UnOeMof4a1MqZ0ioo0J3',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetReservsBasic(_CoverManagerTool):
    name: str = "covermanager_get_reservs_basic"
    description: str | None = 'CoverManager Restaurant / Get Reservs Basic (POST /api/restaurant/get_reservs_basic)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/restaurant/get_reservs_basic',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetReservs(_CoverManagerTool):
    name: str = "covermanager_get_reservs"
    description: str | None = 'CoverManager Restaurant / Get Reservs (GET /api/restaurant/get_reservs/{api_key}/{restaurant}/{date_start}/{date_end}/{page}/{table})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            restaurant: str | None = Field(None, description="Restaurant slug."),
            date_start: str = Field(..., description="Path segment."),
            date_end: str = Field(..., description="Path segment."),
            page: str = Field(..., description="Path segment."),
            table: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            restaurant_val = restaurant or await _resolve_restaurant_slug(self, None)
            if not restaurant_val:
                raise ValueError("restaurant slug is required")
            segments["restaurant"] = restaurant_val
            segments["date_start"] = str(date_start)
            segments["date_end"] = str(date_end)
            segments["page"] = str(page)
            segments["table"] = str(table)
            path = _format_path('/api/restaurant/get_reservs/{api_key}/{restaurant}/{date_start}/{date_end}/{page}/{table}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetReservRestaurant(_CoverManagerTool):
    name: str = "covermanager_get_reserv"
    description: str | None = 'CoverManager Restaurant / Get Reserv (GET /api/restaurant/get_reserv/{api_key}/{token})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            token: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            segments["token"] = str(token)
            path = _format_path('/api/restaurant/get_reserv/{api_key}/{token}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetMap(_CoverManagerTool):
    name: str = "covermanager_get_map"
    description: str | None = 'CoverManager Restaurant / Get Map (GET /api/restaurant/get_map/{api_key}/{restaurant}/{date}/{luner})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            restaurant: str | None = Field(None, description="Restaurant slug."),
            date: str = Field(..., description="Path segment."),
            luner: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            restaurant_val = restaurant or await _resolve_restaurant_slug(self, None)
            if not restaurant_val:
                raise ValueError("restaurant slug is required")
            segments["restaurant"] = restaurant_val
            segments["date"] = str(date)
            segments["luner"] = str(luner)
            path = _format_path('/api/restaurant/get_map/{api_key}/{restaurant}/{date}/{luner}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetTableAvailabilityByRestaurant(_CoverManagerTool):
    name: str = "covermanager_get_table_availability_by_restaurant"
    description: str | None = 'CoverManager Restaurant / Get Table Availability By Restaurant (GET /api/restaurant/table_availability/{api_key}/{restaurant}/{date})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            restaurant: str | None = Field(None, description="Restaurant slug."),
            date: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            restaurant_val = restaurant or await _resolve_restaurant_slug(self, None)
            if not restaurant_val:
                raise ValueError("restaurant slug is required")
            segments["restaurant"] = restaurant_val
            segments["date"] = str(date)
            path = _format_path('/api/restaurant/table_availability/{api_key}/{restaurant}/{date}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerCreateCategory(_CoverManagerTool):
    name: str = "covermanager_create_category"
    description: str | None = 'CoverManager Tags / Create Category (POST /api/categories)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/categories',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerCreateTag(_CoverManagerTool):
    name: str = "covermanager_create_tag"
    description: str | None = 'CoverManager Tags / Create Tag (POST /api/categories/{id}/tags)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/categories/{{id}}/tags',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerCreateTagsByCategory(_CoverManagerTool):
    name: str = "covermanager_create_tags_by_category"
    description: str | None = 'CoverManager Tags / Create Tags By Category (POST /api/categories/{restaurantslug}/{categoryIdornothing})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/categories/{{restaurant-slug}}/{{categoryId-or-nothing}}',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAvailabilityDays(_CoverManagerTool):
    name: str = "covermanager_availability_days"
    description: str | None = 'CoverManager Reserv / Availability days (POST /api/reserv/availability_calendar)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/availability_calendar',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAvailabilityDaysInfoHoursPeople(_CoverManagerTool):
    name: str = "covermanager_availability_days_info_hours_people"
    description: str | None = 'CoverManager Reserv / Availability days info hours people (POST /api/reserv/availability_calendar_total)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/availability_calendar_total',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAvailabilityDaysTotal(_CoverManagerTool):
    name: str = "covermanager_availability_days_total"
    description: str | None = 'CoverManager Reserv / Availability days total (POST /api/reserv/availability_total)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/availability_total',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAvailability(_CoverManagerTool):
    name: str = "covermanager_availability"
    description: str | None = 'CoverManager Reserv / Availability (POST /api/reserv/availability)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/availability',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAvailabilityExtended(_CoverManagerTool):
    name: str = "covermanager_availability_extended"
    description: str | None = 'CoverManager Reserv / Availability Extended (POST /apiV2/availability_extended)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/apiV2/availability_extended',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAvailabilityMessage(_CoverManagerTool):
    name: str = "covermanager_availability_message"
    description: str | None = 'CoverManager Reserv / Availability Message (POST /api/reserv/availability_message)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/availability_message',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerIsReservable(_CoverManagerTool):
    name: str = "covermanager_is_reservable"
    description: str | None = 'CoverManager Reserv / Is reservable (POST /api/reserv/is_reservable)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/is_reservable',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetZones(_CoverManagerTool):
    name: str = "covermanager_get_zones"
    description: str | None = 'CoverManager Reserv / Get zones (POST /api/reserv/get_zones)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/get_zones',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerReserv(_CoverManagerTool):
    name: str = "covermanager_reserv"
    description: str | None = 'CoverManager Reserv / Reserv (POST /api/reserv/reserv)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/reserv',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerUpdateReserv(_CoverManagerTool):
    name: str = "covermanager_update_reserv"
    description: str | None = 'CoverManager Reserv / Update Reserv (POST /api/reserv/update_reserv)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/update_reserv',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerReservWalkInReserv(_CoverManagerTool):
    name: str = "covermanager_reserv_walk_in"
    description: str | None = 'CoverManager Reserv / Reserv Walk In (POST /api/reserv/walk_in)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/walk_in',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerWaitingList(_CoverManagerTool):
    name: str = "covermanager_waiting_list"
    description: str | None = 'CoverManager Reserv / Waiting List (POST /api/reserv/waiting_list)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/waiting_list',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerReservForce(_CoverManagerTool):
    name: str = "covermanager_reserv_force"
    description: str | None = 'CoverManager Reserv / Reserv Force (POST /api/reserv/reserv_force)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/reserv_force',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerCancelClient(_CoverManagerTool):
    name: str = "covermanager_cancel_client"
    description: str | None = 'CoverManager Reserv / Cancel client (POST /api/reserv/cancel_client)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/cancel_client',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerCrossSelling(_CoverManagerTool):
    name: str = "covermanager_cross_selling"
    description: str | None = 'CoverManager Reserv / Cross-Selling (POST /api/reserv/crosselling)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/crosselling',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetWebhookUrl(_CoverManagerTool):
    name: str = "covermanager_set_webhook_url"
    description: str | None = 'CoverManager Reserv / Set webhook url (POST /api/reserv/set_webhook_url)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/set_webhook_url',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetConfirmUrl(_CoverManagerTool):
    name: str = "covermanager_set_confirm_url"
    description: str | None = 'CoverManager Reserv / Set confirm url (POST /api/reserv/set_confirm_url)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/set_confirm_url',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetCancelUrl(_CoverManagerTool):
    name: str = "covermanager_set_cancel_url"
    description: str | None = 'CoverManager Reserv / Set cancel url (POST /api/reserv/set_cancel_url)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/set_cancel_url',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetSeatedStatus(_CoverManagerTool):
    name: str = "covermanager_set_seated_status"
    description: str | None = 'CoverManager Reserv / Set seated status (POST /api/reserv/sit_client)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/sit_client',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetConfirmStatus(_CoverManagerTool):
    name: str = "covermanager_set_confirm_status"
    description: str | None = 'CoverManager Reserv / Set confirm status (POST /api/reserv/confirm_client)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/confirm_client',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerRevertStatus(_CoverManagerTool):
    name: str = "covermanager_revert_status"
    description: str | None = 'CoverManager Reserv / Revert Status (POST /api/reserv/revert_status_reserv)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/revert_status_reserv',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSitClientPending(_CoverManagerTool):
    name: str = "covermanager_sit_client_pending"
    description: str | None = 'CoverManager Reserv / Sit client pending (POST /api/reserv/sit_client_pending)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/sit_client_pending',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerUndoSeatedStatus(_CoverManagerTool):
    name: str = "covermanager_undo_seated_status"
    description: str | None = 'CoverManager Reserv / Undo seated status (POST /api/reserv/undo_sit)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/undo_sit',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetTicketReserv(_CoverManagerTool):
    name: str = "covermanager_set_ticket"
    description: str | None = 'CoverManager Reserv / Set ticket (POST /api/reserv/set_ticket)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/set_ticket',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetTicketParcial(_CoverManagerTool):
    name: str = "covermanager_set_ticket_parcial"
    description: str | None = 'CoverManager Reserv / Set ticket parcial (POST /api/reserv/set_ticket)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/set_ticket',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetCard(_CoverManagerTool):
    name: str = "covermanager_set_card"
    description: str | None = 'CoverManager Reserv / Set card (POST /api/reserv/set_card)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/set_card',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAddExternalPayment(_CoverManagerTool):
    name: str = "covermanager_add_external_payment"
    description: str | None = 'CoverManager Reserv / Add external payment (POST /api/reserv/add_external_payment)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/add_external_payment',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetSecurePaymentInfo(_CoverManagerTool):
    name: str = "covermanager_get_secure_payment_info"
    description: str | None = 'CoverManager Reserv / Get Secure Payment Info (GET /api/reserv/get_secure_payment_info)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="GET",
                path='/api/reserv/get_secure_payment_info',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAddCommentaryClient(_CoverManagerTool):
    name: str = "covermanager_add_commentary_client"
    description: str | None = 'CoverManager Reserv / Add commentary client (POST /api/reserv/commentary_client)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/commentary_client',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerUpdateMinimumSpend(_CoverManagerTool):
    name: str = "covermanager_update_minimum_spend"
    description: str | None = 'CoverManager Reserv / Update Minimum spend (POST /api/reserv/update_minimum_spend)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/update_minimum_spend',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetTickets(_CoverManagerTool):
    name: str = "covermanager_get_tickets"
    description: str | None = 'CoverManager Reserv / Get tickets (GET /api/reserv/get_tickets)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="GET",
                path='/api/reserv/get_tickets',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetProductByAvailability(_CoverManagerTool):
    name: str = "covermanager_get_product_by_availability"
    description: str | None = 'CoverManager Reserv / Get Product by Availability (GET /api/reserv/get_products_by_availability)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="GET",
                path='/api/reserv/get_products_by_availability',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetTablesId(_CoverManagerTool):
    name: str = "covermanager_set_tables_id"
    description: str | None = 'CoverManager Reserv / Set tables id (POST /api/reserv/update_table)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/update_table',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetPayUrl(_CoverManagerTool):
    name: str = "covermanager_get_pay_url"
    description: str | None = 'CoverManager Reserv / Get Pay url (GET /apiV2/reserv/get_pay_url/{api_key}/{restaurant}/{id_reserv})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            restaurant: str | None = Field(None, description="Restaurant slug."),
            id_reserv: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            restaurant_val = restaurant or await _resolve_restaurant_slug(self, None)
            if not restaurant_val:
                raise ValueError("restaurant slug is required")
            segments["restaurant"] = restaurant_val
            segments["id_reserv"] = str(id_reserv)
            path = _format_path('/apiV2/reserv/get_pay_url/{api_key}/{restaurant}/{id_reserv}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSatisfactions(_CoverManagerTool):
    name: str = "covermanager_satisfactions"
    description: str | None = 'CoverManager Reports / Satisfactions (GET /api/report/get_satisfaction/{api_key}/{restaurant}/{date_start}/{date_end}/{page})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            restaurant: str | None = Field(None, description="Restaurant slug."),
            date_start: str = Field(..., description="Path segment."),
            date_end: str = Field(..., description="Path segment."),
            page: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            restaurant_val = restaurant or await _resolve_restaurant_slug(self, None)
            if not restaurant_val:
                raise ValueError("restaurant slug is required")
            segments["restaurant"] = restaurant_val
            segments["date_start"] = str(date_start)
            segments["date_end"] = str(date_end)
            segments["page"] = str(page)
            path = _format_path('/api/report/get_satisfaction/{api_key}/{restaurant}/{date_start}/{date_end}/{page}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetResumenDate(_CoverManagerTool):
    name: str = "covermanager_get_resumen_date"
    description: str | None = 'CoverManager Reports / Get Resumen Date (POST /api/stats/get_resumen_date)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/stats/get_resumen_date',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerMakeSatisfactionSurvey(_CoverManagerTool):
    name: str = "covermanager_make_satisfaction_survey"
    description: str | None = 'CoverManager Reports / Make satisfaction survey (POST /api/stats/make_survey)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/stats/make_survey',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerListClients(_CoverManagerTool):
    name: str = "covermanager_clients_list"
    description: str | None = 'CoverManager Clients / List (POST /api/clients/clients_list)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/clients/clients_list',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAddClient(_CoverManagerTool):
    name: str = "covermanager_add_client"
    description: str | None = 'CoverManager Clients / Add Client (POST /api/clients/add_client)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/clients/add_client',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetClient(_CoverManagerTool):
    name: str = "covermanager_get_client"
    description: str | None = 'CoverManager Clients / Get Client (POST /api/clients/get_client)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/clients/get_client',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerUpdateFoodPreference(_CoverManagerTool):
    name: str = "covermanager_update_food_preference"
    description: str | None = 'CoverManager Clients / Update Food Preference (POST /api/clients/update_food_preference/{api_key})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/clients/update_food_preference/UnOeMof4a1MqZ0ioo0J3',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerManageClientsReview(_CoverManagerTool):
    name: str = "covermanager_manage_clients_review"
    description: str | None = 'CoverManager Clients / Manage Clients Review (POST /api/clients/send_manage_clients_reviews)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/clients/send_manage_clients_reviews',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetClientsReservs(_CoverManagerTool):
    name: str = "covermanager_get_clients_reservs"
    description: str | None = 'CoverManager Clients / Get Clients Reservs (POST /api/clients/get_clients_reservs)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/clients/get_clients_reservs',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetOrders(_CoverManagerTool):
    name: str = "covermanager_get_orders"
    description: str | None = 'CoverManager CoverAtHome / Get Orders (GET /api/coverathome/get_orders/{api_key}/{restaurant}/{date})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            restaurant: str | None = Field(None, description="Restaurant slug."),
            date: str = Field(..., description="Path segment."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            segments: dict[str, str] = {}
            restaurant_val = restaurant or await _resolve_restaurant_slug(self, None)
            if not restaurant_val:
                raise ValueError("restaurant slug is required")
            segments["restaurant"] = restaurant_val
            segments["date"] = str(date)
            path = _format_path('/api/coverathome/get_orders/{api_key}/{restaurant}/{date}', api_key, segments)
            return await _covermanager_request(self, method="GET", path=path)

        super().__init__(handler=_handler, **kwargs)


class CoverManagerUpdateOrderStatus(_CoverManagerTool):
    name: str = "covermanager_update_order_status"
    description: str | None = 'CoverManager CoverAtHome / Update Order Status (POST /api/coverathome/update_order_status)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/coverathome/update_order_status',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerAddPromotionalCode(_CoverManagerTool):
    name: str = "covermanager_add_promotional_code"
    description: str | None = 'CoverManager Promotional Codes / Add Promotional Code (POST /api/promotional_code/add_promotional_code)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/promotional_code/add_promotional_code',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerUpdatePromotionalCodeUpdate(_CoverManagerTool):
    name: str = "covermanager_update_promotional_code_update"
    description: str | None = 'CoverManager Promotional Codes / Update Promotional Code Update (POST /api/promotional_code/update_promotional_code)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/promotional_code/update_promotional_code',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerDeletePromotionalCode(_CoverManagerTool):
    name: str = "covermanager_delete_promotional_code"
    description: str | None = 'CoverManager Promotional Codes / Delete Promotional Code (POST /api/promotional_code/delete_promotional_code)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/promotional_code/delete_promotional_code',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerCheckCode(_CoverManagerTool):
    name: str = "covermanager_check_code"
    description: str | None = 'CoverManager Promotional Codes / Check Code (POST /api/reserv/check_code)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/reserv/check_code',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerCreate(_CoverManagerTool):
    name: str = "covermanager_create"
    description: str | None = 'CoverManager Onthego / Create (POST /api/onthego/create)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/onthego/create',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerDelete(_CoverManagerTool):
    name: str = "covermanager_delete"
    description: str | None = 'CoverManager Onthego / Delete (POST /api/onthego/delete)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/onthego/delete',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerListOnthego(_CoverManagerTool):
    name: str = "covermanager_onthego_list"
    description: str | None = 'CoverManager Onthego / List (POST /api/onthego/list)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/onthego/list',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerEdit(_CoverManagerTool):
    name: str = "covermanager_edit"
    description: str | None = 'CoverManager Onthego / Edit (POST /api/onthego/edit)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/onthego/edit',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetPaysPays(_CoverManagerTool):
    name: str = "covermanager_get_pays"
    description: str | None = 'CoverManager Pays / Get Pays (POST /api/pays/get_pays)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/pays/get_pays',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetPaysTypes(_CoverManagerTool):
    name: str = "covermanager_get_pays_types"
    description: str | None = 'CoverManager Pays / Get Pays Types (POST /api/pays/get_external_pays_types)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/pays/get_external_pays_types',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetExternalPaysPays(_CoverManagerTool):
    name: str = "covermanager_get_external_pays"
    description: str | None = 'CoverManager Pays / Get External Pays (POST /api/pays/get_pays)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/pays/get_pays',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetRefundsPays(_CoverManagerTool):
    name: str = "covermanager_get_refunds"
    description: str | None = 'CoverManager Pays / Get Refunds (POST /api/pays/get_refunds)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/pays/get_refunds',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetProducts(_CoverManagerTool):
    name: str = "covermanager_get_products"
    description: str | None = 'CoverManager Pays / Get Products (POST /api/pays/get_products)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/pays/get_products',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetMapAlt(_CoverManagerTool):
    name: str = "covermanager_get_map"
    description: str | None = 'CoverManager Multilicenses / Get Map (POST /api/multilicense/get_map)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/multilicense/get_map',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerReservWalkInMultilicenses(_CoverManagerTool):
    name: str = "covermanager_reserv_walk_in"
    description: str | None = 'CoverManager Multilicenses / Reserv Walk In (POST /api/multilicense/walk_in)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/multilicense/walk_in',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetReserv(_CoverManagerTool):
    name: str = "covermanager_get_reserv"
    description: str | None = 'CoverManager Multilicenses / Get reserv (POST /api/multilicense/get_reserv)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/multilicense/get_reserv',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetPays(_CoverManagerTool):
    name: str = "covermanager_get_pays"
    description: str | None = 'CoverManager Multilicenses / Get pays (POST /api/multilicense/get_pays)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/multilicense/get_pays',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetRefundsMultilicenses(_CoverManagerTool):
    name: str = "covermanager_get_refunds"
    description: str | None = 'CoverManager Multilicenses / Get Refunds (POST /api/multilicense/get_refunds)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/multilicense/get_refunds',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetExternalPaysMultilicenses(_CoverManagerTool):
    name: str = "covermanager_get_external_pays"
    description: str | None = 'CoverManager Multilicenses / Get External Pays (POST /api/multilicense/get_external_pays)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/multilicense/get_external_pays',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetWebhookChannel(_CoverManagerTool):
    name: str = "covermanager_get_webhook_channel"
    description: str | None = 'CoverManager Webhooks / Get Webhook channel (GET /api/webhooks/get_webhook_channel/{api_key})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="GET",
                path='/api/webhooks/get_webhook_channel/UnOeMof4a1MqZ0ioo0J3',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetWebhookChannel(_CoverManagerTool):
    name: str = "covermanager_set_webhook_channel"
    description: str | None = 'CoverManager Webhooks / Set Webhook Channel (POST /api/webhooks/set_webhook_channel)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/webhooks/set_webhook_channel/',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerGetWebhookTpvWebhooks(_CoverManagerTool):
    name: str = "covermanager_get_webhook_tpv"
    description: str | None = 'CoverManager Webhooks / Get Webhook TPV (GET /api/webhooks/get_webhook_channel/{api_key})'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="GET",
                path='/api/webhooks/get_webhook_channel/UnOeMof4a1MqZ0ioo0J3',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)


class CoverManagerSetWebhookTpv(_CoverManagerTool):
    name: str = "covermanager_set_webhook_tpv"
    description: str | None = 'CoverManager Webhooks / Set Webhook TPV (POST /api/webhooks/set_webhook_tpv)'

    def __init__(self, **kwargs: Any) -> None:
        async def _handler(
            payload: dict[str, Any] = Field(..., description="JSON request body."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query parameters."),
        ) -> Any:
            return await _covermanager_request(
                self,
                method="POST",
                path='/api/webhooks/set_webhook_tpv/',
                params=query_params,
                json_body=payload,
            )

        super().__init__(handler=_handler, **kwargs)
