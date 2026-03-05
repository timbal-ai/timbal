import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_PLACES_BASE_URL = "https://places.googleapis.com/v1"
_ADDRESS_VALIDATION_URL = "https://addressvalidation.googleapis.com/v1:validateAddress"

_TEXT_SEARCH_FIELDS = (
    "places.id,places.types,places.nationalPhoneNumber,places.internationalPhoneNumber,"
    "places.location,places.viewport,places.formattedAddress,places.googleMapsUri,"
    "places.websiteUri,places.adrFormatAddress,places.businessStatus,places.displayName,"
    "places.primaryType,places.shortFormattedAddress,places.postalAddress"
)

_PLACE_DETAILS_FIELDS = (
    "id,types,nationalPhoneNumber,internationalPhoneNumber,"
    "location,viewport,formattedAddress,googleMapsUri,"
    "websiteUri,adrFormatAddress,businessStatus,displayName,"
    "primaryType,shortFormattedAddress,postalAddress,reviews"
)


async def _resolve_api_key(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Google Maps API key not found. Set GOOGLE_MAPS_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class GoogleMapsTextSearch(Tool):
    name: str = "google_maps_text_search"
    description: str | None = (
        "Search for places using a text query. Returns place IDs, addresses, "
        "coordinates, and business info. Use this for location lookup and validation."
    )
    integration: Annotated[str, Integration("google_maps")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _text_search(
            query: str = Field(..., description="Text query, e.g. 'Ciudad del Transporte Imarcoain Navarra Spain'"),
            language_code: str = Field("en", description="BCP-47 language code for results"),
            max_results: int = Field(5, description="Max number of results (1-20)"),
            region_code: str | None = Field(None, description="CLDR country code to bias results, e.g. 'ES', 'US'"),
            location_bias_lat: float | None = Field(None, description="Latitude to bias results toward"),
            location_bias_lng: float | None = Field(None, description="Longitude to bias results toward"),
            location_bias_radius: float = Field(5000.0, description="Radius in meters for location bias"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {
                "textQuery": query,
                "languageCode": language_code,
                "pageSize": min(max(max_results, 1), 20),
            }
            if region_code:
                body["regionCode"] = region_code
            if location_bias_lat is not None and location_bias_lng is not None:
                body["locationBias"] = {
                    "circle": {
                        "center": {"latitude": location_bias_lat, "longitude": location_bias_lng},
                        "radius": location_bias_radius,
                    }
                }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_PLACES_BASE_URL}/places:searchText",
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": api_key,
                        "X-Goog-FieldMask": _TEXT_SEARCH_FIELDS,
                    },
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_text_search, **kwargs)


class GoogleMapsPlaceDetails(Tool):
    name: str = "google_maps_place_details"
    description: str | None = (
        "Get detailed information about a place by its place ID. Returns address, "
        "coordinates, opening hours, phone, website, and ratings."
    )
    integration: Annotated[str, Integration("google_maps")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _place_details(
            place_id: str = Field(..., description="Google Maps place ID, e.g. 'ChIJN1t_tDeuEmsRUsoyG83frY4'"),
            language_code: str = Field("en", description="BCP-47 language code for results"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_PLACES_BASE_URL}/places/{place_id}",
                    headers={
                        "X-Goog-Api-Key": api_key,
                        "X-Goog-FieldMask": _PLACE_DETAILS_FIELDS,
                    },
                    params={"languageCode": language_code},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_place_details, **kwargs)


class GoogleMapsNearbySearch(Tool):
    name: str = "google_maps_nearby_search"
    description: str | None = (
        "Search for places near a specific location by type. "
        "Requires coordinates and a place type (e.g. 'restaurant', 'gas_station')."
    )
    integration: Annotated[str, Integration("google_maps")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _nearby_search(
            latitude: float = Field(..., description="Center latitude"),
            longitude: float = Field(..., description="Center longitude"),
            radius: float = Field(1000.0, description="Search radius in meters (max 50000)"),
            included_types: list[str] = Field(..., description="Place types to include, e.g. ['restaurant', 'cafe']"),
            max_results: int = Field(5, description="Max number of results (1-20)"),
            language_code: str = Field("en", description="BCP-47 language code for results"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            body: dict[str, Any] = {
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": latitude, "longitude": longitude},
                        "radius": min(radius, 50000.0),
                    }
                },
                "includedTypes": included_types,
                "maxResultCount": min(max(max_results, 1), 20),
                "languageCode": language_code,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_PLACES_BASE_URL}/places:searchNearby",
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": api_key,
                        "X-Goog-FieldMask": _TEXT_SEARCH_FIELDS,
                    },
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_nearby_search, **kwargs)


class GoogleMapsValidateAddress(Tool):
    name: str = "google_maps_validate_address"
    description: str | None = (
        "Validate and normalize a postal address. Returns a verdict (CONFIRMED/UNCONFIRMED) "
        "for each address component, the standardized address, and geocode coordinates. "
        "Best tool for verifying if an address is real and getting its canonical form."
    )
    integration: Annotated[str, Integration("google_maps")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _validate_address(
            address_lines: list[str] = Field(..., description="Address lines, e.g. ['C/ Italia 14', 'Imarcoain']"),
            region_code: str = Field(..., description="ISO-3166-1 country code, e.g. 'ES', 'US', 'FR'"),
            locality: str | None = Field(None, description="City or town name"),
            administrative_area: str | None = Field(None, description="State, province, or region"),
            postal_code: str | None = Field(None, description="Postal/ZIP code"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            address: dict[str, Any] = {
                "regionCode": region_code,
                "addressLines": address_lines,
            }
            if locality:
                address["locality"] = locality
            if administrative_area:
                address["administrativeArea"] = administrative_area
            if postal_code:
                address["postalCode"] = postal_code

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _ADDRESS_VALIDATION_URL,
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": api_key,
                    },
                    json={"address": address},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_validate_address, **kwargs)
