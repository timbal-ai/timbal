from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

import structlog
from pydantic import BaseModel, Discriminator, GetCoreSchemaHandler, GetJsonSchemaHandler, TypeAdapter
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from .utils import _request

logger = structlog.get_logger("timbal.platform.integrations")

_EXPIRY_BUFFER = timedelta(seconds=60)

_credential_cache: dict[tuple[str, str, str], "IntegrationCredential"] = {}


class BearerIntegrationCredential(BaseModel):
    token_type: Literal["bearer"]
    token: str
    expires_at: datetime | None = None


class ApiKeyIntegrationCredential(BaseModel):
    token_type: Literal["api_key"]
    token: str


IntegrationCredential = Annotated[
    BearerIntegrationCredential | ApiKeyIntegrationCredential,
    Discriminator("token_type"),
]

IntegrationCredentialAdapter = TypeAdapter(IntegrationCredential)


class Integration:
    """Platform integration: Pydantic annotation marker + runtime credential resolver.

    Usage::

        class GmailConfig(BaseModel):
            integration: Annotated[str, Integration("gmail")]

        credential = await config.integration.resolve()
        token = credential.token
    """

    def __init__(self, provider: str, org_integration_id: str | None = None) -> None:
        self.provider = provider
        self._org_integration_id = org_integration_id

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        provider = self.provider
        return core_schema.no_info_plain_validator_function(
            lambda v: v if isinstance(v, Integration) else Integration(provider, v),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v._org_integration_id if isinstance(v, Integration) else v,
                info_arg=False,
            ),
        )

    def __get_pydantic_json_schema__(self, schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        return {
            "type": "string",
            "x-timbal-integration": {"provider": self.provider},
        }

    async def resolve(self) -> BearerIntegrationCredential | ApiKeyIntegrationCredential:
        if self._org_integration_id is None:
            raise ValueError("Cannot resolve an Integration without an org_integration_id.")

        from ..state import get_or_create_run_context

        run_context = get_or_create_run_context()
        if not run_context.platform_config:
            raise ValueError("No platform config available for integration credential requests.")

        platform_config = run_context.platform_config
        subject = platform_config.subject
        if not subject:
            raise ValueError("Platform config must have a subject with org_id to fetch integration credentials.")

        cache_key = (platform_config.host, subject.org_id, self._org_integration_id)
        cached = _credential_cache.get(cache_key)
        if cached is not None:
            if isinstance(cached, BearerIntegrationCredential):
                if cached.expires_at is None or cached.expires_at > datetime.now(UTC) + _EXPIRY_BUFFER:
                    return cached
            else:
                return cached

        path = f"orgs/{subject.org_id}/integrations/{self._org_integration_id}"
        response = await _request("GET", path)
        data = response.json()

        credential = IntegrationCredentialAdapter.validate_python(data)
        _credential_cache[cache_key] = credential
        return credential

    def __repr__(self) -> str:
        return f"Integration(provider={self.provider!r}, id={self._org_integration_id!r})"

    def __str__(self) -> str:
        return self._org_integration_id or ""
