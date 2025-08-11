from collections import UserDict
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from uuid_extensions import uuid7

from .data import BaseData, DataValue


class TimbalPlatformAuthType(str, Enum):
    BEARER = "bearer"
    CUSTOM = "custom"


class TimbalPlatformAuth(BaseModel):
    """Configuration for platform authentication.
    At the moment, supports bearer tokens and custom headers.
    """

    type: TimbalPlatformAuthType
    """Type of authentication to use."""
    token: SecretStr
    """Token included in the authentication header."""
    header_name: str | None = None
    """If type is CUSTOM, this will be the name of the header to use."""

    @property
    def header_key(self) -> str:
        """Format header key for requests authenticating with the platform."""
        if self.type == TimbalPlatformAuthType.BEARER:
            return "Authorization"
        elif self.type == TimbalPlatformAuthType.CUSTOM:
            return self.header_name
        else:
            raise NotImplementedError(f"Unknown auth type: {self.type}")

    @property 
    def header_value(self) -> str:
        """Format header value for requests authenticating with the platform."""
        if self.type == TimbalPlatformAuthType.BEARER:
            return f"Bearer {self.token.get_secret_value()}"
        elif self.type == TimbalPlatformAuthType.CUSTOM:
            return self.token.get_secret_value()
        else:
            raise NotImplementedError(f"Unknown auth type: {self.type}")


class TimbalPlatformSubject(BaseModel):
    """Contains identifiers to the platform resource the run context applies to."""

    org_id: str | None = None
    """Organization identifier."""
    app_id: str | None = None
    """Application identifier."""
    version_id: str | None = None
    """Application version identifier."""


class TimbalPlatformConfig(BaseModel):
    """Complete platform configuration.
    Contains all the information needed to authenticate and identify the platform resource the run context applies to.
    """

    host: str
    """Platform host."""
    cdn: str = "content.timbal.ai"
    """CDN host."""
    auth: TimbalPlatformAuth
    """Platform authentication configuration."""
    subject: TimbalPlatformSubject | None = None
    """Platform subject configuration. i.e. this is the agent/workflow platform identifiers context."""

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, values):
        # Pydantic does not natively support multiple aliases for single fields.
        # We keep the other names for backwards compatibility.
        if "auth_config" in values:
            values["auth"] = values.pop("auth_config")

        if "app_config" in values:
            values["subject"] = values.pop("app_config")
        elif "app" in values:
            values["subject"] = values.pop("app")
        elif "scope" in values:
            values["subject"] = values.pop("scope")

        return values


class RunContextData(UserDict):

    # TODO We should call get_data_key internally for this.
    def __getitem__(self, key: str):
        return super().__getitem__(key).resolve()

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, BaseData):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, DataValue(value=value))

    def as_dict(self) -> dict[str, BaseData]:
        return dict(self.data)


class RunContext(BaseModel):
    """Context for a run.
    This is shared between all steps in an agent/workflow (including nested agents/workflows).
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    id: str = Field(
        default_factory=lambda: uuid7(as_type="str"),
        description="Unique identifier for the run.",
    )
    parent_id: str | None = Field(
        None,
        description="Whether this run is a direct child of another run.",
    )
    idempotency_key: str | None = Field(
        None,
        description="Idempotency key for the run."
    )
    data: RunContextData = Field(
        default_factory=RunContextData,
        description="Data to be shared between steps in an agent or workflow."
    )
    timbal_platform_config: TimbalPlatformConfig | None = Field(
        None,
        description="Platform configuration for the run."
    )
    usage: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Usage data for the run. Stored as key-value pairs."
            "e.g. 'gpt-4.1:input_text_tokens': 1000"
        ),
    )

    def update_usage(self, key: str, value: int) -> None:
        """"""
        if key in self.usage:
            self.usage[key] += value
        else:
            self.usage[key] = value
