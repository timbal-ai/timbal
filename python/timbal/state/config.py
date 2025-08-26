from enum import Enum

from pydantic import BaseModel, SecretStr, model_validator


class PlatformAuthType(str, Enum):
    BEARER = "bearer"
    CUSTOM = "custom"


class PlatformAuth(BaseModel):
    """Configuration for platform authentication.
    At the moment, supports bearer tokens and custom headers.
    """

    type: PlatformAuthType
    """Type of authentication to use."""
    token: SecretStr
    """Token included in the authentication header."""
    header_name: str | None = None
    """If type is CUSTOM, this will be the name of the header to use."""

    @property
    def header_key(self) -> str:
        """Format header key for requests authenticating with the platform."""
        if self.type == PlatformAuthType.BEARER:
            return "Authorization"
        elif self.type == PlatformAuthType.CUSTOM:
            return self.header_name
        else:
            raise NotImplementedError(f"Unknown auth type: {self.type}")

    @property 
    def header_value(self) -> str:
        """Format header value for requests authenticating with the platform."""
        if self.type == PlatformAuthType.BEARER:
            return f"Bearer {self.token.get_secret_value()}"
        elif self.type == PlatformAuthType.CUSTOM:
            return self.token.get_secret_value()
        else:
            raise NotImplementedError(f"Unknown auth type: {self.type}")


class PlatformSubject(BaseModel):
    """Contains identifiers to the platform resource the run context applies to."""

    org_id: str
    """Organization identifier."""
    app_id: str
    """Application identifier."""
    version_id: str | None = None
    """Application version identifier."""


class PlatformConfig(BaseModel):
    """Complete platform configuration.
    Contains all the information needed to authenticate and identify the platform resource the run context applies to.
    """

    host: str
    """Platform host."""
    cdn: str = "content.timbal.ai"
    """CDN host."""
    auth: PlatformAuth
    """Platform authentication configuration."""
    subject: PlatformSubject | None = None
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
