from enum import Enum

from pydantic import BaseModel, ConfigDict


class TimbalPlatformAuthType(str, Enum):
    BEARER = "bearer"
    CUSTOM = "custom"


class TimbalPlatformAuthConfig(BaseModel):
    """Configuration for platform authentication.
    At the moment, supports bearer tokens and custom headers.
    """

    type: TimbalPlatformAuthType
    """Type of authentication to use."""
    token: str
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
            return f"Bearer {self.token}"
        elif self.type == TimbalPlatformAuthType.CUSTOM:
            return self.token
        else:
            raise NotImplementedError(f"Unknown auth type: {self.type}")


class TimbalPlatformAppConfig(BaseModel):
    """Application-specific platform configuration.
    Contains identifiers to the platform resource the run context applies to.
    """

    org_id: str 
    """Organization identifier."""
    app_id: str
    """Application identifier."""
    version_id: str
    """Application version identifier."""


class TimbalPlatformConfig(BaseModel):
    """Complete platform configuration.
    Contains all the information needed to authenticate and identify the platform resource the run context applies to.
    """

    host: str
    """Platform host."""
    cdn: str = "content.timbal.ai"
    """CDN host."""
    auth_config: TimbalPlatformAuthConfig
    """Platform authentication configuration."""
    app_config: TimbalPlatformAppConfig
    """Platform application configuration."""


class RunContext(BaseModel):
    """Context for a run.
    This is shared between all steps in a flow (including nested subflows).
    """
    # Allow for extra fields.
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    """Unique identifier for the run.
    We allow for the id to be None so the dev has more flexibility when trying things out.
    The existance of this field should be enforced when creating a Snapshot and using the state saver.
    """
    parent_id: str | None = None
    """Whether this run is a direct child of another run.
    Can be used to recursively retrieve all runs into a single list -> chat history.
    Can also be used to create a new branch from a specific run -> rewind.
    """
    timbal_platform_config: TimbalPlatformConfig | None = None
    """Platform configuration for the run."""
