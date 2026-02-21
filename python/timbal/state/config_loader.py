import configparser
import os
from pathlib import Path
from typing import NamedTuple

import structlog
from pydantic import SecretStr

from .config import PlatformConfig, PlatformSubject

logger = structlog.get_logger("timbal.state.config_loader")

TIMBAL_CONFIG_DIR = Path.home() / ".timbal"


class FileConfig(NamedTuple):
    """Raw values extracted from ~/.timbal/ config and credentials files."""

    base_url: str | None
    api_key: SecretStr | None
    org: str | None


def _resolve_section_name(profile: str) -> str:
    """Convert a profile name to the INI section name.

    Follows AWS CLI convention:
    - "default" maps to [default]
    - Any other name maps to [profile <name>]
    """
    if profile == "default":
        return "default"
    return f"profile {profile}"


def load_file_config(
    profile: str | None = None,
    config_dir: Path | None = None,
) -> FileConfig:
    """Load configuration from ~/.timbal/config and ~/.timbal/credentials files.

    Args:
        profile: Profile name to load. If None, uses TIMBAL_PROFILE env var
                 or falls back to "default".
        config_dir: Override the config directory (for testing).
    """
    if profile is None:
        profile = os.getenv("TIMBAL_PROFILE", "default")

    base_dir = config_dir or TIMBAL_CONFIG_DIR
    config_path = base_dir / "config"
    credentials_path = base_dir / "credentials"

    section = _resolve_section_name(profile)

    base_url: str | None = None
    org: str | None = None
    api_key: str | None = None

    if config_path.is_file():
        config = configparser.ConfigParser()
        try:
            config.read(config_path)
            if config.has_section(section):
                base_url = config.get(section, "base_url", fallback=None)
                org = config.get(section, "org", fallback=None)
            else:
                logger.debug(
                    f"Profile section '{section}' not found in config file.",
                    config_path=str(config_path),
                    profile=profile,
                )
        except configparser.Error as e:
            logger.warning(
                "Failed to parse timbal config file.",
                config_path=str(config_path),
                error=str(e),
            )

    if credentials_path.is_file():
        credentials = configparser.ConfigParser()
        try:
            credentials.read(credentials_path)
            if credentials.has_section(section):
                raw_key = credentials.get(section, "api_key", fallback=None)
                if raw_key:
                    api_key = SecretStr(raw_key)
            else:
                logger.debug(
                    f"Profile section '{section}' not found in credentials file.",
                    credentials_path=str(credentials_path),
                    profile=profile,
                )
        except configparser.Error as e:
            logger.warning(
                "Failed to parse timbal credentials file.",
                credentials_path=str(credentials_path),
                error=str(e),
            )

    return FileConfig(base_url=base_url, api_key=api_key, org=org)


def _strip_scheme(url: str) -> str:
    """Strip https:// or http:// prefix from a URL to get just the host."""
    if url.startswith("https://"):
        return url[len("https://") :]
    if url.startswith("http://"):
        return url[len("http://") :]
    return url


def resolve_platform_config(
    platform_config: PlatformConfig | None = None,
    profile: str | None = None,
    config_dir: Path | None = None,
) -> PlatformConfig | None:
    """Resolve platform configuration by merging sources with precedence.

    Per-field precedence (highest to lowest):
    1. Fields already set on the input platform_config
    2. Environment variables
    3. ~/.timbal/ config and credentials files

    Args:
        platform_config: Existing config to fill in missing fields for.
        profile: Profile name for file config. Defaults to TIMBAL_PROFILE env var or "default".
        config_dir: Override the config directory (for testing).
    """
    file_config = load_file_config(profile=profile, config_dir=config_dir)
    logger.debug("Loaded file config.", file_config=file_config._asdict())

    if not platform_config:
        # No platform_config — build from scratch with bearer auth
        host = os.getenv("TIMBAL_API_HOST")
        if host:
            logger.debug("Resolved host from TIMBAL_API_HOST.", host=host)
        elif file_config.base_url:
            host = _strip_scheme(file_config.base_url)
            logger.debug("Resolved host from config file.", host=host)

        api_key = os.getenv("TIMBAL_API_KEY") or os.getenv("TIMBAL_API_TOKEN")
        if api_key:
            logger.debug("Resolved api_key from environment variable.")
        else:
            api_key = file_config.api_key
            if api_key:
                logger.debug("Resolved api_key from credentials file.")

        if not host or not api_key:
            logger.debug("Could not resolve platform config.", has_host=bool(host), has_api_key=bool(api_key))
            return None

        platform_config = PlatformConfig(
            host=host,
            auth={"type": "bearer", "token": api_key},  # type: ignore
        )
        logger.debug("Built platform config from scratch.", host=host)
    else:
        logger.debug("Using existing platform config.", host=platform_config.host)

    # Resolve subject fields: explicit > env > file
    existing_subject = platform_config.subject

    org_id = None
    if existing_subject:
        org_id = existing_subject.org_id
    if not org_id:
        org_id = os.getenv("TIMBAL_ORG_ID")
        if org_id:
            logger.debug("Resolved org_id from TIMBAL_ORG_ID.", org_id=org_id)
        else:
            org_id = file_config.org
            if org_id:
                logger.debug("Resolved org_id from config file.", org_id=org_id)
    else:
        logger.debug("Using org_id from existing subject.", org_id=org_id)

    if not org_id:
        logger.debug("No org_id found, skipping subject resolution.")
        return platform_config

    app_id = None
    if existing_subject:
        app_id = existing_subject.app_id
    if not app_id:
        app_id = os.getenv("TIMBAL_APP_ID")

    version_id = None
    if existing_subject:
        version_id = existing_subject.version_id
    if not version_id:
        version_id = os.getenv("TIMBAL_VERSION_ID")

    platform_config.subject = PlatformSubject(
        org_id=org_id,
        app_id=app_id,
        version_id=version_id,
    )
    logger.debug("Resolved platform subject.", org_id=org_id, app_id=app_id, version_id=version_id)

    return platform_config
