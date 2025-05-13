"""Tests for various methods of initializing Timbal platform configuration and run contexts.
It's important to have these and always check they don't error. 
At least at some moment in time we used these APIs, so we should ensure to maintain backwards compatibility.
"""

import os

from timbal.state.context import RunContext, TimbalPlatformConfig


def test_timbal_platform_config():
    _ = TimbalPlatformConfig(
        host="api.timbal.ai",
        auth={
            "type": "custom",
            "token": "token"
        },
        app_config={}
    )

    _ = TimbalPlatformConfig.model_validate({
        "host": "api.timbal.ai",
        "auth_config": {
            "type": "bearer",
            "token": os.getenv("TIMBAL_API_TOKEN")
        },
        "scope": {
            "org_id": "org_id"
        }
    })

    _ = TimbalPlatformConfig(**{
        "host": "api.timbal.ai",
        "auth": {
            "type": "bearer",
            "token": os.getenv("TIMBAL_API_TOKEN")
        },
        "scope": {
            "app_id": "app_id",
        }
    })


def test_run_context():
    _ = RunContext(
        timbal_platform_config=TimbalPlatformConfig(**{
            "host": "api.timbal.ai",
            "auth_config": {
                "type": "bearer",
                "token": os.getenv("TIMBAL_API_TOKEN")
            },
            "app_config": {}
        })
    )

    _ = RunContext.model_validate({
        "timbal_platform_config": {
            "host": "api.timbal.ai",
            "auth_config": {
                "type": "bearer",
                "token": os.getenv("TIMBAL_API_TOKEN")
            },
            "app_config": {}
        }
    })

    _ = RunContext.model_validate({
        "id": "test",
        "timbal_platform_config": {
            "host": "dev.timbal.ai",
            "auth": {
                "type": "custom",
                "token": os.getenv("TIMBAL_API_TOKEN"),
            },
            "app": {
                "org_id": "1",
                "app_id": "18",
                "version_id": "85",
            },
        },
    })
