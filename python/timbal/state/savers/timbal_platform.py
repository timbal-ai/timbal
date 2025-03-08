from typing import Any

import requests
from pydantic import TypeAdapter

from ...types.models import dump
from ..context import RunContext
from ..data import Data
from ..snapshot import Snapshot
from .base import BaseSaver


class TimbalPlatformSaver(BaseSaver):
    """A state saver that stores snapshots in the Timbal platform.

    This state saver is used to store snapshots in the Timbal platform.
    You can see the logs and snapshots history from the platform UI.

    Note:
        This state saver requires a `TimbalPlatformConfig` to be passed within the `RunContext`.
    """

    @staticmethod
    def _load_snapshot_from_res_body(res_body: dict[str, Any]) -> Snapshot:
        res_body["data"] = {
            k: TypeAdapter(Data).validate_python(v)
            for k, v in res_body["data"].items()
        }
        return Snapshot(**res_body)


    def get_last(
        self,
        path: str,
        context: RunContext,
    ) -> Snapshot | None:
        """See base class."""
        if not context.timbal_platform_config:
            raise ValueError("Missing platform configuration for fetching the last snapshot.")

        # TODO

        return None


    def put(
        self, 
        snapshot: Snapshot,
        context: RunContext,
    ) -> None:
        """See base class."""
        if not context.timbal_platform_config:
            raise ValueError("Missing platform configuration for storing the snapshot.")

        # No need to check for anything else, the timbal platform config will already be validated.

        host = context.timbal_platform_config.host

        auth_config = context.timbal_platform_config.auth_config
        headers = {auth_config.header_key: auth_config.header_value}

        app_config = context.timbal_platform_config.app_config
        org_id = app_config.org_id
        app_id = app_config.app_id
        resource_path = f"orgs/{org_id}/apps/{app_id}/runs/{context.id}"

        body = dump(snapshot)

        res = requests.post(
            f"https://{host}/{resource_path}/snapshots", 
            headers=headers,
            json=body,
        )
        res.raise_for_status()
    