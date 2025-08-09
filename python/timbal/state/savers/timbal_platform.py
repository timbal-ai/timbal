from typing import Any

import requests
import structlog
from pydantic import TypeAdapter

from ...types.models import dump
from ..context import RunContext
from ..data import Data
from ..snapshot import Snapshot
from .base import BaseSaver

logger = structlog.get_logger("timbal.state.savers.timbal_platform")


class TimbalPlatformSaver(BaseSaver):
    """A state saver that stores snapshots in the Timbal platform.

    This state saver is used to store snapshots in the Timbal platform.
    You can see the logs and snapshots history from the platform UI.

    Note:
        This state saver requires a `TimbalPlatformConfig` to be passed within the `RunContext`.
    """
    _get_warning_shown = False
    _put_warning_shown = False


    @staticmethod
    def _load_snapshot_from_res_body(res_body: dict[str, Any]) -> Snapshot:
        res_body["data"] = {
            k: TypeAdapter(Data).validate_python(v)
            for k, v in res_body["data"].items()
        }
        return Snapshot(**res_body)


    async def get_last(
        self,
        path: str,
        context: RunContext,
    ) -> Snapshot | None:
        """See base class."""
        if not context.timbal_platform_config:
            if not self._get_warning_shown:
                logger.warning(
                    "TimbalPlatformSaver: Missing config for GET operation. " \
                    "Pass config to the RunContext to enable fetching snapshots from the platform. " \
                    "You can safely ignore this warning if you intend to push this app to the platform later."
                )
                self._get_warning_shown = True
            return None

        if context.parent_id is None:
            return None

        # No need to check for anything else, the timbal platform config will already be validated.

        host = context.timbal_platform_config.host

        auth = context.timbal_platform_config.auth
        headers = {auth.header_key: auth.header_value}

        subject = context.timbal_platform_config.subject
        org_id = subject.org_id
        app_id = subject.app_id
        resource_path = f"orgs/{org_id}/apps/{app_id}/runs/{context.parent_id}"

        # TODO Convert to async method -> create a new tool to fetch snapshots
        res = requests.get(
            f"https://{host}/{resource_path}/snapshots", 
            headers=headers,
            params={"path": path},
        )
        res.raise_for_status()

        res_body = res.json()
        return self._load_snapshot_from_res_body(res_body)


    async def put(
        self, 
        snapshot: Snapshot,
        context: RunContext,
    ) -> None:
        """See base class."""
        if not context.timbal_platform_config:
            if not self._put_warning_shown:
                logger.warning(
                    "TimbalPlatformSaver: Missing config for PUT operation. " \
                    "Pass config to the RunContext to enable storing snapshots on the platform. " \
                    "You can safely ignore this warning if you intend to push this app to the platform later."
                )
                self._put_warning_shown = True
            return None

        # No need to check for anything else, the timbal platform config will already be validated.

        host = context.timbal_platform_config.host

        auth = context.timbal_platform_config.auth
        headers = {auth.header_key: auth.header_value}

        subject = context.timbal_platform_config.subject
        org_id = subject.org_id
        app_id = subject.app_id
        resource_path = f"orgs/{org_id}/apps/{app_id}/runs/{context.id}"

        body = await dump(snapshot)

        # TODO Convert to async method -> create a new tool to upload snapshots
        res = requests.post(
            f"https://{host}/{resource_path}/snapshots", 
            headers=headers,
            json=body,
        )
        res.raise_for_status()
    