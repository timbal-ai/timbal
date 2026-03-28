import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_REST_API_VERSION = "v1"


def _build_base_url(account_id: str) -> str:
    """Build NetSuite REST Record API base URL from account ID."""
    normalized = account_id.strip().lower().replace("_", "-")
    return f"https://{normalized}.suitetalk.api.netsuite.com/services/rest/record/{_REST_API_VERSION}"


def _build_suiteql_url(account_id: str) -> str:
    """Build NetSuite SuiteQL query URL from account ID."""
    normalized = account_id.strip().lower().replace("_", "-")
    return f"https://{normalized}.suitetalk.api.netsuite.com/services/rest/query/v1/suiteql"


def _netsuite_auth_header(
    method: str,
    url: str,
    account_id: str,
    consumer_key: str,
    consumer_secret: str,
    token_id: str,
    token_secret: str,
) -> str:
    """Generate OAuth 1.0a Token-Based Authentication (TBA) header for NetSuite."""
    import base64
    import hashlib
    import hmac
    import time
    import urllib.parse
    import uuid

    nonce = uuid.uuid4().hex
    timestamp = str(int(time.time()))

    params: dict[str, str] = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": nonce,
        "oauth_signature_method": "HMAC-SHA256",
        "oauth_timestamp": timestamp,
        "oauth_token": token_id,
        "oauth_version": "1.0",
    }

    sorted_params = "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}"
        for k, v in sorted(params.items())
    )
    base_string = (
        f"{method.upper()}&"
        f"{urllib.parse.quote(url, safe='')}&"
        f"{urllib.parse.quote(sorted_params, safe='')}"
    )
    signing_key = f"{urllib.parse.quote(consumer_secret, safe='')}&{urllib.parse.quote(token_secret, safe='')}"
    signature = base64.b64encode(
        hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha256).digest()
    ).decode()

    params["oauth_signature"] = signature
    realm = account_id.strip().upper()

    header_parts = [f'realm="{urllib.parse.quote(realm, safe="")}"'] + [
        f'{k}="{urllib.parse.quote(v, safe="")}"' for k, v in sorted(params.items())
    ]
    return "OAuth " + ", ".join(header_parts)


async def _resolve_credentials(tool: Any) -> tuple[str, str, str, str, str]:
    """Resolve NetSuite TBA credentials.

    Integration credentials (type: credentials):
    - account_id: NetSuite Account ID (e.g. 1234567 or TSTDRV1234567)
    - consumer_key: OAuth consumer key from Integration record
    - consumer_secret: OAuth consumer secret from Integration record
    - token_id: Access token ID
    - token_secret: Access token secret
    """
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    account_id = creds.get("account_id") or getattr(tool, "account_id", None) or os.getenv("NETSUITE_ACCOUNT_ID")
    consumer_key = creds.get("consumer_key") or getattr(tool, "consumer_key", None) or os.getenv("NETSUITE_CONSUMER_KEY")
    consumer_secret = creds.get("consumer_secret") or (
        tool.consumer_secret.get_secret_value()
        if getattr(tool, "consumer_secret", None) and tool.consumer_secret
        else None
    ) or os.getenv("NETSUITE_CONSUMER_SECRET")
    token_id = creds.get("token_id") or getattr(tool, "token_id", None) or os.getenv("NETSUITE_TOKEN_ID")
    token_secret = creds.get("token_secret") or (
        tool.token_secret.get_secret_value()
        if getattr(tool, "token_secret", None) and tool.token_secret
        else None
    ) or os.getenv("NETSUITE_TOKEN_SECRET")

    if not account_id or not consumer_key or not consumer_secret or not token_id or not token_secret:
        raise ValueError(
            "NetSuite credentials not found. Configure integration with "
            "account_id, consumer_key, consumer_secret, token_id, token_secret."
        )
    return account_id, consumer_key, consumer_secret, token_id, token_secret


# ===========================================================================
# ACCOUNT
# ===========================================================================

class NetSuiteCreateAccount(Tool):
    """Create a new account in NetSuite."""

    name: str = "netsuite_create_account"
    description: str | None = "Create a new account in NetSuite."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_account(
            acct_name: str = Field(..., description="Account name."),
            acct_number: str | None = Field(None, description="Account number."),
            acct_type: str | None = Field(
                None,
                description="Account type (e.g. 'Bank', 'Income', 'Expense', 'OtherCurrentAsset').",
            ),
            currency: str | None = Field(None, description="Currency internal ID (e.g. '1' for USD)."),
            description: str | None = Field(None, description="Account description."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/account"

            import httpx

            data: dict[str, Any] = {"acctName": acct_name}
            if acct_number:
                data["acctNumber"] = acct_number
            if acct_type:
                data["acctType"] = {"id": acct_type}
            if currency:
                data["currency"] = {"id": currency}
            if description:
                data["description"] = description

            auth = _netsuite_auth_header("POST", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=data,
                )
                response.raise_for_status()
                location = response.headers.get("Location", "")
                created_id = location.rstrip("/").split("/")[-1] if location else None
                if return_record and created_id:
                    auth2 = _netsuite_auth_header("GET", f"{base}/account/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/account/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "account", "record_id": created_id, "location": location}

        super().__init__(handler=_create_account, **kwargs)


class NetSuiteDeleteAccount(Tool):
    """Delete an account from NetSuite by internal ID."""

    name: str = "netsuite_delete_account"
    description: str | None = "Delete an account from NetSuite by internal ID."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_account(
            record_id: str = Field(..., description="Internal ID of the account to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/account/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "account", "record_id": record_id}

        super().__init__(handler=_delete_account, **kwargs)


class NetSuiteGetAccount(Tool):
    """Retrieve a NetSuite account by internal ID."""

    name: str = "netsuite_get_account"
    description: str | None = "Retrieve a NetSuite account by internal ID."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_account(
            record_id: str = Field(..., description="Internal ID of the account to retrieve."),
            fields: str | None = Field(None, description="Comma-separated list of fields to return. Returns all fields if omitted."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/account/{record_id}"

            import httpx

            params: dict[str, str] = {}
            if fields:
                params["fields"] = fields

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    params=params if params else None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_account, **kwargs)


class NetSuiteListAccounts(Tool):
    """List accounts in NetSuite with optional search query."""

    name: str = "netsuite_list_accounts"
    description: str | None = "List accounts in NetSuite with optional search query."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_accounts(
            query: str | None = Field(None, description="Search query to filter accounts by name or number."),
            limit: int = Field(100, description="Maximum number of accounts to return (default 100)."),
            offset: int = Field(0, description="Number of records to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/account"

            import httpx

            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if query:
                params["q"] = query

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth, "Content-Type": "application/json"}, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_accounts, **kwargs)


# ===========================================================================
# ASSEMBLY BUILD
# ===========================================================================

class NetSuiteCreateAssemblyBuild(Tool):
    """Create a new assembly build record in NetSuite."""

    name: str = "netsuite_create_assembly_build"
    description: str | None = "Create a new assembly build record in NetSuite."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_assembly_build(
            item_id: str = Field(..., description="Internal ID of the assembly item to build."),
            quantity: float = Field(..., description="Quantity to build."),
            location_id: str | None = Field(None, description="Internal ID of the location."),
            subsidiary_id: str | None = Field(None, description="Internal ID of the subsidiary."),
            memo: str | None = Field(None, description="Memo / notes for this build."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblybuild"

            import httpx

            data: dict[str, Any] = {"item": {"id": item_id}, "quantity": quantity}
            if location_id:
                data["location"] = {"id": location_id}
            if subsidiary_id:
                data["subsidiary"] = {"id": subsidiary_id}
            if memo:
                data["memo"] = memo

            auth = _netsuite_auth_header("POST", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                location = response.headers.get("Location", "")
                created_id = location.rstrip("/").split("/")[-1] if location else None
                if return_record and created_id:
                    auth2 = _netsuite_auth_header("GET", f"{base}/assemblybuild/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/assemblybuild/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "assemblybuild", "record_id": created_id, "location": location}

        super().__init__(handler=_create_assembly_build, **kwargs)


class NetSuiteDeleteAssemblyBuild(Tool):
    """Delete an assembly build record from NetSuite by internal ID."""

    name: str = "netsuite_delete_assembly_build"
    description: str | None = "Delete an assembly build record from NetSuite by internal ID."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_assembly_build(
            record_id: str = Field(..., description="Internal ID of the assembly build to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblybuild/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "assemblybuild", "record_id": record_id}

        super().__init__(handler=_delete_assembly_build, **kwargs)


class NetSuiteGetAssemblyBuild(Tool):
    """Retrieve a NetSuite assembly build record by internal ID."""

    name: str = "netsuite_get_assembly_build"
    description: str | None = "Retrieve a NetSuite assembly build record by internal ID."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_assembly_build(
            record_id: str = Field(..., description="Internal ID of the assembly build to retrieve."),
            fields: str | None = Field(None, description="Comma-separated list of fields to return. Returns all fields if omitted."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblybuild/{record_id}"

            import httpx

            params: dict[str, str] = {}
            if fields:
                params["fields"] = fields

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    params=params if params else None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_assembly_build, **kwargs)


class NetSuiteListAssemblyBuilds(Tool):
    """List assembly build records in NetSuite."""

    name: str = "netsuite_list_assembly_builds"
    description: str | None = "List assembly build records in NetSuite."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_assembly_builds(
            query: str | None = Field(None, description="Search query to filter assembly builds."),
            limit: int = Field(100, description="Maximum number of records to return (default 100)."),
            offset: int = Field(0, description="Number of records to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblybuild"

            import httpx

            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if query:
                params["q"] = query

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth, "Content-Type": "application/json"}, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_assembly_builds, **kwargs)


class NetSuiteUpdateAssemblyBuild(Tool):
    """Update an existing assembly build record in NetSuite."""

    name: str = "netsuite_update_assembly_build"
    description: str | None = "Update an existing assembly build record in NetSuite."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_assembly_build(
            record_id: str = Field(..., description="Internal ID of the assembly build to update."),
            data: dict[str, Any] = Field(..., description="Fields to update on the assembly build record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblybuild/{record_id}"

            import httpx

            auth = _netsuite_auth_header("PATCH", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"updated": True, "record_type": "assemblybuild", "record_id": record_id}

        super().__init__(handler=_update_assembly_build, **kwargs)


class NetSuiteTransformAssemblyItemToBuild(Tool):
    """Transform a NetSuite assembly item into an assembly build record."""

    name: str = "netsuite_transform_assembly_item_to_build"
    description: str | None = "Transform a NetSuite assembly item into an assembly build record."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _transform_to_build(
            assembly_item_id: str = Field(..., description="Internal ID of the assembly item to transform."),
            quantity: float | None = Field(None, description="Quantity to build."),
            location_id: str | None = Field(None, description="Internal ID of the location."),
            memo: str | None = Field(None, description="Memo / notes for the new build."),
            return_record: bool = Field(False, description="If true, returns the created assembly build record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblyitem/{assembly_item_id}/!transform/assemblybuild"

            import httpx

            data: dict[str, Any] = {}
            if quantity is not None:
                data["quantity"] = quantity
            if location_id:
                data["location"] = {"id": location_id}
            if memo:
                data["memo"] = memo

            auth = _netsuite_auth_header("POST", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                location = response.headers.get("Location", "")
                created_id = location.rstrip("/").split("/")[-1] if location else None
                if return_record and created_id:
                    auth2 = _netsuite_auth_header("GET", f"{base}/assemblybuild/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/assemblybuild/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "assemblybuild", "record_id": created_id, "location": location}

        super().__init__(handler=_transform_to_build, **kwargs)


class NetSuiteTransformAssemblyItemToWorkOrder(Tool):
    """Transform a NetSuite assembly item into a work order."""

    name: str = "netsuite_transform_assembly_item_to_work_order"
    description: str | None = "Transform a NetSuite assembly item into a work order."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _transform_to_work_order(
            assembly_item_id: str = Field(..., description="Internal ID of the assembly item to transform."),
            quantity: float | None = Field(None, description="Quantity for the work order."),
            location_id: str | None = Field(None, description="Internal ID of the location."),
            memo: str | None = Field(None, description="Memo / notes for the work order."),
            return_record: bool = Field(False, description="If true, returns the created work order record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblyitem/{assembly_item_id}/!transform/workorder"

            import httpx

            data: dict[str, Any] = {}
            if quantity is not None:
                data["quantity"] = quantity
            if location_id:
                data["location"] = {"id": location_id}
            if memo:
                data["memo"] = memo

            auth = _netsuite_auth_header("POST", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                location = response.headers.get("Location", "")
                created_id = location.rstrip("/").split("/")[-1] if location else None
                if return_record and created_id:
                    auth2 = _netsuite_auth_header("GET", f"{base}/workorder/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/workorder/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "workorder", "record_id": created_id, "location": location}

        super().__init__(handler=_transform_to_work_order, **kwargs)


class NetSuiteTransformAssemblyToUnbuild(Tool):
    """Transform a NetSuite assembly build into an assembly unbuild (reverse the build)."""

    name: str = "netsuite_transform_assembly_to_unbuild"
    description: str | None = "Transform a NetSuite assembly build into an assembly unbuild (reverse the build)."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _transform_to_unbuild(
            assembly_build_id: str = Field(..., description="Internal ID of the assembly build to unbuild."),
            quantity: float | None = Field(None, description="Quantity to unbuild."),
            memo: str | None = Field(None, description="Memo / notes for the unbuild."),
            return_record: bool = Field(False, description="If true, returns the created assembly unbuild record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblybuild/{assembly_build_id}/!transform/assemblyunbuild"

            import httpx

            data: dict[str, Any] = {}
            if quantity is not None:
                data["quantity"] = quantity
            if memo:
                data["memo"] = memo

            auth = _netsuite_auth_header("POST", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                location = response.headers.get("Location", "")
                created_id = location.rstrip("/").split("/")[-1] if location else None
                if return_record and created_id:
                    auth2 = _netsuite_auth_header("GET", f"{base}/assemblyunbuild/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/assemblyunbuild/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "assemblyunbuild", "record_id": created_id, "location": location}

        super().__init__(handler=_transform_to_unbuild, **kwargs)


# ===========================================================================
# ASSEMBLY ITEM
# ===========================================================================

class NetSuiteCreateAssemblyItem(Tool):
    """Create a new assembly item (Bill of Materials) in NetSuite."""

    name: str = "netsuite_create_assembly_item"
    description: str | None = "Create a new assembly item (Bill of Materials) in NetSuite."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_assembly_item(
            item_name: str = Field(..., description="Name / display name of the assembly item."),
            subsidiary_id: str = Field(..., description="Internal ID of the subsidiary this item belongs to."),
            sales_price: float | None = Field(None, description="Sales price of the assembled item."),
            cost: float | None = Field(None, description="Cost of the assembled item."),
            description: str | None = Field(None, description="Description of the assembly item."),
            upc_code: str | None = Field(None, description="UPC / barcode of the assembly item."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblyitem"

            import httpx

            data: dict[str, Any] = {
                "itemId": item_name,
                "subsidiary": {"items": [{"id": subsidiary_id}]},
            }
            if sales_price is not None:
                data["salesPrice"] = sales_price
            if cost is not None:
                data["cost"] = cost
            if description:
                data["description"] = description
            if upc_code:
                data["upcCode"] = upc_code

            auth = _netsuite_auth_header("POST", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                location = response.headers.get("Location", "")
                created_id = location.rstrip("/").split("/")[-1] if location else None
                if return_record and created_id:
                    auth2 = _netsuite_auth_header("GET", f"{base}/assemblyitem/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/assemblyitem/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "assemblyitem", "record_id": created_id, "location": location}

        super().__init__(handler=_create_assembly_item, **kwargs)


class NetSuiteDeleteAssemblyItem(Tool):
    """Delete an assembly item from NetSuite by internal ID."""

    name: str = "netsuite_delete_assembly_item"
    description: str | None = "Delete an assembly item from NetSuite by internal ID."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_assembly_item(
            record_id: str = Field(..., description="Internal ID of the assembly item to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblyitem/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "assemblyitem", "record_id": record_id}

        super().__init__(handler=_delete_assembly_item, **kwargs)


class NetSuiteGetAssemblyItem(Tool):
    """Retrieve a NetSuite assembly item by internal ID."""

    name: str = "netsuite_get_assembly_item"
    description: str | None = "Retrieve a NetSuite assembly item by internal ID."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_assembly_item(
            record_id: str = Field(..., description="Internal ID of the assembly item to retrieve."),
            fields: str | None = Field(None, description="Comma-separated list of fields to return. Returns all fields if omitted."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblyitem/{record_id}"

            import httpx

            params: dict[str, str] = {}
            if fields:
                params["fields"] = fields

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    params=params if params else None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_assembly_item, **kwargs)


class NetSuiteListAssemblyItems(Tool):
    """List assembly items in NetSuite."""

    name: str = "netsuite_list_assembly_items"
    description: str | None = "List assembly items in NetSuite."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_assembly_items(
            query: str | None = Field(None, description="Search query to filter assembly items by name."),
            limit: int = Field(100, description="Maximum number of records to return (default 100)."),
            offset: int = Field(0, description="Number of records to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblyitem"

            import httpx

            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if query:
                params["q"] = query

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth, "Content-Type": "application/json"}, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_assembly_items, **kwargs)


class NetSuiteUpdateAssemblyItem(Tool):
    """Update an existing assembly item in NetSuite."""

    name: str = "netsuite_update_assembly_item"
    description: str | None = "Update an existing assembly item in NetSuite."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_assembly_item(
            record_id: str = Field(..., description="Internal ID of the assembly item to update."),
            data: dict[str, Any] = Field(..., description="Fields to update on the assembly item record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/assemblyitem/{record_id}"

            import httpx

            auth = _netsuite_auth_header("PATCH", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"updated": True, "record_type": "assemblyitem", "record_id": record_id}

        super().__init__(handler=_update_assembly_item, **kwargs)


# ===========================================================================
# SUITEQL (generic query)
# ===========================================================================

class NetSuiteSuiteQL(Tool):
    """Run a SuiteQL query against NetSuite for flexible, SQL-like record searches."""

    name: str = "netsuite_suiteql"
    description: str | None = "Run a SuiteQL query against NetSuite for flexible, SQL-like record searches."
    integration: Annotated[str, Integration("netsuite")] | None = None
    account_id: str | None = None
    consumer_key: str | None = None
    consumer_secret: SecretStr | None = None
    token_id: str | None = None
    token_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "account_id": self.account_id,
                "consumer_key": self.consumer_key,
                "consumer_secret": self.consumer_secret,
                "token_id": self.token_id,
                "token_secret": self.token_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _suiteql(
            query: str = Field(
                ...,
                description="SuiteQL query string (SQL-like). E.g. \"SELECT id, companyName FROM customer WHERE companyName LIKE '%Acme%'\".",
            ),
            limit: int = Field(100, description="Maximum number of rows to return (default 100, max 1000)."),
            offset: int = Field(0, description="Number of rows to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            url = _build_suiteql_url(account_id)

            import httpx

            payload = {"q": query}
            params = {"limit": min(limit, 1000), "offset": offset}
            auth = _netsuite_auth_header("POST", url, account_id, consumer_key, consumer_secret, token_id, token_secret)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={"Authorization": auth, "Content-Type": "application/json", "Prefer": "transient"},
                    json=payload,
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_suiteql, **kwargs)
