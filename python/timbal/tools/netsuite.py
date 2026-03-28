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


# ===========================================================================
# BILLING ACCOUNT
# ===========================================================================

class NetSuiteCreateBillingAccount(Tool):
    """Create a new billing account in NetSuite."""

    name: str = "netsuite_create_billing_account"
    description: str | None = "Create a new billing account in NetSuite."
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
        async def _create_billing_account(
            customer: str = Field(..., description="Internal ID of the customer this billing account belongs to."),
            name: str | None = Field(None, description="Name of the billing account."),
            currency: str | None = Field(None, description="Currency internal ID (e.g. '1' for USD)."),
            billing_schedule: str | None = Field(None, description="Internal ID of the billing schedule to associate."),
            memo: str | None = Field(None, description="Memo or notes for the billing account."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingaccount"

            import httpx

            data: dict[str, Any] = {"customer": {"id": customer}}
            if name:
                data["name"] = name
            if currency:
                data["currency"] = {"id": currency}
            if billing_schedule:
                data["billingSchedule"] = {"id": billing_schedule}
            if memo:
                data["memo"] = memo

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
                    auth2 = _netsuite_auth_header("GET", f"{base}/billingaccount/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/billingaccount/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "billingaccount", "record_id": created_id, "location": location}

        super().__init__(handler=_create_billing_account, **kwargs)


class NetSuiteDeleteBillingAccount(Tool):
    """Delete a billing account from NetSuite by internal ID."""

    name: str = "netsuite_delete_billing_account"
    description: str | None = "Delete a billing account from NetSuite by internal ID."
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
        async def _delete_billing_account(
            record_id: str = Field(..., description="Internal ID of the billing account to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingaccount/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "billingaccount", "record_id": record_id}

        super().__init__(handler=_delete_billing_account, **kwargs)


class NetSuiteGetBillingAccount(Tool):
    """Retrieve a NetSuite billing account by internal ID."""

    name: str = "netsuite_get_billing_account"
    description: str | None = "Retrieve a NetSuite billing account by internal ID."
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
        async def _get_billing_account(
            record_id: str = Field(..., description="Internal ID of the billing account to retrieve."),
            fields: str | None = Field(None, description="Comma-separated list of fields to return. Returns all fields if omitted."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingaccount/{record_id}"

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

        super().__init__(handler=_get_billing_account, **kwargs)


class NetSuiteListBillingAccounts(Tool):
    """List billing accounts in NetSuite with optional filtering."""

    name: str = "netsuite_list_billing_accounts"
    description: str | None = "List billing accounts in NetSuite with optional filtering."
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
        async def _list_billing_accounts(
            query: str | None = Field(None, description="Search query to filter billing accounts."),
            limit: int = Field(100, description="Maximum number of billing accounts to return (default 100)."),
            offset: int = Field(0, description="Number of records to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingaccount"

            import httpx

            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if query:
                params["q"] = query

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth, "Content-Type": "application/json"}, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_billing_accounts, **kwargs)


class NetSuiteUpdateBillingAccount(Tool):
    """Update an existing billing account in NetSuite."""

    name: str = "netsuite_update_billing_account"
    description: str | None = "Update an existing billing account in NetSuite."
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
        async def _update_billing_account(
            record_id: str = Field(..., description="Internal ID of the billing account to update."),
            data: dict[str, Any] = Field(..., description="Fields to update on the billing account record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingaccount/{record_id}"

            import httpx

            auth = _netsuite_auth_header("PATCH", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"updated": True, "record_type": "billingaccount", "record_id": record_id}

        super().__init__(handler=_update_billing_account, **kwargs)


# ===========================================================================
# BILLING SCHEDULE
# ===========================================================================

class NetSuiteCreateBillingSchedule(Tool):
    """Create a new billing schedule in NetSuite."""

    name: str = "netsuite_create_billing_schedule"
    description: str | None = "Create a new billing schedule in NetSuite."
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
        async def _create_billing_schedule(
            name: str = Field(..., description="Name of the billing schedule."),
            frequency: str | None = Field(None, description="Billing frequency (e.g. 'MONTHLY', 'QUARTERLY', 'ANNUALLY')."),
            recurrence_count: int | None = Field(None, description="Number of billing recurrences."),
            memo: str | None = Field(None, description="Memo or notes for the billing schedule."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingschedule"

            import httpx

            data: dict[str, Any] = {"name": name}
            if frequency:
                data["frequency"] = {"id": frequency}
            if recurrence_count is not None:
                data["recurrenceCount"] = recurrence_count
            if memo:
                data["memo"] = memo

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
                    auth2 = _netsuite_auth_header("GET", f"{base}/billingschedule/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/billingschedule/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "billingschedule", "record_id": created_id, "location": location}

        super().__init__(handler=_create_billing_schedule, **kwargs)


class NetSuiteDeleteBillingSchedule(Tool):
    """Delete a billing schedule from NetSuite by internal ID."""

    name: str = "netsuite_delete_billing_schedule"
    description: str | None = "Delete a billing schedule from NetSuite by internal ID."
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
        async def _delete_billing_schedule(
            record_id: str = Field(..., description="Internal ID of the billing schedule to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingschedule/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "billingschedule", "record_id": record_id}

        super().__init__(handler=_delete_billing_schedule, **kwargs)


class NetSuiteGetBillingSchedule(Tool):
    """Retrieve a NetSuite billing schedule by internal ID."""

    name: str = "netsuite_get_billing_schedule"
    description: str | None = "Retrieve a NetSuite billing schedule by internal ID."
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
        async def _get_billing_schedule(
            record_id: str = Field(..., description="Internal ID of the billing schedule to retrieve."),
            fields: str | None = Field(None, description="Comma-separated list of fields to return. Returns all fields if omitted."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingschedule/{record_id}"

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

        super().__init__(handler=_get_billing_schedule, **kwargs)


class NetSuiteListBillingSchedules(Tool):
    """List billing schedules in NetSuite with optional filtering."""

    name: str = "netsuite_list_billing_schedules"
    description: str | None = "List billing schedules in NetSuite with optional filtering."
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
        async def _list_billing_schedules(
            query: str | None = Field(None, description="Search query to filter billing schedules."),
            limit: int = Field(100, description="Maximum number of billing schedules to return (default 100)."),
            offset: int = Field(0, description="Number of records to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingschedule"

            import httpx

            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if query:
                params["q"] = query

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth, "Content-Type": "application/json"}, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_billing_schedules, **kwargs)


class NetSuiteUpdateBillingSchedule(Tool):
    """Update an existing billing schedule in NetSuite."""

    name: str = "netsuite_update_billing_schedule"
    description: str | None = "Update an existing billing schedule in NetSuite."
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
        async def _update_billing_schedule(
            record_id: str = Field(..., description="Internal ID of the billing schedule to update."),
            data: dict[str, Any] = Field(..., description="Fields to update on the billing schedule record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingschedule/{record_id}"

            import httpx

            auth = _netsuite_auth_header("PATCH", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"updated": True, "record_type": "billingschedule", "record_id": record_id}

        super().__init__(handler=_update_billing_schedule, **kwargs)


class NetSuiteUpsertBillingAccount(Tool):
    """Upsert (create or update) a billing account in NetSuite using an external ID."""

    name: str = "netsuite_upsert_billing_account"
    description: str | None = "Upsert (create or update) a billing account in NetSuite using an external ID."
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
        async def _upsert_billing_account(
            external_id: str = Field(..., description="External ID used to identify and upsert the billing account."),
            data: dict[str, Any] = Field(..., description="Fields to set on the billing account record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingaccount/eid:{external_id}"

            import httpx

            auth = _netsuite_auth_header("PUT", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.put(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"upserted": True, "record_type": "billingaccount", "external_id": external_id}

        super().__init__(handler=_upsert_billing_account, **kwargs)


class NetSuiteUpsertBillingSchedule(Tool):
    """Upsert (create or update) a billing schedule in NetSuite using an external ID."""

    name: str = "netsuite_upsert_billing_schedule"
    description: str | None = "Upsert (create or update) a billing schedule in NetSuite using an external ID."
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
        async def _upsert_billing_schedule(
            external_id: str = Field(..., description="External ID used to identify and upsert the billing schedule."),
            data: dict[str, Any] = Field(..., description="Fields to set on the billing schedule record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/billingschedule/eid:{external_id}"

            import httpx

            auth = _netsuite_auth_header("PUT", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.put(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"upserted": True, "record_type": "billingschedule", "external_id": external_id}

        super().__init__(handler=_upsert_billing_schedule, **kwargs)


# ===========================================================================
# CALENDAR EVENT
# ===========================================================================

class NetSuiteCreateCalendarEvent(Tool):
    """Create a new calendar event in NetSuite."""

    name: str = "netsuite_create_calendar_event"
    description: str | None = "Create a new calendar event in NetSuite."
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
        async def _create_calendar_event(
            title: str = Field(..., description="Title of the calendar event."),
            start_date: str = Field(..., description="Start date/time of the event (ISO 8601 format, e.g. '2024-01-15T10:00:00')."),
            end_date: str | None = Field(None, description="End date/time of the event (ISO 8601 format)."),
            location: str | None = Field(None, description="Location of the event."),
            message: str | None = Field(None, description="Description or message body of the event."),
            all_day_event: bool = Field(False, description="If true, marks the event as an all-day event."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/calendarevent"

            import httpx

            data: dict[str, Any] = {"title": title, "startDate": start_date}
            if end_date:
                data["endDate"] = end_date
            if location:
                data["location"] = location
            if message:
                data["message"] = message
            if all_day_event:
                data["allDayEvent"] = all_day_event

            auth = _netsuite_auth_header("POST", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    json=data,
                )
                response.raise_for_status()
                location_hdr = response.headers.get("Location", "")
                created_id = location_hdr.rstrip("/").split("/")[-1] if location_hdr else None
                if return_record and created_id:
                    auth2 = _netsuite_auth_header("GET", f"{base}/calendarevent/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/calendarevent/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "calendarevent", "record_id": created_id, "location": location_hdr}

        super().__init__(handler=_create_calendar_event, **kwargs)


class NetSuiteDeleteCalendarEvent(Tool):
    """Delete a calendar event from NetSuite by internal ID."""

    name: str = "netsuite_delete_calendar_event"
    description: str | None = "Delete a calendar event from NetSuite by internal ID."
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
        async def _delete_calendar_event(
            record_id: str = Field(..., description="Internal ID of the calendar event to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/calendarevent/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "calendarevent", "record_id": record_id}

        super().__init__(handler=_delete_calendar_event, **kwargs)


class NetSuiteGetCalendarEvent(Tool):
    """Retrieve a NetSuite calendar event by internal ID."""

    name: str = "netsuite_get_calendar_event"
    description: str | None = "Retrieve a NetSuite calendar event by internal ID."
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
        async def _get_calendar_event(
            record_id: str = Field(..., description="Internal ID of the calendar event to retrieve."),
            fields: str | None = Field(None, description="Comma-separated list of fields to return. Returns all fields if omitted."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/calendarevent/{record_id}"

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

        super().__init__(handler=_get_calendar_event, **kwargs)


class NetSuiteListCalendarEvents(Tool):
    """List calendar events in NetSuite with optional filtering."""

    name: str = "netsuite_list_calendar_events"
    description: str | None = "List calendar events in NetSuite with optional filtering."
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
        async def _list_calendar_events(
            query: str | None = Field(None, description="Search query to filter calendar events."),
            limit: int = Field(100, description="Maximum number of calendar events to return (default 100)."),
            offset: int = Field(0, description="Number of records to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/calendarevent"

            import httpx

            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if query:
                params["q"] = query

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth, "Content-Type": "application/json"}, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_calendar_events, **kwargs)


class NetSuiteUpdateCalendarEvent(Tool):
    """Update an existing calendar event in NetSuite."""

    name: str = "netsuite_update_calendar_event"
    description: str | None = "Update an existing calendar event in NetSuite."
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
        async def _update_calendar_event(
            record_id: str = Field(..., description="Internal ID of the calendar event to update."),
            data: dict[str, Any] = Field(..., description="Fields to update on the calendar event record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/calendarevent/{record_id}"

            import httpx

            auth = _netsuite_auth_header("PATCH", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"updated": True, "record_type": "calendarevent", "record_id": record_id}

        super().__init__(handler=_update_calendar_event, **kwargs)


class NetSuiteUpsertCalendarEvent(Tool):
    """Upsert (create or update) a calendar event in NetSuite using an external ID."""

    name: str = "netsuite_upsert_calendar_event"
    description: str | None = "Upsert (create or update) a calendar event in NetSuite using an external ID."
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
        async def _upsert_calendar_event(
            external_id: str = Field(..., description="External ID used to identify and upsert the calendar event."),
            data: dict[str, Any] = Field(..., description="Fields to set on the calendar event record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/calendarevent/eid:{external_id}"

            import httpx

            auth = _netsuite_auth_header("PUT", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.put(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"upserted": True, "record_type": "calendarevent", "external_id": external_id}

        super().__init__(handler=_upsert_calendar_event, **kwargs)


# ===========================================================================
# FINANCE — INTERCOMPANY JOURNAL ENTRY
# ===========================================================================

class NetSuiteCustomSuiteQL(Tool):
    """Run a custom SuiteQL query against NetSuite with full control over the query string."""

    name: str = "netsuite_custom_suiteql"
    description: str | None = "Run a custom SuiteQL query against NetSuite with full control over the query string."
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
        async def _custom_suiteql(
            query: str = Field(
                ...,
                description="Full SuiteQL query string (SQL-like). E.g. \"SELECT id, tranId FROM transaction WHERE type = 'Journal' ORDER BY id DESC\".",
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

        super().__init__(handler=_custom_suiteql, **kwargs)


class NetSuiteCreateIntercompanyJournalEntry(Tool):
    """Create a new intercompany journal entry in NetSuite."""

    name: str = "netsuite_create_intercompany_journal_entry"
    description: str | None = "Create a new intercompany journal entry in NetSuite."
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
        async def _create_intercompany_journal_entry(
            subsidiary: str = Field(..., description="Internal ID of the subsidiary for the journal entry."),
            tran_date: str | None = Field(None, description="Transaction date (ISO 8601 format, e.g. '2024-01-15')."),
            memo: str | None = Field(None, description="Memo or description for the journal entry."),
            lines: list[dict[str, Any]] | None = Field(None, description="List of line items. Each line should include 'account' (id), 'debit' or 'credit' amount, and optional 'memo'."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/intercompanyjournalentry"

            import httpx

            data: dict[str, Any] = {"subsidiary": {"id": subsidiary}}
            if tran_date:
                data["tranDate"] = tran_date
            if memo:
                data["memo"] = memo
            if lines:
                data["line"] = {"items": lines}

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
                    auth2 = _netsuite_auth_header("GET", f"{base}/intercompanyjournalentry/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/intercompanyjournalentry/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "intercompanyjournalentry", "record_id": created_id, "location": location}

        super().__init__(handler=_create_intercompany_journal_entry, **kwargs)


class NetSuiteDeleteIntercompanyJournalEntry(Tool):
    """Delete an intercompany journal entry from NetSuite by internal ID."""

    name: str = "netsuite_delete_intercompany_journal_entry"
    description: str | None = "Delete an intercompany journal entry from NetSuite by internal ID."
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
        async def _delete_intercompany_journal_entry(
            record_id: str = Field(..., description="Internal ID of the intercompany journal entry to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/intercompanyjournalentry/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "intercompanyjournalentry", "record_id": record_id}

        super().__init__(handler=_delete_intercompany_journal_entry, **kwargs)


class NetSuiteGetIntercompanyJournalEntry(Tool):
    """Retrieve a NetSuite intercompany journal entry by internal ID."""

    name: str = "netsuite_get_intercompany_journal_entry"
    description: str | None = "Retrieve a NetSuite intercompany journal entry by internal ID."
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
        async def _get_intercompany_journal_entry(
            record_id: str = Field(..., description="Internal ID of the intercompany journal entry to retrieve."),
            fields: str | None = Field(None, description="Comma-separated list of fields to return. Returns all fields if omitted."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/intercompanyjournalentry/{record_id}"

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

        super().__init__(handler=_get_intercompany_journal_entry, **kwargs)


class NetSuiteListIntercompanyJournalEntries(Tool):
    """List intercompany journal entries in NetSuite with optional filtering."""

    name: str = "netsuite_list_intercompany_journal_entries"
    description: str | None = "List intercompany journal entries in NetSuite with optional filtering."
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
        async def _list_intercompany_journal_entries(
            query: str | None = Field(None, description="Search query to filter intercompany journal entries."),
            limit: int = Field(100, description="Maximum number of entries to return (default 100)."),
            offset: int = Field(0, description="Number of records to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/intercompanyjournalentry"

            import httpx

            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if query:
                params["q"] = query

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth, "Content-Type": "application/json"}, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_intercompany_journal_entries, **kwargs)


class NetSuiteUpdateIntercompanyJournalEntry(Tool):
    """Update an existing intercompany journal entry in NetSuite."""

    name: str = "netsuite_update_intercompany_journal_entry"
    description: str | None = "Update an existing intercompany journal entry in NetSuite."
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
        async def _update_intercompany_journal_entry(
            record_id: str = Field(..., description="Internal ID of the intercompany journal entry to update."),
            data: dict[str, Any] = Field(..., description="Fields to update on the intercompany journal entry record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/intercompanyjournalentry/{record_id}"

            import httpx

            auth = _netsuite_auth_header("PATCH", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"updated": True, "record_type": "intercompanyjournalentry", "record_id": record_id}

        super().__init__(handler=_update_intercompany_journal_entry, **kwargs)


# ===========================================================================
# INVENTORY — BIN TRANSFER
# ===========================================================================

class NetSuiteCreateBinTransfer(Tool):
    """Create a new bin transfer record in NetSuite."""

    name: str = "netsuite_create_bin_transfer"
    description: str | None = "Create a new bin transfer record in NetSuite."
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
        async def _create_bin_transfer(
            subsidiary: str | None = Field(None, description="Internal ID of the subsidiary."),
            tran_date: str | None = Field(None, description="Transaction date (ISO 8601, e.g. '2024-01-15')."),
            memo: str | None = Field(None, description="Memo or notes for the bin transfer."),
            inventory: list[dict[str, Any]] | None = Field(None, description="List of inventory lines. Each line should include 'item' (id), 'quantity', 'fromBin' (id), and 'toBin' (id)."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/bintransfer"

            import httpx

            data: dict[str, Any] = {}
            if subsidiary:
                data["subsidiary"] = {"id": subsidiary}
            if tran_date:
                data["tranDate"] = tran_date
            if memo:
                data["memo"] = memo
            if inventory:
                data["inventory"] = {"items": inventory}

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
                    auth2 = _netsuite_auth_header("GET", f"{base}/bintransfer/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/bintransfer/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "bintransfer", "record_id": created_id, "location": location}

        super().__init__(handler=_create_bin_transfer, **kwargs)


class NetSuiteDeleteBinTransfer(Tool):
    """Delete a bin transfer from NetSuite by internal ID."""

    name: str = "netsuite_delete_bin_transfer"
    description: str | None = "Delete a bin transfer from NetSuite by internal ID."
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
        async def _delete_bin_transfer(
            record_id: str = Field(..., description="Internal ID of the bin transfer to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/bintransfer/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "bintransfer", "record_id": record_id}

        super().__init__(handler=_delete_bin_transfer, **kwargs)


class NetSuiteGetBinTransfer(Tool):
    """Retrieve a NetSuite bin transfer by internal ID."""

    name: str = "netsuite_get_bin_transfer"
    description: str | None = "Retrieve a NetSuite bin transfer by internal ID."
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
        async def _get_bin_transfer(
            record_id: str = Field(..., description="Internal ID of the bin transfer to retrieve."),
            fields: str | None = Field(None, description="Comma-separated list of fields to return. Returns all fields if omitted."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/bintransfer/{record_id}"

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

        super().__init__(handler=_get_bin_transfer, **kwargs)


class NetSuiteListBinTransfers(Tool):
    """List bin transfers in NetSuite with optional filtering."""

    name: str = "netsuite_list_bin_transfers"
    description: str | None = "List bin transfers in NetSuite with optional filtering."
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
        async def _list_bin_transfers(
            query: str | None = Field(None, description="Search query to filter bin transfers."),
            limit: int = Field(100, description="Maximum number of bin transfers to return (default 100)."),
            offset: int = Field(0, description="Number of records to skip for pagination."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/bintransfer"

            import httpx

            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if query:
                params["q"] = query

            auth = _netsuite_auth_header("GET", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={"Authorization": auth, "Content-Type": "application/json"}, params=params)
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_bin_transfers, **kwargs)


class NetSuiteUpdateBinTransfer(Tool):
    """Update an existing bin transfer in NetSuite."""

    name: str = "netsuite_update_bin_transfer"
    description: str | None = "Update an existing bin transfer in NetSuite."
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
        async def _update_bin_transfer(
            record_id: str = Field(..., description="Internal ID of the bin transfer to update."),
            data: dict[str, Any] = Field(..., description="Fields to update on the bin transfer record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/bintransfer/{record_id}"

            import httpx

            auth = _netsuite_auth_header("PATCH", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.patch(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"updated": True, "record_type": "bintransfer", "record_id": record_id}

        super().__init__(handler=_update_bin_transfer, **kwargs)


class NetSuiteUpsertBinTransfer(Tool):
    """Upsert (create or update) a bin transfer in NetSuite using an external ID."""

    name: str = "netsuite_upsert_bin_transfer"
    description: str | None = "Upsert (create or update) a bin transfer in NetSuite using an external ID."
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
        async def _upsert_bin_transfer(
            external_id: str = Field(..., description="External ID used to identify and upsert the bin transfer."),
            data: dict[str, Any] = Field(..., description="Fields to set on the bin transfer record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/bintransfer/eid:{external_id}"

            import httpx

            auth = _netsuite_auth_header("PUT", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.put(url, headers={"Authorization": auth, "Content-Type": "application/json"}, json=data)
                response.raise_for_status()
                return {"upserted": True, "record_type": "bintransfer", "external_id": external_id}

        super().__init__(handler=_upsert_bin_transfer, **kwargs)


# ===========================================================================
# PERIOD — ACCOUNTING PERIOD
# ===========================================================================

class NetSuiteCreateAccountingPeriod(Tool):
    """Create a new accounting period in NetSuite."""

    name: str = "netsuite_create_accounting_period"
    description: str | None = "Create a new accounting period in NetSuite."
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
        async def _create_accounting_period(
            period_name: str = Field(..., description="Name of the accounting period (e.g. 'Jan 2024')."),
            start_date: str = Field(..., description="Start date of the period (ISO 8601, e.g. '2024-01-01')."),
            end_date: str = Field(..., description="End date of the period (ISO 8601, e.g. '2024-01-31')."),
            fiscal_calendar: str | None = Field(None, description="Internal ID of the fiscal calendar."),
            is_quarter: bool = Field(False, description="If true, marks this period as a quarter."),
            is_year: bool = Field(False, description="If true, marks this period as a year."),
            return_record: bool = Field(False, description="If true, returns the created record."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/accountingperiod"

            import httpx

            data: dict[str, Any] = {
                "periodName": period_name,
                "startDate": start_date,
                "endDate": end_date,
            }
            if fiscal_calendar:
                data["fiscalCalendar"] = {"id": fiscal_calendar}
            if is_quarter:
                data["isQuarter"] = is_quarter
            if is_year:
                data["isYear"] = is_year

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
                    auth2 = _netsuite_auth_header("GET", f"{base}/accountingperiod/{created_id}", account_id, consumer_key, consumer_secret, token_id, token_secret)
                    r2 = await client.get(f"{base}/accountingperiod/{created_id}", headers={"Authorization": auth2, "Content-Type": "application/json"})
                    r2.raise_for_status()
                    return r2.json()
                return {"created": True, "record_type": "accountingperiod", "record_id": created_id, "location": location}

        super().__init__(handler=_create_accounting_period, **kwargs)


class NetSuiteDeleteAccountingPeriod(Tool):
    """Delete an accounting period from NetSuite by internal ID."""

    name: str = "netsuite_delete_accounting_period"
    description: str | None = "Delete an accounting period from NetSuite by internal ID."
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
        async def _delete_accounting_period(
            record_id: str = Field(..., description="Internal ID of the accounting period to delete."),
        ) -> Any:
            account_id, consumer_key, consumer_secret, token_id, token_secret = await _resolve_credentials(self)
            base = _build_base_url(account_id)
            url = f"{base}/accountingperiod/{record_id}"

            import httpx

            auth = _netsuite_auth_header("DELETE", url, account_id, consumer_key, consumer_secret, token_id, token_secret)
            async with httpx.AsyncClient() as client:
                response = await client.delete(url, headers={"Authorization": auth, "Content-Type": "application/json"})
                response.raise_for_status()
                return {"deleted": True, "record_type": "accountingperiod", "record_id": record_id}

        super().__init__(handler=_delete_accounting_period, **kwargs)
