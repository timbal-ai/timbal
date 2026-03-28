import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_DATAVERSE_API_VERSION = "v9.2"


def _dataverse_base(environment_url: str) -> str:
    """Build Dataverse Web API base URL from environment URL."""
    base = environment_url.rstrip("/")
    if "/api/data" not in base:
        base = f"{base}/api/data/{_DATAVERSE_API_VERSION}"
    return base


async def _get_token_from_client_credentials(
    tenant_id: str, client_id: str, client_secret: str, resource: str
) -> str:
    """Obtain access token via OAuth 2.0 client credentials flow for Dataverse."""
    import httpx

    scope = f"{resource.rstrip('/')}/.default"
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": scope,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return response.json()["access_token"]


def _build_environment_url(environment_name: str) -> str:
    """Build Dataverse URL from environment name (e.g. org name)."""
    org = environment_name.strip().lower()
    return f"https://{org}.crm.dynamics.com"


async def _resolve_credentials(tool: Any) -> tuple[str, str]:
    """Resolve token and environment_url from integration credentials.

    Integration credentials (type: credentials):
    - tenant_id: Azure AD tenant ID
    - environment_name: Org/environment name (e.g. Production, myorg)
    - client_id: Azure AD app client ID
    - client_secret: Azure AD app client secret
    """
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    tenant_id = creds.get("tenant_id") or getattr(tool, "tenant_id", None) or os.getenv("DYNAMICS_SALES_TENANT_ID")
    environment_name = (
        creds.get("environment_name")
        or getattr(tool, "environment_name", None)
        or os.getenv("DYNAMICS_SALES_ENVIRONMENT_NAME")
    )
    client_id = creds.get("client_id") or getattr(tool, "client_id", None) or os.getenv("DYNAMICS_SALES_CLIENT_ID")
    client_secret = creds.get("client_secret") or (
        tool.client_secret.get_secret_value() if getattr(tool, "client_secret", None) and tool.client_secret else None
    ) or os.getenv("DYNAMICS_SALES_CLIENT_SECRET")

    environment_url = _build_environment_url(environment_name)

    resource = environment_url.rstrip("/")
    token = await _get_token_from_client_credentials(
        tenant_id, client_id, client_secret, resource
    )

    if not token or not environment_url:
        raise ValueError(
            "Dynamics 365 Sales credentials not found. Configure integration with "
            "tenant_id, environment_name, client_id, client_secret."
        )
    return token, environment_url


class DynamicsSalesFindContact(Tool):
    """Search for a contact by id, name, or using a custom filter. See the documentation."""

    name: str = "dynamics_sales_find_contact"
    description: str | None = "Search for a contact by id, name, or using a custom filter. See the documentation."
    integration: Annotated[str, Integration("dynamics_sales")] | None = None
    tenant_id: str | None = None
    environment_name: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "tenant_id": self.tenant_id,
                "environment_name": self.environment_name,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _find_contact(
            contact_id: str | None = Field(
                None,
                description="Contact ID (GUID) to retrieve a specific contact. If provided, name and filter are ignored.",
            ),
            name: str | None = Field(
                None,
                description="Search contacts by name (partial match on fullname). Ignored if contact_id is provided.",
            ),
            filter_query: str | None = Field(
                None,
                description="OData $filter expression, e.g. \"contains(fullname,'Smith')\" or \"emailaddress1 eq 'test@example.com'\". Ignored if contact_id or name is provided.",
            ),
            select: str | None = Field(
                None,
                description="Comma-separated list of properties to return, e.g. 'fullname,emailaddress1,telephone1'",
            ),
            top: int = Field(10, description="Maximum number of contacts to return (default 10, max 5000)"),
        ) -> Any:
            token, environment_url = await _resolve_credentials(self)
            base = _dataverse_base(environment_url)

            import httpx

            if contact_id:
                url = f"{base}/contacts({contact_id})"
                params: dict[str, str | int] = {}
                if select:
                    params["$select"] = select
            else:
                url = f"{base}/contacts"
                params = {"$top": min(top, 5000)}
                if filter_query:
                    params["$filter"] = filter_query
                elif name:
                    escaped = name.replace("'", "''")
                    params["$filter"] = f"contains(fullname,'{escaped}')"
                if select:
                    params["$select"] = select

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "OData-MaxVersion": "4.0",
                        "OData-Version": "4.0",
                        "Accept": "application/json",
                    },
                    params=params if params else None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_find_contact, **kwargs)


class DynamicsSalesCreateCustomEntity(Tool):
    """Create a custom entity. See the documentation."""

    name: str = "dynamics_sales_create_custom_entity"
    description: str | None = "Create a custom entity. See the documentation."
    integration: Annotated[str, Integration("dynamics_sales")] | None = None
    tenant_id: str | None = None
    environment_name: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "tenant_id": self.tenant_id,
                "environment_name": self.environment_name,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_custom_entity(
            entity_set_name: str = Field(
                ...,
                description="Entity set name for the custom table (e.g. cr123_myentities). Found in Power Apps → Tables → Advanced → Copy set name.",
            ),
            data: dict[str, Any] = Field(
                ...,
                description="JSON object with entity attributes. Use logical names for columns (e.g. cr123_name, cr123_description).",
            ),
            return_representation: bool = Field(
                False,
                description="If true, returns the created record in the response.",
            ),
        ) -> Any:
            token, environment_url = await _resolve_credentials(self)
            base = _dataverse_base(environment_url)
            url = f"{base}/{entity_set_name}"

            import httpx

            headers: dict[str, str] = {
                "Authorization": f"Bearer {token}",
                "OData-MaxVersion": "4.0",
                "OData-Version": "4.0",
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
            }
            if return_representation:
                headers["Prefer"] = "return=representation"

            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                if return_representation and response.content:
                    return response.json()
                entity_id = response.headers.get("OData-EntityId")
                return {"created": True, "entity_id": entity_id, "entity_set_name": entity_set_name}

        super().__init__(handler=_create_custom_entity, **kwargs)


class DynamicsSalesCreateOpportunity(Tool):
    """Create an opportunity (oportunitat) in Microsoft Dynamics 365 Sales CRM."""

    name: str = "dynamics_sales_create_opportunity"
    description: str | None = "Create an opportunity (oportunitat) in Microsoft Dynamics 365 Sales CRM."
    integration: Annotated[str, Integration("dynamics_sales")] | None = None
    tenant_id: str | None = None
    environment_name: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "tenant_id": self.tenant_id,
                "environment_name": self.environment_name,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_opportunity(
            name: str = Field(
                ...,
                description="Name of the opportunity / project (Nombre del Proyecto).",
            ),
            account_id: str | None = Field(
                None,
                description="GUID of the account (Cuenta) to associate with the opportunity.",
            ),
            contact_id: str | None = Field(
                None,
                description="GUID of the primary contact to associate with the opportunity.",
            ),
            budget_amount: float | None = Field(
                None,
                description="Estimated budget / revenue amount (Presupuesto) in the org's base currency.",
            ),
            description: str | None = Field(
                None,
                description="Additional notes or description for the opportunity.",
            ),
            estimated_close_date: str | None = Field(
                None,
                description="Estimated close date in ISO 8601 format (YYYY-MM-DD), e.g. '2026-06-30'.",
            ),
            owner_id: str | None = Field(
                None,
                description="GUID of the system user who owns this opportunity (Propietario). Defaults to the integration user if omitted.",
            ),
            return_representation: bool = Field(
                False,
                description="If true, returns the created opportunity record in the response.",
            ),
        ) -> Any:
            token, environment_url = await _resolve_credentials(self)
            base = _dataverse_base(environment_url)
            url = f"{base}/opportunities"

            import httpx

            data: dict[str, Any] = {"name": name}
            if budget_amount is not None:
                data["budgetamount"] = budget_amount
            if description is not None:
                data["description"] = description
            if estimated_close_date is not None:
                data["estimatedclosedate"] = estimated_close_date
            if account_id is not None:
                data["customerid_account@odata.bind"] = f"/accounts({account_id})"
            if contact_id is not None:
                data["customerid_contact@odata.bind"] = f"/contacts({contact_id})"
            if owner_id is not None:
                data["ownerid@odata.bind"] = f"/systemusers({owner_id})"

            headers: dict[str, str] = {
                "Authorization": f"Bearer {token}",
                "OData-MaxVersion": "4.0",
                "OData-Version": "4.0",
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
            }
            if return_representation:
                headers["Prefer"] = "return=representation"

            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                if return_representation and response.content:
                    return response.json()
                entity_id = response.headers.get("OData-EntityId")
                return {"created": True, "opportunity_id": entity_id}

        super().__init__(handler=_create_opportunity, **kwargs)
