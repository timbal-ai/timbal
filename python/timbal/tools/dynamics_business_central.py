import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BC_API_SCOPE = "https://api.businesscentral.dynamics.com/.default"
_BC_API_BASE = "https://api.businesscentral.dynamics.com/v2.0"


def _bc_url(tenant_id: str, environment_name: str, company_id: str, path: str) -> str:
    """Build Business Central API URL from tenant, environment, and company."""
    base = f"{_BC_API_BASE}/{tenant_id}/{environment_name}/api/v2.0"
    return f"{base}/companies({company_id})/{path}"


async def _get_token_from_client_credentials(
    tenant_id: str, client_id: str, client_secret: str
) -> str:
    """Obtain access token via OAuth 2.0 client credentials flow."""
    import httpx

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": _BC_API_SCOPE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return response.json()["access_token"]


async def _resolve_credentials(tool: Any) -> tuple[str, str, str, str]:
    """Resolve token, tenant_id, environment_name, company_id from integration credentials.

    Integration credentials (type: credentials):
    - tenant_id: Azure AD tenant ID
    - environment_name: BC environment (e.g. Production, Sandbox)
    - client_id: Azure AD app client ID
    - client_secret: Azure AD app client secret

    company_id: Required for BC API. Set DYNAMICS_BC_COMPANY_ID or in integration.
    """
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    tenant_id = creds.get("tenant_id") or getattr(tool, "tenant_id", None) or os.getenv("DYNAMICS_BC_TENANT_ID")
    environment_name = (
        creds.get("environment_name")
        or getattr(tool, "environment_name", None)
        or os.getenv("DYNAMICS_BC_ENVIRONMENT", "Production")
    )
    client_id = creds.get("client_id") or getattr(tool, "client_id", None) or os.getenv("DYNAMICS_BC_CLIENT_ID")
    client_secret = creds.get("client_secret") or (
        tool.client_secret.get_secret_value() if getattr(tool, "client_secret", None) and tool.client_secret else None
    ) or os.getenv("DYNAMICS_BC_CLIENT_SECRET")
    company_id = creds.get("company_id") or getattr(tool, "company_id", None) or os.getenv("DYNAMICS_BC_COMPANY_ID")

    token = await _get_token_from_client_credentials(tenant_id, client_id, client_secret)

    if not token or not tenant_id:
        raise ValueError(
            "Dynamics 365 Business Central credentials not found. Configure integration with "
            "tenant_id, environment_name, client_id, client_secret."
        )
    if not company_id:
        raise ValueError(
            "Dynamics 365 Business Central company_id is required. Set DYNAMICS_BC_COMPANY_ID."
        )
    return token, tenant_id, environment_name, company_id


class DynamicsBCCreateCustomer(Tool):
    """Creates a new customer in Dynamics 365 Business Central. See the documentation."""

    name: str = "dynamics_bc_create_customer"
    description: str | None = "Creates a new customer. See the documentation."
    integration: Annotated[str, Integration("dynamics_business_central")] | None = None
    tenant_id: str | None = None
    environment_name: str | None = None
    company_id: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "tenant_id": self.tenant_id,
                "environment_name": self.environment_name,
                "company_id": self.company_id,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_customer(
            display_name: str = Field(..., description="Customer display name"),
            customer_type: str = Field("Company", description="Customer type: 'Company' or 'Person'"),
            address_line1: str | None = Field(None, description="Address line 1"),
            address_line2: str | None = Field(None, description="Address line 2"),
            city: str | None = Field(None, description="City"),
            state: str | None = Field(None, description="State or region"),
            country: str | None = Field(None, description="Country code (e.g. US, GB)"),
            postal_code: str | None = Field(None, description="Postal/ZIP code"),
            phone_number: str | None = Field(None, description="Phone number"),
            email: str | None = Field(None, description="Email address"),
            website: str | None = Field(None, description="Website URL"),
            tax_liable: bool | None = Field(None, description="Whether customer is tax liable"),
            currency_code: str | None = Field(None, description="Currency code (e.g. USD, EUR)"),
        ) -> Any:
            token, tenant_id, environment_name, company_id = await _resolve_credentials(self)

            payload: dict[str, Any] = {"displayName": display_name, "type": customer_type}
            if address_line1 is not None:
                payload["addressLine1"] = address_line1
            if address_line2 is not None:
                payload["addressLine2"] = address_line2
            if city is not None:
                payload["city"] = city
            if state is not None:
                payload["state"] = state
            if country is not None:
                payload["country"] = country
            if postal_code is not None:
                payload["postalCode"] = postal_code
            if phone_number is not None:
                payload["phoneNumber"] = phone_number
            if email is not None:
                payload["email"] = email
            if website is not None:
                payload["website"] = website
            if tax_liable is not None:
                payload["taxLiable"] = tax_liable
            if currency_code is not None:
                payload["currencyCode"] = currency_code

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _bc_url(tenant_id, environment_name, company_id, "customers"),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_customer, **kwargs)


class DynamicsBCUpdateCustomer(Tool):

    name: str = "dynamics_bc_update_customer"
    description: str | None = "Updates an existing customer. See the documentation."
    integration: Annotated[str, Integration("dynamics_business_central")] | None = None
    tenant_id: str | None = None
    environment_name: str | None = None
    company_id: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "tenant_id": self.tenant_id,
                "environment_name": self.environment_name,
                "company_id": self.company_id,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_customer(
            customer_id: str = Field(..., description="Business Central customer ID (GUID)"),
            display_name: str | None = Field(None, description="Customer display name"),
            address_line1: str | None = Field(None, description="Address line 1"),
            address_line2: str | None = Field(None, description="Address line 2"),
            city: str | None = Field(None, description="City"),
            state: str | None = Field(None, description="State or region"),
            country: str | None = Field(None, description="Country code"),
            postal_code: str | None = Field(None, description="Postal/ZIP code"),
            phone_number: str | None = Field(None, description="Phone number"),
            email: str | None = Field(None, description="Email address"),
            website: str | None = Field(None, description="Website URL"),
        ) -> Any:
            token, tenant_id, environment_name, company_id = await _resolve_credentials(self)

            payload: dict[str, Any] = {}
            if display_name is not None:
                payload["displayName"] = display_name
            if address_line1 is not None:
                payload["addressLine1"] = address_line1
            if address_line2 is not None:
                payload["addressLine2"] = address_line2
            if city is not None:
                payload["city"] = city
            if state is not None:
                payload["state"] = state
            if country is not None:
                payload["country"] = country
            if postal_code is not None:
                payload["postalCode"] = postal_code
            if phone_number is not None:
                payload["phoneNumber"] = phone_number
            if email is not None:
                payload["email"] = email
            if website is not None:
                payload["website"] = website

            if not payload:
                return {"updated": False, "message": "No fields to update"}

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    _bc_url(tenant_id, environment_name, company_id, f"customers({customer_id})"),
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_customer, **kwargs)


class DynamicsBCGetSalesOrder(Tool):

    name: str = "dynamics_bc_get_sales_order"
    description: str | None = "Retrieves a sales order by ID. See the documentation."
    integration: Annotated[str, Integration("dynamics_business_central")] | None = None
    tenant_id: str | None = None
    environment_name: str | None = None
    company_id: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "tenant_id": self.tenant_id,
                "environment_name": self.environment_name,
                "company_id": self.company_id,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sales_order(
            sales_order_id: str = Field(..., description="Business Central sales order ID (GUID)"),
            expand: str | None = Field(
                None,
                description="OData $expand for related entities, e.g. 'customerFinancialDetails,picture'",
            ),
        ) -> Any:
            token, tenant_id, environment_name, company_id = await _resolve_credentials(self)

            url = _bc_url(tenant_id, environment_name, company_id, f"salesOrders({sales_order_id})")
            params: dict[str, str] = {}
            if expand:
                params["$expand"] = expand

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    params=params if params else None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_sales_order, **kwargs)
