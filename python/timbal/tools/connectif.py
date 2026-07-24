"""Connectif HTTP API tools.

Auth uses an API key (``keyId:keySecret``) in the Authorization header:
``Authorization: apiKey {apiKey}``.

Integration credentials (type: credentials):
- api_key: Full API key string (key id and secret joined with ``:``)
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal
from urllib.parse import quote

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_API_ROOT = "https://api.connectif.cloud"


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    text = str(value).strip()
    return text or None


def _join_url(path: str) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    return f"{_API_ROOT}{normalized}"


async def _resolve_api_key(tool: Any) -> str:
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    full_key = (
        _secret_value(creds.get("api_key"))
        or _secret_value(getattr(tool, "api_key", None))
        or _secret_value(os.getenv("CONNECTIF_API_KEY"))
    )
    if full_key:
        return full_key

    key_id = _secret_value(creds.get("api_key_id")) or _secret_value(os.getenv("CONNECTIF_API_KEY_ID"))
    key_secret = _secret_value(creds.get("api_key_secret")) or _secret_value(
        os.getenv("CONNECTIF_API_KEY_SECRET")
    )
    if key_id and key_secret:
        return f"{key_id}:{key_secret}"

    raise ValueError(
        "Connectif API key not found. Configure integration with api_key, or set CONNECTIF_API_KEY "
        "(format keyId:keySecret), or CONNECTIF_API_KEY_ID + CONNECTIF_API_KEY_SECRET."
    )


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"apiKey {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def _connectif_request(
    tool: Any,
    *,
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    import httpx

    api_key = await _resolve_api_key(tool)
    url = _join_url(path)
    headers = _auth_headers(api_key)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.request(
            method.upper(),
            url,
            headers=headers,
            params=params or None,
            json=json_body,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()


def _connectif_config_fields(tool: Any) -> dict[str, Any]:
    return {
        "integration": tool.integration,
        "api_key": tool.api_key,
    }


class _ConnectifTool(Tool):
    integration: Annotated[str, Integration("connectif")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config(_connectif_config_fields(self))}


# ---------------------------------------------------------------------------
# Escape hatch + store
# ---------------------------------------------------------------------------


class ConnectifRequest(_ConnectifTool):
    """Call any Connectif HTTP API endpoint."""

    name: str = "connectif_request"
    description: str | None = (
        "Call any Connectif HTTP API endpoint with method, path, optional query params and JSON body."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _request(
            method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = Field("GET", description="HTTP method."),
            path: str = Field(..., description="API path (e.g. 'contacts/email/user@example.com')."),
            query_params: dict[str, Any] | None = Field(None, description="Optional query string parameters."),
            body: dict[str, Any] | None = Field(None, description="Optional JSON request body."),
        ) -> Any:
            return await _connectif_request(
                self,
                method=method,
                path=path,
                params=query_params,
                json_body=body,
            )

        super().__init__(handler=_request, **kwargs)


class ConnectifGetStore(_ConnectifTool):
    """Get Connectif store details (health/metadata check)."""

    name: str = "connectif_get_store"
    description: str | None = "Get Connectif store details."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_store() -> Any:
            return await _connectif_request(self, method="GET", path="/store/")

        super().__init__(handler=_get_store, **kwargs)


# ---------------------------------------------------------------------------
# Contact fields
# ---------------------------------------------------------------------------


class ConnectifListContactFields(_ConnectifTool):
    """List contact field definitions for the account."""

    name: str = "connectif_list_contact_fields"
    description: str | None = "List contact field definitions (use field ids when patching contacts)."

    def __init__(self, **kwargs: Any) -> None:
        async def _list() -> Any:
            return await _connectif_request(self, method="GET", path="/contact-fields/")

        super().__init__(handler=_list, **kwargs)


class ConnectifGetContactField(_ConnectifTool):
    """Get a contact field definition by id."""

    name: str = "connectif_get_contact_field"
    description: str | None = "Get a contact field definition by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(field_id: str = Field(..., description="Contact field id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/contact-fields/{field_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifCreateContactField(_ConnectifTool):
    """Create a contact field definition."""

    name: str = "connectif_create_contact_field"
    description: str | None = "Create a contact field definition."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            field: dict[str, Any] = Field(..., description="Contact field definition JSON body."),
        ) -> Any:
            return await _connectif_request(self, method="POST", path="/contact-fields/", json_body=field)

        super().__init__(handler=_create, **kwargs)


class ConnectifDeleteContactField(_ConnectifTool):
    """Delete a contact field by id."""

    name: str = "connectif_delete_contact_field"
    description: str | None = "Delete a contact field by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(field_id: str = Field(..., description="Contact field id.")) -> Any:
            return await _connectif_request(self, method="DELETE", path=f"/contact-fields/{field_id}")

        super().__init__(handler=_delete, **kwargs)


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


class ConnectifGetContactByEmail(_ConnectifTool):
    """Get a contact by email."""

    name: str = "connectif_get_contact_by_email"
    description: str | None = "Get a contact by email."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(email: str = Field(..., description="Contact email address.")) -> Any:
            encoded = quote(email, safe="")
            return await _connectif_request(self, method="GET", path=f"/contacts/email/{encoded}")

        super().__init__(handler=_get, **kwargs)


class ConnectifGetContactByExternalId(_ConnectifTool):
    """Get a contact by external id."""

    name: str = "connectif_get_contact_by_external_id"
    description: str | None = "Get a contact by external id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(external_id: str = Field(..., description="Contact external id.")) -> Any:
            encoded = quote(external_id, safe="")
            return await _connectif_request(self, method="GET", path=f"/contacts/external-id/{encoded}")

        super().__init__(handler=_get, **kwargs)


class ConnectifUpsertContactByEmail(_ConnectifTool):
    """Create or update a contact by email (PATCH upsert)."""

    name: str = "connectif_upsert_contact_by_email"
    description: str | None = "Create or update a contact by email. Creates the contact if it does not exist."

    def __init__(self, **kwargs: Any) -> None:
        async def _upsert(
            email: str = Field(..., description="Contact email address."),
            updates: dict[str, Any] = Field(
                ...,
                description=(
                    "Contact fields to set. Standard keys: _name, _surname, _externalId, _birthdate, "
                    "_mobilePhone, _emailStatus, _newsletterSubscriptionStatus, _smsSubscriptionStatus, "
                    "_points, plus custom field ids."
                ),
            ),
        ) -> Any:
            encoded = quote(email, safe="")
            return await _connectif_request(
                self,
                method="PATCH",
                path=f"/contacts/email/{encoded}",
                json_body=updates,
            )

        super().__init__(handler=_upsert, **kwargs)


class ConnectifUpsertContactByExternalId(_ConnectifTool):
    """Create or update a contact by external id (PATCH upsert)."""

    name: str = "connectif_upsert_contact_by_external_id"
    description: str | None = "Create or update a contact by external id. Creates the contact if it does not exist."

    def __init__(self, **kwargs: Any) -> None:
        async def _upsert(
            external_id: str = Field(..., description="Contact external id."),
            updates: dict[str, Any] = Field(..., description="Contact fields to set (same schema as email upsert)."),
        ) -> Any:
            encoded = quote(external_id, safe="")
            return await _connectif_request(
                self,
                method="PATCH",
                path=f"/contacts/external-id/{encoded}",
                json_body=updates,
            )

        super().__init__(handler=_upsert, **kwargs)


class ConnectifDeleteContactByEmail(_ConnectifTool):
    """Delete a contact by email."""

    name: str = "connectif_delete_contact_by_email"
    description: str | None = "Delete a contact by email."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(email: str = Field(..., description="Contact email address.")) -> Any:
            encoded = quote(email, safe="")
            return await _connectif_request(self, method="DELETE", path=f"/contacts/email/{encoded}")

        super().__init__(handler=_delete, **kwargs)


class ConnectifDeleteContactByExternalId(_ConnectifTool):
    """Delete a contact by external id."""

    name: str = "connectif_delete_contact_by_external_id"
    description: str | None = "Delete a contact by external id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(external_id: str = Field(..., description="Contact external id.")) -> Any:
            encoded = quote(external_id, safe="")
            return await _connectif_request(self, method="DELETE", path=f"/contacts/external-id/{encoded}")

        super().__init__(handler=_delete, **kwargs)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


class ConnectifUpsertProduct(_ConnectifTool):
    """Create or update a product by store product id (PATCH upsert)."""

    name: str = "connectif_upsert_product"
    description: str | None = "Create or update a product by store product id."

    def __init__(self, **kwargs: Any) -> None:
        async def _upsert(
            product_id: str = Field(..., description="Product id in your store."),
            updates: dict[str, Any] = Field(
                ...,
                description="Product fields: name, unitPrice, categories, brand, description, availability, etc.",
            ),
        ) -> Any:
            encoded = quote(product_id, safe="")
            return await _connectif_request(
                self,
                method="PATCH",
                path=f"/products/product-id/{encoded}",
                json_body=updates,
            )

        super().__init__(handler=_upsert, **kwargs)


class ConnectifGetProductByStoreId(_ConnectifTool):
    """Get a product by store product id."""

    name: str = "connectif_get_product_by_store_id"
    description: str | None = "Get a product by store product id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(product_id: str = Field(..., description="Product id in your store.")) -> Any:
            encoded = quote(product_id, safe="")
            return await _connectif_request(self, method="GET", path=f"/products/product-id/{encoded}")

        super().__init__(handler=_get, **kwargs)


class ConnectifDeleteProductByStoreId(_ConnectifTool):
    """Delete a product by store product id."""

    name: str = "connectif_delete_product_by_store_id"
    description: str | None = "Delete a product by store product id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(product_id: str = Field(..., description="Product id in your store.")) -> Any:
            encoded = quote(product_id, safe="")
            return await _connectif_request(self, method="DELETE", path=f"/products/product-id/{encoded}")

        super().__init__(handler=_delete, **kwargs)


class ConnectifGetProduct(_ConnectifTool):
    """Get a product by Connectif internal id."""

    name: str = "connectif_get_product"
    description: str | None = "Get a product by Connectif internal id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(product_id: str = Field(..., description="Connectif product id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/products/{product_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifDeleteProduct(_ConnectifTool):
    """Delete a product by Connectif internal id."""

    name: str = "connectif_delete_product"
    description: str | None = "Delete a product by Connectif internal id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(product_id: str = Field(..., description="Connectif product id.")) -> Any:
            return await _connectif_request(self, method="DELETE", path=f"/products/{product_id}")

        super().__init__(handler=_delete, **kwargs)


# ---------------------------------------------------------------------------
# Purchases
# ---------------------------------------------------------------------------


class ConnectifCreatePurchase(_ConnectifTool):
    """Create a purchase record."""

    name: str = "connectif_create_purchase"
    description: str | None = (
        "Create a purchase. Requires purchaseId, products[], and contactEmail or contactExternalId."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            purchase: dict[str, Any] = Field(..., description="Purchase JSON body."),
            trigger_purchase_event: bool | None = Field(
                None,
                description="Whether to trigger the On-Purchase workflow node. Defaults to true in Connectif.",
            ),
        ) -> Any:
            params: dict[str, Any] = {}
            if trigger_purchase_event is not None:
                params["triggerPurchaseEvent"] = trigger_purchase_event
            return await _connectif_request(
                self,
                method="POST",
                path="/purchases/",
                params=params or None,
                json_body=purchase,
            )

        super().__init__(handler=_create, **kwargs)


class ConnectifGetPurchaseByStoreId(_ConnectifTool):
    """Get a purchase by store purchase id."""

    name: str = "connectif_get_purchase_by_store_id"
    description: str | None = "Get a purchase by store purchase id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(purchase_id: str = Field(..., description="Purchase id in your store.")) -> Any:
            encoded = quote(purchase_id, safe="")
            return await _connectif_request(self, method="GET", path=f"/purchases/purchase-id/{encoded}")

        super().__init__(handler=_get, **kwargs)


class ConnectifDeletePurchaseByStoreId(_ConnectifTool):
    """Delete a purchase by store purchase id."""

    name: str = "connectif_delete_purchase_by_store_id"
    description: str | None = "Delete a purchase by store purchase id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(purchase_id: str = Field(..., description="Purchase id in your store.")) -> Any:
            encoded = quote(purchase_id, safe="")
            return await _connectif_request(self, method="DELETE", path=f"/purchases/purchase-id/{encoded}")

        super().__init__(handler=_delete, **kwargs)


class ConnectifGetPurchase(_ConnectifTool):
    """Get a purchase by Connectif internal id."""

    name: str = "connectif_get_purchase"
    description: str | None = "Get a purchase by Connectif internal id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(purchase_id: str = Field(..., description="Connectif purchase id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/purchases/{purchase_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifDeletePurchase(_ConnectifTool):
    """Delete a purchase by Connectif internal id."""

    name: str = "connectif_delete_purchase"
    description: str | None = "Delete a purchase by Connectif internal id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(purchase_id: str = Field(..., description="Connectif purchase id.")) -> Any:
            return await _connectif_request(self, method="DELETE", path=f"/purchases/{purchase_id}")

        super().__init__(handler=_delete, **kwargs)


# ---------------------------------------------------------------------------
# Custom events + workflows + coupon sets
# ---------------------------------------------------------------------------


class ConnectifCreateCustomEventByAlias(_ConnectifTool):
    """Create a custom event by event type alias."""

    name: str = "connectif_create_custom_event_by_alias"
    description: str | None = "Trigger a custom event by event type alias."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            alias: str = Field(..., description="Custom event type alias."),
            event: dict[str, Any] = Field(..., description="Custom event JSON body."),
        ) -> Any:
            encoded = quote(alias, safe="")
            return await _connectif_request(
                self,
                method="POST",
                path=f"/custom-events/alias/{encoded}",
                json_body=event,
            )

        super().__init__(handler=_create, **kwargs)


class ConnectifCreateCustomEventById(_ConnectifTool):
    """Create a custom event by event type id."""

    name: str = "connectif_create_custom_event_by_id"
    description: str | None = "Trigger a custom event by event type id."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            event_type_id: str = Field(..., description="Custom event type id."),
            event: dict[str, Any] = Field(..., description="Custom event JSON body."),
        ) -> Any:
            return await _connectif_request(
                self,
                method="POST",
                path=f"/custom-events/id/{event_type_id}",
                json_body=event,
            )

        super().__init__(handler=_create, **kwargs)


class ConnectifListWorkflows(_ConnectifTool):
    """List workflows."""

    name: str = "connectif_list_workflows"
    description: str | None = "List Connectif workflows."

    def __init__(self, **kwargs: Any) -> None:
        async def _list() -> Any:
            return await _connectif_request(self, method="GET", path="/workflows/")

        super().__init__(handler=_list, **kwargs)


class ConnectifGetWorkflow(_ConnectifTool):
    """Get a workflow by id."""

    name: str = "connectif_get_workflow"
    description: str | None = "Get a workflow by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(workflow_id: str = Field(..., description="Workflow id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/workflows/{workflow_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifListCouponSets(_ConnectifTool):
    """List coupon sets."""

    name: str = "connectif_list_coupon_sets"
    description: str | None = "List coupon sets (vouchers)."

    def __init__(self, **kwargs: Any) -> None:
        async def _list() -> Any:
            return await _connectif_request(self, method="GET", path="/coupon-sets/")

        super().__init__(handler=_list, **kwargs)


class ConnectifGetCouponSet(_ConnectifTool):
    """Get a coupon set by id."""

    name: str = "connectif_get_coupon_set"
    description: str | None = "Get a coupon set by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(coupon_set_id: str = Field(..., description="Coupon set id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/coupon-sets/{coupon_set_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifListCustomEventTypes(_ConnectifTool):
    """List custom event types."""

    name: str = "connectif_list_custom_event_types"
    description: str | None = "List custom event types."

    def __init__(self, **kwargs: Any) -> None:
        async def _list() -> Any:
            return await _connectif_request(self, method="GET", path="/custom-event-types/")

        super().__init__(handler=_list, **kwargs)


class ConnectifCreateCustomEventType(_ConnectifTool):
    """Create a custom event type."""

    name: str = "connectif_create_custom_event_type"
    description: str | None = "Create a custom event type."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            event_type: dict[str, Any] = Field(..., description="Custom event type JSON body."),
        ) -> Any:
            return await _connectif_request(self, method="POST", path="/custom-event-types/", json_body=event_type)

        super().__init__(handler=_create, **kwargs)


class ConnectifGetCustomEventType(_ConnectifTool):
    """Get a custom event type by id."""

    name: str = "connectif_get_custom_event_type"
    description: str | None = "Get a custom event type by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(event_type_id: str = Field(..., description="Custom event type id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/custom-event-types/{event_type_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifDeleteCustomEventType(_ConnectifTool):
    """Delete a custom event type by id."""

    name: str = "connectif_delete_custom_event_type"
    description: str | None = "Delete a custom event type by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(event_type_id: str = Field(..., description="Custom event type id.")) -> Any:
            return await _connectif_request(self, method="DELETE", path=f"/custom-event-types/{event_type_id}")

        super().__init__(handler=_delete, **kwargs)


# ---------------------------------------------------------------------------
# Imports + exports
# ---------------------------------------------------------------------------


class ConnectifListImports(_ConnectifTool):
    """List imports."""

    name: str = "connectif_list_imports"
    description: str | None = "List bulk imports."

    def __init__(self, **kwargs: Any) -> None:
        async def _list() -> Any:
            return await _connectif_request(self, method="GET", path="/imports/")

        super().__init__(handler=_list, **kwargs)


class ConnectifCreateImport(_ConnectifTool):
    """Create a bulk import."""

    name: str = "connectif_create_import"
    description: str | None = "Create a bulk import."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            import_job: dict[str, Any] = Field(..., description="Import job JSON body."),
        ) -> Any:
            return await _connectif_request(self, method="POST", path="/imports/", json_body=import_job)

        super().__init__(handler=_create, **kwargs)


class ConnectifGetImport(_ConnectifTool):
    """Get an import by id."""

    name: str = "connectif_get_import"
    description: str | None = "Get an import by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(import_id: str = Field(..., description="Import id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/imports/{import_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifDeleteImport(_ConnectifTool):
    """Delete an import by id."""

    name: str = "connectif_delete_import"
    description: str | None = "Delete an import by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(import_id: str = Field(..., description="Import id.")) -> Any:
            return await _connectif_request(self, method="DELETE", path=f"/imports/{import_id}")

        super().__init__(handler=_delete, **kwargs)


class ConnectifListExports(_ConnectifTool):
    """List exports."""

    name: str = "connectif_list_exports"
    description: str | None = "List data exports."

    def __init__(self, **kwargs: Any) -> None:
        async def _list() -> Any:
            return await _connectif_request(self, method="GET", path="/exports/")

        super().__init__(handler=_list, **kwargs)


class ConnectifCreateExport(_ConnectifTool):
    """Create a data export."""

    name: str = "connectif_create_export"
    description: str | None = "Create a data export."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            export_job: dict[str, Any] = Field(..., description="Export job JSON body."),
        ) -> Any:
            return await _connectif_request(self, method="POST", path="/exports/", json_body=export_job)

        super().__init__(handler=_create, **kwargs)


class ConnectifCreateDataExplorerExport(_ConnectifTool):
    """Create a data explorer export."""

    name: str = "connectif_create_data_explorer_export"
    description: str | None = "Create a data explorer export."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            export_job: dict[str, Any] = Field(..., description="Data explorer export JSON body."),
        ) -> Any:
            return await _connectif_request(
                self,
                method="POST",
                path="/exports/type/data-explorer",
                json_body=export_job,
            )

        super().__init__(handler=_create, **kwargs)


class ConnectifGetExport(_ConnectifTool):
    """Get an export by id."""

    name: str = "connectif_get_export"
    description: str | None = "Get an export by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(export_id: str = Field(..., description="Export id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/exports/{export_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifDeleteExport(_ConnectifTool):
    """Delete an export by id."""

    name: str = "connectif_delete_export"
    description: str | None = "Delete an export by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(export_id: str = Field(..., description="Export id.")) -> Any:
            return await _connectif_request(self, method="DELETE", path=f"/exports/{export_id}")

        super().__init__(handler=_delete, **kwargs)


# ---------------------------------------------------------------------------
# Custom integrations
# ---------------------------------------------------------------------------


class ConnectifListCustomIntegrations(_ConnectifTool):
    """List custom integrations."""

    name: str = "connectif_list_custom_integrations"
    description: str | None = "List custom integrations."

    def __init__(self, **kwargs: Any) -> None:
        async def _list() -> Any:
            return await _connectif_request(self, method="GET", path="/custom-integrations/")

        super().__init__(handler=_list, **kwargs)


class ConnectifCreateCustomIntegration(_ConnectifTool):
    """Create a custom integration."""

    name: str = "connectif_create_custom_integration"
    description: str | None = "Create a custom integration."

    def __init__(self, **kwargs: Any) -> None:
        async def _create(
            integration: dict[str, Any] = Field(..., description="Custom integration JSON body."),
        ) -> Any:
            return await _connectif_request(
                self,
                method="POST",
                path="/custom-integrations/",
                json_body=integration,
            )

        super().__init__(handler=_create, **kwargs)


class ConnectifGetCustomIntegration(_ConnectifTool):
    """Get a custom integration by id."""

    name: str = "connectif_get_custom_integration"
    description: str | None = "Get a custom integration by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _get(integration_id: str = Field(..., description="Custom integration id.")) -> Any:
            return await _connectif_request(self, method="GET", path=f"/custom-integrations/{integration_id}")

        super().__init__(handler=_get, **kwargs)


class ConnectifPatchCustomIntegration(_ConnectifTool):
    """Update a custom integration by id."""

    name: str = "connectif_patch_custom_integration"
    description: str | None = "Update a custom integration by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _patch(
            integration_id: str = Field(..., description="Custom integration id."),
            updates: dict[str, Any] = Field(..., description="Custom integration fields to update."),
        ) -> Any:
            return await _connectif_request(
                self,
                method="PATCH",
                path=f"/custom-integrations/{integration_id}",
                json_body=updates,
            )

        super().__init__(handler=_patch, **kwargs)


class ConnectifDeleteCustomIntegration(_ConnectifTool):
    """Delete a custom integration by id."""

    name: str = "connectif_delete_custom_integration"
    description: str | None = "Delete a custom integration by id."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete(integration_id: str = Field(..., description="Custom integration id.")) -> Any:
            return await _connectif_request(self, method="DELETE", path=f"/custom-integrations/{integration_id}")

        super().__init__(handler=_delete, **kwargs)
