import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_CONTENT_BASE = "https://shoppingcontent.googleapis.com/content/v2.1"


def _resolve_merchant_id(tool: Any, merchant_id: str | None) -> str:
    resolved = merchant_id or tool.default_merchant_id or os.getenv("GOOGLE_MERCHANT_ID")
    if not resolved:
        raise ValueError(
            "Merchant ID is required. Pass merchant_id, set default_merchant_id on the tool, or set GOOGLE_MERCHANT_ID."
        )
    return str(resolved)


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    if tool.token is not None:
        return tool.token.get_secret_value()
    raise ValueError("Google Merchant Center credentials not found. Configure an integration or pass token.")


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class GoogleMerchantListAccounts(Tool):
    name: str = "google_merchant_list_accounts"
    description: str | None = "List Merchant Center accounts accessible to the authenticated user."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), **self._annotate_config({"integration": self.integration, "token": self.token})}

    def __init__(self, **kwargs: Any) -> None:
        async def _list_accounts(
            max_results: int = Field(250, description="Maximum accounts to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"maxResults": max_results}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CONTENT_BASE}/accounts",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_accounts, **kwargs)


class GoogleMerchantGetAccount(Tool):
    name: str = "google_merchant_get_account"
    description: str | None = "Get Merchant Center account details."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_account(
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
            account_id: str | None = Field(None, description="Sub-account ID (defaults to merchant_id)"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            acct = account_id or merchant
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CONTENT_BASE}/{merchant}/accounts/{acct}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_account, **kwargs)


class GoogleMerchantListProducts(Tool):
    name: str = "google_merchant_list_products"
    description: str | None = "List products in a Merchant Center account."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_products(
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
            max_results: int = Field(250, description="Maximum products to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            params: dict[str, Any] = {"maxResults": max_results}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CONTENT_BASE}/{merchant}/products",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_products, **kwargs)


class GoogleMerchantGetProduct(Tool):
    name: str = "google_merchant_get_product"
    description: str | None = "Get a product from Merchant Center by product ID."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_product(
            product_id: str = Field(
                ..., description="Product ID in channel:contentLanguage:targetCountry:offerId format"
            ),
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CONTENT_BASE}/{merchant}/products/{product_id}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_product, **kwargs)


class GoogleMerchantInsertProduct(Tool):
    name: str = "google_merchant_insert_product"
    description: str | None = "Insert a product into Merchant Center."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _insert_product(
            product: dict = Field(..., description="Product resource object"),
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(
                    f"{_CONTENT_BASE}/{merchant}/products",
                    headers=_auth_headers(token),
                    json=product,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_insert_product, **kwargs)


class GoogleMerchantUpdateProduct(Tool):
    name: str = "google_merchant_update_product"
    description: str | None = "Update an existing Merchant Center product."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_product(
            product_id: str = Field(..., description="Product ID"),
            product: dict = Field(..., description="Updated product resource object"),
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.put(
                    f"{_CONTENT_BASE}/{merchant}/products/{product_id}",
                    headers=_auth_headers(token),
                    json=product,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_product, **kwargs)


class GoogleMerchantDeleteProduct(Tool):
    name: str = "google_merchant_delete_product"
    description: str | None = "Delete a product from Merchant Center."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_product(
            product_id: str = Field(..., description="Product ID to delete"),
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.delete(
                    f"{_CONTENT_BASE}/{merchant}/products/{product_id}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return {"deleted": True, "productId": product_id}

        super().__init__(handler=_delete_product, **kwargs)


class GoogleMerchantListProductStatuses(Tool):
    name: str = "google_merchant_list_product_statuses"
    description: str | None = "List product statuses and issues in Merchant Center."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_product_statuses(
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
            max_results: int = Field(250, description="Maximum statuses to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            params: dict[str, Any] = {"maxResults": max_results}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CONTENT_BASE}/{merchant}/productstatuses",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_product_statuses, **kwargs)


class GoogleMerchantGetProductStatus(Tool):
    name: str = "google_merchant_get_product_status"
    description: str | None = "Get status and issues for a specific Merchant Center product."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_product_status(
            product_id: str = Field(..., description="Product ID"),
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CONTENT_BASE}/{merchant}/productstatuses/{product_id}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_product_status, **kwargs)


class GoogleMerchantListDatafeeds(Tool):
    name: str = "google_merchant_list_datafeeds"
    description: str | None = "List datafeeds configured in Merchant Center."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_datafeeds(
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
            max_results: int = Field(250, description="Maximum datafeeds to return"),
            page_token: str | None = Field(None, description="Pagination token"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            params: dict[str, Any] = {"maxResults": max_results}
            if page_token:
                params["pageToken"] = page_token

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CONTENT_BASE}/{merchant}/datafeeds",
                    headers=_auth_headers(token),
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_datafeeds, **kwargs)


class GoogleMerchantGetDatafeed(Tool):
    name: str = "google_merchant_get_datafeed"
    description: str | None = "Get a Merchant Center datafeed by ID."
    integration: Annotated[str, Integration("google_merchant_center")] | None = None
    token: SecretStr | None = None
    default_merchant_id: str | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token, "default_merchant_id": self.default_merchant_id}
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_datafeed(
            datafeed_id: str = Field(..., description="Datafeed ID"),
            merchant_id: str | None = Field(None, description="Merchant Center account ID"),
        ) -> Any:
            token = await _resolve_token(self)
            merchant = _resolve_merchant_id(self, merchant_id)
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.get(
                    f"{_CONTENT_BASE}/{merchant}/datafeeds/{datafeed_id}",
                    headers=_auth_headers(token),
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_datafeed, **kwargs)
