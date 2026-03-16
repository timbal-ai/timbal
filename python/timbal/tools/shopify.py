import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_API_VERSION = "2024-01"


def _shopify_base(shop_url: str) -> str:
    return f"https://{shop_url}/admin/api/{_API_VERSION}"


async def _resolve_credentials(tool: Any) -> tuple[str, str]:
    """Resolve Shopify token and shop_url from integration, explicit fields, or env vars."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"], credentials["shop_url"]
    token = tool.token.get_secret_value() if tool.token else os.getenv("SHOPIFY_TOKEN")
    shop_url = tool.shop_url if tool.shop_url else os.getenv("SHOPIFY_SHOP")
    if not token or not shop_url:
        raise ValueError(
            "Shopify credentials not found. Set SHOPIFY_TOKEN and SHOPIFY_SHOP environment variables, "
            "pass token and shop_url in config, or configure an integration."
        )
    return token, shop_url


class ShopifyGetShopDetails(Tool):
    name: str = "shopify_get_shop_details"
    description: str | None = "Get the details of a Shopify store (name, email, currency, timezone, plan, etc.)."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_shop_details() -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop_url)}/shop.json",
                    headers={"X-Shopify-Access-Token": token},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_shop_details, **kwargs)


class ShopifyGetProducts(Tool):
    name: str = "shopify_get_products"
    description: str | None = "Retrieve a list of products from a Shopify store."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_products(
            limit: int = Field(50, description="Maximum number of products to return"),
            page_info: str | None = Field(None, description="Cursor for pagination from a previous response's Link header"),
            title: str | None = Field(None, description="Filter products by title"),
            vendor: str | None = Field(None, description="Filter products by vendor"),
            product_type: str | None = Field(None, description="Filter products by type"),
            status: str | None = Field(None, description="Filter products by status: 'active', 'archived', or 'draft'"),
            ids: list[str] | None = Field(None, description="Filter products by specific product IDs"),
        ) -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {"limit": limit}
            if page_info:
                params["page_info"] = page_info
            if title:
                params["title"] = title
            if vendor:
                params["vendor"] = vendor
            if product_type:
                params["product_type"] = product_type
            if status:
                params["status"] = status
            if ids:
                params["ids"] = ",".join(ids)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop_url)}/products.json",
                    headers={"X-Shopify-Access-Token": token},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_products, **kwargs)


class ShopifyGetProduct(Tool):
    name: str = "shopify_get_product"
    description: str | None = "Retrieve a single product from Shopify by product ID."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_product(product_id: str = Field(..., description="Shopify product ID")) -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop_url)}/products/{product_id}.json",
                    headers={"X-Shopify-Access-Token": token},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_product, **kwargs)


class ShopifyCreateProduct(Tool):
    name: str = "shopify_create_product"
    description: str | None = "Create a product in a Shopify store."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_product(
            title: str = Field(..., description="Product title (required)"),
            body_html: str | None = Field(None, description="Product description in HTML format"),
            vendor: str | None = Field(None, description="Product vendor name"),
            product_type: str | None = Field(None, description="Product type/category"),
            tags: str | None = Field(None, description="Product tags (comma-separated)"),
            status: str = Field("draft", description="Product status: 'active', 'archived', or 'draft'"),
            variants: list[dict[str, Any]] | None = Field(None, description="List of variant objects, e.g. [{'price': '9.99', 'sku': 'SKU-001'}]"),
            options: list[dict[str, Any]] | None = Field(None, description="List of product options, e.g. [{'name': 'Size', 'values': ['S', 'M', 'L']}]"),
            images: list[dict[str, Any]] | None = Field(None, description="List of image objects, e.g. [{'src': 'https://example.com/image.jpg'}]"),
        ) -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            product: dict[str, Any] = {"title": title, "status": status}
            if body_html:
                product["body_html"] = body_html
            if vendor:
                product["vendor"] = vendor
            if product_type:
                product["product_type"] = product_type
            if tags:
                product["tags"] = tags
            if variants:
                product["variants"] = variants
            if options:
                product["options"] = options
            if images:
                product["images"] = images

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_shopify_base(shop_url)}/products.json",
                    headers={"X-Shopify-Access-Token": token},
                    json={"product": product},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_product, **kwargs)


class ShopifyDeleteProduct(Tool):
    name: str = "shopify_delete_product"
    description: str | None = "Delete a product from Shopify, including all associated variants and media."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_product(product_id: str = Field(..., description="Shopify product ID")) -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_shopify_base(shop_url)}/products/{product_id}.json",
                    headers={"X-Shopify-Access-Token": token},
                )
                response.raise_for_status()
                return {"deleted": True, "product_id": product_id}

        super().__init__(handler=_delete_product, **kwargs)


class ShopifyGetInventoryLevel(Tool):
    name: str = "shopify_get_inventory_level"
    description: str | None = "Get the inventory level for a specific inventory item at one or more locations."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_inventory_level(
            inventory_item_ids: list[str] = Field(..., description="List of inventory item IDs to get levels for"),
            location_ids: list[str] | None = Field(None, description="List of location IDs to filter by"),
        ) -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            params: dict[str, Any] = {
                "inventory_item_ids": ",".join(inventory_item_ids),
            }
            if location_ids:
                params["location_ids"] = ",".join(location_ids)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop_url)}/inventory_levels.json",
                    headers={"X-Shopify-Access-Token": token},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_inventory_level, **kwargs)


class ShopifyAdjustInventory(Tool):
    name: str = "shopify_adjust_inventory"
    description: str | None = "Adjust inventory levels for a specific inventory item at a location."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _adjust_inventory(
            inventory_item_id: str = Field(..., description="Inventory item ID to adjust"),
            location_id: str = Field(..., description="Location ID for the inventory adjustment"),
            adjustment: int = Field(..., description="Inventory adjustment amount (positive to increase, negative to decrease)"),
        ) -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_shopify_base(shop_url)}/inventory_levels/adjust.json",
                    headers={"X-Shopify-Access-Token": token},
                    json={
                        "inventory_item_id": inventory_item_id,
                        "location_id": location_id,
                        "available_adjustment": adjustment,
                    },
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_adjust_inventory, **kwargs)


class ShopifyUpdateInventoryTracking(Tool):
    name: str = "shopify_update_inventory_tracking"
    description: str | None = "Enable or disable inventory tracking for a specific inventory item."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_inventory_tracking(
            inventory_item_id: str = Field(..., description="Inventory item ID to update tracking for"),
            tracked: bool = Field(..., description="Whether to track inventory for this item (true/false)"),
        ) -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_shopify_base(shop_url)}/inventory_items/{inventory_item_id}.json",
                    headers={"X-Shopify-Access-Token": token},
                    json={"inventory_item": {"id": inventory_item_id, "tracked": tracked}},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_inventory_tracking, **kwargs)


class ShopifyGetVariantInventoryItem(Tool):
    name: str = "shopify_get_variant_inventory_item"
    description: str | None = "Get the inventory item ID and details for a specific product variant."
    integration: Annotated[str, Integration("shopify")] | None = None
    token: SecretStr | None = None
    shop_url: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "token": self.token, "shop_url": self.shop_url}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_variant_inventory_item(variant_id: str = Field(..., description="Shopify product variant ID")) -> Any:
            token, shop_url = await _resolve_credentials(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop_url)}/variants/{variant_id}.json",
                    headers={"X-Shopify-Access-Token": token},
                    params={"fields": "id,inventory_item_id,sku,title,inventory_quantity"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_variant_inventory_item, **kwargs)
