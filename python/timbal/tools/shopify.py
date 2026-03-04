from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_SHOPIFY_API_VERSION = "2024-01"


def _shopify_base(shop: str) -> str:
    return f"https://{shop}/admin/api/{_SHOPIFY_API_VERSION}"


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------


class GetShopDetails(Tool):
    name: str = "shopify_get_shop_details"
    description: str | None = "Get the details of a Shopify store (name, email, currency, timezone, plan, etc.)."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_shop_details() -> Any:
            """
            Gets shop details using the shop_url and token from integration credentials.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop)}/shop.json",
                    headers={"X-Shopify-Access-Token": token},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/GetShopDetails"

        super().__init__(handler=_get_shop_details, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


class GetProducts(Tool):
    name: str = "shopify_get_products"
    description: str | None = "Retrieve a list of products from a Shopify store."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_products(
            limit: int = 50,
            page_info: str | None = None,
            title: str | None = None,
            vendor: str | None = None,
            product_type: str | None = None,
            status: str | None = None,
            ids: list[str] | None = None,
        ) -> Any:
            """
            Gets products using shop_url and token from integration credentials.
            status: "active", "archived", or "draft"
            page_info: cursor for pagination from a previous response's Link header.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

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
                    f"{_shopify_base(shop)}/products.json",
                    headers={"X-Shopify-Access-Token": token},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/GetProducts"

        super().__init__(handler=_get_products, metadata=metadata, **kwargs)


class GetProduct(Tool):
    name: str = "shopify_get_product"
    description: str | None = "Retrieve a single product from Shopify by product ID."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_product(product_id: str) -> Any:
            """
            Gets a single product using shop_url and token from integration credentials.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop)}/products/{product_id}.json",
                    headers={"X-Shopify-Access-Token": token},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/GetProduct"

        super().__init__(handler=_get_product, metadata=metadata, **kwargs)


class CreateProduct(Tool):
    name: str = "shopify_create_product"
    description: str | None = "Create a product in a Shopify store."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_product(
            title: str,
            body_html: str | None = None,
            vendor: str | None = None,
            product_type: str | None = None,
            tags: str | None = None,
            status: str = "draft",
            variants: list[dict[str, Any]] | None = None,
            options: list[dict[str, Any]] | None = None,
            images: list[dict[str, Any]] | None = None,
        ) -> Any:
            """
            Creates a product using shop_url and token from integration credentials.
            status: "active", "archived", or "draft"
            variants: list of variant objects, e.g. [{"price": "9.99", "sku": "SKU-001"}]
            options: list of option objects, e.g. [{"name": "Size", "values": ["S", "M", "L"]}]
            images: list of image objects, e.g. [{"src": "https://..."}]
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

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
                    f"{_shopify_base(shop)}/products.json",
                    headers={"X-Shopify-Access-Token": token},
                    json={"product": product},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/CreateProduct"

        super().__init__(handler=_create_product, metadata=metadata, **kwargs)


class DeleteProduct(Tool):
    name: str = "shopify_delete_product"
    description: str | None = "Delete a product from Shopify, including all associated variants and media."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_product(product_id: str) -> Any:
            """
            Deletes a product using shop_url and token from integration credentials.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_shopify_base(shop)}/products/{product_id}.json",
                    headers={"X-Shopify-Access-Token": token},
                )
                response.raise_for_status()
                return {"deleted": True, "product_id": product_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/DeleteProduct"

        super().__init__(handler=_delete_product, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


class GetInventoryLevel(Tool):
    name: str = "shopify_get_inventory_level"
    description: str | None = "Get the inventory level for a specific inventory item at one or more locations."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_inventory_level(
            inventory_item_ids: list[str],
            location_ids: list[str] | None = None,
        ) -> Any:
            """
            Gets inventory levels using shop_url and token from integration credentials.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

            params: dict[str, Any] = {
                "inventory_item_ids": ",".join(inventory_item_ids),
            }
            if location_ids:
                params["location_ids"] = ",".join(location_ids)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop)}/inventory_levels.json",
                    headers={"X-Shopify-Access-Token": token},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/GetInventoryLevel"

        super().__init__(handler=_get_inventory_level, metadata=metadata, **kwargs)


class AdjustInventory(Tool):
    name: str = "shopify_adjust_inventory"
    description: str | None = "Adjust inventory levels for a specific inventory item at a location."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _adjust_inventory(
            inventory_item_id: str,
            location_id: str,
            adjustment: int,
        ) -> Any:
            """
            Adjusts inventory using shop_url and token from integration credentials.
            adjustment: positive to increase stock, negative to decrease.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_shopify_base(shop)}/inventory_levels/adjust.json",
                    headers={"X-Shopify-Access-Token": token},
                    json={
                        "inventory_item_id": inventory_item_id,
                        "location_id": location_id,
                        "available_adjustment": adjustment,
                    },
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/AdjustInventory"

        super().__init__(handler=_adjust_inventory, metadata=metadata, **kwargs)


class UpdateInventoryTracking(Tool):
    name: str = "shopify_update_inventory_tracking"
    description: str | None = "Enable or disable inventory tracking for a specific inventory item."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_inventory_tracking(
            inventory_item_id: str,
            tracked: bool,
        ) -> Any:
            """
            Updates inventory tracking using shop_url and token from integration credentials.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_shopify_base(shop)}/inventory_items/{inventory_item_id}.json",
                    headers={"X-Shopify-Access-Token": token},
                    json={"inventory_item": {"id": inventory_item_id, "tracked": tracked}},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/UpdateInventoryTracking"

        super().__init__(handler=_update_inventory_tracking, metadata=metadata, **kwargs)


class GetVariantInventoryItem(Tool):
    name: str = "shopify_get_variant_inventory_item"
    description: str | None = "Get the inventory item ID and details for a specific product variant."
    integration: Annotated[str, Integration("shopify")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_variant_inventory_item(variant_id: str) -> Any:
            """
            Gets variant inventory item using shop_url and token from integration credentials.
            """
            assert isinstance(self.integration, Integration)
            credentials = await self.integration.resolve()
            assert "token" in credentials
            assert "shop_url" in credentials
            
            token = credentials["token"]
            shop = credentials["shop_url"]

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_shopify_base(shop)}/variants/{variant_id}.json",
                    headers={"X-Shopify-Access-Token": token},
                    params={"fields": "id,inventory_item_id,sku,title,inventory_quantity"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Shopify/GetVariantInventoryItem"

        super().__init__(handler=_get_variant_inventory_item, metadata=metadata, **kwargs)
