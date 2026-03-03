from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_STRIPE_API_BASE = "https://api.stripe.com/v1"


class ListCharges(Tool):
    name: str = "stripe_list_charges"
    description: str | None = "List charges from Stripe with optional filtering by customer, amount, or date range."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_charges(
            limit: int = 10,
            customer: str | None = None,
            starting_after: str | None = None,
            ending_before: str | None = None,
            created_gte: int | None = None,
            created_lte: int | None = None,
        ) -> Any:
            """
            customer: Stripe customer ID to filter charges by.
            starting_after / ending_before: charge ID cursors for pagination.
            created_gte / created_lte: Unix timestamps to filter by creation date.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"limit": limit}
            if customer:
                params["customer"] = customer
            if starting_after:
                params["starting_after"] = starting_after
            if ending_before:
                params["ending_before"] = ending_before
            if created_gte:
                params["created[gte]"] = created_gte
            if created_lte:
                params["created[lte]"] = created_lte

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/charges",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/ListCharges"

        super().__init__(handler=_list_charges, metadata=metadata, **kwargs)


class CreateCustomer(Tool):
    name: str = "stripe_create_customer"
    description: str | None = "Create a new Stripe customer."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_customer(
            email: str | None = None,
            name: str | None = None,
            phone: str | None = None,
            description: str | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            metadata: key-value pairs to attach to the customer object (max 50 keys).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if email:
                data["email"] = email
            if name:
                data["name"] = name
            if phone:
                data["phone"] = phone
            if description:
                data["description"] = description
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/customers",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreateCustomer"

        super().__init__(handler=_create_customer, metadata=metadata, **kwargs)


class SearchCustomer(Tool):
    name: str = "stripe_search_customer"
    description: str | None = "Search for Stripe customers using a query string."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_customer(
            query: str,
            limit: int = 10,
            page: str | None = None,
        ) -> Any:
            """
            query: Stripe search query string.
            Examples:
              - "email:'customer@example.com'"
              - "name:'John Doe'"
              - "metadata['user_id']:'12345'"
            page: cursor for the next page, from a previous response's next_page field.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"query": query, "limit": limit}
            if page:
                params["page"] = page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/customers/search",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/SearchCustomer"

        super().__init__(handler=_search_customer, metadata=metadata, **kwargs)


class CreatePayment(Tool):
    name: str = "stripe_create_payment"
    description: str | None = "Create a Stripe PaymentIntent to collect a payment."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_payment(
            amount: int,
            currency: str,
            customer: str | None = None,
            payment_method: str | None = None,
            description: str | None = None,
            confirm: bool = False,
            automatic_payment_methods: bool = True,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            amount: amount in the smallest currency unit (e.g. cents for USD). e.g. 1099 = $10.99.
            currency: three-letter ISO currency code, lowercase, e.g. "usd", "eur".
            customer: Stripe customer ID to associate with the payment.
            payment_method: Stripe payment method ID (e.g. "pm_card_visa"). Required if confirm=True.
            confirm: if True, confirms the PaymentIntent immediately after creation.
            automatic_payment_methods: enable automatic payment method selection (recommended for new integrations).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {
                "amount": amount,
                "currency": currency,
                "confirm": str(confirm).lower(),
            }
            if automatic_payment_methods:
                data["automatic_payment_methods[enabled]"] = "true"
            if customer:
                data["customer"] = customer
            if payment_method:
                data["payment_method"] = payment_method
            if description:
                data["description"] = description
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payment_intents",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreatePayment"

        super().__init__(handler=_create_payment, metadata=metadata, **kwargs)


class SendRefund(Tool):
    name: str = "stripe_send_refund"
    description: str | None = "Refund a Stripe charge or PaymentIntent, fully or partially."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_refund(
            charge: str | None = None,
            payment_intent: str | None = None,
            amount: int | None = None,
            reason: str | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            charge: Stripe charge ID to refund (e.g. "ch_xxx"). Provide either charge or payment_intent.
            payment_intent: Stripe PaymentIntent ID to refund (e.g. "pi_xxx").
            amount: amount to refund in the smallest currency unit. Omit to refund the full charge.
            reason: "duplicate", "fraudulent", or "requested_by_customer".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if charge:
                data["charge"] = charge
            if payment_intent:
                data["payment_intent"] = payment_intent
            if amount is not None:
                data["amount"] = amount
            if reason:
                data["reason"] = reason
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/refunds",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/SendRefund"

        super().__init__(handler=_send_refund, metadata=metadata, **kwargs)
