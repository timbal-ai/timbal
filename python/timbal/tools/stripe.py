import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_STRIPE_API_BASE = "https://api.stripe.com/v1"


async def _resolve_api_key(tool: Any) -> str:
    """Resolve Stripe API key from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["api_key"]
    if tool.api_key is not None:
        return tool.api_key.get_secret_value()
    env_key = os.getenv("STRIPE_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Stripe API key not found. Set STRIPE_API_KEY environment variable, "
        "pass api_key in config, or configure an integration."
    )


class ListCharges(Tool):
    name: str = "stripe_list_charges"
    description: str | None = "List charges from Stripe with optional filtering by customer, amount, or date range."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_charges(
            limit: int = Field(10, description="Maximum number of charges to return"),
            customer: str | None = Field(None, description="Stripe customer ID to filter charges by"),
            starting_after: str | None = Field(None, description="Charge ID cursor for pagination (start after this charge)"),
            ending_before: str | None = Field(None, description="Charge ID cursor for pagination (end before this charge)"),
            created_gte: int | None = Field(None, description="Unix timestamp to filter charges created after this time"),
            created_lte: int | None = Field(None, description="Unix timestamp to filter charges created before this time"),
        ) -> Any:
            api_key = await _resolve_api_key(self)

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

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/charges",
                    headers={"Authorization": f"Bearer {api_key}"},
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
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_customer(
            email: str | None = Field(None, description="Customer email address"),
            name: str | None = Field(None, description="Customer full name"),
            phone: str | None = Field(None, description="Customer phone number"),
            description: str | None = Field(None, description="Customer description or notes"),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the customer object (max 50 keys)"),
        ) -> Any:
            api_key = await _resolve_api_key(self)

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

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/customers",
                    headers={"Authorization": f"Bearer {api_key}"},
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
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_customer(
            query: str = Field(..., description="Stripe search query string. Examples: email:'customer@example.com', name:'John Doe', metadata['user_id']:'12345'"),
            limit: int = Field(10, description="Maximum number of customers to return"),
            page: str | None = Field(None, description="Cursor for the next page, from a previous response's next_page field"),
        ) -> Any:
            api_key = await _resolve_api_key(self)

            params: dict[str, Any] = {"query": query, "limit": limit}
            if page:
                params["page"] = page

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/customers/search",
                    headers={"Authorization": f"Bearer {api_key}"},
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
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_payment(
            amount: int = Field(..., description="Amount in the smallest currency unit (e.g. cents for USD). e.g. 1099 = $10.99"),
            currency: str = Field(..., description="Three-letter ISO currency code, lowercase, e.g. 'usd', 'eur'"),
            customer: str | None = Field(None, description="Stripe customer ID to associate with the payment"),
            payment_method: str | None = Field(None, description="Stripe payment method ID (e.g. 'pm_card_visa'). Required if confirm=True"),
            description: str | None = Field(None, description="Payment description"),
            confirm: bool = Field(False, description="Whether to confirm the payment immediately"),
            automatic_payment_methods: bool = Field(True, description="Whether to use automatic payment methods"),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the payment object"),
        ) -> Any:
            api_key = await _resolve_api_key(self)

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

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payment_intents",
                    headers={"Authorization": f"Bearer {api_key}"},
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
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "api_key": self.api_key},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_refund(
            charge: str | None = Field(None, description="Stripe charge ID to refund (e.g. 'ch_xxx'). Provide either charge or payment_intent"),
            payment_intent: str | None = Field(None, description="Stripe PaymentIntent ID to refund (e.g. 'pi_xxx')"),
            amount: int | None = Field(None, description="Amount to refund in the smallest currency unit. Omit to refund the full charge"),
            reason: str | None = Field(None, description="Refund reason: 'duplicate', 'fraudulent', or 'requested_by_customer'"),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the refund object"),
        ) -> Any:
            api_key = await _resolve_api_key(self)

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

            import httpx

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


class UpdateCustomer(Tool):
    name: str = "stripe_update_customer"
    description: str | None = "Update an existing Stripe customer's details."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_customer(
            customer_id: str,
            email: str | None = None,
            name: str | None = None,
            phone: str | None = None,
            description: str | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            customer_id: Stripe customer ID (e.g. "cus_xxx").
            metadata: key-value pairs to attach to the customer object.
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
                    f"{_STRIPE_API_BASE}/customers/{customer_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/UpdateCustomer"

        super().__init__(handler=_update_customer, metadata=metadata, **kwargs)


class RetrieveCustomer(Tool):
    name: str = "stripe_retrieve_customer"
    description: str | None = "Retrieve the details of an existing Stripe customer."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_customer(customer_id: str) -> Any:
            """
            customer_id: Stripe customer ID (e.g. "cus_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/customers/{customer_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrieveCustomer"

        super().__init__(handler=_retrieve_customer, metadata=metadata, **kwargs)


class ListCustomers(Tool):
    name: str = "stripe_list_customers"
    description: str | None = "List or find Stripe customers."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_customers(
            limit: int = 10,
            email: str | None = None,
            starting_after: str | None = None,
            ending_before: str | None = None,
            created_gte: int | None = None,
            created_lte: int | None = None,
        ) -> Any:
            """
            email: filter customers by exact email address.
            starting_after / ending_before: customer ID cursors for pagination.
            created_gte / created_lte: Unix timestamps to filter by creation date.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"limit": limit}
            if email:
                params["email"] = email
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
                    f"{_STRIPE_API_BASE}/customers",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/ListCustomers"

        super().__init__(handler=_list_customers, metadata=metadata, **kwargs)


class DeleteCustomer(Tool):
    name: str = "stripe_delete_customer"
    description: str | None = "Permanently delete a Stripe customer and all associated data."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_customer(customer_id: str) -> Any:
            """
            customer_id: Stripe customer ID to delete (e.g. "cus_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_STRIPE_API_BASE}/customers/{customer_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/DeleteCustomer"

        super().__init__(handler=_delete_customer, metadata=metadata, **kwargs)


class UpdatePaymentIntent(Tool):
    name: str = "stripe_update_payment_intent"
    description: str | None = "Update an existing Stripe payment intent."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_payment_intent(
            payment_intent_id: str,
            amount: int | None = None,
            currency: str | None = None,
            customer: str | None = None,
            description: str | None = None,
            payment_method: str | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            payment_intent_id: Stripe PaymentIntent ID (e.g. "pi_xxx").
            amount: updated amount in the smallest currency unit.
            metadata: key-value pairs to attach to the payment intent.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if amount is not None:
                data["amount"] = amount
            if currency:
                data["currency"] = currency
            if customer:
                data["customer"] = customer
            if description:
                data["description"] = description
            if payment_method:
                data["payment_method"] = payment_method
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payment_intents/{payment_intent_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/UpdatePaymentIntent"

        super().__init__(handler=_update_payment_intent, metadata=metadata, **kwargs)


class RetrievePaymentIntent(Tool):
    name: str = "stripe_retrieve_payment_intent"
    description: str | None = "Retrieve the details of a previously created Stripe payment intent."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_payment_intent(payment_intent_id: str) -> Any:
            """
            payment_intent_id: Stripe PaymentIntent ID (e.g. "pi_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/payment_intents/{payment_intent_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrievePaymentIntent"

        super().__init__(handler=_retrieve_payment_intent, metadata=metadata, **kwargs)


class ListPaymentIntents(Tool):
    name: str = "stripe_list_payment_intents"
    description: str | None = "List Stripe payment intents with optional filtering."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_payment_intents(
            limit: int = 10,
            customer: str | None = None,
            starting_after: str | None = None,
            ending_before: str | None = None,
            created_gte: int | None = None,
            created_lte: int | None = None,
        ) -> Any:
            """
            customer: Stripe customer ID to filter by.
            starting_after / ending_before: PaymentIntent ID cursors for pagination.
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
                    f"{_STRIPE_API_BASE}/payment_intents",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/ListPaymentIntents"

        super().__init__(handler=_list_payment_intents, metadata=metadata, **kwargs)


class ConfirmPaymentIntent(Tool):
    name: str = "stripe_confirm_payment_intent"
    description: str | None = "Confirm that a customer intends to pay with the current or provided payment method."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _confirm_payment_intent(
            payment_intent_id: str,
            payment_method: str | None = None,
            return_url: str | None = None,
        ) -> Any:
            """
            payment_intent_id: Stripe PaymentIntent ID (e.g. "pi_xxx").
            payment_method: payment method ID to attach and confirm with.
            return_url: URL to redirect to after confirmation for redirect-based payment methods.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if payment_method:
                data["payment_method"] = payment_method
            if return_url:
                data["return_url"] = return_url

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payment_intents/{payment_intent_id}/confirm",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/ConfirmPaymentIntent"

        super().__init__(handler=_confirm_payment_intent, metadata=metadata, **kwargs)


class CapturePaymentIntent(Tool):
    name: str = "stripe_capture_payment_intent"
    description: str | None = "Capture the funds of an existing uncaptured Stripe payment intent."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _capture_payment_intent(
            payment_intent_id: str,
            amount_to_capture: int | None = None,
        ) -> Any:
            """
            payment_intent_id: Stripe PaymentIntent ID (e.g. "pi_xxx").
            amount_to_capture: amount to capture in the smallest currency unit. Defaults to full amount.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if amount_to_capture is not None:
                data["amount_to_capture"] = amount_to_capture

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payment_intents/{payment_intent_id}/capture",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CapturePaymentIntent"

        super().__init__(handler=_capture_payment_intent, metadata=metadata, **kwargs)


class CancelPaymentIntent(Tool):
    name: str = "stripe_cancel_payment_intent"
    description: str | None = "Cancel a Stripe PaymentIntent."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_payment_intent(
            payment_intent_id: str,
            cancellation_reason: str | None = None,
        ) -> Any:
            """
            payment_intent_id: Stripe PaymentIntent ID (e.g. "pi_xxx").
            cancellation_reason: "duplicate", "fraudulent", "requested_by_customer", or "abandoned".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if cancellation_reason:
                data["cancellation_reason"] = cancellation_reason

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payment_intents/{payment_intent_id}/cancel",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CancelPaymentIntent"

        super().__init__(handler=_cancel_payment_intent, metadata=metadata, **kwargs)


class UpdateRefund(Tool):
    name: str = "stripe_update_refund"
    description: str | None = "Update the metadata on a Stripe refund."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_refund(
            refund_id: str,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            refund_id: Stripe refund ID (e.g. "re_xxx").
            metadata: key-value pairs to update on the refund object.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/refunds/{refund_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/UpdateRefund"

        super().__init__(handler=_update_refund, metadata=metadata, **kwargs)


class RetrieveRefund(Tool):
    name: str = "stripe_retrieve_refund"
    description: str | None = "Retrieve the details of an existing Stripe refund."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_refund(refund_id: str) -> Any:
            """
            refund_id: Stripe refund ID (e.g. "re_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/refunds/{refund_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrieveRefund"

        super().__init__(handler=_retrieve_refund, metadata=metadata, **kwargs)


class ListRefunds(Tool):
    name: str = "stripe_list_refunds"
    description: str | None = "List or find Stripe refunds."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_refunds(
            limit: int = 10,
            charge: str | None = None,
            payment_intent: str | None = None,
            starting_after: str | None = None,
            ending_before: str | None = None,
        ) -> Any:
            """
            charge: Stripe charge ID to filter refunds by.
            payment_intent: Stripe PaymentIntent ID to filter refunds by.
            starting_after / ending_before: refund ID cursors for pagination.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"limit": limit}
            if charge:
                params["charge"] = charge
            if payment_intent:
                params["payment_intent"] = payment_intent
            if starting_after:
                params["starting_after"] = starting_after
            if ending_before:
                params["ending_before"] = ending_before

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/refunds",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/ListRefunds"

        super().__init__(handler=_list_refunds, metadata=metadata, **kwargs)


class CreateInvoice(Tool):
    name: str = "stripe_create_invoice"
    description: str | None = "Create a Stripe invoice for a customer."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_invoice(
            customer: str,
            collection_method: str = "charge_automatically",
            days_until_due: int | None = None,
            description: str | None = None,
            auto_advance: bool = True,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            customer: Stripe customer ID (e.g. "cus_xxx").
            collection_method: "charge_automatically" or "send_invoice".
            days_until_due: number of days from creation until due; required when collection_method is "send_invoice".
            auto_advance: if True, Stripe will automatically finalize and send the invoice.
            metadata: key-value pairs to attach to the invoice.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {
                "customer": customer,
                "collection_method": collection_method,
                "auto_advance": str(auto_advance).lower(),
            }
            if days_until_due is not None:
                data["days_until_due"] = days_until_due
            if description:
                data["description"] = description
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/invoices",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreateInvoice"

        super().__init__(handler=_create_invoice, metadata=metadata, **kwargs)


class UpdateInvoice(Tool):
    name: str = "stripe_update_invoice"
    description: str | None = "Update an existing Stripe invoice."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_invoice(
            invoice_id: str,
            collection_method: str | None = None,
            days_until_due: int | None = None,
            description: str | None = None,
            auto_advance: bool | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            invoice_id: Stripe invoice ID (e.g. "in_xxx").
            collection_method: "charge_automatically" or "send_invoice".
            days_until_due: days from creation to due date; required for "send_invoice".
            metadata: key-value pairs to update on the invoice.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if collection_method:
                data["collection_method"] = collection_method
            if days_until_due is not None:
                data["days_until_due"] = days_until_due
            if description:
                data["description"] = description
            if auto_advance is not None:
                data["auto_advance"] = str(auto_advance).lower()
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/invoices/{invoice_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/UpdateInvoice"

        super().__init__(handler=_update_invoice, metadata=metadata, **kwargs)


class RetrieveInvoice(Tool):
    name: str = "stripe_retrieve_invoice"
    description: str | None = "Retrieve the details of an existing Stripe invoice."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_invoice(invoice_id: str) -> Any:
            """
            invoice_id: Stripe invoice ID (e.g. "in_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/invoices/{invoice_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrieveInvoice"

        super().__init__(handler=_retrieve_invoice, metadata=metadata, **kwargs)


class ListInvoices(Tool):
    name: str = "stripe_list_invoices"
    description: str | None = "List or find Stripe invoices."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_invoices(
            limit: int = 10,
            customer: str | None = None,
            status: str | None = None,
            subscription: str | None = None,
            starting_after: str | None = None,
            ending_before: str | None = None,
        ) -> Any:
            """
            customer: Stripe customer ID to filter by.
            status: "draft", "open", "paid", "uncollectible", or "void".
            subscription: Stripe subscription ID to filter by.
            starting_after / ending_before: invoice ID cursors for pagination.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"limit": limit}
            if customer:
                params["customer"] = customer
            if status:
                params["status"] = status
            if subscription:
                params["subscription"] = subscription
            if starting_after:
                params["starting_after"] = starting_after
            if ending_before:
                params["ending_before"] = ending_before

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/invoices",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/ListInvoices"

        super().__init__(handler=_list_invoices, metadata=metadata, **kwargs)


class SendInvoice(Tool):
    name: str = "stripe_send_invoice"
    description: str | None = "Manually send a Stripe invoice to the customer outside the normal schedule."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_invoice(invoice_id: str) -> Any:
            """
            invoice_id: Stripe invoice ID to send (e.g. "in_xxx"). Invoice must be finalized first.
            Note: no emails are sent in test mode.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/invoices/{invoice_id}/send",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/SendInvoice"

        super().__init__(handler=_send_invoice, metadata=metadata, **kwargs)


class FinalizeInvoice(Tool):
    name: str = "stripe_finalize_invoice"
    description: str | None = "Finalize a Stripe draft invoice, making it ready to be paid."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _finalize_invoice(
            invoice_id: str,
            auto_advance: bool | None = None,
        ) -> Any:
            """
            invoice_id: Stripe draft invoice ID to finalize (e.g. "in_xxx").
            auto_advance: controls whether Stripe continues to automatically advance the invoice's status.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if auto_advance is not None:
                data["auto_advance"] = str(auto_advance).lower()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/invoices/{invoice_id}/finalize",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/FinalizeInvoice"

        super().__init__(handler=_finalize_invoice, metadata=metadata, **kwargs)


class VoidInvoice(Tool):
    name: str = "stripe_void_invoice"
    description: str | None = "Void a Stripe invoice."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _void_invoice(invoice_id: str) -> Any:
            """
            invoice_id: Stripe invoice ID to void (e.g. "in_xxx"). Must be an open invoice.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/invoices/{invoice_id}/void",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/VoidInvoice"

        super().__init__(handler=_void_invoice, metadata=metadata, **kwargs)


class WriteOffInvoice(Tool):
    name: str = "stripe_write_off_invoice"
    description: str | None = "Mark a Stripe invoice as uncollectible (write off)."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _write_off_invoice(invoice_id: str) -> Any:
            """
            invoice_id: Stripe invoice ID to mark as uncollectible (e.g. "in_xxx"). Must be an open invoice.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/invoices/{invoice_id}/mark_uncollectible",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/WriteOffInvoice"

        super().__init__(handler=_write_off_invoice, metadata=metadata, **kwargs)


class DeleteOrVoidInvoice(Tool):
    name: str = "stripe_delete_or_void_invoice"
    description: str | None = "Delete a draft Stripe invoice, or void a non-draft or subscription invoice."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_or_void_invoice(invoice_id: str) -> Any:
            """
            invoice_id: Stripe invoice ID (e.g. "in_xxx").
            If the invoice is a draft, it will be deleted. Otherwise it will be voided.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {token}"}
                get_response = await client.get(
                    f"{_STRIPE_API_BASE}/invoices/{invoice_id}",
                    headers=headers,
                )
                get_response.raise_for_status()
                invoice = get_response.json()

                if invoice.get("status") == "draft":
                    response = await client.delete(
                        f"{_STRIPE_API_BASE}/invoices/{invoice_id}",
                        headers=headers,
                    )
                else:
                    response = await client.post(
                        f"{_STRIPE_API_BASE}/invoices/{invoice_id}/void",
                        headers=headers,
                    )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/DeleteOrVoidInvoice"

        super().__init__(handler=_delete_or_void_invoice, metadata=metadata, **kwargs)


class CreateInvoiceLineItem(Tool):
    name: str = "stripe_create_invoice_line_item"
    description: str | None = "Add a line item to a Stripe invoice."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_invoice_line_item(
            customer: str,
            invoice: str | None = None,
            amount: int | None = None,
            currency: str | None = None,
            price: str | None = None,
            quantity: int | None = None,
            description: str | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            customer: Stripe customer ID (e.g. "cus_xxx").
            invoice: Stripe invoice ID to attach this item to immediately (e.g. "in_xxx").
            amount: unit amount in the smallest currency unit. Required if price is not provided.
            currency: three-letter ISO currency code. Required if price is not provided.
            price: Stripe price ID to use for this item.
            quantity: quantity of the item (default 1).
            metadata: key-value pairs to attach to the invoice item.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {"customer": customer}
            if invoice:
                data["invoice"] = invoice
            if amount is not None:
                data["amount"] = amount
            if currency:
                data["currency"] = currency
            if price:
                data["price"] = price
            if quantity is not None:
                data["quantity"] = quantity
            if description:
                data["description"] = description
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/invoiceitems",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreateInvoiceLineItem"

        super().__init__(handler=_create_invoice_line_item, metadata=metadata, **kwargs)


class UpdateInvoiceLineItem(Tool):
    name: str = "stripe_update_invoice_line_item"
    description: str | None = "Update a line item on a Stripe invoice."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_invoice_line_item(
            invoice_item_id: str,
            amount: int | None = None,
            description: str | None = None,
            quantity: int | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            invoice_item_id: Stripe invoice item ID (e.g. "ii_xxx").
            amount: updated amount in the smallest currency unit.
            quantity: updated quantity.
            metadata: key-value pairs to update on the invoice item.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if amount is not None:
                data["amount"] = amount
            if description:
                data["description"] = description
            if quantity is not None:
                data["quantity"] = quantity
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/invoiceitems/{invoice_item_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/UpdateInvoiceLineItem"

        super().__init__(handler=_update_invoice_line_item, metadata=metadata, **kwargs)


class RetrieveInvoiceLineItem(Tool):
    name: str = "stripe_retrieve_invoice_line_item"
    description: str | None = "Retrieve a single line item from a Stripe invoice."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_invoice_line_item(invoice_item_id: str) -> Any:
            """
            invoice_item_id: Stripe invoice item ID (e.g. "ii_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/invoiceitems/{invoice_item_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrieveInvoiceLineItem"

        super().__init__(handler=_retrieve_invoice_line_item, metadata=metadata, **kwargs)


class DeleteInvoiceLineItem(Tool):
    name: str = "stripe_delete_invoice_line_item"
    description: str | None = "Delete a line item from a Stripe invoice."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_invoice_line_item(invoice_item_id: str) -> Any:
            """
            invoice_item_id: Stripe invoice item ID to delete (e.g. "ii_xxx").
            The item must not be attached to a finalized invoice.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_STRIPE_API_BASE}/invoiceitems/{invoice_item_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/DeleteInvoiceLineItem"

        super().__init__(handler=_delete_invoice_line_item, metadata=metadata, **kwargs)


class CreateSubscription(Tool):
    name: str = "stripe_create_subscription"
    description: str | None = "Create a Stripe subscription for a customer."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_subscription(
            customer: str,
            price_ids: list[str],
            trial_period_days: int | None = None,
            cancel_at_period_end: bool = False,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            customer: Stripe customer ID (e.g. "cus_xxx").
            price_ids: list of Stripe price IDs to subscribe the customer to.
            trial_period_days: number of trial days before charging begins.
            cancel_at_period_end: if True, cancels the subscription at the end of the current period.
            metadata: key-value pairs to attach to the subscription.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {
                "customer": customer,
                "cancel_at_period_end": str(cancel_at_period_end).lower(),
            }
            for i, price_id in enumerate(price_ids):
                data[f"items[{i}][price]"] = price_id
            if trial_period_days is not None:
                data["trial_period_days"] = trial_period_days
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/subscriptions",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreateSubscription"

        super().__init__(handler=_create_subscription, metadata=metadata, **kwargs)


class SearchSubscriptions(Tool):
    name: str = "stripe_search_subscriptions"
    description: str | None = "Search for Stripe subscriptions using a query string."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_subscriptions(
            query: str,
            limit: int = 10,
            page: str | None = None,
        ) -> Any:
            """
            query: Stripe search query string.
            Examples:
              - "status:'active'"
              - "customer:'cus_xxx'"
              - "metadata['plan']:'premium'"
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
                    f"{_STRIPE_API_BASE}/subscriptions/search",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/SearchSubscriptions"

        super().__init__(handler=_search_subscriptions, metadata=metadata, **kwargs)


class CancelSubscription(Tool):
    name: str = "stripe_cancel_subscription"
    description: str | None = "Cancel a Stripe subscription."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_subscription(
            subscription_id: str,
            cancel_at_period_end: bool = False,
        ) -> Any:
            """
            subscription_id: Stripe subscription ID (e.g. "sub_xxx").
            cancel_at_period_end: if True, the subscription remains active until the end of the
              current period and then cancels. If False, cancels immediately.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            if cancel_at_period_end:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{_STRIPE_API_BASE}/subscriptions/{subscription_id}",
                        headers={"Authorization": f"Bearer {token}"},
                        data={"cancel_at_period_end": "true"},
                    )
                    response.raise_for_status()
                    return response.json()
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.delete(
                        f"{_STRIPE_API_BASE}/subscriptions/{subscription_id}",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    response.raise_for_status()
                    return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CancelSubscription"

        super().__init__(handler=_cancel_subscription, metadata=metadata, **kwargs)


class CreateProduct(Tool):
    name: str = "stripe_create_product"
    description: str | None = "Create a new Stripe product."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_product(
            name: str,
            description: str | None = None,
            active: bool = True,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            name: the product's name.
            active: whether the product is available for purchase.
            metadata: key-value pairs to attach to the product.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {
                "name": name,
                "active": str(active).lower(),
            }
            if description:
                data["description"] = description
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/products",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreateProduct"

        super().__init__(handler=_create_product, metadata=metadata, **kwargs)


class RetrieveProduct(Tool):
    name: str = "stripe_retrieve_product"
    description: str | None = "Retrieve a Stripe product by ID."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_product(product_id: str) -> Any:
            """
            product_id: Stripe product ID (e.g. "prod_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/products/{product_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrieveProduct"

        super().__init__(handler=_retrieve_product, metadata=metadata, **kwargs)


class CreatePrice(Tool):
    name: str = "stripe_create_price"
    description: str | None = "Create a new price for an existing Stripe product. The price can be recurring or one-time."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_price(
            unit_amount: int,
            currency: str,
            product: str,
            recurring_interval: str | None = None,
            recurring_interval_count: int | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            unit_amount: price in the smallest currency unit (e.g. cents). e.g. 1099 = $10.99.
            currency: three-letter ISO currency code, lowercase (e.g. "usd").
            product: Stripe product ID this price belongs to (e.g. "prod_xxx").
            recurring_interval: billing interval for recurring prices: "day", "week", "month", or "year".
              Omit for one-time prices.
            recurring_interval_count: number of intervals between each billing cycle (default 1).
            metadata: key-value pairs to attach to the price.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {
                "unit_amount": unit_amount,
                "currency": currency,
                "product": product,
            }
            if recurring_interval:
                data["recurring[interval]"] = recurring_interval
                if recurring_interval_count is not None:
                    data["recurring[interval_count]"] = recurring_interval_count
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/prices",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreatePrice"

        super().__init__(handler=_create_price, metadata=metadata, **kwargs)


class RetrievePrice(Tool):
    name: str = "stripe_retrieve_price"
    description: str | None = "Retrieve the details of an existing Stripe product price."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_price(price_id: str) -> Any:
            """
            price_id: Stripe price ID (e.g. "price_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/prices/{price_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrievePrice"

        super().__init__(handler=_retrieve_price, metadata=metadata, **kwargs)


class CreatePayout(Tool):
    name: str = "stripe_create_payout"
    description: str | None = "Create a Stripe payout to send funds to a bank account or debit card."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_payout(
            amount: int,
            currency: str,
            description: str | None = None,
            statement_descriptor: str | None = None,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            amount: amount to send in the smallest currency unit.
            currency: three-letter ISO currency code, lowercase (e.g. "usd").
            statement_descriptor: text that appears on the recipient's bank statement (max 22 chars).
            metadata: key-value pairs to attach to the payout.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {
                "amount": amount,
                "currency": currency,
            }
            if description:
                data["description"] = description
            if statement_descriptor:
                data["statement_descriptor"] = statement_descriptor
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payouts",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreatePayout"

        super().__init__(handler=_create_payout, metadata=metadata, **kwargs)


class UpdatePayout(Tool):
    name: str = "stripe_update_payout"
    description: str | None = "Update the metadata on a Stripe payout."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_payout(
            payout_id: str,
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            payout_id: Stripe payout ID (e.g. "po_xxx").
            metadata: key-value pairs to update on the payout.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payouts/{payout_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/UpdatePayout"

        super().__init__(handler=_update_payout, metadata=metadata, **kwargs)


class RetrievePayout(Tool):
    name: str = "stripe_retrieve_payout"
    description: str | None = "Retrieve the details of an existing Stripe payout."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_payout(payout_id: str) -> Any:
            """
            payout_id: Stripe payout ID (e.g. "po_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/payouts/{payout_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrievePayout"

        super().__init__(handler=_retrieve_payout, metadata=metadata, **kwargs)


class ListPayouts(Tool):
    name: str = "stripe_list_payouts"
    description: str | None = "List or find Stripe payouts."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_payouts(
            limit: int = 10,
            status: str | None = None,
            starting_after: str | None = None,
            ending_before: str | None = None,
            arrival_date_gte: int | None = None,
            arrival_date_lte: int | None = None,
        ) -> Any:
            """
            status: filter by payout status: "pending", "paid", "failed", or "canceled".
            starting_after / ending_before: payout ID cursors for pagination.
            arrival_date_gte / arrival_date_lte: Unix timestamps to filter by expected arrival date.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"limit": limit}
            if status:
                params["status"] = status
            if starting_after:
                params["starting_after"] = starting_after
            if ending_before:
                params["ending_before"] = ending_before
            if arrival_date_gte:
                params["arrival_date[gte]"] = arrival_date_gte
            if arrival_date_lte:
                params["arrival_date[lte]"] = arrival_date_lte

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/payouts",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/ListPayouts"

        super().__init__(handler=_list_payouts, metadata=metadata, **kwargs)


class CancelOrReversePayout(Tool):
    name: str = "stripe_cancel_or_reverse_payout"
    description: str | None = "Cancel a pending Stripe payout or reverse a paid payout."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_or_reverse_payout(
            payout_id: str,
            action: str = "cancel",
            metadata: dict[str, str] | None = None,
        ) -> Any:
            """
            payout_id: Stripe payout ID (e.g. "po_xxx").
            action: "cancel" for pending payouts, or "reverse" for paid payouts.
            metadata: key-value pairs to attach (only used for reverse action).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {}
            if action == "reverse" and metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/payouts/{payout_id}/{action}",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CancelOrReversePayout"

        super().__init__(handler=_cancel_or_reverse_payout, metadata=metadata, **kwargs)


class RetrieveBalance(Tool):
    name: str = "stripe_retrieve_balance"
    description: str | None = "Retrieve the current Stripe account balance."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_balance() -> Any:
            """
            Returns the current account balance broken down by currency and fund type
            (available, pending, etc.).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/balance",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrieveBalance"

        super().__init__(handler=_retrieve_balance, metadata=metadata, **kwargs)


class ListBalanceHistory(Tool):
    name: str = "stripe_list_balance_history"
    description: str | None = "List all Stripe balance transactions."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_balance_history(
            limit: int = 10,
            type: str | None = None,
            starting_after: str | None = None,
            ending_before: str | None = None,
            created_gte: int | None = None,
            created_lte: int | None = None,
        ) -> Any:
            """
            type: filter by transaction type (e.g. "charge", "refund", "payout", "transfer").
            starting_after / ending_before: balance transaction ID cursors for pagination.
            created_gte / created_lte: Unix timestamps to filter by creation date.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"limit": limit}
            if type:
                params["type"] = type
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
                    f"{_STRIPE_API_BASE}/balance_transactions",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/ListBalanceHistory"

        super().__init__(handler=_list_balance_history, metadata=metadata, **kwargs)


class RetrieveCheckoutSession(Tool):
    name: str = "stripe_retrieve_checkout_session"
    description: str | None = "Retrieve a Stripe Checkout Session by ID."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_checkout_session(session_id: str) -> Any:
            """
            session_id: Stripe Checkout Session ID (e.g. "cs_xxx").
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/checkout/sessions/{session_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrieveCheckoutSession"

        super().__init__(handler=_retrieve_checkout_session, metadata=metadata, **kwargs)


class RetrieveCheckoutSessionLineItems(Tool):
    name: str = "stripe_retrieve_checkout_session_line_items"
    description: str | None = "Retrieve the line items from a Stripe Checkout Session."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_checkout_session_line_items(
            session_id: str,
            limit: int = 10,
            starting_after: str | None = None,
        ) -> Any:
            """
            session_id: Stripe Checkout Session ID (e.g. "cs_xxx").
            starting_after: line item ID cursor for pagination.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            params: dict[str, Any] = {"limit": limit}
            if starting_after:
                params["starting_after"] = starting_after

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_STRIPE_API_BASE}/checkout/sessions/{session_id}/line_items",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/RetrieveCheckoutSessionLineItems"

        super().__init__(handler=_retrieve_checkout_session_line_items, metadata=metadata, **kwargs)


class CreateBillingMeter(Tool):
    name: str = "stripe_create_billing_meter"
    description: str | None = "Create a Stripe billing meter for metered billing."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_billing_meter(
            display_name: str,
            event_name: str,
            aggregation_formula: str = "sum",
        ) -> Any:
            """
            display_name: human-readable name for the billing meter shown in the Stripe dashboard.
            event_name: name of the billing meter event to record usage against.
            aggregation_formula: how to aggregate usage events: "sum" (default) or "count".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {
                "display_name": display_name,
                "event_name": event_name,
                "default_aggregation[formula]": aggregation_formula,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/billing/meters",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreateBillingMeter"

        super().__init__(handler=_create_billing_meter, metadata=metadata, **kwargs)


class CreateUsageRecord(Tool):
    name: str = "stripe_create_usage_record"
    description: str | None = "Create a usage record for metered billing on a subscription item."
    integration: Annotated[str, Integration("stripe")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_usage_record(
            subscription_item_id: str,
            quantity: int,
            timestamp: int | None = None,
            action: str = "increment",
        ) -> Any:
            """
            subscription_item_id: Stripe subscription item ID (e.g. "si_xxx").
            quantity: usage quantity to record.
            timestamp: Unix timestamp for when the usage occurred. Defaults to current time.
            action: "increment" (default) to add to existing usage, or "set" to overwrite.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            data: dict[str, Any] = {
                "quantity": quantity,
                "action": action,
            }
            if timestamp is not None:
                data["timestamp"] = timestamp

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_STRIPE_API_BASE}/subscription_items/{subscription_item_id}/usage_records",
                    headers={"Authorization": f"Bearer {token}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Stripe/CreateUsageRecord"

        super().__init__(handler=_create_usage_record, metadata=metadata, **kwargs)
