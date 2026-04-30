import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_BASE_URL = "https://api.stripe.com/v1"


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
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
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
            import httpx

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
                    f"{_BASE_URL}/charges",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_charges, **kwargs)


class CreateCustomer(Tool):
    name: str = "stripe_create_customer"
    description: str | None = "Create a new Stripe customer."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
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
            import httpx

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
                    f"{_BASE_URL}/customers",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_customer, **kwargs)


class SearchCustomer(Tool):
    name: str = "stripe_search_customer"
    description: str | None = "Search for Stripe customers using a query string."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_customer(
            query: str = Field(..., description="Stripe search query string. Examples: email:'customer@example.com', name:'John Doe', metadata['user_id']:'12345'"),
            limit: int = Field(10, description="Maximum number of customers to return"),
            page: str | None = Field(None, description="Cursor for the next page, from a previous response's next_page field"),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {"query": query, "limit": limit}
            if page:
                params["page"] = page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/customers/search",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_customer, **kwargs)


class CreatePayment(Tool):
    name: str = "stripe_create_payment"
    description: str | None = "Create a Stripe PaymentIntent to collect a payment."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
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
            import httpx

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
                    f"{_BASE_URL}/payment_intents",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_payment, **kwargs)


class SendRefund(Tool):
    name: str = "stripe_send_refund"
    description: str | None = "Refund a Stripe charge or PaymentIntent, fully or partially."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
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
            import httpx

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
                    f"{_BASE_URL}/refunds",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_send_refund, **kwargs)


class UpdateCustomer(Tool):
    name: str = "stripe_update_customer"
    description: str | None = "Update an existing Stripe customer's details."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_customer(
            customer_id: str = Field(..., description="Stripe customer ID to update (e.g. 'cus_xxx')."),
            email: str | None = Field(None, description="Updated customer email address"),
            name: str | None = Field(None, description="Updated customer full name"),
            phone: str | None = Field(None, description="Updated customer phone number"),
            description: str | None = Field(None, description="Updated customer description or notes"),
            metadata: dict[str, str] | None = Field(
                None, description="Key-value pairs to attach to the customer object (max 50 keys)"
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/customers/{customer_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_customer, **kwargs)


class RetrieveCustomer(Tool):
    name: str = "stripe_retrieve_customer"
    description: str | None = "Retrieve the details of an existing Stripe customer."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_customer(
            customer_id: str = Field(..., description="Stripe customer ID to retrieve (e.g. 'cus_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/customers/{customer_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_customer, **kwargs)


class ListCustomers(Tool):
    name: str = "stripe_list_customers"
    description: str | None = "List or find Stripe customers."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_customers(
            limit: int = Field(10, description="Maximum number of customers to return"),
            email: str | None = Field(None, description="Filter customers by exact email address"),
            starting_after: str | None = Field(
                None, description="Customer ID cursor for pagination (return results after this customer)."
            ),
            ending_before: str | None = Field(
                None, description="Customer ID cursor for pagination (return results before this customer)."
            ),
            created_gte: int | None = Field(
                None, description="Unix timestamp to filter customers created after this time"
            ),
            created_lte: int | None = Field(
                None, description="Unix timestamp to filter customers created before this time"
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/customers",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_customers, **kwargs)


class DeleteCustomer(Tool):
    name: str = "stripe_delete_customer"
    description: str | None = "Permanently delete a Stripe customer and all associated data."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_customer(
            customer_id: str = Field(..., description="Stripe customer ID to delete (e.g. 'cus_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_BASE_URL}/customers/{customer_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_delete_customer, **kwargs)


class UpdatePaymentIntent(Tool):
    name: str = "stripe_update_payment_intent"
    description: str | None = "Update an existing Stripe payment intent."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_payment_intent(
            payment_intent_id: str = Field(..., description="Stripe PaymentIntent ID (e.g. 'pi_xxx')."),
            amount: int | None = Field(None, description="Updated amount in the smallest currency unit."),
            currency: str | None = Field(None, description="Three-letter ISO currency code, lowercase."),
            customer: str | None = Field(None, description="Stripe customer ID to associate."),
            description: str | None = Field(None, description="Payment intent description."),
            payment_method: str | None = Field(None, description="Stripe payment method ID to attach."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the payment intent."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/payment_intents/{payment_intent_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_payment_intent, **kwargs)


class RetrievePaymentIntent(Tool):
    name: str = "stripe_retrieve_payment_intent"
    description: str | None = "Retrieve the details of a previously created Stripe payment intent."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_payment_intent(
            payment_intent_id: str = Field(
                ..., description="Stripe PaymentIntent ID to retrieve (e.g. 'pi_xxx')."
            ),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/payment_intents/{payment_intent_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_payment_intent, **kwargs)


class ListPaymentIntents(Tool):
    name: str = "stripe_list_payment_intents"
    description: str | None = "List Stripe payment intents with optional filtering."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_payment_intents(
            limit: int = Field(10, description="Maximum number of payment intents to return."),
            customer: str | None = Field(None, description="Stripe customer ID to filter by."),
            starting_after: str | None = Field(None, description="PaymentIntent ID cursor for pagination (start after this)."),
            ending_before: str | None = Field(None, description="PaymentIntent ID cursor for pagination (end before this)."),
            created_gte: int | None = Field(None, description="Unix timestamp to filter payment intents created after this time."),
            created_lte: int | None = Field(None, description="Unix timestamp to filter payment intents created before this time."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/payment_intents",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_payment_intents, **kwargs)


class ConfirmPaymentIntent(Tool):
    name: str = "stripe_confirm_payment_intent"
    description: str | None = "Confirm that a customer intends to pay with the current or provided payment method."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _confirm_payment_intent(
            payment_intent_id: str = Field(..., description="Stripe PaymentIntent ID (e.g. 'pi_xxx')."),
            payment_method: str | None = Field(None, description="Payment method ID to attach and confirm with."),
            return_url: str | None = Field(None, description="URL to redirect to after confirmation for redirect-based payment methods."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {}
            if payment_method:
                data["payment_method"] = payment_method
            if return_url:
                data["return_url"] = return_url

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/payment_intents/{payment_intent_id}/confirm",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_confirm_payment_intent, **kwargs)


class CapturePaymentIntent(Tool):
    name: str = "stripe_capture_payment_intent"
    description: str | None = "Capture the funds of an existing uncaptured Stripe payment intent."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _capture_payment_intent(
            payment_intent_id: str = Field(..., description="Stripe PaymentIntent ID (e.g. 'pi_xxx')."),
            amount_to_capture: int | None = Field(None, description="Amount to capture in the smallest currency unit. Defaults to full amount."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {}
            if amount_to_capture is not None:
                data["amount_to_capture"] = amount_to_capture

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/payment_intents/{payment_intent_id}/capture",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_capture_payment_intent, **kwargs)


class CancelPaymentIntent(Tool):
    name: str = "stripe_cancel_payment_intent"
    description: str | None = "Cancel a Stripe PaymentIntent."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_payment_intent(
            payment_intent_id: str = Field(..., description="Stripe PaymentIntent ID (e.g. 'pi_xxx')."),
            cancellation_reason: str | None = Field(None, description="Reason: 'duplicate', 'fraudulent', 'requested_by_customer', or 'abandoned'."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {}
            if cancellation_reason:
                data["cancellation_reason"] = cancellation_reason

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/payment_intents/{payment_intent_id}/cancel",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_cancel_payment_intent, **kwargs)


class UpdateRefund(Tool):
    name: str = "stripe_update_refund"
    description: str | None = "Update the metadata on a Stripe refund."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_refund(
            refund_id: str = Field(..., description="Stripe refund ID (e.g. 're_xxx')."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to update on the refund object."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {}
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/refunds/{refund_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_refund, **kwargs)


class RetrieveRefund(Tool):
    name: str = "stripe_retrieve_refund"
    description: str | None = "Retrieve the details of an existing Stripe refund."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_refund(
            refund_id: str = Field(..., description="Stripe refund ID (e.g. 're_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/refunds/{refund_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_refund, **kwargs)


class ListRefunds(Tool):
    name: str = "stripe_list_refunds"
    description: str | None = "List or find Stripe refunds."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_refunds(
            limit: int = Field(10, description="Maximum number of refunds to return."),
            charge: str | None = Field(None, description="Stripe charge ID to filter refunds by."),
            payment_intent: str | None = Field(None, description="Stripe PaymentIntent ID to filter refunds by."),
            starting_after: str | None = Field(None, description="Refund ID cursor for pagination (start after this)."),
            ending_before: str | None = Field(None, description="Refund ID cursor for pagination (end before this)."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/refunds",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_refunds, **kwargs)


class CreateInvoice(Tool):
    name: str = "stripe_create_invoice"
    description: str | None = "Create a Stripe invoice for a customer."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_invoice(
            customer: str = Field(..., description="Stripe customer ID (e.g. 'cus_xxx')."),
            collection_method: str = Field("charge_automatically", description="'charge_automatically' or 'send_invoice'."),
            days_until_due: int | None = Field(None, description="Days from creation until due; required when collection_method is 'send_invoice'."),
            description: str | None = Field(None, description="Invoice description."),
            auto_advance: bool = Field(True, description="If True, Stripe will automatically finalize and send the invoice."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the invoice."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/invoices",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_invoice, **kwargs)


class UpdateInvoice(Tool):
    name: str = "stripe_update_invoice"
    description: str | None = "Update an existing Stripe invoice."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_invoice(
            invoice_id: str = Field(..., description="Stripe invoice ID (e.g. 'in_xxx')."),
            collection_method: str | None = Field(None, description="'charge_automatically' or 'send_invoice'."),
            days_until_due: int | None = Field(None, description="Days from creation to due date; required for 'send_invoice'."),
            description: str | None = Field(None, description="Invoice description."),
            auto_advance: bool | None = Field(None, description="Whether Stripe automatically advances the invoice status."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to update on the invoice."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/invoices/{invoice_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_invoice, **kwargs)


class RetrieveInvoice(Tool):
    name: str = "stripe_retrieve_invoice"
    description: str | None = "Retrieve the details of an existing Stripe invoice."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_invoice(
            invoice_id: str = Field(..., description="Stripe invoice ID (e.g. 'in_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/invoices/{invoice_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_invoice, **kwargs)


class ListInvoices(Tool):
    name: str = "stripe_list_invoices"
    description: str | None = "List or find Stripe invoices."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_invoices(
            limit: int = Field(10, description="Maximum number of invoices to return."),
            customer: str | None = Field(None, description="Stripe customer ID to filter by."),
            status: str | None = Field(None, description="'draft', 'open', 'paid', 'uncollectible', or 'void'."),
            subscription: str | None = Field(None, description="Stripe subscription ID to filter by."),
            starting_after: str | None = Field(None, description="Invoice ID cursor for pagination (start after this)."),
            ending_before: str | None = Field(None, description="Invoice ID cursor for pagination (end before this)."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/invoices",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_invoices, **kwargs)


class SendInvoice(Tool):
    name: str = "stripe_send_invoice"
    description: str | None = "Manually send a Stripe invoice to the customer outside the normal schedule."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_invoice(
            invoice_id: str = Field(..., description="Stripe invoice ID to send (e.g. 'in_xxx'). Invoice must be finalized first. No emails sent in test mode."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/invoices/{invoice_id}/send",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_send_invoice, **kwargs)


class FinalizeInvoice(Tool):
    name: str = "stripe_finalize_invoice"
    description: str | None = "Finalize a Stripe draft invoice, making it ready to be paid."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _finalize_invoice(
            invoice_id: str = Field(..., description="Stripe draft invoice ID to finalize (e.g. 'in_xxx')."),
            auto_advance: bool | None = Field(None, description="Whether Stripe continues to automatically advance the invoice's status."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {}
            if auto_advance is not None:
                data["auto_advance"] = str(auto_advance).lower()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/invoices/{invoice_id}/finalize",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_finalize_invoice, **kwargs)


class VoidInvoice(Tool):
    name: str = "stripe_void_invoice"
    description: str | None = "Void a Stripe invoice."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _void_invoice(
            invoice_id: str = Field(..., description="Stripe invoice ID to void (e.g. 'in_xxx'). Must be an open invoice."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/invoices/{invoice_id}/void",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_void_invoice, **kwargs)


class WriteOffInvoice(Tool):
    name: str = "stripe_write_off_invoice"
    description: str | None = "Mark a Stripe invoice as uncollectible (write off)."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _write_off_invoice(
            invoice_id: str = Field(..., description="Stripe invoice ID to mark as uncollectible (e.g. 'in_xxx'). Must be an open invoice."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/invoices/{invoice_id}/mark_uncollectible",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_write_off_invoice, **kwargs)


class DeleteOrVoidInvoice(Tool):
    name: str = "stripe_delete_or_void_invoice"
    description: str | None = "Delete a draft Stripe invoice, or void a non-draft or subscription invoice."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_or_void_invoice(
            invoice_id: str = Field(..., description="Stripe invoice ID (e.g. 'in_xxx'). Draft invoices are deleted; others are voided."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                get_response = await client.get(
                    f"{_BASE_URL}/invoices/{invoice_id}",
                    headers=headers,
                )
                get_response.raise_for_status()
                invoice = get_response.json()

                if invoice.get("status") == "draft":
                    response = await client.delete(
                        f"{_BASE_URL}/invoices/{invoice_id}",
                        headers=headers,
                    )
                else:
                    response = await client.post(
                        f"{_BASE_URL}/invoices/{invoice_id}/void",
                        headers=headers,
                    )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_delete_or_void_invoice, **kwargs)


class CreateInvoiceLineItem(Tool):
    name: str = "stripe_create_invoice_line_item"
    description: str | None = "Add a line item to a Stripe invoice."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_invoice_line_item(
            customer: str = Field(..., description="Stripe customer ID (e.g. 'cus_xxx')."),
            invoice: str | None = Field(None, description="Stripe invoice ID to attach this item to immediately (e.g. 'in_xxx')."),
            amount: int | None = Field(None, description="Unit amount in the smallest currency unit. Required if price is not provided."),
            currency: str | None = Field(None, description="Three-letter ISO currency code. Required if price is not provided."),
            price: str | None = Field(None, description="Stripe price ID to use for this item."),
            quantity: int | None = Field(None, description="Quantity of the item (default 1)."),
            description: str | None = Field(None, description="Line item description."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the invoice item."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/invoiceitems",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_invoice_line_item, **kwargs)


class UpdateInvoiceLineItem(Tool):
    name: str = "stripe_update_invoice_line_item"
    description: str | None = "Update a line item on a Stripe invoice."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_invoice_line_item(
            invoice_item_id: str = Field(..., description="Stripe invoice item ID (e.g. 'ii_xxx')."),
            amount: int | None = Field(None, description="Updated amount in the smallest currency unit."),
            description: str | None = Field(None, description="Updated line item description."),
            quantity: int | None = Field(None, description="Updated quantity."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to update on the invoice item."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/invoiceitems/{invoice_item_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_invoice_line_item, **kwargs)


class RetrieveInvoiceLineItem(Tool):
    name: str = "stripe_retrieve_invoice_line_item"
    description: str | None = "Retrieve a single line item from a Stripe invoice."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_invoice_line_item(
            invoice_item_id: str = Field(..., description="Stripe invoice item ID (e.g. 'ii_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/invoiceitems/{invoice_item_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_invoice_line_item, **kwargs)


class DeleteInvoiceLineItem(Tool):
    name: str = "stripe_delete_invoice_line_item"
    description: str | None = "Delete a line item from a Stripe invoice."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_invoice_line_item(
            invoice_item_id: str = Field(..., description="Stripe invoice item ID to delete (e.g. 'ii_xxx'). Must not be attached to a finalized invoice."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_BASE_URL}/invoiceitems/{invoice_item_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_delete_invoice_line_item, **kwargs)


class CreateSubscription(Tool):
    name: str = "stripe_create_subscription"
    description: str | None = "Create a Stripe subscription for a customer."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_subscription(
            customer: str = Field(..., description="Stripe customer ID (e.g. 'cus_xxx')."),
            price_ids: list[str] = Field(..., description="List of Stripe price IDs to subscribe the customer to."),
            trial_period_days: int | None = Field(None, description="Number of trial days before charging begins."),
            cancel_at_period_end: bool = Field(False, description="If True, cancels the subscription at the end of the current period."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the subscription."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/subscriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_subscription, **kwargs)


class SearchSubscriptions(Tool):
    name: str = "stripe_search_subscriptions"
    description: str | None = "Search for Stripe subscriptions using a query string."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_subscriptions(
            query: str = Field(..., description="Stripe search query string. E.g. status:'active', customer:'cus_xxx', metadata['plan']:'premium'."),
            limit: int = Field(10, description="Maximum number of subscriptions to return."),
            page: str | None = Field(None, description="Cursor for the next page, from a previous response's next_page field."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {"query": query, "limit": limit}
            if page:
                params["page"] = page

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/subscriptions/search",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_subscriptions, **kwargs)


class CancelSubscription(Tool):
    name: str = "stripe_cancel_subscription"
    description: str | None = "Cancel a Stripe subscription."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_subscription(
            subscription_id: str = Field(..., description="Stripe subscription ID (e.g. 'sub_xxx')."),
            cancel_at_period_end: bool = Field(False, description="If True, remains active until period end then cancels. If False, cancels immediately."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            if cancel_at_period_end:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{_BASE_URL}/subscriptions/{subscription_id}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        data={"cancel_at_period_end": "true"},
                    )
                    response.raise_for_status()
                    return response.json()
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.delete(
                        f"{_BASE_URL}/subscriptions/{subscription_id}",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    response.raise_for_status()
                    return response.json()

        super().__init__(handler=_cancel_subscription, **kwargs)


class CreateProduct(Tool):
    name: str = "stripe_create_product"
    description: str | None = "Create a new Stripe product."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_product(
            name: str = Field(..., description="The product's name."),
            description: str | None = Field(None, description="Product description."),
            active: bool = Field(True, description="Whether the product is available for purchase."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the product."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/products",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_product, **kwargs)


class RetrieveProduct(Tool):
    name: str = "stripe_retrieve_product"
    description: str | None = "Retrieve a Stripe product by ID."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_product(
            product_id: str = Field(..., description="Stripe product ID (e.g. 'prod_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/products/{product_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_product, **kwargs)


class ListProducts(Tool):
    name: str = "stripe_list_products"
    description: str | None = "List Stripe products with optional filtering."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_products(
            limit: int = Field(10, description="Maximum number of products to return (max 100)."),
            active: bool | None = Field(None, description="Filter by active status. True returns only active products, false only inactive."),
            starting_after: str | None = Field(None, description="Product ID cursor for pagination (return results after this product)."),
            ending_before: str | None = Field(None, description="Product ID cursor for pagination (return results before this product)."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {"limit": limit}
            if active is not None:
                params["active"] = str(active).lower()
            if starting_after:
                params["starting_after"] = starting_after
            if ending_before:
                params["ending_before"] = ending_before

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/products",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_products, **kwargs)


class CreatePrice(Tool):
    name: str = "stripe_create_price"
    description: str | None = "Create a new price for an existing Stripe product. The price can be recurring or one-time."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_price(
            unit_amount: int = Field(..., description="Price in the smallest currency unit (e.g. cents). 1099 = $10.99."),
            currency: str = Field(..., description="Three-letter ISO currency code, lowercase (e.g. 'usd')."),
            product: str = Field(..., description="Stripe product ID this price belongs to (e.g. 'prod_xxx')."),
            recurring_interval: str | None = Field(None, description="Billing interval for recurring: 'day', 'week', 'month', or 'year'. Omit for one-time."),
            recurring_interval_count: int | None = Field(None, description="Number of intervals between each billing cycle (default 1)."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the price."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/prices",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_price, **kwargs)


class RetrievePrice(Tool):
    name: str = "stripe_retrieve_price"
    description: str | None = "Retrieve the details of an existing Stripe product price."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_price(
            price_id: str = Field(..., description="Stripe price ID (e.g. 'price_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/prices/{price_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_price, **kwargs)


class CreatePayout(Tool):
    name: str = "stripe_create_payout"
    description: str | None = "Create a Stripe payout to send funds to a bank account or debit card."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_payout(
            amount: int = Field(..., description="Amount to send in the smallest currency unit."),
            currency: str = Field(..., description="Three-letter ISO currency code, lowercase (e.g. 'usd')."),
            description: str | None = Field(None, description="Payout description."),
            statement_descriptor: str | None = Field(None, description="Text on recipient's bank statement (max 22 chars)."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach to the payout."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/payouts",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_payout, **kwargs)


class UpdatePayout(Tool):
    name: str = "stripe_update_payout"
    description: str | None = "Update the metadata on a Stripe payout."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_payout(
            payout_id: str = Field(..., description="Stripe payout ID (e.g. 'po_xxx')."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to update on the payout."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {}
            if metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/payouts/{payout_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_payout, **kwargs)


class RetrievePayout(Tool):
    name: str = "stripe_retrieve_payout"
    description: str | None = "Retrieve the details of an existing Stripe payout."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_payout(
            payout_id: str = Field(..., description="Stripe payout ID (e.g. 'po_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/payouts/{payout_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_payout, **kwargs)


class ListPayouts(Tool):
    name: str = "stripe_list_payouts"
    description: str | None = "List or find Stripe payouts."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_payouts(
            limit: int = Field(10, description="Maximum number of payouts to return."),
            status: str | None = Field(None, description="Filter by status: 'pending', 'paid', 'failed', or 'canceled'."),
            starting_after: str | None = Field(None, description="Payout ID cursor for pagination (start after this)."),
            ending_before: str | None = Field(None, description="Payout ID cursor for pagination (end before this)."),
            arrival_date_gte: int | None = Field(None, description="Unix timestamp to filter payouts arriving after this date."),
            arrival_date_lte: int | None = Field(None, description="Unix timestamp to filter payouts arriving before this date."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/payouts",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_payouts, **kwargs)


class CancelOrReversePayout(Tool):
    name: str = "stripe_cancel_or_reverse_payout"
    description: str | None = "Cancel a pending Stripe payout or reverse a paid payout."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_or_reverse_payout(
            payout_id: str = Field(..., description="Stripe payout ID (e.g. 'po_xxx')."),
            action: str = Field("cancel", description="'cancel' for pending payouts, or 'reverse' for paid payouts."),
            metadata: dict[str, str] | None = Field(None, description="Key-value pairs to attach (only used for reverse action)."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {}
            if action == "reverse" and metadata:
                for k, v in metadata.items():
                    data[f"metadata[{k}]"] = v

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/payouts/{payout_id}/{action}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_cancel_or_reverse_payout, **kwargs)


class RetrieveBalance(Tool):
    name: str = "stripe_retrieve_balance"
    description: str | None = "Retrieve the current Stripe account balance."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_balance() -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/balance",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_balance, **kwargs)


class ListBalanceHistory(Tool):
    name: str = "stripe_list_balance_history"
    description: str | None = "List all Stripe balance transactions."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_balance_history(
            limit: int = Field(10, description="Maximum number of balance transactions to return."),
            type: str | None = Field(None, description="Filter by type: 'charge', 'refund', 'payout', 'transfer'."),
            starting_after: str | None = Field(None, description="Balance transaction ID cursor for pagination (start after this)."),
            ending_before: str | None = Field(None, description="Balance transaction ID cursor for pagination (end before this)."),
            created_gte: int | None = Field(None, description="Unix timestamp to filter transactions created after this time."),
            created_lte: int | None = Field(None, description="Unix timestamp to filter transactions created before this time."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

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
                    f"{_BASE_URL}/balance_transactions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_balance_history, **kwargs)


class RetrieveCheckoutSession(Tool):
    name: str = "stripe_retrieve_checkout_session"
    description: str | None = "Retrieve a Stripe Checkout Session by ID."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_checkout_session(
            session_id: str = Field(..., description="Stripe Checkout Session ID (e.g. 'cs_xxx')."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/checkout/sessions/{session_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_checkout_session, **kwargs)


class RetrieveCheckoutSessionLineItems(Tool):
    name: str = "stripe_retrieve_checkout_session_line_items"
    description: str | None = "Retrieve the line items from a Stripe Checkout Session."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _retrieve_checkout_session_line_items(
            session_id: str = Field(..., description="Stripe Checkout Session ID (e.g. 'cs_xxx')."),
            limit: int = Field(10, description="Maximum number of line items to return."),
            starting_after: str | None = Field(None, description="Line item ID cursor for pagination."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            params: dict[str, Any] = {"limit": limit}
            if starting_after:
                params["starting_after"] = starting_after

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_BASE_URL}/checkout/sessions/{session_id}/line_items",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_retrieve_checkout_session_line_items, **kwargs)


class CreateBillingMeter(Tool):
    name: str = "stripe_create_billing_meter"
    description: str | None = "Create a Stripe billing meter for metered billing."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_billing_meter(
            display_name: str = Field(..., description="Human-readable name for the billing meter shown in the Stripe dashboard."),
            event_name: str = Field(..., description="Name of the billing meter event to record usage against."),
            aggregation_formula: str = Field("sum", description="How to aggregate usage events: 'sum' (default) or 'count'."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {
                "display_name": display_name,
                "event_name": event_name,
                "default_aggregation[formula]": aggregation_formula,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/billing/meters",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_billing_meter, **kwargs)


class CreateUsageRecord(Tool):
    name: str = "stripe_create_usage_record"
    description: str | None = "Create a usage record for metered billing on a subscription item."
    integration: Annotated[str, Integration("stripe")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_usage_record(
            subscription_item_id: str = Field(..., description="Stripe subscription item ID (e.g. 'si_xxx')."),
            quantity: int = Field(..., description="Usage quantity to record."),
            timestamp: int | None = Field(None, description="Unix timestamp for when the usage occurred. Defaults to current time."),
            action: str = Field("increment", description="'increment' (default) to add to existing usage, or 'set' to overwrite."),
        ) -> Any:
            api_key = await _resolve_api_key(self)
            import httpx

            data: dict[str, Any] = {
                "quantity": quantity,
                "action": action,
            }
            if timestamp is not None:
                data["timestamp"] = timestamp

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_BASE_URL}/subscription_items/{subscription_item_id}/usage_records",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_usage_record, **kwargs)
