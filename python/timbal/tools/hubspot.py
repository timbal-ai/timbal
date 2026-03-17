import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_API_BASE = "https://api.hubapi.com"


async def _resolve_token(tool: Any) -> str:
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["token"]
    if tool.token is not None:
        return tool.token.get_secret_value()
    env_key = os.getenv("HUBSPOT_TOKEN")
    if env_key:
        return env_key
    raise ValueError(
        "HubSpot token not found. Set HUBSPOT_TOKEN environment variable, "
        "pass token in config, or configure an integration."
    )


class HubSpotListContacts(Tool):
    name: str = "hubspot_list_contacts"
    description: str | None = "List HubSpot contacts with optional filtering."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_contacts(
            limit: int = Field(10, description="Number of contacts to return (max 100)"),
            after: str | None = Field(None, description="Pagination cursor to get next page of results"),
            properties: list[str] | None = Field(None, description="List of contact properties to return"),
            archived: bool = Field(False, description="Include archived contacts in results"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"limit": limit, "archived": archived}
            if after:
                params["after"] = after
            if properties:
                params["properties"] = ",".join(properties)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/contacts",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_contacts, **kwargs)


class HubSpotGetContact(Tool):
    name: str = "hubspot_get_contact"
    description: str | None = "Retrieve a specific HubSpot contact by ID with detailed information."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_contact(
            contact_id: str = Field(..., description="The HubSpot contact ID to retrieve"),
            properties: list[str] | None = Field(None, description="List of properties to retrieve"),
            associations: list[str] | None = Field(None, description="List of association types to retrieve"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {}
            if properties:
                params["properties"] = ",".join(properties)
            if associations:
                params["associations"] = ",".join(associations)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/contacts/{contact_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_contact, **kwargs)


class HubSpotCreateContact(Tool):
    name: str = "hubspot_create_contact"
    description: str | None = "Create a new HubSpot contact."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_contact(
            email: str = Field(..., description="The email address of the contact"),
            firstname: str | None = Field(None, description="The first name of the contact"),
            lastname: str | None = Field(None, description="The last name of the contact"),
            phone: str | None = Field(None, description="The phone number of the contact"),
            company: str | None = Field(None, description="The company name of the contact"),
            jobtitle: str | None = Field(None, description="The job title of the contact"),
            website: str | None = Field(None, description="The website of the contact"),
            lifecyclestage: str | None = Field(None, description="The lifecycle stage of the contact"),
            hs_lead_status: str | None = Field(None, description="The lead status of the contact"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {"email": email}
            if firstname:
                properties["firstname"] = firstname
            if lastname:
                properties["lastname"] = lastname
            if phone:
                properties["phone"] = phone
            if company:
                properties["company"] = company
            if jobtitle:
                properties["jobtitle"] = jobtitle
            if website:
                properties["website"] = website
            if lifecyclestage:
                properties["lifecyclestage"] = lifecyclestage
            if hs_lead_status:
                properties["hs_lead_status"] = hs_lead_status

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/contacts",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_contact, **kwargs)


class HubSpotUpdateContact(Tool):
    name: str = "hubspot_update_contact"
    description: str | None = "Update an existing HubSpot contact."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_contact(
            contact_id: str = Field(..., description="The HubSpot contact ID to update"),
            email: str | None = Field(None, description="The email address of the contact"),
            firstname: str | None = Field(None, description="The first name of the contact"),
            lastname: str | None = Field(None, description="The last name of the contact"),
            phone: str | None = Field(None, description="The phone number of the contact"),
            company: str | None = Field(None, description="The company name of the contact"),
            jobtitle: str | None = Field(None, description="The job title of the contact"),
            website: str | None = Field(None, description="The website of the contact"),
            lifecyclestage: str | None = Field(None, description="The lifecycle stage of the contact"),
            hs_lead_status: str | None = Field(None, description="The lead status of the contact"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {}
            if email:
                properties["email"] = email
            if firstname:
                properties["firstname"] = firstname
            if lastname:
                properties["lastname"] = lastname
            if phone:
                properties["phone"] = phone
            if company:
                properties["company"] = company
            if jobtitle:
                properties["jobtitle"] = jobtitle
            if website:
                properties["website"] = website
            if lifecyclestage:
                properties["lifecyclestage"] = lifecyclestage
            if hs_lead_status:
                properties["hs_lead_status"] = hs_lead_status

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/crm/v3/objects/contacts/{contact_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_contact, **kwargs)


class HubSpotSearchContacts(Tool):
    name: str = "hubspot_search_contacts"
    description: str | None = "Search for HubSpot contacts using advanced filters."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_contacts(
            filter_groups: list[dict[str, Any]] = Field(
                ...,
                description=(
                    "Required. List of filter groups. Each group: {'filters': [{'propertyName': str, 'operator': 'EQ'|'CONTAINS_TOKEN'|'CONTAINS'|..., 'value': str}]}. "
                    "E.g. email contains 'example.com': [{'filters': [{'propertyName': 'email', 'operator': 'CONTAINS_TOKEN', 'value': 'example.com'}]}]"
                ),
            ),
            properties: list[str] | None = Field(None, description="List of properties to retrieve"),
            sorts: list[dict[str, Any]] | None = Field(None, description="list of {'propertyName': ..., 'direction': 'ASCENDING' | 'DESCENDING'}"),
            limit: int = Field(10, description="Maximum number of results to return"),
            after: int = Field(0, description="Cursor for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {
                "filterGroups": filter_groups,
                "limit": limit,
                "after": after,
            }
            if properties:
                body["properties"] = properties
            if sorts:
                body["sorts"] = sorts

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/contacts/search",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_contacts, **kwargs)


class HubSpotMergeContacts(Tool):
    name: str = "hubspot_merge_contacts"
    description: str | None = "Merge two HubSpot contacts into one."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _merge_contacts(
            primary_object_id: str = Field(..., description="Primary contact ID to merge into"),
            object_id_to_merge: str = Field(..., description="Contact ID to be merged")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/contacts/merge",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"primaryObjectId": primary_object_id, "objectIdToMerge": object_id_to_merge},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_merge_contacts, **kwargs)


class HubSpotGdprDeleteContact(Tool):
    name: str = "hubspot_gdpr_delete_contact"
    description: str | None = "Permanently delete a contact and all associated content to follow GDPR."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _gdpr_delete_contact(
            object_id: str | None = Field(None, description="Contact ID to delete"),
            email: str | None = Field(None, description="Email address of the contact to delete"),
        ) -> Any:
            """
            Provide either object_id or email to identify the contact.
            """
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {}
            if object_id:
                body["objectId"] = object_id
            if email:
                body["email"] = email

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/contacts/gdpr-delete",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return {"deleted": True}

        super().__init__(handler=_gdpr_delete_contact, **kwargs)


class HubSpotListCompanies(Tool):
    name: str = "hubspot_list_companies"
    description: str | None = "List HubSpot companies with optional filtering."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_companies(
            limit: int = Field(10, description="Maximum number of results to return"),
            after: str | None = Field(None, description="Cursor for pagination"),
            properties: list[str] | None = Field(None, description="List of properties to retrieve"),
            archived: bool = Field(False, description="Whether to include archived companies"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"limit": limit, "archived": archived}
            if after:
                params["after"] = after
            if properties:
                params["properties"] = ",".join(properties)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/companies",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_companies, **kwargs)


class HubSpotGetCompany(Tool):
    name: str = "hubspot_get_company"
    description: str | None = "Retrieve a specific HubSpot company by ID with detailed information."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_company(
            company_id: str = Field(..., description="Company ID to retrieve"),
            properties: list[str] | None = Field(None, description="List of properties to retrieve"),
            associations: list[str] | None = Field(None, description="List of associations to retrieve"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {}
            if properties:
                params["properties"] = ",".join(properties)
            if associations:
                params["associations"] = ",".join(associations)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/companies/{company_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_company, **kwargs)


class HubSpotCreateCompany(Tool):
    name: str = "hubspot_create_company"
    description: str | None = "Create a new HubSpot company."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_company(
            name: str = Field(..., description="Company name"),
            domain: str | None = Field(None, description="Company domain"),
            industry: str | None = Field(None, description="Company industry"),
            city: str | None = Field(None, description="Company city"),
            country: str | None = Field(None, description="Company country"),
            phone: str | None = Field(None, description="Company phone"),
            website: str | None = Field(None, description="Company website"),
            description: str | None = Field(None, description="Company description"),
            numberofemployees: int | None = Field(None, description="Number of employees"),
            annualrevenue: str | None = Field(None, description="Annual revenue"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {"name": name}
            if domain:
                properties["domain"] = domain
            if industry:
                properties["industry"] = industry
            if city:
                properties["city"] = city
            if country:
                properties["country"] = country
            if phone:
                properties["phone"] = phone
            if website:
                properties["website"] = website
            if description:
                properties["description"] = description
            if numberofemployees is not None:
                properties["numberofemployees"] = numberofemployees
            if annualrevenue:
                properties["annualrevenue"] = annualrevenue

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/companies",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_company, **kwargs)


class HubSpotUpdateCompany(Tool):
    name: str = "hubspot_update_company"
    description: str | None = "Update an existing HubSpot company."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_company(
            company_id: str = Field(..., description="Company ID to update"),
            name: str | None = Field(None, description="Company name"),
            domain: str | None = Field(None, description="Company domain"),
            industry: str | None = Field(None, description="Company industry"),
            city: str | None = Field(None, description="Company city"),
            country: str | None = Field(None, description="Company country"),
            phone: str | None = Field(None, description="Company phone"),
            website: str | None = Field(None, description="Company website"),
            description: str | None = Field(None, description="Company description"),
            numberofemployees: int | None = Field(None, description="Number of employees"),
            annualrevenue: str | None = Field(None, description="Annual revenue"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {}
            if name:
                properties["name"] = name
            if domain:
                properties["domain"] = domain
            if industry:
                properties["industry"] = industry
            if city:
                properties["city"] = city
            if country:
                properties["country"] = country
            if phone:
                properties["phone"] = phone
            if website:
                properties["website"] = website
            if description:
                properties["description"] = description
            if numberofemployees is not None:
                properties["numberofemployees"] = numberofemployees
            if annualrevenue:
                properties["annualrevenue"] = annualrevenue

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/crm/v3/objects/companies/{company_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                if response.status_code >= 400:
                    error_details = response.json()
                    raise ValueError(f"HubSpot API Error: {response.status_code} - {error_details}")
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_company, **kwargs)


class HubSpotSearchCompanies(Tool):
    name: str = "hubspot_search_companies"
    description: str | None = "Search for HubSpot companies using advanced filters and sorting options."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_companies(
            filter_groups: list[dict[str, Any]] = Field(
                ...,
                description=(
                    "Required. List of filter groups. Each group: {'filters': [{'propertyName': str, 'operator': 'EQ'|'CONTAINS_TOKEN'|'CONTAINS'|..., 'value': str}]}. "
                    "E.g. domain equals 'example.com': [{'filters': [{'propertyName': 'domain', 'operator': 'EQ', 'value': 'example.com'}]}]"
                ),
            ),
            properties: list[str] | None = Field(None, description="List of properties to retrieve"),
            sorts: list[dict[str, Any]] | None = Field(None, description="List of sorts"),
            limit: int = Field(10, description="Maximum number of results to return"),
            after: int = Field(0, description="Cursor for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {
                "filterGroups": filter_groups,
                "limit": limit,
                "after": after,
            }
            if properties:
                body["properties"] = properties
            if sorts:
                body["sorts"] = sorts

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/companies/search",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_companies, **kwargs)


class HubSpotListDeals(Tool):
    name: str = "hubspot_list_deals"
    description: str | None = "List or search HubSpot deals with optional filtering."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deals(
            limit: int = Field(10, description="Maximum number of results to return"),
            after: str | None = Field(None, description="Cursor for pagination"),
            properties: list[str] | None = Field(None, description="List of properties to retrieve"),
            archived: bool = Field(False, description="Whether to include archived deals"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"limit": limit, "archived": archived}
            if after:
                params["after"] = after
            if properties:
                params["properties"] = ",".join(properties)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/deals",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_deals, **kwargs)


class HubSpotGetDeal(Tool):
    name: str = "hubspot_get_deal"
    description: str | None = "Retrieve a specific HubSpot deal by ID with detailed information."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_deal(
            deal_id: str = Field(..., description="Deal ID to retrieve"),
            properties: list[str] | None = Field(None, description="List of properties to retrieve"),
            associations: list[str] | None = Field(None, description="List of associations to retrieve"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {}
            if properties:
                params["properties"] = ",".join(properties)
            if associations:
                params["associations"] = ",".join(associations)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/deals/{deal_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_deal, **kwargs)


class HubSpotCreateDeal(Tool):
    name: str = "hubspot_create_deal"
    description: str | None = "Create a new HubSpot deal."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_deal(
            dealname: str = Field(..., description="Deal name"),
            pipeline: str | None = Field(None, description="Pipeline ID (use hubspot_get_deal_pipelines to discover available IDs)."),
            dealstage: str | None = Field(None, description="Stage ID within the pipeline."),
            amount: str | None = Field(None, description="Deal value as a string, e.g. '5000'"),
            closedate: str | None = Field(None, description="Expected close date in ISO 8601 format e.g. '2026-06-30'"),
            hubspot_owner_id: str | None = Field(None, description="Owner ID"),
            description: str | None = Field(None, description="Deal description"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {"dealname": dealname}
            if pipeline:
                properties["pipeline"] = pipeline
            if dealstage:
                properties["dealstage"] = dealstage
            if amount:
                properties["amount"] = amount
            if closedate:
                properties["closedate"] = closedate
            if hubspot_owner_id:
                properties["hubspot_owner_id"] = hubspot_owner_id
            if description:
                properties["description"] = description

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/deals",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_deal, **kwargs)


class HubSpotUpdateDeal(Tool):
    name: str = "hubspot_update_deal"
    description: str | None = "Update an existing HubSpot deal."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_deal(
            deal_id: str = Field(..., description="Deal ID to update"),
            dealname: str | None = Field(None, description="Deal name"),
            pipeline: str | None = Field(None, description="Pipeline ID"),
            dealstage: str | None = Field(None, description="Stage ID within the pipeline."),
            amount: str | None = Field(None, description="Deal value as a string, e.g. '5000'"),
            closedate: str | None = Field(None, description="Expected close date in ISO 8601 format e.g. '2026-06-30'"),
            hubspot_owner_id: str | None = Field(None, description="Owner ID"),
            description: str | None = Field(None, description="Deal description"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {}
            if dealname:
                properties["dealname"] = dealname
            if pipeline:
                properties["pipeline"] = pipeline
            if dealstage:
                properties["dealstage"] = dealstage
            if amount:
                properties["amount"] = amount
            if closedate:
                properties["closedate"] = closedate
            if hubspot_owner_id:
                properties["hubspot_owner_id"] = hubspot_owner_id
            if description:
                properties["description"] = description

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/crm/v3/objects/deals/{deal_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_deal, **kwargs)


class HubSpotGetDealPipelines(Tool):
    name: str = "hubspot_get_deal_pipelines"
    description: str | None = "Get all deal pipelines and their stages. Use this before creating a deal to discover valid pipeline and dealstage IDs."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_deal_pipelines() -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/pipelines/deals",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_deal_pipelines, **kwargs)


class HubSpotListTickets(Tool):
    name: str = "hubspot_list_tickets"
    description: str | None = "List HubSpot tickets with optional filtering."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_tickets(
            limit: int = Field(10, description="Maximum number of tickets to return"),
            after: str | None = Field(None, description="Cursor for pagination"),
            properties: list[str] | None = Field(None, description="Properties to include in the response"),
            archived: bool = Field(False, description="Whether to include archived tickets"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"limit": limit, "archived": archived}
            if after:
                params["after"] = after
            if properties:
                params["properties"] = ",".join(properties)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/tickets",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_tickets, **kwargs)


class HubSpotGetTicket(Tool):
    name: str = "hubspot_get_ticket"
    description: str | None = "Get a specific HubSpot ticket by ID."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_ticket(
            ticket_id: str = Field(..., description="Ticket ID to retrieve"),
            properties: list[str] | None = Field(None, description="Properties to include in the response"),
            associations: list[str] | None = Field(None, description="Associations to include in the response"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {}
            if properties:
                params["properties"] = ",".join(properties)
            if associations:
                params["associations"] = ",".join(associations)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/tickets/{ticket_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_ticket, **kwargs)


class HubSpotCreateTicket(Tool):
    name: str = "hubspot_create_ticket"
    description: str | None = "Create a new HubSpot ticket."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_ticket(
            subject: str = Field(..., description="Ticket subject"),
            hs_pipeline: str | None = Field(None, description="Pipeline ID. Use hubspot_get_deal_pipelines (tickets share the same pipeline API) to find valid IDs."),
            hs_pipeline_stage: str | None = Field(None, description="Pipeline stage ID. Use hubspot_get_deal_pipelines (tickets share the same pipeline API) to find valid IDs."),
            content: str | None = Field(None, description="Ticket content"),
            hs_ticket_priority: str | None = Field(None, description="Ticket priority: 'LOW', 'MEDIUM', or 'HIGH'"),
            hubspot_owner_id: str | None = Field(None, description="Owner ID"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {"subject": subject}
            if hs_pipeline:
                properties["hs_pipeline"] = hs_pipeline
            if hs_pipeline_stage:
                properties["hs_pipeline_stage"] = hs_pipeline_stage
            if content:
                properties["content"] = content
            if hs_ticket_priority:
                properties["hs_ticket_priority"] = hs_ticket_priority
            if hubspot_owner_id:
                properties["hubspot_owner_id"] = hubspot_owner_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/tickets",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_ticket, **kwargs)


class HubSpotUpdateTicket(Tool):
    name: str = "hubspot_update_ticket"
    description: str | None = "Update an existing HubSpot ticket."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_ticket(
            ticket_id: str = Field(..., description="Ticket ID to update"),
            subject: str | None = Field(None, description="Ticket subject"),
            hs_pipeline: str | None = Field(None, description="Pipeline ID. Use hubspot_get_deal_pipelines (tickets share the same pipeline API) to find valid IDs."),
            hs_pipeline_stage: str | None = Field(None, description="Pipeline stage ID. Use hubspot_get_deal_pipelines (tickets share the same pipeline API) to find valid IDs."),
            content: str | None = Field(None, description="Ticket content"),
            hs_ticket_priority: str | None = Field(None, description="Ticket priority: 'LOW', 'MEDIUM', or 'HIGH'"),
            hubspot_owner_id: str | None = Field(None, description="Owner ID"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {}
            if subject:
                properties["subject"] = subject
            if hs_pipeline:
                properties["hs_pipeline"] = hs_pipeline
            if hs_pipeline_stage:
                properties["hs_pipeline_stage"] = hs_pipeline_stage
            if content:
                properties["content"] = content
            if hs_ticket_priority:
                properties["hs_ticket_priority"] = hs_ticket_priority
            if hubspot_owner_id:
                properties["hubspot_owner_id"] = hubspot_owner_id

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/crm/v3/objects/tickets/{ticket_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_ticket, **kwargs)


class HubSpotDeleteTicket(Tool):
    name: str = "hubspot_delete_ticket"
    description: str | None = "Archive/delete a HubSpot ticket."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_ticket(ticket_id: str = Field(..., description="Ticket ID to delete")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_API_BASE}/crm/v3/objects/tickets/{ticket_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "ticket_id": ticket_id}

        super().__init__(handler=_delete_ticket, **kwargs)


class HubSpotMergeTickets(Tool):
    name: str = "hubspot_merge_tickets"
    description: str | None = "Merge two HubSpot tickets into one."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _merge_tickets(
            primary_object_id: str = Field(..., description="Primary ticket ID to merge into"),
            object_id_to_merge: str = Field(..., description="Ticket ID to be merged")
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/tickets/merge",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"primaryObjectId": primary_object_id, "objectIdToMerge": object_id_to_merge},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_merge_tickets, **kwargs)


class HubSpotListProducts(Tool):
    name: str = "hubspot_list_products"
    description: str | None = "List HubSpot products with optional filtering."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_products(
            limit: int = Field(10, description="Maximum number of products to return"),
            after: str | None = Field(None, description="Cursor for pagination"),
            properties: list[str] | None = Field(None, description="List of properties to include in the response"),
            archived: bool = Field(False, description="Whether to include archived products"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"limit": limit, "archived": archived}
            if after:
                params["after"] = after
            if properties:
                params["properties"] = ",".join(properties)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/products",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_products, **kwargs)


class HubSpotGetProduct(Tool):
    name: str = "hubspot_get_product"
    description: str | None = "Get a specific HubSpot product by ID."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_product(
            product_id: str = Field(..., description="Product ID to retrieve"),
            properties: list[str] | None = Field(None, description="List of properties to include in the response"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {}
            if properties:
                params["properties"] = ",".join(properties)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/products/{product_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_product, **kwargs)


class HubSpotCreateProduct(Tool):
    name: str = "hubspot_create_product"
    description: str | None = "Create a new HubSpot product."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_product(
            name: str = Field(..., description="Product name"),
            price: str | None = Field(None, description="Product price"),
            description: str | None = Field(None, description="Product description"),
            hs_sku: str | None = Field(None, description="Product SKU"),
            hs_cost_of_goods_sold: str | None = Field(None, description="Cost of goods sold"),
            hs_recurring_billing_period: str | None = Field(None, description="Recurring billing period"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {"name": name}
            if price:
                properties["price"] = price
            if description:
                properties["description"] = description
            if hs_sku:
                properties["hs_sku"] = hs_sku
            if hs_cost_of_goods_sold:
                properties["hs_cost_of_goods_sold"] = hs_cost_of_goods_sold
            if hs_recurring_billing_period:
                properties["hs_recurring_billing_period"] = hs_recurring_billing_period

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/crm/v3/objects/products",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_product, **kwargs)


class HubSpotUpdateProduct(Tool):
    name: str = "hubspot_update_product"
    description: str | None = "Update an existing HubSpot product."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_product(
            product_id: str = Field(..., description="Product ID to update"),
            name: str | None = Field(None, description="Product name"),
            price: str | None = Field(None, description="Product price"),
            description: str | None = Field(None, description="Product description"),
            hs_sku: str | None = Field(None, description="Product SKU"),
            hs_cost_of_goods_sold: str | None = Field(None, description="Cost of goods sold"),
            hs_recurring_billing_period: str | None = Field(None, description="Recurring billing period"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            properties: dict[str, Any] = {}
            if name:
                properties["name"] = name
            if price:
                properties["price"] = price
            if description:
                properties["description"] = description
            if hs_sku:
                properties["hs_sku"] = hs_sku
            if hs_cost_of_goods_sold:
                properties["hs_cost_of_goods_sold"] = hs_cost_of_goods_sold
            if hs_recurring_billing_period:
                properties["hs_recurring_billing_period"] = hs_recurring_billing_period

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/crm/v3/objects/products/{product_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"properties": properties},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_product, **kwargs)


class HubSpotDeleteProduct(Tool):
    name: str = "hubspot_delete_product"
    description: str | None = "Archive/delete a HubSpot product."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_product(product_id: str = Field(..., description="Product ID to delete")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_API_BASE}/crm/v3/objects/products/{product_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "product_id": product_id}

        super().__init__(handler=_delete_product, **kwargs)


class HubSpotGetEngagements(Tool):
    name: str = "hubspot_get_engagements"
    description: str | None = "Get engagement data (calls, emails, meetings, etc.) for a contact."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_engagements(
            contact_id: str = Field(..., description="Contact ID to get engagements for"),
            limit: int = Field(20, description="Maximum number of engagements to return"),
            offset: int = Field(0, description="Offset for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/engagements/v1/engagements/associated/contact/{contact_id}/paged",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"limit": limit, "offset": offset},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_engagements, **kwargs)


class HubSpotGetEngagement(Tool):
    name: str = "hubspot_get_engagement"
    description: str | None = "Get a specific HubSpot engagement by ID."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_engagement(engagement_id: str = Field(..., description="Engagement ID to retrieve")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/engagements/v1/engagements/{engagement_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_engagement, **kwargs)


class HubSpotListEngagements(Tool):
    name: str = "hubspot_list_engagements"
    description: str | None = "List HubSpot engagements with optional filtering."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_engagements(
            limit: int = Field(20, description="Maximum number of engagements to return"),
            offset: int = Field(0, description="Offset for pagination"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/engagements/v1/engagements/paged",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"limit": limit, "offset": offset},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_engagements, **kwargs)


class HubSpotGetRecentEngagements(Tool):
    name: str = "hubspot_get_recent_engagements"
    description: str | None = "Get recently created or updated HubSpot engagements."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_recent_engagements(
            limit: int = Field(20, description="Maximum number of engagements to return"),
            offset: int = Field(0, description="Offset for pagination"),
            since: int | None = Field(None, description="Unix timestamp in milliseconds — only return engagements modified after this time"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"count": limit, "offset": offset}
            if since:
                params["since"] = since

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/engagements/v1/engagements/recent/modified",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_recent_engagements, **kwargs)


class HubSpotGetCallDispositions(Tool):
    name: str = "hubspot_get_call_dispositions"
    description: str | None = "Get all possible dispositions for sales calls in HubSpot."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_call_dispositions() -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/calling/v1/dispositions",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_call_dispositions, **kwargs)


class HubSpotCreateEngagement(Tool):
    name: str = "hubspot_create_engagement"
    description: str | None = "Create a new HubSpot engagement (email, call, meeting, task, or note)."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_engagement(
            engagement_type: str = Field(..., description="Engagement type: EMAIL, CALL, MEETING, TASK, or NOTE"),
            metadata: dict[str, Any] = Field(..., description="Type-specific fields (e.g. {\"body\": \"...\"} for NOTE, {\"subject\": \"...\"} for EMAIL)"),
            associations: dict[str, list[int]] | None = Field(None, description="Associations with contacts, companies, deals, or tickets: {\"contactIds\": [...], \"companyIds\": [...], \"dealIds\": [...], \"ticketIds\": [...]}"),
            active: bool = Field(True, description="Whether the engagement is active"),
            timestamp: int | None = Field(None, description="Unix timestamp in milliseconds for the engagement time"),
            owner_id: int | None = Field(None, description="Owner ID of the engagement"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            engagement: dict[str, Any] = {"type": engagement_type, "active": active}
            if timestamp:
                engagement["timestamp"] = timestamp
            if owner_id:
                engagement["ownerId"] = owner_id

            body: dict[str, Any] = {"engagement": engagement, "metadata": metadata}
            if associations:
                body["associations"] = associations

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/engagements/v1/engagements",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_engagement, **kwargs)


class UpdateEngagement(Tool):
    name: str = "hubspot_update_engagement"
    description: str | None = "Update an existing HubSpot engagement."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_engagement(
            engagement_id: str = Field(..., description="The ID of the engagement to update"),
            engagement: dict[str, Any] | None = Field(None, description="Updated engagement fields"),
            metadata: dict[str, Any] | None = Field(None, description="Updated metadata"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {}
            if engagement:
                body["engagement"] = engagement
            if metadata:
                body["metadata"] = metadata

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{_API_BASE}/engagements/v1/engagements/{engagement_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_update_engagement, **kwargs)


class HubSpotDeleteEngagement(Tool):
    name: str = "hubspot_delete_engagement"
    description: str | None = "Delete a HubSpot engagement."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_engagement(engagement_id: str = Field(..., description="Engagement ID to delete")) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_API_BASE}/engagements/v1/engagements/{engagement_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "engagement_id": engagement_id}

        super().__init__(handler=_delete_engagement, **kwargs)


class HubSpotSendEmail(Tool):
    name: str = "hubspot_send_email"
    description: str | None = "Send a transactional email to a HubSpot contact."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _send_email(
            email_id: int = Field(..., description="ID of the HubSpot transactional email template to send"),
            to_email: str = Field(..., description="Recipient email address"),
            contact_properties: dict[str, str] | None = Field(None, description="Override contact properties for personalization tokens"),
            custom_properties: dict[str, Any] | None = Field(None, description="Additional custom personalization tokens"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            body: dict[str, Any] = {
                "emailId": email_id,
                "message": {"to": to_email},
            }
            if contact_properties:
                body["contactProperties"] = [
                    {"name": k, "value": v} for k, v in contact_properties.items()
                ]
            if custom_properties:
                body["customProperties"] = [
                    {"name": k, "value": v} for k, v in custom_properties.items()
                ]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{_API_BASE}/marketing/v3/transactional/single-email/send",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_send_email, **kwargs)


class HubSpotGetAssociations(Tool):
    name: str = "hubspot_get_associations"
    description: str | None = "Get all associations for a specific object (contact, company, deal, ticket)."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_associations(
            from_object_type: str = Field(..., description="The type of object to get associations from (e.g., 'contacts', 'companies', 'deals', 'tickets')"),
            object_id: str = Field(..., description="The ID of the object to get associations from"),
            to_object_type: str = Field(..., description="The type of object to get associations to (e.g., 'contacts', 'companies', 'deals', 'tickets')"),
            after: str | None = Field(None, description="Cursor for pagination"),
            limit: int = Field(500, description="Maximum number of associations to return"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            params: dict[str, Any] = {"limit": limit}
            if after:
                params["after"] = after

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/objects/{from_object_type}/{object_id}/associations/{to_object_type}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_associations, **kwargs)


class HubSpotCreateAssociation(Tool):
    name: str = "hubspot_create_association"
    description: str | None = "Create an association between two HubSpot objects (e.g. link a contact to a company)."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_association(
            from_object_type: str = Field(..., description="The type of object to create an association from (e.g., 'contacts', 'companies', 'deals', 'tickets')"),
            from_object_id: str = Field(..., description="The ID of the object to create an association from"),
            to_object_type: str = Field(..., description="The type of object to create an association to (e.g., 'contacts', 'companies', 'deals', 'tickets')"),
            to_object_id: str = Field(..., description="The ID of the object to create an association to"),
            association_type: str = Field(..., description="The type of association to create (e.g., 'contact_to_company', 'deal_to_contact')"),
        ) -> Any:
            """
            association_type: e.g. "contact_to_company", "deal_to_contact". See GetAssociationTypes.
            """
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{_API_BASE}/crm/v3/objects/{from_object_type}/{from_object_id}"
                    f"/associations/{to_object_type}/{to_object_id}/{association_type}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_create_association, **kwargs)


class HubSpotDeleteAssociation(Tool):
    name: str = "hubspot_delete_association"
    description: str | None = "Remove an association between two HubSpot objects."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_association(
            from_object_type: str = Field(..., description="The type of object to delete an association from (e.g., 'contacts', 'companies', 'deals', 'tickets')"),
            from_object_id: str = Field(..., description="The ID of the object to delete an association from"),
            to_object_type: str = Field(..., description="The type of object to delete an association to (e.g., 'contacts', 'companies', 'deals', 'tickets')"),
            to_object_id: str = Field(..., description="The ID of the object to delete an association to"),
            association_type: str = Field(..., description="The type of association to delete (e.g., 'contact_to_company', 'deal_to_contact')"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{_API_BASE}/crm/v3/objects/{from_object_type}/{from_object_id}"
                    f"/associations/{to_object_type}/{to_object_id}/{association_type}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True}

        super().__init__(handler=_delete_association, **kwargs)


class HubSpotGetAssociationTypes(Tool):
    name: str = "hubspot_get_association_types"
    description: str | None = "Get all available association types and labels between two HubSpot object types."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_association_types(
            from_object_type: str = Field(..., description="The type of object to get association types from (e.g., 'contacts', 'companies', 'deals', 'tickets')"),
            to_object_type: str = Field(..., description="The type of object to get association types to (e.g., 'contacts', 'companies', 'deals', 'tickets')"),
        ) -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/associations/{from_object_type}/{to_object_type}/types",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_association_types, **kwargs)


class HubSpotGetUsers(Tool):
    name: str = "hubspot_get_users"
    description: str | None = "Get HubSpot users/owners with their IDs and information."
    integration: Annotated[str, Integration("hubspot")] | None = None
    token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "token": self.token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_users() -> Any:
            token = await _resolve_token(self)
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_API_BASE}/crm/v3/owners",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_users, **kwargs)