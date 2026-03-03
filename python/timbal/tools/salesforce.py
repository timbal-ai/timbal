from typing import Annotated, Any

import httpx

from ..core.tool import Tool
from ..platform.integrations import Integration

_SF_API_VERSION = "v60.0"


def _sf(instance_url: str, path: str) -> str:
    return f"{instance_url}/services/data/{_SF_API_VERSION}/{path}"


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


class CreateCase(Tool):
    name: str = "salesforce_create_case"
    description: str | None = "Create a new case in Salesforce."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_case(
            instance_url: str,
            subject: str,
            status: str = "New",
            priority: str = "Medium",
            origin: str | None = None,
            description: str | None = None,
            account_id: str | None = None,
            contact_id: str | None = None,
            case_type: str | None = None,
            reason: str | None = None,
        ) -> Any:
            """
            instance_url: Salesforce org URL, e.g. "https://myorg.my.salesforce.com"
            status: e.g. "New", "Working", "Closed".
            priority: "Low", "Medium", or "High".
            origin: e.g. "Phone", "Email", "Web".
            case_type: e.g. "Question", "Problem", "Feature Request".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {"Subject": subject, "Status": status, "Priority": priority}
            if origin:
                payload["Origin"] = origin
            if description:
                payload["Description"] = description
            if account_id:
                payload["AccountId"] = account_id
            if contact_id:
                payload["ContactId"] = contact_id
            if case_type:
                payload["Type"] = case_type
            if reason:
                payload["Reason"] = reason

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _sf(instance_url, "sobjects/Case/"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/CreateCase"
        super().__init__(handler=_create_case, metadata=metadata, **kwargs)


class UpdateCase(Tool):
    name: str = "salesforce_update_case"
    description: str | None = "Update an existing Salesforce case."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_case(
            instance_url: str,
            case_id: str,
            subject: str | None = None,
            status: str | None = None,
            priority: str | None = None,
            origin: str | None = None,
            description: str | None = None,
            account_id: str | None = None,
            contact_id: str | None = None,
            case_type: str | None = None,
            reason: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {}
            if subject:
                payload["Subject"] = subject
            if status:
                payload["Status"] = status
            if priority:
                payload["Priority"] = priority
            if origin:
                payload["Origin"] = origin
            if description:
                payload["Description"] = description
            if account_id:
                payload["AccountId"] = account_id
            if contact_id:
                payload["ContactId"] = contact_id
            if case_type:
                payload["Type"] = case_type
            if reason:
                payload["Reason"] = reason

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    _sf(instance_url, f"sobjects/Case/{case_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return {"updated": True, "case_id": case_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/UpdateCase"
        super().__init__(handler=_update_case, metadata=metadata, **kwargs)


class DeleteCase(Tool):
    name: str = "salesforce_delete_case"
    description: str | None = "Delete a Salesforce case."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_case(instance_url: str, case_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    _sf(instance_url, f"sobjects/Case/{case_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "case_id": case_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/DeleteCase"
        super().__init__(handler=_delete_case, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Comments (CaseComment)
# ---------------------------------------------------------------------------


class CreateComment(Tool):
    name: str = "salesforce_create_comment"
    description: str | None = "Create a comment on a Salesforce case."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_comment(
            instance_url: str,
            parent_id: str,
            comment_body: str,
            is_published: bool = True,
        ) -> Any:
            """
            parent_id: ID of the Case this comment belongs to.
            is_published: if True, the comment is visible to the customer portal.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _sf(instance_url, "sobjects/CaseComment/"),
                    headers={"Authorization": f"Bearer {token}"},
                    json={"ParentId": parent_id, "CommentBody": comment_body, "IsPublished": is_published},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/CreateComment"
        super().__init__(handler=_create_comment, metadata=metadata, **kwargs)


class UpdateComment(Tool):
    name: str = "salesforce_update_comment"
    description: str | None = "Update a Salesforce case comment."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_comment(
            instance_url: str,
            comment_id: str,
            comment_body: str | None = None,
            is_published: bool | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {}
            if comment_body:
                payload["CommentBody"] = comment_body
            if is_published is not None:
                payload["IsPublished"] = is_published

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    _sf(instance_url, f"sobjects/CaseComment/{comment_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return {"updated": True, "comment_id": comment_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/UpdateComment"
        super().__init__(handler=_update_comment, metadata=metadata, **kwargs)


class DeleteComment(Tool):
    name: str = "salesforce_delete_comment"
    description: str | None = "Delete a Salesforce case comment."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_comment(instance_url: str, comment_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    _sf(instance_url, f"sobjects/CaseComment/{comment_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "comment_id": comment_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/DeleteComment"
        super().__init__(handler=_delete_comment, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


class CreateContact(Tool):
    name: str = "salesforce_create_contact"
    description: str | None = "Create a new contact in Salesforce."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_contact(
            instance_url: str,
            last_name: str,
            first_name: str | None = None,
            email: str | None = None,
            phone: str | None = None,
            title: str | None = None,
            department: str | None = None,
            account_id: str | None = None,
            mailing_city: str | None = None,
            mailing_country: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {"LastName": last_name}
            if first_name:
                payload["FirstName"] = first_name
            if email:
                payload["Email"] = email
            if phone:
                payload["Phone"] = phone
            if title:
                payload["Title"] = title
            if department:
                payload["Department"] = department
            if account_id:
                payload["AccountId"] = account_id
            if mailing_city:
                payload["MailingCity"] = mailing_city
            if mailing_country:
                payload["MailingCountry"] = mailing_country

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _sf(instance_url, "sobjects/Contact/"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/CreateContact"
        super().__init__(handler=_create_contact, metadata=metadata, **kwargs)


class UpdateContact(Tool):
    name: str = "salesforce_update_contact"
    description: str | None = "Update an existing Salesforce contact."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_contact(
            instance_url: str,
            contact_id: str,
            last_name: str | None = None,
            first_name: str | None = None,
            email: str | None = None,
            phone: str | None = None,
            title: str | None = None,
            department: str | None = None,
            account_id: str | None = None,
            mailing_city: str | None = None,
            mailing_country: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {}
            if last_name:
                payload["LastName"] = last_name
            if first_name:
                payload["FirstName"] = first_name
            if email:
                payload["Email"] = email
            if phone:
                payload["Phone"] = phone
            if title:
                payload["Title"] = title
            if department:
                payload["Department"] = department
            if account_id:
                payload["AccountId"] = account_id
            if mailing_city:
                payload["MailingCity"] = mailing_city
            if mailing_country:
                payload["MailingCountry"] = mailing_country

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    _sf(instance_url, f"sobjects/Contact/{contact_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return {"updated": True, "contact_id": contact_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/UpdateContact"
        super().__init__(handler=_update_contact, metadata=metadata, **kwargs)


class DeleteContact(Tool):
    name: str = "salesforce_delete_contact"
    description: str | None = "Delete a Salesforce contact."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_contact(instance_url: str, contact_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    _sf(instance_url, f"sobjects/Contact/{contact_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "contact_id": contact_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/DeleteContact"
        super().__init__(handler=_delete_contact, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Leads
# ---------------------------------------------------------------------------


class CreateLead(Tool):
    name: str = "salesforce_create_lead"
    description: str | None = "Create a new lead in Salesforce."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_lead(
            instance_url: str,
            last_name: str,
            company: str,
            first_name: str | None = None,
            email: str | None = None,
            phone: str | None = None,
            title: str | None = None,
            lead_source: str | None = None,
            status: str = "Open - Not Contacted",
            industry: str | None = None,
            city: str | None = None,
            country: str | None = None,
            description: str | None = None,
        ) -> Any:
            """
            lead_source: e.g. "Web", "Phone Inquiry", "Partner Referral", "Purchased List".
            status: e.g. "Open - Not Contacted", "Working - Contacted", "Closed - Converted".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {
                "LastName": last_name,
                "Company": company,
                "Status": status,
            }
            if first_name:
                payload["FirstName"] = first_name
            if email:
                payload["Email"] = email
            if phone:
                payload["Phone"] = phone
            if title:
                payload["Title"] = title
            if lead_source:
                payload["LeadSource"] = lead_source
            if industry:
                payload["Industry"] = industry
            if city:
                payload["City"] = city
            if country:
                payload["Country"] = country
            if description:
                payload["Description"] = description

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _sf(instance_url, "sobjects/Lead/"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/CreateLead"
        super().__init__(handler=_create_lead, metadata=metadata, **kwargs)


class GetLead(Tool):
    name: str = "salesforce_get_lead"
    description: str | None = "Retrieve a specific Salesforce lead by ID."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_lead(
            instance_url: str,
            lead_id: str,
            fields: list[str] | None = None,
        ) -> Any:
            """
            fields: list of API field names to return. Returns all fields if omitted.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            url = _sf(instance_url, f"sobjects/Lead/{lead_id}")
            if fields:
                url += f"?fields={','.join(fields)}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/GetLead"
        super().__init__(handler=_get_lead, metadata=metadata, **kwargs)


class UpdateLead(Tool):
    name: str = "salesforce_update_lead"
    description: str | None = "Update an existing Salesforce lead."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_lead(
            instance_url: str,
            lead_id: str,
            last_name: str | None = None,
            first_name: str | None = None,
            company: str | None = None,
            email: str | None = None,
            phone: str | None = None,
            title: str | None = None,
            lead_source: str | None = None,
            status: str | None = None,
            industry: str | None = None,
            description: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {}
            if last_name:
                payload["LastName"] = last_name
            if first_name:
                payload["FirstName"] = first_name
            if company:
                payload["Company"] = company
            if email:
                payload["Email"] = email
            if phone:
                payload["Phone"] = phone
            if title:
                payload["Title"] = title
            if lead_source:
                payload["LeadSource"] = lead_source
            if status:
                payload["Status"] = status
            if industry:
                payload["Industry"] = industry
            if description:
                payload["Description"] = description

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    _sf(instance_url, f"sobjects/Lead/{lead_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return {"updated": True, "lead_id": lead_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/UpdateLead"
        super().__init__(handler=_update_lead, metadata=metadata, **kwargs)


class DeleteLead(Tool):
    name: str = "salesforce_delete_lead"
    description: str | None = "Delete a Salesforce lead."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_lead(instance_url: str, lead_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    _sf(instance_url, f"sobjects/Lead/{lead_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "lead_id": lead_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/DeleteLead"
        super().__init__(handler=_delete_lead, metadata=metadata, **kwargs)


class SearchLeads(Tool):
    name: str = "salesforce_search_leads"
    description: str | None = "Search for Salesforce leads using a SOSL query."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_leads(
            instance_url: str,
            search_term: str,
            fields: list[str] | None = None,
            limit: int = 20,
        ) -> Any:
            """
            search_term: keyword(s) to search for across lead fields.
            fields: list of Lead fields to return, e.g. ["Id", "FirstName", "LastName", "Email", "Status"].
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            returning_fields = ", ".join(fields) if fields else "Id, FirstName, LastName, Email, Company, Status, Phone"
            sosl = f"FIND {{{search_term}}} IN ALL FIELDS RETURNING Lead({returning_fields} LIMIT {limit})"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _sf(instance_url, "search/"),
                    headers={"Authorization": f"Bearer {token}"},
                    params={"q": sosl},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/SearchLeads"
        super().__init__(handler=_search_leads, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------


class CreateOpportunity(Tool):
    name: str = "salesforce_create_opportunity"
    description: str | None = "Create a new opportunity in Salesforce."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_opportunity(
            instance_url: str,
            name: str,
            stage_name: str,
            close_date: str,
            amount: float | None = None,
            account_id: str | None = None,
            probability: float | None = None,
            lead_source: str | None = None,
            description: str | None = None,
            owner_id: str | None = None,
        ) -> Any:
            """
            stage_name: e.g. "Prospecting", "Qualification", "Proposal/Price Quote",
                        "Negotiation/Review", "Closed Won", "Closed Lost".
            close_date: expected close date in YYYY-MM-DD format.
            probability: win probability percentage (0–100).
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {
                "Name": name,
                "StageName": stage_name,
                "CloseDate": close_date,
            }
            if amount is not None:
                payload["Amount"] = amount
            if account_id:
                payload["AccountId"] = account_id
            if probability is not None:
                payload["Probability"] = probability
            if lead_source:
                payload["LeadSource"] = lead_source
            if description:
                payload["Description"] = description
            if owner_id:
                payload["OwnerId"] = owner_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _sf(instance_url, "sobjects/Opportunity/"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/CreateOpportunity"
        super().__init__(handler=_create_opportunity, metadata=metadata, **kwargs)


class UpdateOpportunity(Tool):
    name: str = "salesforce_update_opportunity"
    description: str | None = "Update an existing Salesforce opportunity."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_opportunity(
            instance_url: str,
            opportunity_id: str,
            name: str | None = None,
            stage_name: str | None = None,
            close_date: str | None = None,
            amount: float | None = None,
            account_id: str | None = None,
            probability: float | None = None,
            lead_source: str | None = None,
            description: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {}
            if name:
                payload["Name"] = name
            if stage_name:
                payload["StageName"] = stage_name
            if close_date:
                payload["CloseDate"] = close_date
            if amount is not None:
                payload["Amount"] = amount
            if account_id:
                payload["AccountId"] = account_id
            if probability is not None:
                payload["Probability"] = probability
            if lead_source:
                payload["LeadSource"] = lead_source
            if description:
                payload["Description"] = description

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    _sf(instance_url, f"sobjects/Opportunity/{opportunity_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return {"updated": True, "opportunity_id": opportunity_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/UpdateOpportunity"
        super().__init__(handler=_update_opportunity, metadata=metadata, **kwargs)


class DeleteOpportunity(Tool):
    name: str = "salesforce_delete_opportunity"
    description: str | None = "Delete a Salesforce opportunity."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_opportunity(instance_url: str, opportunity_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    _sf(instance_url, f"sobjects/Opportunity/{opportunity_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "opportunity_id": opportunity_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/DeleteOpportunity"
        super().__init__(handler=_delete_opportunity, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class CreateTask(Tool):
    name: str = "salesforce_create_task"
    description: str | None = "Create a new task in Salesforce."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_task(
            instance_url: str,
            subject: str,
            status: str = "Not Started",
            priority: str = "Normal",
            what_id: str | None = None,
            who_id: str | None = None,
            activity_date: str | None = None,
            description: str | None = None,
            owner_id: str | None = None,
            task_type: str | None = None,
        ) -> Any:
            """
            what_id: related record ID (e.g. an Opportunity or Account ID).
            who_id: related contact or lead ID.
            activity_date: due date in YYYY-MM-DD format.
            status: e.g. "Not Started", "In Progress", "Completed", "Waiting on someone else", "Deferred".
            priority: "High", "Normal", or "Low".
            task_type: e.g. "Call", "Email", "Meeting".
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {"Subject": subject, "Status": status, "Priority": priority}
            if what_id:
                payload["WhatId"] = what_id
            if who_id:
                payload["WhoId"] = who_id
            if activity_date:
                payload["ActivityDate"] = activity_date
            if description:
                payload["Description"] = description
            if owner_id:
                payload["OwnerId"] = owner_id
            if task_type:
                payload["Type"] = task_type

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _sf(instance_url, "sobjects/Task/"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/CreateTask"
        super().__init__(handler=_create_task, metadata=metadata, **kwargs)


class GetTask(Tool):
    name: str = "salesforce_get_task"
    description: str | None = "Retrieve a specific Salesforce task by ID."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_task(
            instance_url: str,
            task_id: str,
            fields: list[str] | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            url = _sf(instance_url, f"sobjects/Task/{task_id}")
            if fields:
                url += f"?fields={','.join(fields)}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/GetTask"
        super().__init__(handler=_get_task, metadata=metadata, **kwargs)


class UpdateTask(Tool):
    name: str = "salesforce_update_task"
    description: str | None = "Update an existing Salesforce task."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_task(
            instance_url: str,
            task_id: str,
            subject: str | None = None,
            status: str | None = None,
            priority: str | None = None,
            activity_date: str | None = None,
            description: str | None = None,
            owner_id: str | None = None,
        ) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            payload: dict[str, Any] = {}
            if subject:
                payload["Subject"] = subject
            if status:
                payload["Status"] = status
            if priority:
                payload["Priority"] = priority
            if activity_date:
                payload["ActivityDate"] = activity_date
            if description:
                payload["Description"] = description
            if owner_id:
                payload["OwnerId"] = owner_id

            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    _sf(instance_url, f"sobjects/Task/{task_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                )
                response.raise_for_status()
                return {"updated": True, "task_id": task_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/UpdateTask"
        super().__init__(handler=_update_task, metadata=metadata, **kwargs)


class DeleteTask(Tool):
    name: str = "salesforce_delete_task"
    description: str | None = "Delete a Salesforce task."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_task(instance_url: str, task_id: str) -> Any:
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    _sf(instance_url, f"sobjects/Task/{task_id}"),
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                return {"deleted": True, "task_id": task_id}

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/DeleteTask"
        super().__init__(handler=_delete_task, metadata=metadata, **kwargs)


class SearchTasks(Tool):
    name: str = "salesforce_search_tasks"
    description: str | None = "Search for Salesforce tasks using a SOSL query."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_tasks(
            instance_url: str,
            search_term: str,
            fields: list[str] | None = None,
            limit: int = 20,
        ) -> Any:
            """
            search_term: keyword(s) to search for across task fields.
            fields: list of Task fields to return, e.g. ["Id", "Subject", "Status", "Priority", "ActivityDate"].
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            returning_fields = ", ".join(fields) if fields else "Id, Subject, Status, Priority, ActivityDate, OwnerId"
            sosl = f"FIND {{{search_term}}} IN ALL FIELDS RETURNING Task({returning_fields} LIMIT {limit})"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _sf(instance_url, "search/"),
                    headers={"Authorization": f"Bearer {token}"},
                    params={"q": sosl},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/SearchTasks"
        super().__init__(handler=_search_tasks, metadata=metadata, **kwargs)


# ---------------------------------------------------------------------------
# Generic / Query
# ---------------------------------------------------------------------------


class QuerySalesforce(Tool):
    name: str = "salesforce_query"
    description: str | None = "Execute a SOQL query against Salesforce and return matching records."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _query_salesforce(
            instance_url: str,
            soql: str,
            all_rows: bool = False,
        ) -> Any:
            """
            soql: SOQL query string.
              Example: "SELECT Id, Name, Email FROM Contact WHERE CreatedDate = TODAY"
            all_rows: if True, includes deleted and archived records in results.
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            endpoint = "queryAll/" if all_rows else "query/"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    _sf(instance_url, endpoint),
                    headers={"Authorization": f"Bearer {token}"},
                    params={"q": soql},
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/Query"
        super().__init__(handler=_query_salesforce, metadata=metadata, **kwargs)


class CreateRecord(Tool):
    name: str = "salesforce_create_record"
    description: str | None = "Create a record for any Salesforce object type."
    integration: Annotated[str, Integration("salesforce")]

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            "integration": {"type": "string", "value": self.integration},
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_record(
            instance_url: str,
            object_type: str,
            fields: dict[str, Any],
        ) -> Any:
            """
            object_type: Salesforce API object name, e.g. "Account", "CustomObject__c".
            fields: dict of Salesforce API field names to values.
              Example: {"Name": "Acme Corp", "BillingCity": "San Francisco"}
            """
            assert isinstance(self.integration, Integration)
            credential = await self.integration.resolve()
            token = credential.token

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _sf(instance_url, f"sobjects/{object_type}/"),
                    headers={"Authorization": f"Bearer {token}"},
                    json=fields,
                )
                response.raise_for_status()
                return response.json()

        metadata = kwargs.pop("metadata", {})
        metadata["type"] = "Salesforce/CreateRecord"
        super().__init__(handler=_create_record, metadata=metadata, **kwargs)
