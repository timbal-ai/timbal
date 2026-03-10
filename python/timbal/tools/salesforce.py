import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_SF_API_VERSION = "v60.0"


async def _resolve_token(tool: Any) -> str:
    """Resolve Salesforce security token from integration, explicit field, or env var."""
    if isinstance(tool.integration, Integration):
        credentials = await tool.integration.resolve()
        return credentials["security_token"]
    if tool.security_token is not None:
        return tool.security_token.get_secret_value()
    env_key = os.getenv("SALESFORCE_SECURITY_TOKEN")
    if env_key:
        return env_key
    raise ValueError(
        "Salesforce security token not found. Set SALESFORCE_SECURITY_TOKEN environment variable, "
        "pass security_token in config, or configure an integration."
    )


def _sf(instance_url: str, path: str) -> str:
    return f"{instance_url}/services/data/{_SF_API_VERSION}/{path}"


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


class CreateCase(Tool):
    name: str = "salesforce_create_case"
    description: str | None = "Create a new case in Salesforce."
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_case(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            subject: str = Field(..., description="Case subject/title"),
            status: str = Field("New", description="Case status: e.g. 'New', 'Working', 'Closed'"),
            priority: str = Field("Medium", description="Case priority: e.g. 'High', 'Medium', 'Low'"),
            origin: str | None = Field(None, description="Case origin: e.g. 'Web', 'Email', 'Phone'"),
            description: str | None = Field(None, description="Case description/details"),
            account_id: str | None = Field(None, description="Salesforce Account ID"),
            contact_id: str | None = Field(None, description="Salesforce Contact ID"),
            case_type: str | None = Field(None, description="Case type: e.g. 'Problem', 'Feature Request', 'Question'"),
            reason: str | None = Field(None, description="Case reason: e.g. 'Equipment failure', 'User error'"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_case(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            case_id: str = Field(..., description="Salesforce Case ID"),
            subject: str | None = Field(None, description="Case subject/title"),
            status: str | None = Field(None, description="Case status: e.g. 'New', 'Working', 'Closed'"),
            priority: str | None = Field(None, description="Case priority: e.g. 'High', 'Medium', 'Low'"),
            origin: str | None = Field(None, description="Case origin: e.g. 'Web', 'Email', 'Phone'"),
            description: str | None = Field(None, description="Case description/details"),
            account_id: str | None = Field(None, description="Salesforce Account ID"),
            contact_id: str | None = Field(None, description="Salesforce Contact ID"),
            case_type: str | None = Field(None, description="Case type: e.g. 'Problem', 'Feature Request'"),
            reason: str | None = Field(None, description="Case reason: e.g. 'Equipment failure', 'User error'"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_case(
            instance_url: str = Field(..., description="Salesforce org URL"),
            case_id: str = Field(..., description="Salesforce case ID")
        ) -> Any:
            token = await _resolve_token(self)

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_comment(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            parent_id: str = Field(..., description="ID of the Case this comment belongs to"),
            comment_body: str = Field(..., description="Comment content/body text"),
            is_published: bool = Field(True, description="If True, the comment is visible to the customer portal"),
        ) -> Any:
            token = await _resolve_token(self)

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_comment(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            comment_id: str = Field(..., description="Salesforce Case Comment ID to update"),
            comment_body: str | None = Field(None, description="Updated comment content/body text"),
            is_published: bool | None = Field(None, description="Updated visibility status: True if visible to customer portal"),
        ) -> Any:
            token = await _resolve_token(self)

            payload: dict[str, Any] = {}
            if comment_body:
                payload["CommentBody"] = comment_body
            if is_published is not None:
                payload["IsPublished"] = is_published

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_comment(
            instance_url: str = Field(..., description="Salesforce org URL"),
            comment_id: str = Field(..., description="Salesforce comment ID")
        ) -> Any:
            token = await _resolve_token(self)

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_contact(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            last_name: str = Field(..., description="Contact last name (required)"),
            first_name: str | None = Field(None, description="Contact first name"),
            email: str | None = Field(None, description="Contact email address"),
            phone: str | None = Field(None, description="Contact phone number"),
            title: str | None = Field(None, description="Contact job title"),
            department: str | None = Field(None, description="Contact department"),
            account_id: str | None = Field(None, description="Salesforce Account ID to associate with contact"),
            mailing_city: str | None = Field(None, description="Contact mailing city"),
            mailing_country: str | None = Field(None, description="Contact mailing country"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_contact(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            contact_id: str = Field(..., description="Salesforce Contact ID to update"),
            last_name: str | None = Field(None, description="Updated contact last name"),
            first_name: str | None = Field(None, description="Updated contact first name"),
            email: str | None = Field(None, description="Updated contact email address"),
            phone: str | None = Field(None, description="Updated contact phone number"),
            title: str | None = Field(None, description="Updated contact job title"),
            department: str | None = Field(None, description="Updated contact department"),
            account_id: str | None = Field(None, description="Updated Salesforce Account ID to associate with contact"),
            mailing_city: str | None = Field(None, description="Updated contact mailing city"),
            mailing_country: str | None = Field(None, description="Updated contact mailing country"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_contact(
            instance_url: str = Field(..., description="Salesforce org URL"),
            contact_id: str = Field(..., description="Salesforce contact ID")
        ) -> Any:
            token = await _resolve_token(self)

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_lead(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            last_name: str = Field(..., description="Lead last name (required)"),
            company: str = Field(..., description="Lead company name (required)"),
            first_name: str | None = Field(None, description="Lead first name"),
            email: str | None = Field(None, description="Lead email address"),
            phone: str | None = Field(None, description="Lead phone number"),
            title: str | None = Field(None, description="Lead job title"),
            lead_source: str | None = Field(None, description="Lead source: e.g. 'Web', 'Phone Inquiry', 'Partner Referral', 'Purchased List'"),
            status: str = Field("Open - Not Contacted", description="Lead status: e.g. 'Open - Not Contacted', 'Working - Contacted', 'Closed - Converted'"),
            industry: str | None = Field(None, description="Lead industry"),
            city: str | None = Field(None, description="Lead city"),
            country: str | None = Field(None, description="Lead country"),
            description: str | None = Field(None, description="Lead description or notes"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_lead(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            lead_id: str = Field(..., description="Salesforce Lead ID to retrieve"),
            fields: list[str] | None = Field(None, description="List of API field names to return. Returns all fields if omitted"),
        ) -> Any:
            token = await _resolve_token(self)

            url = _sf(instance_url, f"sobjects/Lead/{lead_id}")
            if fields:
                url += f"?fields={','.join(fields)}"

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_lead(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            lead_id: str = Field(..., description="Salesforce Lead ID to update"),
            last_name: str | None = Field(None, description="Lead last name"),
            first_name: str | None = Field(None, description="Lead first name"),
            company: str | None = Field(None, description="Lead company"),
            email: str | None = Field(None, description="Lead email"),
            phone: str | None = Field(None, description="Lead phone"),
            title: str | None = Field(None, description="Lead title"),
            lead_source: str | None = Field(None, description="Lead source"),
            status: str | None = Field(None, description="Lead status"),
            industry: str | None = Field(None, description="Lead industry"),
            description: str | None = Field(None, description="Lead description"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_lead(
            instance_url: str = Field(..., description="Salesforce org URL"),
            lead_id: str = Field(..., description="Salesforce lead ID")
        ) -> Any:
            token = await _resolve_token(self)

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_leads(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            search_term: str = Field(..., description="Keyword(s) to search for across lead fields."),
            fields: list[str] | None = Field(None, description="list of Lead fields to return, e.g. ['Id', 'FirstName', 'LastName', 'Email', 'Status']."),
            limit: int = Field(20, description="Maximum number of leads to return"),
        ) -> Any:
            token = await _resolve_token(self)

            returning_fields = ", ".join(fields) if fields else "Id, FirstName, LastName, Email, Company, Status, Phone"
            sosl = f"FIND {{{search_term}}} IN ALL FIELDS RETURNING Lead({returning_fields} LIMIT {limit})"

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_opportunity(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            name: str = Field(..., description="Opportunity name (required)"),
            stage_name: str = Field(..., description="Opportunity stage: e.g. 'Prospecting', 'Qualification', 'Proposal/Price Quote', 'Negotiation/Review', 'Closed Won', 'Closed Lost'"),
            close_date: str = Field(..., description="Expected close date in YYYY-MM-DD format"),
            amount: float | None = Field(None, description="Opportunity amount/value"),
            account_id: str | None = Field(None, description="Salesforce Account ID to associate with opportunity"),
            probability: float | None = Field(None, description="Probability of closing (0-100)"),
            lead_source: str | None = Field(None, description="Lead source: e.g. 'Web', 'Phone', 'Referral'"),
            description: str | None = Field(None, description="Opportunity description or notes"),
            owner_id: str | None = Field(None, description="Salesforce User ID for opportunity owner"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_opportunity(
            instance_url: str = Field(..., description="Salesforce org URL"),
            opportunity_id: str = Field(..., description="Salesforce opportunity ID"),
            name: str | None = Field(None, description="Opportunity name"),
            stage_name: str | None = Field(None, description="Opportunity stage"),
            close_date: str | None = Field(None, description="Close date in YYYY-MM-DD format"),
            amount: float | None = Field(None, description="Opportunity amount"),
            account_id: str | None = Field(None, description="Account ID"),
            probability: float | None = Field(None, description="Probability of closing (0-100)"),
            lead_source: str | None = Field(None, description="Lead source: e.g. 'Web', 'Phone', 'Referral'"),
            description: str | None = Field(None, description="Opportunity description or notes"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_opportunity(
            instance_url: str = Field(..., description="Salesforce org URL"),
            opportunity_id: str = Field(..., description="Salesforce opportunity ID")
        ) -> Any:
            token = await _resolve_token(self)

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_task(
            instance_url: str = Field(..., description="Salesforce org URL"),
            subject: str = Field(..., description="Task subject/description"),
            status: str = Field("Not Started", description="Task status:  'Not Started', 'In Progress', 'Completed', 'Waiting on someone else', 'Deferred'"),
            priority: str = Field("Normal", description="Task priority. Possibilities: 'High', 'Normal', or 'Low'"),
            what_id: str | None = Field(None, description="Related record ID (e.g. Opportunity or Account ID)"),
            who_id: str | None = Field(None, description="Related contact or lead ID"),
            activity_date: str | None = Field(None, description="Due date in YYYY-MM-DD format"),
            description: str | None = Field(None, description="Task description"),
            owner_id: str | None = Field(None, description="Owner ID"),
            task_type: str | None = Field(None, description="Task type (e.g. Call, Email, Meeting)"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_task(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            task_id: str = Field(..., description="Salesforce Task ID to retrieve"),
            fields: list[str] | None = Field(None, description="List of API field names to return. Returns all fields if omitted"),
        ) -> Any:
            token = await _resolve_token(self)

            url = _sf(instance_url, f"sobjects/Task/{task_id}")
            if fields:
                url += f"?fields={','.join(fields)}"

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_task(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            task_id: str = Field(..., description="Salesforce Task ID to update"),
            subject: str | None = Field(None, description="Updated task subject/title"),
            status: str | None = Field(None, description="Updated task status: e.g. 'Not Started', 'In Progress', 'Completed'"),
            priority: str | None = Field(None, description="Updated task priority: e.g. 'High', 'Normal', 'Low'"),
            activity_date: str | None = Field(None, description="Updated activity date in YYYY-MM-DD format"),
            description: str | None = Field(None, description="Updated task description or notes"),
            owner_id: str | None = Field(None, description="Updated Salesforce User ID for task owner"),
        ) -> Any:
            token = await _resolve_token(self)

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

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_task(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            task_id: str = Field(..., description="Salesforce Task ID to delete"),
        ) -> Any:
            token = await _resolve_token(self)

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_tasks(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            search_term: str = Field(..., description="Keyword(s) to search for across task fields"),
            fields: list[str] | None = Field(None, description="List of Task fields to return, e.g. ['Id', 'Subject', 'Status', 'Priority', 'ActivityDate']"),
            limit: int = Field(20, description="Maximum number of tasks to return"),
        ) -> Any:
            token = await _resolve_token(self)

            returning_fields = ", ".join(fields) if fields else "Id, Subject, Status, Priority, ActivityDate, OwnerId"
            sosl = f"FIND {{{search_term}}} IN ALL FIELDS RETURNING Task({returning_fields} LIMIT {limit})"

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _query_salesforce(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            soql: str = Field(..., description="SOQL query string. Example: 'SELECT Id, Name, Email FROM Contact WHERE CreatedDate = TODAY'"),
            all_rows: bool = Field(False, description="If True, includes deleted and archived records in results"),
        ) -> Any:
            token = await _resolve_token(self)

            endpoint = "queryAll/" if all_rows else "query/"

            import httpx

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
    integration: Annotated[str, Integration("salesforce")] | None = None
    security_token: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config(
                {"integration": self.integration, "security_token": self.security_token},
                required={"integration"},
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_record(
            instance_url: str = Field(..., description="Salesforce org URL, e.g. 'https://myorg.my.salesforce.com'"),
            object_type: str = Field(..., description="Salesforce API object name, e.g. 'Account', 'CustomObject__c'"),
            fields: dict[str, Any] = Field(..., description="Dict of Salesforce API field names to values. Example: {'Name': 'Acme Corp', 'BillingCity': 'San Francisco'}"),
        ) -> Any:
            token = await _resolve_token(self)

            import httpx

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
