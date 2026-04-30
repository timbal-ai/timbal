import os
from typing import Annotated, Any

from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration

_ZOHO_ACCOUNTS_DOMAIN = {
    "com": "https://accounts.zoho.com",
    "eu": "https://accounts.zoho.eu",
    "in": "https://accounts.zoho.in",
    "au": "https://accounts.zoho.com.au",
    "jp": "https://accounts.zoho.jp",
    "ca": "https://accounts.zohocloud.ca",
}

_ZOHO_API_DOMAIN = {
    "com": "https://www.zohoapis.com",
    "eu": "https://www.zohoapis.eu",
    "in": "https://www.zohoapis.in",
    "au": "https://www.zohoapis.com.au",
    "jp": "https://www.zohoapis.jp",
    "ca": "https://www.zohoapis.ca",
}

_API_VERSION = "v3"


def _build_api_base(data_center: str) -> str:
    domain = _ZOHO_API_DOMAIN.get(data_center.lower(), _ZOHO_API_DOMAIN["com"])
    return f"{domain}/crm/{_API_VERSION}"


async def _get_access_token(client_id: str, client_secret: str, refresh_token: str, data_center: str) -> str:
    """Exchange refresh token for a new access token via Zoho OAuth 2.0."""
    import httpx

    accounts_domain = _ZOHO_ACCOUNTS_DOMAIN.get(data_center.lower(), _ZOHO_ACCOUNTS_DOMAIN["com"])
    token_url = f"{accounts_domain}/oauth/v2/token"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
            },
        )
        response.raise_for_status()
        result = response.json()
        if "access_token" not in result:
            raise ValueError(f"Failed to obtain Zoho CRM access token: {result}")
        return result["access_token"]


async def _resolve_credentials(tool: Any) -> tuple[str, str]:
    """Resolve Zoho CRM access token and API base URL.

    Integration credentials (type: credentials):
    - client_id: Zoho OAuth client ID
    - client_secret: Zoho OAuth client secret
    - refresh_token: Zoho OAuth refresh token (long-lived)
    - data_center: Zoho data center region: 'com' (US, default), 'eu', 'in', 'au', 'jp', 'ca'
    """
    creds: dict[str, Any] = {}
    if isinstance(getattr(tool, "integration", None), Integration):
        creds = await tool.integration.resolve()

    client_id = creds.get("client_id") or getattr(tool, "client_id", None) or os.getenv("ZOHO_CRM_CLIENT_ID")
    client_secret = creds.get("client_secret") or (
        tool.client_secret.get_secret_value()
        if getattr(tool, "client_secret", None) and tool.client_secret
        else None
    ) or os.getenv("ZOHO_CRM_CLIENT_SECRET")
    refresh_token = creds.get("refresh_token") or (
        tool.refresh_token.get_secret_value()
        if getattr(tool, "refresh_token", None) and tool.refresh_token
        else None
    ) or os.getenv("ZOHO_CRM_REFRESH_TOKEN")
    data_center = creds.get("data_center") or getattr(tool, "data_center", None) or os.getenv("ZOHO_CRM_DATA_CENTER", "com")

    if not client_id or not client_secret or not refresh_token:
        raise ValueError(
            "Zoho CRM credentials not found. Configure integration with "
            "client_id, client_secret, refresh_token, and optionally data_center."
        )

    access_token = await _get_access_token(client_id, client_secret, refresh_token, data_center)
    api_base = _build_api_base(data_center)
    return access_token, api_base


class ZohoCrmGetObject(Tool):
    """Gets record data given its id."""

    name: str = "zoho_crm_get_object"
    description: str | None = "Gets record data given its id."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _get_object(
            module: str = Field(
                ...,
                description="Zoho CRM module API name (e.g. 'Leads', 'Contacts', 'Accounts', 'Deals').",
            ),
            record_id: str = Field(
                ...,
                description="Unique ID of the record to retrieve.",
            ),
            fields: str | None = Field(
                None,
                description="Comma-separated list of fields to return. Returns all fields if omitted.",
            ),
        ) -> Any:
            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/{module}/{record_id}"

            import httpx

            params: dict[str, str] = {}
            if fields:
                params["fields"] = fields

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
                    params=params if params else None,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_get_object, **kwargs)


class ZohoCrmUpdateObject(Tool):
    """Updates existing entities in the module. See the documentation."""

    name: str = "zoho_crm_update_object"
    description: str | None = "Updates existing entities in the module. See the documentation."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _update_object(
            module: str = Field(
                ...,
                description="Zoho CRM module API name (e.g. 'Leads', 'Contacts', 'Deals').",
            ),
            record_id: str = Field(
                ...,
                description="Unique ID of the record to update.",
            ),
            data: dict[str, Any] = Field(
                ...,
                description="JSON object with the fields to update. Only the provided fields will be changed.",
            ),
        ) -> Any:
            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/{module}/{record_id}"

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    url,
                    headers={
                        "Authorization": f"Zoho-oauthtoken {access_token}",
                        "Content-Type": "application/json",
                    },
                    json={"data": [data]},
                )
                response.raise_for_status()
                result = response.json()
                detail = result.get("data", [{}])[0]
                return {
                    "updated": detail.get("status") == "success",
                    "record_id": record_id,
                    "module": module,
                    "details": detail,
                }

        super().__init__(handler=_update_object, **kwargs)


class ZohoCrmSearchObjects(Tool):
    """Retrieves the records that match your search criteria."""

    name: str = "zoho_crm_search_objects"
    description: str | None = "Retrieves the records that match your search criteria."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _search_objects(
            module: str = Field(
                ...,
                description="Zoho CRM module API name (e.g. 'Leads', 'Contacts', 'Accounts', 'Deals').",
            ),
            criteria: str | None = Field(
                None,
                description="Field criteria filter, e.g. \"(Last_Name:equals:Smith)\" or \"(Amount:greater_than:1000)and(Stage:equals:Closed Won)\".",
            ),
            email: str | None = Field(
                None,
                description="Search by exact email address. Ignored if criteria is provided.",
            ),
            phone: str | None = Field(
                None,
                description="Search by phone number. Ignored if criteria or email is provided.",
            ),
            word: str | None = Field(
                None,
                description="Full-text word search across the module. Ignored if criteria, email, or phone is provided.",
            ),
            fields: str | None = Field(
                None,
                description="Comma-separated list of fields to return in results.",
            ),
            per_page: int = Field(20, description="Number of records per page (default 20, max 200)."),
            page: int = Field(1, description="Page number for pagination (default 1)."),
        ) -> Any:
            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/{module}/search"

            import httpx

            params: dict[str, Any] = {
                "per_page": min(per_page, 200),
                "page": page,
            }
            if criteria:
                params["criteria"] = criteria
            elif email:
                params["email"] = email
            elif phone:
                params["phone"] = phone
            elif word:
                params["word"] = word

            if fields:
                params["fields"] = fields

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_search_objects, **kwargs)


class ZohoCrmListObjects(Tool):
    """Gets the list of available records from a module."""

    name: str = "zoho_crm_list_objects"
    description: str | None = "Gets the list of available records from a module."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_objects(
            module: str = Field(
                ...,
                description="Zoho CRM module API name (e.g. 'Leads', 'Contacts', 'Accounts', 'Deals').",
            ),
            fields: str | None = Field(
                None,
                description="Comma-separated list of fields to return. Returns all fields if omitted.",
            ),
            sort_by: str | None = Field(
                None,
                description="Field API name to sort results by (e.g. 'Created_Time').",
            ),
            sort_order: str | None = Field(
                None,
                description="Sort direction: 'asc' or 'desc'.",
            ),
            per_page: int = Field(20, description="Number of records per page (default 20, max 200)."),
            page: int = Field(1, description="Page number for pagination (default 1)."),
        ) -> Any:
            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/{module}"

            import httpx

            params: dict[str, Any] = {
                "per_page": min(per_page, 200),
                "page": page,
            }
            if fields:
                params["fields"] = fields
            if sort_by:
                params["sort_by"] = sort_by
            if sort_order:
                params["sort_order"] = sort_order

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
                    params=params,
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_objects, **kwargs)


class ZohoCrmListModules(Tool):
    """Retrieves a list of all the modules available in your CRM account."""

    name: str = "zoho_crm_list_modules"
    description: str | None = "Retrieves a list of all the modules available in your CRM account."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_modules() -> Any:
            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/settings/modules"

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_modules, **kwargs)


class ZohoCrmListFields(Tool):
    """Gets the field metadata for the specified module."""

    name: str = "zoho_crm_list_fields"
    description: str | None = "Gets the field metadata for the specified module."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _list_fields(
            module: str = Field(
                ...,
                description="Zoho CRM module API name to retrieve field metadata for (e.g. 'Leads', 'Contacts', 'Deals').",
            ),
        ) -> Any:
            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/settings/fields"

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
                    params={"module": module},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_list_fields, **kwargs)


class ZohoCrmUploadAttachment(Tool):
    """Uploads an attachment file to Zoho CRM from a URL or file path. See the documentation."""

    name: str = "zoho_crm_upload_attachment"
    description: str | None = "Uploads an attachment file to Zoho CRM from a URL or file path. See the documentation."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_attachment(
            module: str = Field(
                ...,
                description="Zoho CRM module API name (e.g. 'Leads', 'Contacts', 'Deals').",
            ),
            record_id: str = Field(
                ...,
                description="Unique ID of the record to attach the file to.",
            ),
            attachment_url: str | None = Field(
                None,
                description="Public URL of the file to attach. If provided, file_path is ignored.",
            ),
            file_path: str | None = Field(
                None,
                description="Local file path of the file to upload as an attachment. Used only if attachment_url is not provided.",
            ),
        ) -> Any:
            if not attachment_url and not file_path:
                raise ValueError("Either attachment_url or file_path must be provided.")

            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/{module}/{record_id}/Attachments"

            import httpx

            headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}

            async with httpx.AsyncClient() as client:
                if attachment_url:
                    response = await client.post(
                        url,
                        headers=headers,
                        params={"attachmentUrl": attachment_url},
                    )
                else:
                    assert file_path is not None
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                    file_name = file_path.split("/")[-1]
                    response = await client.post(
                        url,
                        headers=headers,
                        files={"file": (file_name, file_content)},
                    )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_upload_attachment, **kwargs)


class ZohoCrmDownloadAttachment(Tool):
    """Downloads an attachment file from Zoho CRM, saves it in the temporary file system and exports the file path."""

    name: str = "zoho_crm_download_attachment"
    description: str | None = "Downloads an attachment file from Zoho CRM, saves it in the temporary file system and exports the file path."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _download_attachment(
            module: str = Field(
                ...,
                description="Zoho CRM module API name (e.g. 'Leads', 'Contacts', 'Deals').",
            ),
            record_id: str = Field(
                ...,
                description="Unique ID of the record the attachment belongs to.",
            ),
            attachment_id: str = Field(
                ...,
                description="Unique ID of the attachment to download.",
            ),
        ) -> Any:
            import tempfile

            import httpx

            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/{module}/{record_id}/Attachments/{attachment_id}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
                )
                response.raise_for_status()

                content_disposition = response.headers.get("Content-Disposition", "")
                file_name = attachment_id
                if "filename=" in content_disposition:
                    file_name = content_disposition.split("filename=")[-1].strip().strip('"')

                content = response.content

            suffix = f"_{file_name}" if file_name else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            return {"file_path": tmp_path, "file_name": file_name, "size_bytes": len(content)}

        super().__init__(handler=_download_attachment, **kwargs)


class ZohoCrmCreateObject(Tool):
    """Create a new object/module entry. See the documentation."""

    name: str = "zoho_crm_create_object"
    description: str | None = "Create a new object/module entry. See the documentation."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _create_object(
            module: str = Field(
                ...,
                description="Zoho CRM module API name (e.g. 'Leads', 'Contacts', 'Accounts', 'Deals').",
            ),
            data: dict[str, Any] = Field(
                ...,
                description="JSON object with record field values. Use Zoho field API names (e.g. {'Last_Name': 'Smith', 'Email': 'smith@example.com', 'Amount': 5000}).",
            ),
        ) -> Any:
            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/{module}"

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Zoho-oauthtoken {access_token}",
                        "Content-Type": "application/json",
                    },
                    json={"data": [data]},
                )
                response.raise_for_status()
                result = response.json()
                detail = result.get("data", [{}])[0]
                return {
                    "created": detail.get("status") == "success",
                    "record_id": detail.get("details", {}).get("id"),
                    "module": module,
                    "details": detail,
                }

        super().__init__(handler=_create_object, **kwargs)


class ZohoCrmConvertLead(Tool):
    """Converts a Lead into a Contact or an Account. See the documentation."""

    name: str = "zoho_crm_convert_lead"
    description: str | None = "Converts a Lead into a Contact or an Account. See the documentation."
    integration: Annotated[str, Integration("zoho_crm")] | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    data_center: str | None = None

    def get_config(self) -> dict[str, Any]:
        """See base class."""
        return {
            **super().get_config(),
            **self._annotate_config({
                "integration": self.integration,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "data_center": self.data_center,
            }),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _convert_lead(
            lead_id: str = Field(
                ...,
                description="Unique ID of the Lead record to convert.",
            ),
            overwrite: bool = Field(
                False,
                description="If true, existing Contact/Account data will be overwritten with lead data.",
            ),
            notify_lead_owner: bool = Field(
                False,
                description="If true, notifies the lead owner about the conversion.",
            ),
            notify_new_entity_owner: bool = Field(
                False,
                description="If true, notifies the owner of the newly created Contact/Account.",
            ),
            accounts: dict[str, Any] | None = Field(
                None,
                description="Account data for the conversion. Use {'id': 'existing_account_id'} to link an existing account, or provide field values to create a new one.",
            ),
            contacts: dict[str, Any] | None = Field(
                None,
                description="Contact data for the conversion. Use {'id': 'existing_contact_id'} to link an existing contact, or provide field values to create a new one.",
            ),
            deals: dict[str, Any] | None = Field(
                None,
                description="Deal data to create during conversion (e.g. {'Deal_Name': 'New Deal', 'Amount': 5000}). Omit to skip deal creation.",
            ),
        ) -> Any:
            access_token, api_base = await _resolve_credentials(self)
            url = f"{api_base}/Leads/{lead_id}/actions/convert"

            import httpx

            payload: dict[str, Any] = {
                "overwrite": overwrite,
                "notify_lead_owner": notify_lead_owner,
                "notify_new_entity_owner": notify_new_entity_owner,
            }
            if accounts is not None:
                payload["Accounts"] = accounts
            if contacts is not None:
                payload["Contacts"] = contacts
            if deals is not None:
                payload["Deals"] = deals

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Zoho-oauthtoken {access_token}",
                        "Content-Type": "application/json",
                    },
                    json={"data": [payload]},
                )
                response.raise_for_status()
                return response.json()

        super().__init__(handler=_convert_lead, **kwargs)
