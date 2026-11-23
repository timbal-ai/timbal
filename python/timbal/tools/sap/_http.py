"""Published SAP OData/CDS and SOAP operations and inbound SAP Agent Gateway A2A."""

import asyncio
import hashlib
import json
import re
import time
import weakref
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlsplit
from uuid import uuid4
from xml.etree import ElementTree as ET

import httpx
from pydantic import BaseModel, Field, JsonValue, SecretStr, StringConstraints, field_validator, model_validator

from ...platform.integrations import Integration
from ._access import Ref, SAPCall, SAPCustomizing, SAPHTTPError, SAPTool, Scalar, parse_host

Property = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
TokenProvider = Callable[[], Awaitable[str]]
_SOAP = {"1.1": "http://schemas.xmlsoap.org/soap/envelope/", "1.2": "http://www.w3.org/2003/05/soap-envelope"}
_SAP_CLIENT = re.compile(r"^\d{3}$")
#: Seconds before expiry at which a cached token is renewed (half the lifetime, if shorter).
_TOKEN_MARGIN = 120
_TOKEN_TIMEOUT = 15
_clock = time.monotonic
# (token URL, client ID, hash of the secret) -> (access token, renew at)
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
# One grant at a time per client. Per event loop, since an asyncio.Lock must not be shared
# across loops; weak keys so a closed loop takes its locks along.
_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = weakref.WeakKeyDictionary()


def https_endpoint(value: str) -> str:
    url = httpx.URL(value)
    if url.scheme != "https" or not url.host or url.userinfo or url.fragment:
        raise ValueError("Configure an HTTPS SAP endpoint without embedded credentials or a fragment.")
    return value


def _lock(key: str) -> asyncio.Lock:
    locks = _LOCKS.setdefault(asyncio.get_running_loop(), {})
    lock = locks.get(key)
    if lock is None:
        lock = locks[key] = asyncio.Lock()
    return lock


def _token_key(token_url: str, client_id: str, client_secret: str) -> str:
    secret = hashlib.sha256(client_secret.encode()).hexdigest()
    return hashlib.sha256("\x00".join((token_url, client_id, secret)).encode()).hexdigest()


async def client_credentials_token(token_url: str, client_id: str, client_secret: str) -> tuple[str, str]:
    """An access token from the OAuth 2.0 client-credentials grant, and its cache key.

    Cached in this process until shortly before it expires. Errors never carry the secret or a token.
    """
    try:
        https_endpoint(token_url)
    except (ValueError, httpx.InvalidURL):
        raise ValueError("The SAP connection's token_url must be an HTTPS URL without credentials.") from None
    key = _token_key(token_url, client_id, client_secret)
    async with _lock(key):
        cached = _TOKEN_CACHE.get(key)
        if cached and _clock() < cached[1]:
            return cached[0], key
        async with httpx.AsyncClient(timeout=_TOKEN_TIMEOUT, follow_redirects=False) as client:
            response = await client.post(
                token_url,
                auth=(client_id, client_secret),
                data={"grant_type": "client_credentials"},
                headers={"Accept": "application/json"},
            )
        if 300 <= response.status_code < 400:
            raise RuntimeError("SAP's token endpoint answered with a redirect, which is not followed.")
        if not response.is_success:
            raise SAPHTTPError("token endpoint", response.status_code)
        try:
            body = response.json()
        except ValueError:
            body = None
        token = body.get("access_token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token:
            raise RuntimeError("SAP's token endpoint did not issue an access token.")
        token_type = body.get("token_type")
        if not isinstance(token_type, str) or token_type.lower() != "bearer":
            raise RuntimeError("SAP's token endpoint issued a token that is not a bearer token.")
        lifetime = body.get("expires_in")
        try:
            lifetime = float(lifetime)
        except (TypeError, ValueError):
            lifetime = 0
        if lifetime > 0:
            _TOKEN_CACHE[key] = (token, _clock() + lifetime - min(_TOKEN_MARGIN, lifetime / 2))
        return token, key


def _sap_connection(creds: dict[str, Any]) -> dict[str, Any]:
    """host, port and SAP client of a ``sap`` connection."""
    host, url_port = parse_host(creds.get("host"))
    port = creds.get("port")
    try:
        port = int(port) if port not in (None, "") else (url_port or 443)
    except (TypeError, ValueError):
        raise ValueError("The SAP connection's port must be a number.") from None
    if not 1 <= port <= 65535 or (url_port and url_port != port):
        raise ValueError("The SAP connection's host and port disagree, or the port is out of range.")
    sap_client = creds.get("sap_client") or None
    if sap_client is not None and not _SAP_CLIENT.match(str(sap_client)):
        raise ValueError("The SAP connection's sap_client must be three digits.")
    return {"host": host, "port": port, "sap_client": sap_client and str(sap_client)}


class SAPTokenExchange(BaseModel):
    """Obtain a SAP-audience access token for the current caller via SAP Cloud Identity Services."""

    endpoint: Ref
    resource: Ref
    client_id: Ref
    client_secret: SecretStr = Field(exclude=True, repr=False)
    subject_token: TokenProvider = Field(exclude=True, repr=False)
    subject_token_type: Literal[
        "urn:ietf:params:oauth:token-type:access_token", "urn:ietf:params:oauth:token-type:id_token"
    ] = "urn:ietf:params:oauth:token-type:access_token"

    _check_endpoint = field_validator("endpoint")(https_endpoint)

    async def __call__(self) -> str:
        subject = await self.subject_token()
        if not subject:
            raise PermissionError("SAP token exchange requires an authenticated user.")
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            response = await client.post(
                self.endpoint,
                auth=(self.client_id, self.client_secret.get_secret_value()),
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                    "resource": self.resource,
                    "subject_token_type": self.subject_token_type,
                    "subject_token": subject,
                },
            )
        if not response.is_success:
            raise SAPHTTPError("token exchange", response.status_code)
        token = response.json().get("access_token")
        if not isinstance(token, str) or not token or token == subject:
            raise RuntimeError("SAP token exchange did not issue a distinct access token.")
        return token


class _HTTPTool(SAPTool):
    endpoint: Ref
    object_name: Ref
    # Supplies the bearer token instead of the connection's OAuth client (e.g. SAPTokenExchange).
    token_provider: TokenProvider | None = Field(default=None, exclude=True)
    # A ``sap`` connection: host, port, sap_client and the OAuth client (token_url, client_id,
    # client_secret) the tool obtains short-lived tokens from.
    integration: Annotated[str, Integration("sap")] | None = None

    _check_endpoint = field_validator("endpoint")(https_endpoint)

    async def _client(self, call: SAPCall) -> httpx.AsyncClient:
        creds = await self.integration.resolve() if self.integration else None
        if creds is not None:
            call.connection = _sap_connection(creds)
            self._pinned(call, self.endpoint)
        if self.token_provider:
            token = await self.token_provider()
        elif creds is not None:
            token_url, client_id, secret = (creds.get(k) for k in ("token_url", "client_id", "client_secret"))
            if not (token_url and client_id and secret):
                raise ValueError(
                    "The SAP connection needs token_url, client_id and client_secret, or configure a token_provider."
                )
            token, call.token_key = await client_credentials_token(str(token_url), str(client_id), str(secret))
        else:
            token = None
        if not token:
            raise ValueError("Configure a provider of SAP-audience access tokens for this endpoint.")
        return httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"}, follow_redirects=False, timeout=self.access.timeout_seconds
        )

    @staticmethod
    def _pinned(call: SAPCall, url: str) -> str:
        """Refuse a URL off the connection's host and port; add the connection's SAP client when absent."""
        if call.connection is None:
            return url
        target = httpx.URL(url)
        if (target.scheme, target.host.lower(), target.port or 443) != (
            "https",
            call.connection["host"],
            call.connection["port"],
        ):
            # Whatever the token's source, it goes only to the host the connection names.
            raise PermissionError("The SAP endpoint is not on the connection's host and port.")
        sap_client = call.connection["sap_client"]
        if sap_client and not any(key.lower() == "sap-client" for key in target.params.keys()):
            target = target.copy_add_param("sap-client", sap_client)
        return str(target)

    async def _send(
        self,
        call: SAPCall,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers_only: bool = False,
        **kwargs,
    ) -> tuple[bytes, httpx.Headers]:
        async with client.stream(method, self._pinned(call, url), **kwargs) as response:
            if not response.is_success:
                if response.status_code == 401 and call.token_key:
                    # SAP refused the cached token (revoked or rotated): the next call gets a new one.
                    _TOKEN_CACHE.pop(call.token_key, None)
                # The URL/query and SAP error body can contain customer data.
                raise SAPHTTPError(self.object_name, response.status_code)
            if headers_only:
                return b"", response.headers
            content = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=min(65536, self.access.max_bytes + 1)):
                if len(content) + len(chunk) > self.access.max_bytes:
                    raise ValueError("SAP response exceeds the configured byte limit.")
                content.extend(chunk)
            return bytes(content), response.headers


def _literal(value: Scalar) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return json.dumps(value, allow_nan=False)


class SAPODataRead(_HTTPTool):
    name: str = "sap_odata_read"
    description: str = "Read one page of the configured SAP OData entity set using equality filters."
    read_only: Literal[True] = True
    cds: bool = False
    properties: Annotated[tuple[Property, ...], Field(min_length=1)]
    order_by: Annotated[tuple[Property, ...], Field(min_length=1)]

    async def _invoke(
        self,
        filters: dict[str, Scalar] | None = None,
        limit: int | None = None,
        offset: int = 0,
        skiptoken: str | None = None,
    ) -> dict:
        async def run(call):
            call.account_input([filters, limit, offset, skiptoken])
            size = self.access.max_rows if limit is None else limit
            if not 1 <= size <= self.access.max_rows or offset < 0 or (skiptoken is not None and offset):
                raise ValueError("Invalid SAP page size, offset or continuation.")
            query = {"$select": ",".join(self.properties), "$top": str(size), "$orderby": ",".join(self.order_by)}
            if filters:
                if not filters.keys() <= set(self.properties):
                    raise ValueError("Filter property is outside the configured projection.")
                query["$filter"] = " and ".join(f"{k} eq {_literal(v)}" for k, v in filters.items())
            if skiptoken is not None:
                query["$skiptoken"] = skiptoken
            elif offset:
                query["$skip"] = str(offset)
            async with await self._client(call) as client:
                body, _ = await self._send(
                    call,
                    client,
                    "GET",
                    str(httpx.URL(self.endpoint).copy_merge_params(query)),
                    headers={"Accept": "application/json"},
                )
            data = json.loads(body)
            collection = data.get("d", data)
            rows = collection.get("results", collection.get("value"))
            if not isinstance(rows, list) or len(rows) > size:
                raise ValueError("SAP returned an invalid or oversized OData page.")
            # Extract only opaque paging values; every new request is rebuilt from the configured URL.
            link = collection.get("__next") or collection.get("@odata.nextLink") or collection.get("odata.nextLink")
            page = {"rows": [{k: row[k] for k in self.properties if k in row} for row in rows]}
            if link:
                params = httpx.URL(link).params
                if "$skiptoken" in params:
                    page["next_skiptoken"] = params["$skiptoken"]
                elif "$skip" in params:
                    page["next_offset"] = int(params["$skip"])
                else:
                    raise ValueError("Unsupported OData continuation link.")
            elif len(rows) == size and skiptoken is None:
                page["next_offset"] = offset + size
            return page

        return await self.access.invoke(
            self.name, self.object_name, run, route="cds_view" if self.cds else "published_api"
        )


class _PayloadTool(_HTTPTool):
    input_model: type[BaseModel] = Field(exclude=True)
    document_types: tuple[Ref, ...]
    customizing: SAPCustomizing | None = None

    transport_query_parameter: Ref | None = None

    @model_validator(mode="after")
    def _check_customizing_system(self):
        if self.customizing:
            if self.transport_query_parameter and self.transport_query_parameter.lower() == "sap-client":
                raise ValueError("The transport parameter must be distinct from sap-client.")
            system = httpx.URL(https_endpoint(self.customizing.nonproduction_system))
            endpoint = httpx.URL(self.endpoint)
            if system.path != "/" or system.query or system.netloc != endpoint.netloc:
                raise ValueError("Customizing endpoint must match the designated non-production HTTPS origin.")
        return self

    def _target(self, url: str) -> str:
        if not self.customizing:
            return url
        target = httpx.URL(url).copy_set_param("sap-client", self.customizing.client)
        if self.transport_query_parameter:
            target = target.copy_set_param(self.transport_query_parameter, self.customizing.transport_request)
        return str(target)


class SAPODataWrite(_PayloadTool):
    """One configured modifying operation, optionally with session-bound CSRF handling."""

    name: str = "sap_odata_write"
    description: str = "Submit the validated payload to the configured SAP OData operation."
    read_only: Literal[False] = False
    method: Literal["POST", "PUT", "PATCH", "DELETE"]
    csrf_endpoint: Ref | None = None
    etag: str | None = None
    key_fields: tuple[Property, ...] = ()

    @model_validator(mode="after")
    def _same_origin(self):
        if self.csrf_endpoint:
            origin = urlsplit(https_endpoint(self.csrf_endpoint))
            target = urlsplit(self.endpoint)
            if (origin.hostname, origin.port or 443) != (target.hostname, target.port or 443):
                raise ValueError("The SAP CSRF endpoint must have the same origin as the operation.")
        return self

    async def _invoke(self, payload: BaseModel):
        async def run(call):
            data = payload.model_dump(mode="json", by_alias=True, exclude_unset=True)
            call.account_input(data)
            target = httpx.URL(self._target(self.endpoint))
            if self.key_fields:
                keys = []
                for key in self.key_fields:
                    value = data.pop(key, None)
                    if not isinstance(value, str | int | float):
                        raise ValueError(f"Missing or invalid OData entity key: {key}")
                    keys.append(f"{key}={quote(_literal(value), safe='')}")
                target = target.copy_with(path=f"{target.path.rstrip('/')}({','.join(keys)})")
            headers = {"Accept": "application/json"}
            if self.etag:
                headers["If-Match"] = self.etag
            async with await self._client(call) as client:
                if self.csrf_endpoint:
                    _, fetched = await self._send(
                        call,
                        client,
                        "GET",
                        self._target(self.csrf_endpoint),
                        headers={"X-CSRF-Token": "Fetch"},
                        headers_only=True,
                    )
                    if not (token := fetched.get("X-CSRF-Token")):
                        raise RuntimeError("SAP did not issue a CSRF token.")
                    headers["X-CSRF-Token"] = token
                    self.access._charge(call.consumer, start=False)
                kwargs = {"json": data} if self.method != "DELETE" else {}
                body, _ = await self._send(call, client, self.method, str(target), headers=headers, **kwargs)
            return json.loads(body) if body else None

        return await self.access.invoke(
            self.name,
            self.object_name,
            run,
            route="published_api",
            document_types=self.document_types,
            customizing=self.customizing,
        )


class SAPSOAPOperation(_PayloadTool):
    """Bind a single WSDL operation, including BAPIs published as SOAP services."""

    name: str = "sap_soap_operation"
    description: str = "Invoke the configured SAP SOAP operation with validated inputs."
    read_only: bool
    version: Literal["1.1", "1.2"] = "1.1"
    action: Ref
    operation_tag: Ref
    # The application maps the input schema to the element prescribed by its WSDL.
    build_element: Callable[[BaseModel], ET.Element] = Field(exclude=True)

    async def _invoke(self, payload: BaseModel):
        async def run(call):
            namespace = _SOAP[self.version]
            element = self.build_element(payload)
            if element.tag != self.operation_tag:
                raise ValueError("SOAP message does not match the registered WSDL operation.")
            envelope = ET.Element(f"{{{namespace}}}Envelope")
            ET.SubElement(envelope, f"{{{namespace}}}Body").append(element)
            content = ET.tostring(envelope, encoding="utf-8")
            call.account_input(content.decode("utf-8"))
            action = '"' + self.action + '"'
            headers = (
                {"Content-Type": "text/xml; charset=utf-8", "SOAPAction": action}
                if self.version == "1.1"
                else {"Content-Type": f"application/soap+xml; charset=utf-8; action={action}"}
            )
            async with await self._client(call) as client:
                body, _ = await self._send(
                    call,
                    client,
                    "POST",
                    self._target(self.endpoint),
                    headers=headers,
                    content=content,
                )
            if not body:
                return None
            root = ET.fromstring(body, parser=ET.XMLParser(target=_SOAPTree()))
            soap_body = root.find(f"{{{namespace}}}Body")
            if soap_body is None or soap_body.find(f"{{{namespace}}}Fault") is not None:
                raise RuntimeError("SAP returned an invalid SOAP envelope or a SOAP fault.")
            return ET.tostring(soap_body, encoding="unicode")

        return await self.access.invoke(
            self.name,
            self.object_name,
            run,
            route="published_api",
            document_types=self.document_types,
            customizing=self.customizing,
        )


class _SOAPTree(ET.TreeBuilder):
    def doctype(self, *_args):
        raise ValueError("Unexpected DTD in SAP SOAP response.")


class _AgentGateway(_HTTPTool):
    # This provider must obtain a gateway-audience token on behalf of the current named user.
    token_provider: TokenProvider = Field(exclude=True)


class SAPAgentSend(_AgentGateway):
    name: str = "sap_agent_send"
    description: str = "Send a message to the configured SAP agent through the SAP Agent Gateway."
    read_only: Literal[False] = False
    document_types: tuple[Ref, ...]

    request_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    callback_url: Ref | None = None
    callback_token: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @field_validator("callback_url")
    @classmethod
    def _https_callback(cls, value: str | None) -> str | None:
        return https_endpoint(value) if value is not None else None

    async def _invoke(self, text: str, context_id: str | None = None, task_id: str | None = None) -> dict:
        async def run(call: SAPCall):
            message = {"messageId": str(uuid4()), "role": "ROLE_USER", "parts": [{"text": text}]}
            if context_id:
                message["contextId"] = context_id
            if task_id:
                message["taskId"] = task_id
            request = {"message": message}
            if self.request_metadata:
                request["metadata"] = self.request_metadata
            if self.callback_url:
                callback = {"url": self.callback_url}
                if self.callback_token:
                    callback["token"] = self.callback_token.get_secret_value()
                request["configuration"] = {"pushNotification": callback}
            call.account_input(request)
            async with await self._client(call) as client:
                body, _ = await self._send(call, client, "POST", self.endpoint, json=request)
            return json.loads(body)

        return await self.access.invoke(
            self.name, self.object_name, run, route="sap_agent", document_types=self.document_types
        )


class SAPAgentGetTask(_AgentGateway):
    name: str = "sap_agent_get_task"
    description: str = "Retrieve a task from the configured SAP agent through the SAP Agent Gateway."
    read_only: Literal[True] = True

    async def _invoke(self, task_id: str) -> dict:
        async def run(call: SAPCall):
            call.account_input(task_id)
            if not task_id.isprintable() or task_id in ("", ".", "..") or "/" in task_id or "\\" in task_id:
                raise ValueError("Invalid SAP Agent Gateway task ID.")
            url = httpx.URL(self.endpoint)
            url = url.copy_with(path=url.path.rstrip("/") + "/" + quote(task_id, safe=""))
            async with await self._client(call) as client:
                body, _ = await self._send(call, client, "GET", str(url))
            return json.loads(body)

        return await self.access.invoke(self.name, self.object_name, run, route="sap_agent")
