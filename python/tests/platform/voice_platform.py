"""In-process fake of the platform's outbound-call API (monolith#485 contract). Places no real calls."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx

RESERVED = {"call_id", "direction", "from", "to", "user_number", "agent_number"}
_CALLS = re.compile(
    r"^/orgs/(?P<org>[^/]+)/projects/(?P<project>[^/]+)/workforce/(?P<wf>[^/]+)/voice/calls(?:/(?P<id>[^/]+)(?:/(?P<op>[a-z-]+))?)?$"
)
_RESULT = re.compile(r"^/call-control/(?P<id>[^/]+)/result$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FakeVoicePlatform:
    def __init__(self, *, org: str = "org_1", project: str = "77") -> None:
        self.org = org
        self.project = project
        self.enabled = True
        self.legacy = False
        self.calls: dict[str, dict[str, Any]] = {}
        self.keys: dict[str, tuple[tuple, str]] = {}
        self.dials = 0
        self.posts: list[dict[str, Any]] = []
        self.timeout_after_accept = 0
        self.tokens: dict[str, str] = {}

    # -- platform-side transitions -------------------------------------------------

    def finish(
        self,
        call_id: str,
        *,
        status: str = "completed",
        outcome: Any = None,
        outcome_status: str | None = None,
        end_reason: str | None = "participant_left",
    ) -> dict[str, Any]:
        view = self.calls[call_id]
        view.update(status=status, end_reason=end_reason, ended_at=_now(), finalized_at=_now())
        if outcome is not None:
            view.update(outcome=outcome, outcome_status="reported", outcome_reported_at=_now())
        else:
            view["outcome_status"] = outcome_status or "unavailable"
        return self._public(call_id)

    # -- HTTP --------------------------------------------------------------------

    def _json(self, status: int, body: Any, headers: dict[str, str] | None = None) -> httpx.Response:
        return httpx.Response(status, json=body, headers=headers or {})

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        result = _RESULT.match(path)
        if result:
            return self._report(request, result.group("id"))
        match = _CALLS.match(path)
        if not match:
            return self._json(404, {"message": "not found"})
        if match.group("org") != self.org or match.group("project") != self.project:
            return self._json(404, {"code": "NOT_FOUND", "message": "call not found"})
        call_id, op = match.group("id"), match.group("op")
        if request.method == "POST" and call_id is None:
            return self._create(request, match.group("wf"))
        if call_id not in self.calls:
            return self._json(404, {"code": "NOT_FOUND", "message": "call not found"})
        if request.method == "GET" and op is None:
            return self._json(200, self.view(call_id))
        if request.method == "POST" and op == "cancel":
            view = self.calls[call_id]
            if view.get("finalized_at") is None:
                self.finish(call_id, status="cancelled", end_reason="cancelled")
            return self._json(202, self.view(call_id))
        return self._json(405, {"message": "method not allowed"})

    def _create(self, request: httpx.Request, workforce: str) -> httpx.Response:
        if not self.enabled:
            return self._json(409, {"code": "CONFLICT", "message": "Outbound SIP is not enabled on this platform"})
        body = json.loads(request.content or b"{}")
        self.posts.append(
            {"key": request.headers.get("Idempotency-Key"), "body": body, "rev": request.url.params.get("rev")}
        )
        context = body.get("context") or {}
        reserved = sorted(RESERVED & set(context))
        if reserved:
            return self._json(400, {"code": "BAD_REQUEST", "message": f"reserved context keys: {reserved}"})
        key = request.headers.get("Idempotency-Key")
        rev = request.url.params.get("rev")
        fingerprint = (
            workforce,
            rev,
            body.get("to"),
            body.get("from_number"),
            body.get("trunk_id"),
            body.get("greeting"),
        )
        if key and key in self.keys:
            seen, call_id = self.keys[key]
            if seen != fingerprint:
                return self._json(
                    422, {"code": "IDEMPOTENCY_KEY_REUSED", "message": "Idempotency key reused for a different request"}
                )
            return self._json(202, self._public(call_id), {"idempotent-replayed": "true"})
        self.dials += 1
        call_id = f"call_{self.dials}"
        self.calls[call_id] = {
            "id": call_id,
            "direction": "outbound",
            "status": "queued",
            "to_number": body.get("to"),
            "from_number": body.get("from_number"),
            "created_at": _now(),
            "outcome_status": "pending",
            "outcome": None,
            "finalized_at": None,
            "origin": body.get("origin"),
            "_context": context,
            "_outcome_schema": body.get("outcome_schema"),
        }
        self.tokens[call_id] = f"secret-token-{call_id}"
        if key:
            self.keys[key] = (fingerprint, call_id)
        if self.timeout_after_accept > 0:
            self.timeout_after_accept -= 1
            raise httpx.ReadTimeout("timed out after the platform accepted the call", request=request)
        return self._json(202, self._public(call_id))

    def _public(self, call_id: str) -> dict[str, Any]:
        view = {k: v for k, v in self.calls[call_id].items() if not k.startswith("_")}
        if self.legacy:
            view.pop("outcome_status", None)
        return view

    def view(self, call_id: str) -> dict[str, Any]:
        return self._public(call_id)

    def seed(self, outcome_schema: dict[str, Any] | None = None) -> tuple[str, str]:
        """A connected call with an outcome schema, as the agent's side sees it. Returns (id, token)."""
        self.dials += 1
        call_id = f"call_{self.dials}"
        self.calls[call_id] = {
            "id": call_id,
            "status": "active",
            "outcome_status": "pending",
            "outcome": None,
            "finalized_at": None,
            "_outcome_schema": outcome_schema,
        }
        self.tokens[call_id] = f"secret-token-{call_id}"
        return call_id, self.tokens[call_id]

    def _report(self, request: httpx.Request, call_id: str) -> httpx.Response:
        if request.headers.get("X-Timbal-Sip-Call-Token") != self.tokens.get(call_id):
            return self._json(403, {"code": "FORBIDDEN", "message": "invalid call token"})
        view = self.calls[call_id]
        if view.get("finalized_at") is not None:
            return self._json(409, {"code": "CONFLICT", "message": "result already final"})
        outcome = json.loads(request.content or b"{}").get("outcome")
        schema = view.get("_outcome_schema") or {}
        problems = []
        for name in schema.get("required", []):
            if not isinstance(outcome, dict) or name not in outcome:
                problems.append(f"/{name}: is required")
        for name, prop in (schema.get("properties") or {}).items():
            allowed = prop.get("enum")
            if allowed and isinstance(outcome, dict) and name in outcome and outcome[name] not in allowed:
                problems.append(f"/{name}: must be one of {allowed}")
        if problems:
            return self._json(422, {"code": "OUTCOME_INVALID", "message": "; ".join(problems)})
        view.update(outcome=outcome, outcome_status="reported", outcome_reported_at=_now())
        return self._json(200, {"ok": True})

    # -- wiring ------------------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def request_client(self):
        transport = self.transport()

        @asynccontextmanager
        async def _client(timeout):
            async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
                yield client

        return _client

    def outcome_client(self):
        transport = self.transport()

        def _client():
            return httpx.AsyncClient(transport=transport, base_url="https://api.timbal.ai")

        return _client
