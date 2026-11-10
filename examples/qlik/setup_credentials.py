# ruff: noqa: T201
"""Personal mode: sign in to Qlik Cloud once and get a refresh token for the tools.

Runs the OAuth authorization-code flow (with PKCE) against your tenant: it opens the
Qlik sign-in page, catches the redirect on http://localhost:8765/callback, exchanges the
code and prints the QLIK_REFRESH_TOKEN line to paste into .env. The tools exchange that
refresh token for access tokens as they need them.

Prerequisites (see README.md):
- QLIK_TENANT_URL, QLIK_CLIENT_ID and QLIK_CLIENT_SECRET in .env
- http://localhost:8765/callback listed in the OAuth client's redirect URLs

Usage:
    python setup_credentials.py
"""

import asyncio
import base64
import hashlib
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from dotenv import load_dotenv
from timbal.tools.qlik import PERSONAL_SCOPES, QlikWhoAmI, normalize_tenant_url

PORT = int(os.getenv("QLIK_CALLBACK_PORT", "8765"))
REDIRECT_URI = f"http://localhost:{PORT}/callback"


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


def _wait_for_code(expected_state: str) -> str:
    """Serve one request on the callback URL and return the authorization code."""
    result: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            query = parse_qs(urlparse(self.path).query)
            if query.get("state", [""])[0] != expected_state:
                result["error"] = "state mismatch"
            elif "error" in query:
                result["error"] = query["error"][0]
            else:
                result["code"] = query.get("code", [""])[0]
            ok = "code" in result
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            message = "Signed in to Qlik. You can close this tab." if ok else f"Qlik sign-in failed: {result}"
            self.wfile.write(message.encode())

        def log_message(self, *_args: object) -> None:
            return

    server = HTTPServer(("localhost", PORT), Handler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    thread.join(timeout=300)
    server.server_close()
    if "code" not in result:
        raise SystemExit(f"No authorization code received: {result.get('error', 'timed out')}")
    return result["code"]


async def main() -> None:
    load_dotenv()
    tenant = normalize_tenant_url(os.environ["QLIK_TENANT_URL"])
    client_id = os.environ["QLIK_CLIENT_ID"]
    client_secret = os.environ["QLIK_CLIENT_SECRET"]

    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    authorize_url = f"{tenant}/oauth/authorize?" + urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": PERSONAL_SCOPES,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    print(f"Opening the Qlik sign-in page:\n{authorize_url}\n")
    webbrowser.open(authorize_url)
    code = _wait_for_code(state)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{tenant}/oauth/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": client_id,
                "client_secret": client_secret,
                "code_verifier": verifier,
            },
        )
    if response.status_code != 200:
        raise SystemExit(f"Token exchange failed: {response.status_code} {response.text[:300]}")
    tokens = response.json()
    if not tokens.get("refresh_token"):
        raise SystemExit("Qlik returned no refresh token: check that the client allows the offline_access scope.")

    me = await QlikWhoAmI(tenant_url=tenant, token=tokens["access_token"])().collect()
    who = me.output if me.status.code == "success" else {}
    print(f"Signed in as {who.get('name')} <{who.get('email')}>.\n")
    print("Add these lines to .env:\n")
    print("QLIK_AUTH_MODE=personal")
    print(f"QLIK_REFRESH_TOKEN={tokens['refresh_token']}")


if __name__ == "__main__":
    asyncio.run(main())
