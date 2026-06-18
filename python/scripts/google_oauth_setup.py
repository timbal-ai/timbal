#!/usr/bin/env python3
"""Generate Google OAuth refresh tokens for marketing tools.

Opens your default browser for consent, then prints env vars to copy into .env.

Usage::

    uv run python python/scripts/google_oauth_setup.py
    uv run python python/scripts/google_oauth_setup.py --scope analytics
    uv run python python/scripts/google_oauth_setup.py --scope all

Requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env (or environment).
Add http://127.0.0.1:8765/oauth2callback as an authorized redirect URI in Google Cloud Console.
"""

from __future__ import annotations

import argparse
import os
import secrets
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
from dotenv import load_dotenv

REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT = 8765
REDIRECT_PATH = "/oauth2callback"
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}{REDIRECT_PATH}"

SCOPES = {
    "analytics": "https://www.googleapis.com/auth/analytics.readonly",
    "search_console": "https://www.googleapis.com/auth/webmasters.readonly",
}

ENV_KEYS = {
    "analytics": "GOOGLE_ANALYTICS_REFRESH_TOKEN",
    "search_console": "GOOGLE_SEARCH_CONSOLE_REFRESH_TOKEN",
}


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    auth_code: str | None = None
    auth_error: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != REDIRECT_PATH:
            self.send_error(404)
            return

        params = urllib.parse.parse_qs(parsed.query)
        if "error" in params:
            _OAuthCallbackHandler.auth_error = params["error"][0]
            body = f"OAuth error: {_OAuthCallbackHandler.auth_error}. You can close this tab."
            self.send_response(400)
        elif "code" in params:
            _OAuthCallbackHandler.auth_code = params["code"][0]
            body = "Authorization successful. You can close this tab and return to the terminal."
            self.send_response(200)
        else:
            body = "Missing OAuth code."
            self.send_response(400)

        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def _exchange_code(*, client_id: str, client_secret: str, code: str) -> dict:
    response = httpx.post(
        "https://oauth2.googleapis.com/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _authorize(*, client_id: str, scopes: list[str]) -> str:
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)

    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.auth_error = None

    server = HTTPServer((REDIRECT_HOST, REDIRECT_PORT), _OAuthCallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print(f"Opening browser for Google OAuth consent ({len(scopes)} scope(s))...")
    print(f"If it does not open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)
    thread.join(timeout=300)
    server.server_close()

    if _OAuthCallbackHandler.auth_error:
        raise RuntimeError(f"OAuth authorization failed: {_OAuthCallbackHandler.auth_error}")
    if not _OAuthCallbackHandler.auth_code:
        raise RuntimeError("Timed out waiting for OAuth callback on http://127.0.0.1:8765")
    return _OAuthCallbackHandler.auth_code


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

    parser = argparse.ArgumentParser(description="Generate Google OAuth refresh tokens.")
    parser.add_argument(
        "--scope",
        choices=[*SCOPES.keys(), "all"],
        default="all",
        help="Which API scope(s) to authorize (default: all combined)",
    )
    args = parser.parse_args()

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env first.")

    if args.scope == "all":
        selected = list(SCOPES.keys())
    else:
        selected = [args.scope]

    scope_urls = [SCOPES[name] for name in selected]
    code = _authorize(client_id=client_id, scopes=scope_urls)
    tokens = _exchange_code(client_id=client_id, client_secret=client_secret, code=code)

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise SystemExit(
            "No refresh_token returned. Revoke app access at "
            "https://myaccount.google.com/permissions and run again with prompt=consent."
        )

    print("\nAdd to your .env:\n")
    if args.scope == "all":
        print(f"GOOGLE_REFRESH_TOKEN={refresh_token}")
        print("\nOr use the same token for each product env var:")
        for name in SCOPES:
            print(f"{ENV_KEYS[name]}={refresh_token}")
    else:
        print(f"{ENV_KEYS[args.scope]}={refresh_token}")

    print("\nAccess token (short-lived, for quick tests):")
    print(tokens.get("access_token", ""))


if __name__ == "__main__":
    main()
