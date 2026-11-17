# ruff: noqa: T201
"""OAuth: sign in to Notion once and save the tokens for the tools.

Runs Notion's OAuth authorization-code flow for a public integration: it opens Notion's consent
page (where you pick the pages the integration may see), catches the redirect on
http://localhost:8766/callback, exchanges the code and saves the tokens to NOTION_TOKEN_FILE
(default .notion_token.json, owner-only). Notion issues a new refresh token on every refresh;
the tools write the newest one back to that file.

Prerequisites (see README.md):
- NOTION_CLIENT_ID and NOTION_CLIENT_SECRET in .env (the public integration's OAuth client)
- http://localhost:8766/callback listed in the integration's redirect URIs

Usage:
    python setup_credentials.py
"""

import asyncio
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

from dotenv import load_dotenv
from timbal.tools._oauth_store import write_token_file
from timbal.tools.notion import AUTHORIZE_URL, NotionWhoAmI, exchange_code

PORT = int(os.getenv("NOTION_CALLBACK_PORT", "8766"))
REDIRECT_URI = f"http://localhost:{PORT}/callback"
TOKEN_FILE = os.getenv("NOTION_TOKEN_FILE") or ".notion_token.json"


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
            message = "Connected to Notion. You can close this tab." if ok else f"Notion sign-in failed: {result}"
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
    client_id = os.environ["NOTION_CLIENT_ID"]
    client_secret = os.environ["NOTION_CLIENT_SECRET"]

    state = secrets.token_urlsafe(16)
    authorize_url = f"{AUTHORIZE_URL}?" + urlencode(
        {
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "owner": "user",
            "state": state,
        }
    )
    print(f"Opening Notion's consent page (choose the pages to share):\n{authorize_url}\n")
    webbrowser.open(authorize_url)
    code = _wait_for_code(state)

    tokens = await exchange_code(client_id, client_secret, code, REDIRECT_URI)
    me = await NotionWhoAmI(token=tokens["access_token"])().collect()
    who = me.output if me.status.code == "success" else {}
    print(f"Connected to the workspace {tokens.get('workspace_name') or who.get('workspace_name')!r}.")

    if not tokens.get("refresh_token"):
        # Nothing to rotate: the access token alone is the credential, valid until revoked.
        print("Notion returned no refresh token. Add this line to .env (keep it secret):\n")
        print("NOTION_AUTH_METHOD=oauth")
        print(f"NOTION_ACCESS_TOKEN={tokens['access_token']}")
        return

    write_token_file(
        TOKEN_FILE,
        {
            "provider": "notion",
            "client_id": client_id,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expires_at": None,
        },
    )
    print(f"Tokens saved to {TOKEN_FILE} (keep it out of version control).\n")
    print("Make sure .env has:\n")
    print("NOTION_AUTH_METHOD=oauth")
    print(f"NOTION_TOKEN_FILE={TOKEN_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
