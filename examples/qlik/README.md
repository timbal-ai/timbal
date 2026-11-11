# Qlik Analyst with Timbal

This example connects a Timbal agent to **Qlik Cloud (Qlik Sense)** so it can answer business
questions with figures from your Qlik apps. It finds the app, reads the app's data model, and
computes `dimensions x measures` through the Qlik Engine API.

You supply three values: the **tenant URL**, the **client ID** and the **client secret** of a
Qlik OAuth client.

## What's in here

| File | Purpose |
| --- | --- |
| `agent.py` | The agent. Run it with a question, or with no arguments for interactive mode. |
| `skills/qlik/SKILL.md` | The agent's instructions for Qlik: when to use which tool, set analysis, rules. |
| `skills/qlik/tools/qlik_tools.py` | The four Qlik tools, loaded with the skill. |
| `setup_credentials.py` | Personal mode only: signs a person in once and saves their tokens to `.qlik_token.json`. |
| `.env.example` | The variables to fill in. |

The tools themselves live in `timbal.tools.qlik`:

| Tool | What it does |
| --- | --- |
| `QlikWhoAmI` (`qlik_whoami`) | Who the tools act as, the roles, and the mode (`personal` / `service`). |
| `QlikListApps` (`qlik_list_apps`) | Apps this identity can open, searchable by name and space, paginated. |
| `QlikGetAppFields` (`qlik_get_app_fields`) | The app's tables, row counts and fields. |
| `QlikQuery` (`qlik_query`) | A hypercube (dimensions x measures) through the Engine API, with set analysis. |

## Required packages

```bash
pip install timbal websockets python-dotenv
```

`websockets` is only needed by `qlik_query` (the Engine API is a websocket); it also comes
with `timbal[server]`.

## Choose an authentication method

The tools support two modes. Choose the one that fits who should see the data.

### Method 1: Personal (user OAuth)

**Use this if:**
- Each person should see only what *their* Qlik roles, spaces and Section Access allow
- The agent answers on behalf of a signed-in user

The tools send that person's access token, so Qlik enforces their permissions. Locally,
`setup_credentials.py` signs you in once and saves a refresh token to a token file; the tools
turn it into access tokens as needed and write the newest refresh token back to that file,
because Qlik can rotate refresh tokens on each use (the old one then stops working). On the
Timbal platform, a personal Qlik integration supplies the user's token instead.

### Method 2: Service (machine-to-machine)

**Use this if:**
- The agent runs unattended (scheduled reports, back-office jobs)
- Everyone using it may see the same data

The tools exchange the client ID and secret for a token through the `client_credentials`
grant. That token acts as the OAuth client's own identity, so the agent sees what that
identity has been given access to, the same for every user.

> **Requirement:** the OAuth client must allow machine-to-machine access and use the
> **Trusted** consent method. Otherwise Qlik refuses the grant with `OAUTH-5`, and the tools
> say so in the error.

## Step 1: Create the OAuth client in Qlik Cloud

You need the **Tenant Admin** role.

1. Open your tenant (`https://<tenant>.<region>.qlikcloud.com`) and go to
   **Administration > Integration > OAuth**.
2. Click **Create new**.
3. **Client type:** `Web`.
4. **Name:** for example `Timbal agent`.
5. **Scopes:** tick `user_default`. For personal mode, also tick `offline_access` so a sign-in
   yields a refresh token.
6. **Redirect URLs** (personal mode): add `http://localhost:8765/callback` for
   `setup_credentials.py`, plus your app's own callback URL if you build a sign-in into it.
7. **Service mode only:** enable **Allow Machine-to-Machine (M2M)** and set the consent
   method to **Trusted**.
8. Click **Create**, then copy the **client ID** and the **client secret**. The secret is shown
   only once.

Labels can differ slightly between tenant versions; the settings are the same.

## Step 2: Fill in `.env`

```bash
cp .env.example .env
```

```bash
QLIK_TENANT_URL=https://your-tenant.eu.qlikcloud.com
QLIK_CLIENT_ID=your_oauth_client_id
QLIK_CLIENT_SECRET=your_oauth_client_secret
QLIK_AUTH_MODE=service   # or personal
ANTHROPIC_API_KEY=your_anthropic_api_key
```

For **service mode**, that's all.

For **personal mode**, sign in once:

```bash
python setup_credentials.py
```

It opens the Qlik sign-in page, waits for the redirect on `http://localhost:8765/callback`,
shows who you signed in as, and saves the tokens to `.qlik_token.json` (readable by you only).
Make sure `.env` has:

```bash
QLIK_AUTH_MODE=personal
QLIK_TOKEN_FILE=.qlik_token.json
```

Keep `.env` and `.qlik_token.json` out of version control (this folder's `.gitignore` does):
the refresh token is your Qlik access. Every process that uses the same token file shares the
current refresh token, so run the agent from this folder or use an absolute path.

## Step 3: Give the identity access to the apps

- **Personal:** nothing to do; the person sees their own apps.
- **Service:** run the agent once and ask *"who are you in Qlik?"* (it calls `qlik_whoami`),
  then add that identity to the spaces whose apps the agent should read. Apps protected by
  Section Access only return the rows that identity is granted.

## Step 4: Run it

```bash
python agent.py "What were total sales by region in 2026?"
```

The agent loads the `qlik` skill, then typically:

1. `qlik_list_apps(query="sales")` to find the app
2. `qlik_get_app_fields(app_id=...)` to read real field names
3. `qlik_query(app_id=..., dimensions=["Region"], measures=["Sum({<Year={2026}>} Sales)"])`

## Using the tools in your own agent

```python
from timbal import Agent
from timbal.tools import QlikGetAppFields, QlikListApps, QlikQuery, QlikWhoAmI

tools = [QlikWhoAmI(), QlikListApps(), QlikGetAppFields(), QlikQuery()]
agent = Agent(name="analyst", model="anthropic/claude-sonnet-4-6", max_tokens=4096, tools=tools)
```

Or copy `skills/qlik/` into your project's skills folder so the tools and their instructions
load together.

Every tool takes the same optional fields, which win over the environment:

| Field | Env var | Mode |
| --- | --- | --- |
| `tenant_url` | `QLIK_TENANT_URL` | both |
| `mode` | `QLIK_AUTH_MODE` | pins `personal` or `service` |
| `token` | `QLIK_ACCESS_TOKEN` | personal |
| `refresh_token` | `QLIK_REFRESH_TOKEN` | personal (with client ID and secret) |
| `token_file` | `QLIK_TOKEN_FILE` | personal: where the newest refresh token is kept |
| `client_id`, `client_secret` | `QLIK_CLIENT_ID`, `QLIK_CLIENT_SECRET` | service (and personal refresh) |
| `api_key` | `QLIK_API_KEY` | service (a Qlik API key instead of OAuth) |
| `integration` | | a Timbal platform integration supplying any of the above |

When `mode` is not set, a person's token is tried before the service credentials. Set
`mode="personal"` wherever a user without a Qlik sign-in must be refused rather than answered
through the service identity.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `OAUTH-5` on the `client_credentials` grant | The client authenticated but is not allowed M2M or is not Trusted. Fix Step 1.7. |
| `OAUTH-5` on the `refresh_token` grant | The refresh token expired, was revoked, or was already used. Run `setup_credentials.py` again, and keep `QLIK_TOKEN_FILE` set. |
| `OAUTH-22` | Wrong client secret for this client ID. |
| `OAUTH-14` | Unknown client ID (or the wrong tenant). |
| `Qlik rejected the personal credentials (401)` | The access token expired or was revoked. Run `setup_credentials.py` again. |
| `the app does not exist or this identity may not open it` | Wrong `app_id`, or the identity lacks access to its space (Step 3). |
| `Bad expression` from `qlik_query` | A field name or the expression syntax is wrong. Check `qlik_get_app_fields`. |
| `invalid redirect_uri` during sign-in | Add `http://localhost:8765/callback` to the client's redirect URLs (Step 1.6). |
