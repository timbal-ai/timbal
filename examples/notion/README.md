# Notion Assistant with Timbal

This example connects a Timbal agent to **Notion** so it can find, read and write your pages and
databases: search by title, read a page as Markdown, query database rows, create pages and rows,
update properties, append content and comment.

It supports both ways Notion lets an app connect: **OAuth** (a public integration the user signs
in to) and an **API key** (an internal integration's secret).

## What's in here

| File | Purpose |
| --- | --- |
| `agent.py` | The agent. Run it with a request, or with no arguments for interactive mode. |
| `skills/notion/SKILL.md` | The agent's instructions for Notion: which tool when, filters, property values, rules. |
| `skills/notion/tools/notion_tools.py` | The ten Notion tools, loaded with the skill. |
| `setup_credentials.py` | OAuth only: signs you in once and saves the tokens to `.notion_token.json`. |
| `.env.example` | The variables to fill in. |

The tools themselves live in `timbal.tools.notion`:

| Tool | What it does |
| --- | --- |
| `NotionWhoAmI` (`notion_whoami`) | The integration, its workspace and owner, and the auth method in use. |
| `NotionSearch` (`notion_search`) | Pages and databases shared with the integration, by title. |
| `NotionGetPage` (`notion_get_page`) | A page's properties and its body as Markdown (nested blocks included). |
| `NotionGetDataSource` (`notion_get_data_source`) | A database's columns, types and select/status options. |
| `NotionQueryDataSource` (`notion_query_data_source`) | Database rows, with Notion filters and sorts, paginated. |
| `NotionCreatePage` (`notion_create_page`) | A sub-page or a database row, with Markdown content. |
| `NotionUpdatePage` (`notion_update_page`) | Rename, set properties, move to or restore from the trash. |
| `NotionAppendContent` (`notion_append_content`) | Markdown appended to a page. |
| `NotionAddComment` / `NotionListComments` | Comment on a page, read its open comments. |

Every ID argument also accepts a notion.so URL. A database ID or a data source ID both work
wherever a database is expected: since Notion API version 2025-09-03, databases hold one or more
*data sources*, and the tools resolve one from the other.

## Required packages

```bash
pip install timbal python-dotenv
```

## Choose an authentication method

### Method 1: OAuth (public integration)

**Use this if:**
- Other people will connect their own Notion workspaces to your app
- Each person should choose which pages the agent can see when they sign in

The person signs in on Notion's consent page and picks the pages to share. Notion returns an
access token and a refresh token. Notion issues a **new refresh token on every refresh**, so the
tools keep the newest one in a token file and read it from there on the next run. Without the
file, a restart would reuse a token that was already spent. On the Timbal platform, a Notion
OAuth integration supplies the token instead.

### Method 2: API key (internal integration)

**Use this if:**
- The agent works in your own workspace
- You want the simplest setup: one secret, no sign-in

An internal integration's secret is a long-lived API key. The integration only sees the pages
you share with it explicitly.

## Step 1: Create the integration in Notion

You need to be a **Workspace Owner** to create an integration.

1. Go to [notion.so/profile/integrations](https://www.notion.so/profile/integrations) and click
   **New integration**.
2. Pick the **workspace** and give it a **name** (e.g. `Timbal assistant`).
3. **Type:**
   - **Internal** for Method 2 (API key).
   - **Public** for Method 1 (OAuth). Notion asks for a company name, website, privacy policy and
     terms URLs, and the **Redirect URIs**: add `http://localhost:8766/callback` for
     `setup_credentials.py`, plus your app's own callback URL if you build a sign-in into it.
4. **Capabilities:** tick *Read content*, *Update content*, *Insert content*, *Read comments* and
   *Insert comments*. *User information without email* is enough for `notion_whoami`.
5. Save, then copy the credentials:
   - Internal: **Internal Integration Secret** (Configuration tab).
   - Public: **OAuth client ID** and **OAuth client secret**.

## Step 2: Fill in `.env`

```bash
cp .env.example .env
```

### Method 2 (API key)

```bash
NOTION_API_KEY=ntn_...
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Then share each page or database the agent should see: open it in Notion, then **`•••` >
Connections** and add your integration. Sharing a page also shares its sub-pages.

### Method 1 (OAuth)

```bash
NOTION_CLIENT_ID=your_oauth_client_id
NOTION_CLIENT_SECRET=your_oauth_client_secret
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Then sign in once:

```bash
python setup_credentials.py
```

It opens Notion's consent page, where you choose the pages to share. It waits for the redirect
on `http://localhost:8766/callback` and saves the tokens to `.notion_token.json`, readable by
you only. Make sure `.env` has:

```bash
NOTION_AUTH_METHOD=oauth
NOTION_TOKEN_FILE=.notion_token.json
```

Keep `.env` and `.notion_token.json` out of version control (this folder's `.gitignore` does).
Every process using the same token file shares the current refresh token, and each refresh is
done once even when several processes refresh at the same time. Run the agent from this folder
or give an absolute path.

## Step 3: Run it

```bash
python agent.py "What's in my Q4 roadmap page?"
python agent.py "Add a task 'Renew SSL certificate' due next Friday to the Tasks database"
```

The agent loads the `notion` skill, then typically searches, reads the database schema, and
queries or writes using the property names it just read.

## Using the tools in your own agent

```python
from timbal import Agent
from timbal.tools import NotionGetPage, NotionQueryDataSource, NotionSearch

agent = Agent(
    name="assistant",
    model="anthropic/claude-sonnet-4-6",
    max_tokens=4096,
    tools=[NotionSearch(), NotionGetPage(), NotionQueryDataSource()],
)
```

Or copy `skills/notion/` into your project's skills folder so the tools and their instructions
load together.

Every tool takes the same optional fields, which win over the environment:

| Field | Env var | Method |
| --- | --- | --- |
| `api_key` | `NOTION_API_KEY` | API key |
| `token` | `NOTION_ACCESS_TOKEN` | OAuth |
| `refresh_token` | `NOTION_REFRESH_TOKEN` | OAuth (with client ID and secret) |
| `token_file` | `NOTION_TOKEN_FILE` | OAuth: where the newest refresh token is kept |
| `client_id`, `client_secret` | `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET` | OAuth refresh |
| `auth_method` | `NOTION_AUTH_METHOD` | pins `oauth` or `api_key` |
| `notion_version` | `NOTION_VERSION` | the `Notion-Version` header (default `2026-03-11`) |
| `integration` | | a Timbal platform integration supplying any of the above |

When `auth_method` is not set, OAuth credentials are tried before the API key. With OAuth, a token
Notion rejects is refreshed once and the call retried. Rate limits (`429`, `529`) are retried
after Notion's `Retry-After`.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `object_not_found` / "not shared with this integration" | Share the page: `•••` > Connections > your integration (OAuth: sign in again and select it). |
| `401 ... check NOTION_API_KEY` | The secret is wrong, or the integration was deleted. Copy it again. |
| `invalid_grant` on the refresh | The refresh token was revoked or already used. Run `setup_credentials.py` again and keep `NOTION_TOKEN_FILE` set. |
| `invalid_client` | Wrong OAuth client ID or secret. |
| `redirect_uri` error during sign-in | Add `http://localhost:8766/callback` to the integration's redirect URIs (Step 1.3). |
| `validation_error` on a filter or property | A property name or option does not match the schema: check `notion_get_data_source`. |
| "Database ... has 2 data sources" | Pass the data source ID the message lists. |
