---
name: notion
description: Read, search and write the user's Notion workspace - find pages and databases, read pages as Markdown, query database rows, create and update pages and rows, append content and comment.
---

## When to use

The user mentions Notion, or asks about something that lives there: a doc, a wiki page, meeting
notes, a roadmap, a task or CRM database. Also when they ask you to write something down,
log it, or file it in Notion.

## Tools

| Tool | Use it to |
| --- | --- |
| `notion_whoami` | Check which workspace and integration you act as, and how (`oauth` / `api_key`). |
| `notion_search` | Find pages and databases by title. `object_type="data_source"` for databases only. |
| `notion_get_page` | Read a page's properties and its body as Markdown. |
| `notion_get_data_source` | Read a database's columns, their types and select/status options. |
| `notion_query_data_source` | List database rows, with a Notion `filter` and `sorts`. |
| `notion_create_page` | Create a sub-page (`parent_type="page"`) or a database row (`parent_type="database"`). |
| `notion_update_page` | Rename a page, set row properties, or move it to the trash (`in_trash=true`). |
| `notion_append_content` | Add Markdown to the end of a page. |
| `notion_add_comment` / `notion_list_comments` | Comment on a page, or read its open comments. |

Every ID argument also accepts a notion.so URL; paste what the user gave you.

## How to work

1. **Find before you act.** Search by the words the user used. If several results fit, pick the
   most recently edited one that matches and say which; ask only when the choice matters.
2. **Read the schema before touching a database.** Call `notion_get_data_source` and use the
   property names and option values exactly as returned (they are case-sensitive).
3. **Filters use Notion's format**, keyed by the property's type:
   - status: `{"property": "Status", "status": {"equals": "Done"}}`
   - select: `{"property": "Priority", "select": {"equals": "High"}}`
   - date: `{"property": "Due", "date": {"on_or_before": "2026-12-31"}}`
   - checkbox: `{"property": "Done", "checkbox": {"equals": true}}`
   - text: `{"property": "Name", "title": {"contains": "launch"}}`
   - combine: `{"and": [...]}` or `{"or": [...]}`
4. **Property values for create/update** use the same type keys:
   `{"Status": {"status": {"name": "Done"}}}`, `{"Due": {"date": {"start": "2026-12-01"}}}`,
   `{"Tags": {"multi_select": [{"name": "ops"}]}}`, `{"Owner": {"people": [{"id": "<user id>"}]}}`,
   `{"Estimate": {"number": 3}}`. Pass the title through `title`, not `properties`.
5. **Write content in Markdown**: headings, `-` lists, `1.` lists, `- [ ]` to-dos, `>` quotes,
   ``` code fences, `---`, **bold**, *italic*, `code` and [links](url).
6. Results page: when `next_cursor` is not null and the answer needs more, call again with it.

## Rules

- Never invent a page, property, option or ID. If something is missing from the schema, say so.
- Before a destructive change (trashing a page, overwriting properties the user did not
  mention), say what you are about to change unless the user asked for exactly that.
- "Not shared with this integration" means the integration cannot see that page. Tell the user
  to open it in Notion, then `•••` > Connections and add the integration. Do not look for
  another route to it.
- Quote what you read faithfully; when you summarise a page, link it (`url`).
