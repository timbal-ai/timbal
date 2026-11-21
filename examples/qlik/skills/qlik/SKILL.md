---
name: qlik
description: Answer business questions with figures from Qlik Cloud (Qlik Sense) apps - find the app, read its data model, then compute dimensions x measures through the Engine API.
---

## When to use

The user asks for a figure, ranking, trend or breakdown that lives in a Qlik Sense app
(sales, margins, stock, production, KPIs...), or asks what Qlik apps or fields exist.

## Tools

| Tool | Use it to |
| --- | --- |
| `qlik_whoami` | Check who the tools act as (`mode`: `personal` = a person, `service` = the app's own identity) and the tenant. |
| `qlik_list_apps` | Find the app. Search by words of its name (`query`); page with `next`. |
| `qlik_get_app_fields` | Read the app's tables, row counts and field names. Always before `qlik_query`. |
| `qlik_query` | Compute a table of `dimensions` x `measures` through the Engine API. |

## How to answer

1. Find the app with `qlik_list_apps`. If several match, pick the one whose name fits the
   question best and say which one you used; ask only when the choice changes the answer.
2. Read its data model with `qlik_get_app_fields`. Use field names exactly as listed:
   they are case-sensitive and may contain spaces or accents.
3. Call `qlik_query`:
   - `dimensions` are field names to group by (`Region`, `Customer`), or a calculated
     dimension starting with `=`.
   - `measures` are Qlik expressions: `Sum(Sales)`, `Count(distinct Customer)`,
     `Sum(Sales) / Sum(Units)`.
   - Filter with set analysis inside the measure, not by fetching everything:
     `Sum({<Year={2026}>} Sales)`, `Sum({<Year={2026}, Month={'Jan','Feb'}>} Sales)`,
     `Sum({<Region-={'Internal'}>} Sales)`. Field names with spaces go in brackets:
     `Sum({<[Order Year]={2026}>} [Net Sales])`.
   - Keep `max_rows` small for rankings (10-50) and say when `truncated` is true.
4. Answer with the figures and name the app and the measure you used, so the user can
   check them in Qlik.

## Rules

- Never invent a field name, app or figure. If a field you need is not in
  `qlik_get_app_fields`, say so and offer the closest fields that exist.
- Measures come back as numbers and dimensions as Qlik's display text. Format numbers for
  the reader; do not re-aggregate rows that a measure already aggregated.
- An Engine error such as "Bad expression" means the expression is wrong: fix the field
  name or the syntax and try once more, then explain what failed.
- Access follows the identity in `qlik_whoami`. In `personal` mode the user sees only what
  their Qlik roles and Section Access allow. An app missing from `qlik_list_apps` or a
  "may not open it" error is a permission boundary, not an outage: say so and do not look
  for another route to the data.
- If a call says the credentials were rejected (401) in `personal` mode, tell the user to
  sign in to Qlik again.
