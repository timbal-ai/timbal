"""Notion tools for the `notion` skill.

Credentials come from the environment (see ../../../.env.example): NOTION_API_KEY for an
internal integration, or NOTION_CLIENT_ID + NOTION_CLIENT_SECRET + NOTION_TOKEN_FILE for OAuth.
To use a Timbal platform integration instead, pass `integration="<org integration id>"`.
"""

from timbal.tools import (
    NotionAddComment,
    NotionAppendContent,
    NotionCreatePage,
    NotionGetDataSource,
    NotionGetPage,
    NotionListComments,
    NotionQueryDataSource,
    NotionSearch,
    NotionUpdatePage,
    NotionWhoAmI,
)

notion_whoami = NotionWhoAmI()
notion_search = NotionSearch()
notion_get_page = NotionGetPage()
notion_get_data_source = NotionGetDataSource()
notion_query_data_source = NotionQueryDataSource()
notion_create_page = NotionCreatePage()
notion_update_page = NotionUpdatePage()
notion_append_content = NotionAppendContent()
notion_add_comment = NotionAddComment()
notion_list_comments = NotionListComments()
