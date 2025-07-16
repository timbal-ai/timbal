import os
from typing import Literal, List
from pydantic import BaseModel, Field
from httpx import AsyncClient

from ...state import get_run_context, resolve_platform_auth
from ...types.field import Field


class IndexModel(BaseModel):
    table: str = Field(description="The name of the table the index is on.")
    name: str = Field(description="The name of the index.")
    columns: List[str] = Field(description="The columns included in the index.")
    type: str = Field(description="The type of the index (e.g., btree, hnsw, etc.).")
    is_unique: bool = Field(description="Whether the index is unique.")
    definition: str = Field(description="The SQL definition of the index.")

class IndexesResponseModel(BaseModel):
    indexes: List[IndexModel] = Field(description="A list of index definitions.")



async def create_index(
    table_name: str = Field(description="The name of the table to create the index on."),
    index_name: str = Field(description="The name of the index to create."),
    index_columns: list[str] = Field(description="The columns to create the index on."),
    index_type: Literal["btree", "hash", "gin", "gist", "brin"] = Field(description="The type of the index to create."),
    index_unique: bool = Field(description="Whether the index should be unique."),
    kb_id: str = Field(description="The ID of the knowledge base."),
    org_id: str = Field(description="The organization ID."),
) -> None:
    """Creates an index on a table in a knowledge base."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/indexes"
    headers = {**headers, "Content-Type": "application/json"}
    payload = {"table_name": table_name, "index_name": index_name, "index_columns": index_columns, "index_type": index_type, "index_unique": index_unique}
    async with AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload, timeout=None)
        res.raise_for_status()

    
async def list_indexes(
    kb_id: str = Field(description="The ID of the knowledge base."),
    org_id: str = Field(description="The organization ID."),
    table_name: str = Field(default=None, description="Optional table name to filter indexes by."),
) -> list[IndexesResponseModel] | None:
    """Lists the indexes in a knowledge base."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/indexes"
    headers = {**headers, "Content-Type": "application/json"}
    
    payload = {}
    if table_name:
        payload["table_name"] = table_name
    
    async with AsyncClient() as client:
        res = await client.get(url, headers=headers, params=payload, timeout=None)
        res.raise_for_status()
        return res.json()


async def delete_index(
    index_name: str = Field(description="The name of the index to delete."),
    kb_id: str = Field(description="The ID of the knowledge base."),
    org_id: str = Field(description="The organization ID."),
) -> None:
    """Deletes an index on a table in a knowledge base."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/indexes/{index_name}"
    headers = {**headers, "Content-Type": "application/json"}
    async with AsyncClient() as client:
        res = await client.delete(url, headers=headers, timeout=None)
        res.raise_for_status()