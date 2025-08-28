from typing import Literal

from pydantic import BaseModel, Field

from ...utils import _platform_api_call, resolve_default


class Index(BaseModel):
    table: str
    """The name of the table the index is on."""
    name: str
    """The name of the index."""
    columns: list[str]
    """The columns included in the index."""
    type: str
    """The type of the index (e.g., btree, hnsw, etc.)."""
    is_unique: bool
    """Whether the index is unique."""
    definition: str
    """The SQL definition of the index."""


async def create_index(
    org_id: str = Field(
        ...,
        description="The organization ID."
    ),
    kb_id: str = Field(
        ...,
        description="The ID of the knowledge base."
    ),
    table_name: str = Field(
        ...,
        description="The name of the table to create the index on."
    ),
    name: str = Field(
        ...,
        description="The name of the index to create."
    ),
    column_names: list[str] = Field(
        ...,
        description="The columns to create the index on."
    ),
    type: Literal["btree", "hash", "gin", "gist", "brin"] = Field(
        "btree",
        description="The type of the index to create."
    ),
    is_unique: bool = Field(
        False,
        description="Whether the index should be unique."
    ),
) -> None:
    """
    Create an index on a table in a knowledge base.

    This function sends a request to the platform API to create a new index on the specified table 
    within a knowledge base, with support for different index types and uniqueness constraints.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.
        table_name (str): The name of the table to create the index on.
        name (str): The name of the index to create.
        column_names (list[str]): The columns to include in the index.
        type (Literal["btree", "hash", "gin", "gist", "brin"]): The type of index to create.
        is_unique (bool): Whether the index should enforce uniqueness.
    """
    type = resolve_default("type", type)
    is_unique = resolve_default("is_unique", is_unique)

    path = f"orgs/{org_id}/kbs/{kb_id}/indexes"
    payload = {
        "table_name": table_name, 
        "name": name, 
        "column_names": column_names, 
        "type": type, 
        "is_unique": is_unique,
    }

    await _platform_api_call("POST", path, json=payload)

    
async def list_indexes(
    kb_id: str = Field(
        ...,
        description="The ID of the knowledge base."
    ),
    org_id: str = Field(
        ...,
        description="The organization ID."
    ),
    table_name: str | None = Field(
        None,
        description="The name of the table to list indexes for."
    ),
) -> list[Index]:
    """
    List all indexes in a knowledge base.

    Sends a request to the platform API to retrieve all index definitions for the specified knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.
        table_name (str | None): The name of the table to list indexes for.

    Returns:
        list[Index]: A list of index definitions.
    """
    table_name = resolve_default("table_name", table_name)

    path = f"orgs/{org_id}/kbs/{kb_id}/indexes"
    params = {}
    if table_name:
        params["table_name"] = table_name

    res = await _platform_api_call("GET", path, params=params)
    return [Index.model_validate(index) for index in res.json().get("indexes", [])]


async def delete_index(
    org_id: str = Field(
        ...,
        description="The organization ID."
    ),
    kb_id: str = Field(
        ...,
        description="The ID of the knowledge base."
    ),
    name: str = Field(
        ...,
        description="The name of the index to delete."
    ),
) -> None:
    """
    Delete an index from a knowledge base.

    Sends a request to the platform API to delete the specified index from the given knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.
        name (str): The name of the index to delete.
    """

    path = f"orgs/{org_id}/kbs/{kb_id}/indexes/{name}"

    await _platform_api_call("DELETE", path)
