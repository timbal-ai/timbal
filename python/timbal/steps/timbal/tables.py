"""
INTERNAL USE ONLY - This module is primarily intended for internal Timbal use.

This module contains functions for interacting with Timbal's Knowledge Base Tables API.
It requires internal authentication tokens and endpoints that may not be available
in the open source distribution. Some functionality may be limited or unavailable
when used outside of the Timbal organization environment.

This is a preview of what will happen under the hood when you want to upload an entire
directory to a knowledge base programmatically or via the CLI.
"""

import json
from typing import Any, Optional, Literal

from httpx import AsyncClient

from ...types.field import Field
from ...state import get_run_context, resolve_platform_auth

from pydantic import BaseModel

class Column(BaseModel):
    name: str = Field(min_length=1, max_length=128, description="Name cannot be empty or longer than 128 characters.")
    data_type: str
    default_value: Optional[str] = None
    is_nullable: bool
    is_unique: bool
    comment: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Comment cannot be empty or longer than 256 characters."
    )



async def create_table(
    table_name: str = Field(description="The name of the table to create."),
    columns: list[Column] = Field(
        description="List of column definitions in Postgres format. Each column should mandatory have 'name' (str), 'data_type' (str), 'default_value' (str), 'is_nullable' (bool), 'is_unique' (bool) and optionally 'comment' (str)."
    ),
    kb_id: str = Field(description="The ID of the knowledge base to create the table in."),
    org_id: str = Field(description="The organization ID.")
) -> dict | None:
    """Creates a new table in a knowledge base."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/tables"
    headers = {**headers, "Content-Type": "application/json"}

    payload = {
        "table_name": table_name,
        "columns": columns,
    }

    async with AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload, timeout=None)
        res.raise_for_status()


async def delete_table(
    table_name: str = Field(description="The name of the table to delete."),
    cascade: bool = Field(description="Whether to cascade the delete to the table's indexes."),
    kb_id: str = Field(description="The ID of the knowledge base to delete the table from."),
    org_id: str = Field(description="The organization ID."),
) -> None:
    """Deletes a table from the knowledge base."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/tables/{table_name}?"
    headers = {**headers, "Content-Type": "application/json"}
    payload = {"cascade": cascade}
    async with AsyncClient() as client:
        res = await client.request(method="DELETE", url=url, headers=headers, content=json.dumps(payload), timeout=None)
        res.raise_for_status()


async def get_table(
    table_name: str = Field(description="The name of the table to get."),
    kb_id: str = Field(description="The ID of the knowledge base to get the table from."),
    org_id: str = Field(description="The organization ID."),
) -> dict | None:
    """
    Gets a table from a knowledge base.
    It returns table_name, comment, columns, constraints and indexes.
    """
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/tables/{table_name}"
    async with AsyncClient() as client:
        res = await client.get(url, headers=headers, timeout=None)
        res.raise_for_status()
        return res.json().get("table")

    

async def import_csv(
    table_name: str = Field(description="The name of the table to upload the CSV to."),
    csv_path: bytes = Field(description="The CSV file as bytes."),
    kb_id: str = Field(description="The ID of the knowledge base containing the table."),
    org_id: str = Field(description="The organization ID."),
    mode: Literal["append", "overwrite"] =  Field(default="overwrite", description="The mode to use for the import."),
) -> dict | None:
    """Uploads a CSV file to a table in a knowledge base."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/tables/{table_name}/csv?mode={mode}"
    headers = {**headers, "Content-Type": "text/csv"}

    try:
        with open(csv_path, "rb") as f:
            csv_data = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: CSV file not found at {csv_path}") from e
    
    async with AsyncClient() as client:
        res = await client.post(url, headers=headers, content=csv_data, timeout=None)
        res.raise_for_status()
        return res.json() if res.text else None


async def import_records(
    table_name: str = Field(description="The name of the table to import records to."),
    records: list[dict[str, Any]] = Field(description="The records to import."),
    kb_id: str = Field(description="The ID of the knowledge base to import records to."),
    org_id: str = Field(description="The organization ID."),
) -> None:
    """Imports records to a table in a knowledge base."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/tables/{table_name}/records"
    payload = {"records": records}
    async with AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload, timeout=None)
        res.raise_for_status()
        return res
    

async def query(
    sql_query: str = Field(description="The SQL query to execute."),
    kb_id: str = Field(description="The ID of the knowledge base."),
    org_id: str = Field(description="The organization ID."),
) -> list[dict[str, Any]] | None:
    """Queries a table in a knowledge base using SQL."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/query"
    headers["Content-Type"] = "application/json"

    payload = {"sql": sql_query}

    async with AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload, timeout=None)
        res.raise_for_status()
        return res.json() if res.text else None
    

async def get_table_definition(
    table_name: str = Field(description="The name of the table to get the definition for."),
    kb_id: str = Field(description="The ID of the knowledge base."),
    org_id: str = Field(description="The organization ID."),
) -> str | None:
    """
    Gets the complete SQL definition of a table including its structure and constraints.    
    This function returns the full CREATE TABLE statement along with any associated
    indexes as a formatted SQL string. 
    """
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/tables/{table_name}?format=definition"
    async with AsyncClient() as client:
        res = await client.get(url, headers=headers, timeout=None)
        res.raise_for_status()
        return res.json().get("table")


async def list_tables(
    kb_id: str = Field(description="The ID of the knowledge base."),
    org_id: str = Field(description="The organization ID."),
    format: Literal["full", "preview", "definition"] = Field(default="preview", description="The format of the tables to get."),
) -> list[dict[str, Any]] | None:
    """
    Gets all tables in a knowledge base.
    The format parameter can be "full", "preview" or "definition".
    - "full" returns the full table definition with all columns, constraints, indexes, etc.
    - "preview" returns a simplified view of the table with only the table name and column names.
    - "definition" returns the full SQL definition of the table including its structure and indexes.
    """
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/tables?format={format}"
    async with AsyncClient() as client:
        res = await client.get(url, headers=headers, timeout=None)
        res.raise_for_status()
        return res.json().get("tables")
