from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ...types.field import Field, resolve_default
from ...utils import _platform_api_call


class Column(BaseModel):
    name: str
    """The name of the column."""
    data_type: str
    """The data type of the column (as a postgres data type)."""
    default_value: str | None = None
    """The default value of the column."""
    is_nullable: bool
    """Whether the column can be null."""
    is_unique: bool
    """Whether the column is unique."""
    comment: str | None = None
    """The comment of the column."""


class Table(BaseModel):
    name: str
    """The name of the table."""
    columns: list[Column]
    """The columns of the table."""
    comment: str | None = None
    """The comment of the table."""
    constraints: list[Any] # TODO Create a proper enum for this
    """The constraints on the table."""


async def create_table(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base to create the table in."),
    name: str = Field(description="The name of the table to create."),
    columns: list[Column] = Field(description="List of column definitions."),
    comment: str | None = Field(
        default=None,
        description="The comment of the table.",
    ),
) -> None:
    """
    Create a new table in a knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID where the table will be created.
        name (str): The name of the table to create.
        columns (list[Column]): A list of column definitions.
        comment (str | None): An optional comment describing the table.
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    name = resolve_default("name", name)
    columns = resolve_default("columns", columns)
    comment = resolve_default("comment", comment)

    # Validate columns
    columns = [column if isinstance(column, Column) else Column.model_validate(column) for column in columns]

    path = f"orgs/{org_id}/kbs/{kb_id}/tables"
    payload = {
        "name": name,
        "columns": [column.model_dump() for column in columns],
        "comment": comment,
    }

    await _platform_api_call("POST", path, json=payload)


async def delete_table(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base to delete the table from."),
    name: str = Field(description="The name of the table to delete."),
    cascade: bool = Field(description="Whether to cascade the delete to the table's indexes."),
) -> None:
    """
    Delete a table in a knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID containing the table.
        name (str): The name of the table to delete.
        cascade (bool): Whether to also delete all indexes and dependent objects associated with the table.
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    name = resolve_default("name", name)
    cascade = resolve_default("cascade", cascade)

    path = f"orgs/{org_id}/kbs/{kb_id}/tables/{name}?"
    payload = {"cascade": cascade}

    await _platform_api_call("DELETE", path, json=payload)


async def get_table(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base to get the table from."),
    name: str = Field(description="The name of the table to get."),
) -> Table:
    """
    Retrieve the full definition of a table from a knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID containing the table.
        name (str): The name of the table to retrieve.

    Returns:
        Table: A Table model containing the table's name, columns, comment, and constraints.
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    name = resolve_default("name", name)

    path = f"orgs/{org_id}/kbs/{kb_id}/tables/{name}"
    params = {"format": "full"}
    
    res = await _platform_api_call("GET", path, params=params)
    return Table.model_validate(res.json().get("table"))


async def get_table_sql(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base."),
    name: str = Field(description="The name of the table to get the definition for."),
) -> str:
    """
    Retrieve the complete SQL definition of a table, including its structure and constraints.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID containing the table.
        name (str): The name of the table to retrieve the definition for.

    Returns:
        str: The full CREATE TABLE statement, including columns, constraints, and any associated indexes, as a formatted SQL string.
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    name = resolve_default("name", name)

    path = f"orgs/{org_id}/kbs/{kb_id}/tables/{name}"
    params = {"format": "definition"}
    
    res = await _platform_api_call("GET", path, params=params)
    return res.json().get("table")


async def get_tables(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base."),
) -> list[Table]:
    """
    List all tables in a knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.

    Returns:
        list[Table]: A list of Table models, each containing the table's name, columns, comment, and constraints.
    """    
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)

    path = f"orgs/{org_id}/kbs/{kb_id}/tables"
    params = {"format": "full"}
    
    res = await _platform_api_call("GET", path, params=params)
    return [Table.model_validate(table) for table in res.json().get("tables", [])]


# TODO Maybe we want to join the list into a single string (LLM oriented)
async def get_tables_sql(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base."),
) -> list[str]:
    """
    List all tables in a knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.

    Returns:
        list[str]: A list of full table SQL definitions.
    """    
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)

    path = f"orgs/{org_id}/kbs/{kb_id}/tables"
    params = {"format": "definition"}
    
    res = await _platform_api_call("GET", path, params=params)
    return res.json().get("tables", [])


async def import_records(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base to import records to."),
    table_name: str = Field(description="The name of the table to import records to."),
    records: list[dict[str, Any]] = Field(description="The records to import."),
) -> None:
    """
    Import records into a table in a knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID containing the table.
        table_name (str): The name of the table to import records to.
        records (list[dict[str, Any]]): The records to import. Each record should be a dictionary where keys match the table's column names.

    Example:
        # For an example table "Documents" with columns: id, filename, content
        await import_records(
            org_id="10",
            kb_id="48",
            table_name="Documents",
            records=[
                {"id": 1, "filename": "foo.txt", "content": "Hello world!"},
                {"id": 2, "filename": "bar.txt", "content": "Another document"}
            ]
        )
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    table_name = resolve_default("table_name", table_name)
    records = resolve_default("records", records)

    path = f"orgs/{org_id}/kbs/{kb_id}/tables/{table_name}/records"
    payload = {"records": records}

    await _platform_api_call("POST", path, json=payload)


async def import_csv(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base containing the table."),
    table_name: str = Field(description="The name of the table to upload the CSV to."),
    csv_path: str | Path = Field(description="The path to the CSV file."),
    mode: Literal["append", "overwrite"] =  Field(
        default="overwrite", 
        description="The mode to use for the import.",
    ),
) -> None:
    """
    Upload a CSV file to a table in a knowledge base.

    This function imports data from a CSV file into an existing table in the specified knowledge base. 
    The CSV file must match the table's schema (column names and types). 
    You can choose to either overwrite the table's contents or append to it.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The ID of the knowledge base containing the table.
        table_name (str): The name of the table to upload the CSV to.
        csv_path (str | Path): The path to the CSV file on disk.
        mode (Literal["append", "overwrite"], optional): Import mode. Use "overwrite" to replace all existing data in the table, or "append" to add to it. Default is "overwrite".
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    table_name = resolve_default("table_name", table_name)
    csv_path = resolve_default("csv_path", csv_path)
    mode = resolve_default("mode", mode)

    path = f"orgs/{org_id}/kbs/{kb_id}/tables/{table_name}/csv?mode={mode}"
    headers = {"Content-Type": "text/csv"}

    with open(csv_path, "rb") as f:
        csv_data = f.read()
    
    await _platform_api_call("POST", path, headers=headers, content=csv_data)


async def search_table(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base containing the table."),
    table_name: str = Field(description="The name of the table to search."),
    query: str = Field(description="The query to search for."),
    embedding_names: list[str] = Field(description="The names of the embeddings to use for the search."),
    # TODO Add more params
    limit: int = Field(
        default=10,
        description="The maximum number of results to return.",
    ),
    offset: int = Field(
        default=0,
        description="The offset to use for pagination.",
    ),
) -> list[dict[str, Any]]:
    """
    Perform a semantic search on a table within a knowledge base using embeddings.

    This function queries the specified table for records most relevant to the provided query string,
    leveraging one or more embedding models for semantic similarity. The search returns the top matching rows
    based on the embeddings and query.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The ID of the knowledge base containing the table.
        table_name (str): The name of the table to search.
        query (str): The natural language query or search phrase.
        embedding_names (list[str]): The names of the embedding models to use for the search.
        limit (int, optional): The maximum number of results to return. Defaults to 10.
        offset (int, optional): The offset for pagination. Defaults to 0.

    Returns:
        list[dict[str, Any]]: A list of ordered records, each as a dictionary. The structure of each record
        depends on the table schema and the specified select columns.
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    table_name = resolve_default("table_name", table_name)
    query = resolve_default("query", query)
    embedding_names = resolve_default("embedding_names", embedding_names)
    limit = resolve_default("limit", limit)
    offset = resolve_default("offset", offset)

    path = f"orgs/{org_id}/kbs/{kb_id}/tables/{table_name}/search"
    payload = {
        "query": query,
        "embedding_names": embedding_names,
        "limit": limit,
        "offset": offset,
    }

    res = await _platform_api_call("POST", path, json=payload)
    return res.json()


async def query(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The ID of the knowledge base."),
    sql: str = Field(description="The SQL query to execute."),
) -> list[dict[str, Any]]:
    """
    Execute a SQL query against a knowledge base table (PostgreSQL dialect).

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.
        sql (str): The SQL query to execute. This must be valid PostgreSQL SQL.

    Returns:
        list[dict[str, Any]]: The query results as a list of dictionaries, where each dictionary represents a row.

    Notes:
        - SQL syntax must follow PostgreSQL conventions.
        - Table and column names are case sensitive. If your identifiers use uppercase or mixed case, you must escape them with double quotes (e.g., "Documents", "FileName").
        - Unescaped identifiers are automatically lowercased by PostgreSQL.

    Example:
        # Count all the documents in the table
        await query(
            org_id="10",
            kb_id="48",
            sql='SELECT COUNT(*) FROM "Documents"'
        )
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    sql = resolve_default("sql", sql)

    path = f"orgs/{org_id}/kbs/{kb_id}/query"
    payload = {"sql": sql}

    res = await _platform_api_call("POST", path, json=payload)
    return res.json()
    