from typing import Any, Literal

from pydantic import BaseModel

from ...types.field import Field, resolve_default
from ...utils import _platform_api_call

EmbeddingStatus = Literal["pending", "queued", "processing", "success", "error"]


class Embedding(BaseModel):
    name: str
    """The name of the embedding."""
    table_name: str
    """The name of the table."""
    column_name: str
    """The name of the column."""
    model: str
    """The model to use for the embedding.
    To check for a complete list of available models for your organization, please
    check the list_embedding_models tool.
    """
    status: EmbeddingStatus
    """The status of the embedding."""
    details: dict[str, Any]
    """Additional information about the embedding."""
    created_at: int 
    """The epoch (ms) when the embedding was created."""
    updated_at: int 
    """The epoch (ms) when the embedding was last updated."""


async def list_embedding_models(
    org_id: str = Field(description="The organization ID."),
) -> list[str]:
    """
    List all available embedding models enabled for a specific organization.

    Args:
        org_id (str): The organization ID.

    Returns:
        list[str]: A list of available embedding models.
    """
    org_id = resolve_default("org_id", org_id)

    path = f"orgs/{org_id}/embedding-models"

    res = await _platform_api_call("GET", path)
    return res.json().get("embedding_models", [])
    

async def create_embedding(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The knowledge base ID."),
    name: str = Field(description="The name of the embedding."),
    table_name: str = Field(description="The name of the table."),
    column_name: str = Field(description="The name of the column."),
    model: str = Field(description="The model to use for the embedding."),
    with_gin_index: bool = Field(description="Whether to create a GIN index for the embedding."),
) -> None:
    """
    Create an embedding for a table column.

    Sends a request to the Timbal platform API to create a new embedding for the specified column
    in a table, using the given model. Optionally, a GIN index can be created for the source column 
    for improved hybrid search performance.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.
        name (str): The name of the embedding.
        table_name (str): The name of the table.
        column_name (str): The name of the column.
        model (str): The embedding model to use.
        with_gin_index (bool): Whether to create a GIN index for the embedding.
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    name = resolve_default("name", name)
    table_name = resolve_default("table_name", table_name)
    column_name = resolve_default("column_name", column_name)
    model = resolve_default("model", model)
    with_gin_index = resolve_default("with_gin_index", with_gin_index)

    path = f"orgs/{org_id}/kbs/{kb_id}/embeddings"
    payload = {
        "name": name, 
        "table_name": table_name, 
        "column_name": column_name, 
        "model": model, 
        "with_gin_index": with_gin_index,
    }

    await _platform_api_call("POST", path, json=payload)


async def list_embeddings(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The knowledge base ID."),
    table_name: str | None = Field(
        default=None,
        description="The name of the table.",
    ),
) -> list[Embedding]:
    """
    List all embeddings for a knowledge base or a specific table.

    Retrieves all embedding definitions for the specified knowledge base, or for a given table if provided.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.
        table_name (str | None): The name of the table (optional).

    Returns:
        list[Embedding]: A list of embedding definitions.
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    table_name = resolve_default("table_name", table_name)

    path = f"orgs/{org_id}/kbs/{kb_id}/embeddings"

    params = {}
    if table_name is not None:
        params["table_name"] = table_name

    res = await _platform_api_call("GET", path, params=params)
    return [Embedding.model_validate(embedding) for embedding in res.json().get("embeddings", [])]


async def delete_embedding(
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The knowledge base ID."),
    name: str = Field(description="The name of the embedding."),
) -> None:
    """
    Delete an embedding from a knowledge base.

    Sends a request to the Timbal platform API to delete the specified embedding from the given knowledge base.

    Args:
        org_id (str): The organization ID.
        kb_id (str): The knowledge base ID.
        name (str): The name of the embedding to delete.
    """
    org_id = resolve_default("org_id", org_id)
    kb_id = resolve_default("kb_id", kb_id)
    name = resolve_default("name", name)

    path = f"orgs/{org_id}/kbs/{kb_id}/embeddings/{name}"

    await _platform_api_call("DELETE", path)
