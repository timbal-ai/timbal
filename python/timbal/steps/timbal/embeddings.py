"""
INTERNAL USE ONLY - Embeddings utilities for Timbal Tables API.

This module provides functions for generating embeddings and performing
similarity searches on tables with embedding columns using vector distance operations.
"""

import json
import os
from typing import List, Dict, Any, Optional, Literal
from httpx import AsyncClient

from ...state import get_run_context, resolve_platform_auth
from ...types.field import Field


# Supported embedding models (updated to use current OpenAI models)
EmbeddingModel = Literal["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]



async def create_embedding(
    name: str = Field(description="The name of the embedding."),
    table_name: str = Field(description="The name of the table."),
    column_name: str = Field(description="The name of the column."),
    model: EmbeddingModel = Field(description="The model to use for the embedding."),
    with_gin_index: bool = Field(description="Whether to create a GIN index for the embedding."),
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The knowledge base ID."),
) -> list[float]:
    """Creates an embedding for a text."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/embeddings"
    headers = {**headers, "Content-Type": "application/json"}
    payload = {"name": name, "table_name": table_name, "column_name": column_name, "model": model, "with_gin_index": with_gin_index}
    async with AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload, timeout=None)
        res.raise_for_status()
        return res.json()

async def list_embeddings(
    table_name: Optional[str] = Field(description="The name of the table."),
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The knowledge base ID."),
) -> list[float]:
    """Lists the embeddings for a model."""
    host, headers = resolve_platform_auth(get_run_context())

    params = {}
    if table_name is not None:
        params["table_name"] = table_name
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/embeddings"
    headers = {**headers, "Content-Type": "application/json"}
    async with AsyncClient() as client:
        res = await client.get(url, headers=headers, params=params, timeout=None)
        res.raise_for_status()
        return res.json()


async def delete_embedding(
    embedding_name: str = Field(description="The name of the embedding."),
    org_id: str = Field(description="The organization ID."),
    kb_id: str = Field(description="The knowledge base ID."),
) -> None:
    """Deletes an embedding for a table column."""
    host, headers = resolve_platform_auth(get_run_context())
    url = f"https://{host}/orgs/{org_id}/kbs/{kb_id}/embeddings/{embedding_name}"
    async with AsyncClient() as client:
        res = await client.delete(url, headers=headers, timeout=None)
        res.raise_for_status()
        return res.json()
