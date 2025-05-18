import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# class ChunkInput(BaseModel):
#     """"""
#     id: str
#     source_id: str
#     content: str
#     metadata: dict[str, Any]


def add_chunks(
    org_id: str,
    kb_id: str,
    chunks: list[Any],
) -> None:
    headers = {"Authorization": f"Bearer {os.getenv('TIMBAL_API_TOKEN')}"}
    body = {"chunks": chunks}

    res = requests.post(
        f"https://{os.getenv('TIMBAL_API_HOST')}/orgs/{org_id}/kbs/{kb_id}/sources/chunks",
        headers=headers,
        json=body,
    )
    res.raise_for_status()


if __name__ == "__main__":
    chunks_path = Path("/Users/dberges/Desktop/timbal-ai/kbs/22/_timbal_sources_chunks/1743692857559.csv")
    df = pd.read_csv(chunks_path)[["source_id", "content"]]
    df = df[df["source_id"].notna()].reset_index(drop=True)
    df["source_id"] = df["source_id"].astype(int).astype(str)

    chunks = [
        {
            "source_id": row["source_id"],
            "content": row["content"],
            "metadata": {},
        }
        for _, row in df.iterrows()
    ]

    add_chunks(org_id="9", kb_id="22", chunks=chunks)
