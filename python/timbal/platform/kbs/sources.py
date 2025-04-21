"""
INTERNAL USE ONLY - This module is primarily intended for internal Timbal use.

This module contains functions for interacting with Timbal's Knowledge Base API.
It requires internal authentication tokens and endpoints that may not be available
in the open source distribution. Some functionality may be limited or unavailable
when used outside of the Timbal organization environment.

This is a preview of what will happen under the hood when you want to upload an entire
directory to a knowledge base programmatically or via the CLI.
"""


import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

ALLOWED_EXTENSIONS = [".pdf", ".md", ".json"]


def upload_source(
    org_id: str,
    kb_id: str,
    source_path: Path,
    parent_path: str | None = None,
    metadata: dict[str, Any] = {},
) -> str:
    content_type, _ = mimetypes.guess_type(source_path)

    with open(source_path, "rb") as f:
        content = f.read()
    content_length = len(content)

    headers = {"Authorization": f"Bearer {os.getenv('TIMBAL_API_TOKEN')}"}
    body = {
        "parent_path": parent_path,
        "is_directory": False,
        "name": source_path.name,
        "metadata": metadata,
        "content_type": content_type,
        "content_length": content_length,
    }

    res = requests.post(
        f"https://{os.getenv('TIMBAL_API_HOST')}/orgs/{org_id}/kbs/{kb_id}/sources",
        headers=headers,
        json=body,
    )
    res.raise_for_status()

    res_body = res.json()
    source_id = res_body.get("org_kb_source_id")
    uploader = res_body.get("uploader")
    if uploader is None:
        return
    
    upload_uri = uploader.get("upload_uri")
    upload_headers = {
        "Content-Type": content_type,
        "Content-Length": str(content_length),
    }

    upload_res = requests.put(
        upload_uri,
        headers=upload_headers,
        data=content,
    )
    upload_res.raise_for_status()

    return source_id


def create_dir(
    org_id: str,
    kb_id: str,
    source_name: str,
    source_parent_path: str | None = None,
    source_metadata: dict[str, Any] = {},
) -> str:
    headers = {"Authorization": f"Bearer {os.getenv('TIMBAL_API_TOKEN')}"}
    body = {
        "is_directory": True,
        "name": source_name,
        "metadata": source_metadata,
        "parent_path": source_parent_path,
        "content_type": None,
        "content_length": None,
    }

    res = requests.post(
        f"https://{os.getenv('TIMBAL_API_HOST')}/orgs/{org_id}/kbs/{kb_id}/sources",
        headers=headers,
        json=body,
    )
    res.raise_for_status()

    res_body = res.json()
    source_id = res_body.get("org_kb_source_id")
    return source_id


def upload_dir(
    org_id: str,
    kb_id: str,
    dir_path: Path,
    dest_path: str | None = None,
) -> None:
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    
    sources_manifest_path = dir_path / "_timbal_sources_manifest.csv"
    sources_manifest_columns = [
        "source_id",                # Null if the file has not been uploaded yet.
        "is_directory",
        "name",
        "source_path",              # Local path.
        "source_parent_path",       # Built from platform parents sources ids.
        "latest_extraction_path",   # Path to the latest extraction of the source.
    ]

    if sources_manifest_path.exists():
        sources = pd.read_csv(sources_manifest_path)[sources_manifest_columns]
    else:
        sources = pd.DataFrame(columns=sources_manifest_columns)
    
    def iterdir(path, parent_path: str | None = None):
        nonlocal sources

        for file in tqdm(path.iterdir()):
            file_posix = file.as_posix()

            if (
                file.is_dir()
                and not file.name.startswith("_") 
                and not file.name.startswith(".")
            ):
                current_source = sources.loc[sources["source_path"] == file_posix]

                # If there's no existing entry in the manifest, create a new one and create the remote dir.
                if current_source.empty:
                    try:
                        source_id = create_dir(
                            org_id, 
                            kb_id, 
                            source_name=file.name, 
                            source_parent_path=parent_path, 
                            source_metadata={},
                        )
                        sources.loc[len(sources)] = {
                            "source_id": source_id,
                            "is_directory": True, 
                            "name": file.name, 
                            "source_path": file_posix,
                            "source_parent_path": parent_path,
                            "latest_extraction_path": None,
                        }
                        if pd.isna(parent_path):
                            source_parent_path = source_id
                        else:
                            source_parent_path = f"{parent_path}.{source_id}"
                        iterdir(file, parent_path=source_parent_path)
                    except Exception as e:
                        print(f"Error creating directory {file.name}: {e}")
                        continue
                else:
                    current_source = current_source.iloc[0].to_dict()
                    if pd.isna(parent_path):
                        source_parent_path = current_source["source_id"]
                    else:
                        source_parent_path = f"{parent_path}.{current_source['source_id']}"
                    iterdir(file, parent_path=source_parent_path)

            elif (
                file.is_file() 
                and file.suffix.lower() in ALLOWED_EXTENSIONS 
                and not file.name.startswith("_") 
                and not file.name.startswith(".")
            ):
                current_source = sources.loc[sources["source_path"] == file_posix]

                if current_source.empty:
                    try:
                        source_id = upload_source(
                            org_id, 
                            kb_id, 
                            source_path=file,
                            parent_path=parent_path, 
                            metadata={},
                        )
                        sources.loc[len(sources)] = {
                            "source_id": source_id,
                            "is_directory": False, 
                            "name": file.name, 
                            "source_path": file_posix,
                            "source_parent_path": parent_path,
                            "latest_extraction_path": None,
                        }
                    except Exception as e:
                        print(f"Error uploading file {file.name}: {e}")
                        continue

    iterdir(dir_path, parent_path=dest_path)

    sources.to_csv(sources_manifest_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python -m timbal.platform.kbs.sources <org_id> <kb_id> <dir_path>")
        sys.exit(1)

    org_id = sys.argv[1]
    kb_id = sys.argv[2]
    dir_path = Path(sys.argv[3]).expanduser().resolve()
    upload_dir(org_id, kb_id, dir_path)
