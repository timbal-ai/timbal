"""
INTERNAL USE ONLY - This module is primarily intended for internal Timbal use.

This module contains functions for interacting with Timbal's Knowledge Base API.
It requires internal authentication tokens and endpoints that may not be available
in the open source distribution. Some functionality may be limited or unavailable
when used outside of the Timbal organization environment.

This is a piece of what happens when sources in a KB are automatically parsed
on the platform. The resulting extractions are uploaded to platform and linked to the original source.
"""


import mimetypes
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def upload_extraction(
    org_id: str,
    kb_id: str,
    source_id: str, 
    extraction_path: Path,
) -> None:
    
    content_type, _ = mimetypes.guess_type(extraction_path)
    if content_type not in ["text/markdown", "text/plain", "application/json"]:
        raise ValueError(f"Unsupported extraction content type: {content_type}")

    with open(extraction_path, "rb") as f:
        content = f.read()
    content_length = len(content)

    headers = {"Authorization": f"Bearer {os.getenv('TIMBAL_API_TOKEN')}"}
    body = {
        "content_type": content_type,
        "content_length": content_length,
    }

    res = requests.post(
        f"https://{os.getenv('TIMBAL_API_HOST')}/orgs/{org_id}/kbs/{kb_id}/sources/{source_id}/extractions",
        headers=headers,
        json=body,
    )
    res.raise_for_status()

    res_body = res.json()
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


def upload_dir(
    org_id: str,
    kb_id: str,
    dir_path: Path,
    dest_path: str | None = None,
) -> None:
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")
    
    sources_manifest_path = dir_path / "_timbal_sources_manifest.csv"
    sources_manifest_dtypes = {
        "source_id": str,
        "is_directory": bool,
        "name": str,
        "source_path": str,
        "source_parent_path": str,
        "latest_extraction_path": str,
    }

    sources = pd.read_csv(sources_manifest_path, dtype=sources_manifest_dtypes)
    sources["relative_source_path"] = sources["source_path"] \
        .apply(lambda x: Path(x).relative_to(dir_path).with_suffix("").as_posix())

    extractions_path = dir_path / "_timbal_sources_extractions"
    extractions_ts = [int(t.stem) for t in extractions_path.iterdir() if t.is_dir()]
    latest_extractions_ts = max(extractions_ts)
    latest_extractions_path = extractions_path / str(latest_extractions_ts)

    def iterdir(path: Path):
        nonlocal sources

        for file in tqdm(path.iterdir()):
            file_posix = file.as_posix()

            if (
                file.is_dir()
                and not file.name.startswith("_") 
                and not file.name.startswith(".")
            ):
                iterdir(file)

            elif (
                file.is_file() 
                and not file.name.startswith("_") 
                and not file.name.startswith(".")
            ):
                relative_path = file.relative_to(latest_extractions_path).with_suffix("").as_posix()
                source = sources.loc[sources["relative_source_path"] == relative_path]
                if source.empty:
                    print(f"Source not found for {file_posix}")
                    continue
                source = source.iloc[0].to_dict()
                if pd.isna(source["latest_extraction_path"]) or source["latest_extraction_path"] != file_posix:
                    source_id = source["source_id"]
                    try:
                        upload_extraction(org_id, kb_id, source_id, file)
                        sources.loc[sources["source_id"] == source_id, "latest_extraction_path"] = file_posix
                    except Exception as e:
                        print(f"Error uploading extraction for {file_posix}: {e}")
                        continue

    iterdir(latest_extractions_path)

    sources.to_csv(sources_manifest_path, index=False, columns=list(sources_manifest_dtypes.keys()))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python -m timbal.platform.kbs.extractions <org_id> <kb_id> <dir_path>")
        sys.exit(1)

    org_id = sys.argv[1]
    kb_id = sys.argv[2]
    dir_path = Path(sys.argv[3]).expanduser().resolve()
    upload_dir(org_id, kb_id, dir_path)
