"""Probe Bizneo HR API auth and placeholder read paths.

Run when Bizneo credentials are available:

  BIZNEO_BASE_URL=https://{tenant}.bizneohr.com \\
  BIZNEO_API_KEY=... \\
  BIZNEO_API_SECRET=... \\
    uv run python scripts/probe_bizneo_hr.py

Optional:
  BIZNEO_API_KEY_HEADER / BIZNEO_API_SECRET_HEADER — override auth header names
  BIZNEO_PROBE_PATHS=jobs,departments,/api/v1/jobs — comma-separated paths to try
"""
# ruff: noqa: T201

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from timbal.tools.bizneo_hr import (
    BizneoHRListCandidates,
    BizneoHRListDepartments,
    BizneoHRListEmployees,
    BizneoHRListJobs,
    BizneoHRListLocations,
    BizneoHRRequest,
    _auth_headers,
    _join_path,
    _resolve_credentials,
)


def _status(code: int) -> str:
    if 200 <= code < 300:
        return "OK"
    if code == 401:
        return "AUTH"
    if code == 403:
        return "FORBIDDEN"
    if code == 404:
        return "NOT_FOUND"
    return "FAIL"


async def _probe_raw_paths(base_url: str, api_key: str, api_secret: str) -> None:
    import httpx

    raw_paths = os.getenv("BIZNEO_PROBE_PATHS", "jobs,departments,locations,candidates,employees")
    paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
    headers = _auth_headers(api_key, api_secret)

    print("\n--- Raw GET probes ---")
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        for path in paths:
            url = _join_path(base_url, path)
            try:
                response = await client.get(url, headers=headers)
                label = _status(response.status_code)
                body = response.text[:200].replace("\n", " ")
                print(f"{label:10} {response.status_code:3} GET {url}  {body}")
            except httpx.HTTPError as exc:
                print(f"FAIL       ERR GET {url}  {exc}")


async def _probe_tools() -> None:
    print("\n--- Tool collect probes ---")
    probes: list[tuple[str, object]] = [
        ("BizneoHRListJobs", BizneoHRListJobs()),
        ("BizneoHRListDepartments", BizneoHRListDepartments()),
        ("BizneoHRListLocations", BizneoHRListLocations()),
        ("BizneoHRListCandidates", BizneoHRListCandidates()),
        ("BizneoHRListEmployees", BizneoHRListEmployees()),
    ]
    for name, tool in probes:
        try:
            out = await tool(page=1, per_page=1).collect()
            code = out.status.code
            err = out.error.get("message") if out.error else ""
            print(f"{code:10} {name}  {err or 'success'}")
        except Exception as exc:
            print(f"FAIL       {name}  {exc}")

    try:
        out = await BizneoHRRequest().collect(method="GET", path="jobs", query_params={"page": 1, "per_page": 1})
        err = out.error.get("message") if out.error else ""
        print(f"{out.status.code:10} BizneoHRRequest  {err or 'success'}")
    except Exception as exc:
        print(f"FAIL       BizneoHRRequest  {exc}")


async def main() -> int:
    missing = [
        name
        for name, val in (
            ("BIZNEO_BASE_URL", os.getenv("BIZNEO_BASE_URL")),
            ("BIZNEO_API_KEY", os.getenv("BIZNEO_API_KEY")),
            ("BIZNEO_API_SECRET", os.getenv("BIZNEO_API_SECRET")),
        )
        if not (val or "").strip()
    ]
    if missing:
        print("Set env vars:", ", ".join(missing))
        return 1

    tool = BizneoHRListJobs()
    base_url, api_key, api_secret = await _resolve_credentials(tool)
    print(f"Base URL: {base_url}")
    print(f"Auth headers: {list(_auth_headers(api_key, api_secret).keys())}")

    await _probe_raw_paths(base_url, api_key, api_secret)
    await _probe_tools()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
