# ruff: noqa: T201
"""Decide what CI runs for this event: the fast lane, the full matrix, or nothing.

Run by the `plan` job of `.github/workflows/ci.yml`; prints `lane=` and `matrix=` for `$GITHUB_OUTPUT`.

- **full**: every OS x Python. The merge queue (`merge_group`), manual runs (`workflow_dispatch`, narrowed
  by its `os` input), and pull requests while the merge queue is not on (`CI_MERGE_QUEUE` repo variable unset).
- **fast**: Ubuntu with the lowest supported Python (what a newer local Python hides most often). Pull
  requests once `CI_MERGE_QUEUE=true`: the merge queue then runs the full matrix before anything lands.
- **none**: a pull request that changes nothing the tests read (docs, other workflows). `CI result` passes.

Environment: EVENT, PR, REPO, MERGE_QUEUE, OS_INPUT (from the workflow), GH_TOKEN for `gh`.
`--files a b c` replaces the PR's changed files (local testing).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

OSES = ["ubuntu-latest", "macos-latest", "windows-latest"]
PYTHONS = ["3.11", "3.12", "3.13"]
FAST = {"os": "ubuntu-latest", "python": "3.11"}
COVERAGE = {"os": "ubuntu-latest", "python": "3.13"}  # one cell of the full matrix measures coverage

# Files the tests (or the CI job itself) depend on. Anything else (docs, other workflows) skips the tests.
TESTED = re.compile(
    r"^(python/|docs/models/|pyproject\.toml$|uv\.lock$|scripts/(audit_models|generate_models|ci_plan)\.py$"
    r"|\.github/workflows/ci\.yml$)"
)


def changed_files(argv: list[str]) -> list[str]:
    if "--files" in argv:
        return argv[argv.index("--files") + 1 :]
    pr, repo = os.environ["PR"], os.environ["REPO"]
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{pr}/files", "--paginate", "--jq", ".[].filename"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def plan(event: str, files: list[str] | None, merge_queue: bool, os_input: str) -> tuple[str, list[dict]]:
    if event == "pull_request":
        if not any(TESTED.match(f) for f in files or []):
            return "none", []
        lane = "fast" if merge_queue else "full"
    else:  # merge_group, workflow_dispatch
        lane = "full"
    if lane == "fast":
        return lane, [{**FAST, "coverage": False}]
    oses = [os_input] if os_input and os_input != "all" else OSES
    cells = [{"os": o, "python": p} for o in oses for p in PYTHONS]
    return lane, [{**c, "coverage": c == COVERAGE} for c in cells]


def main(argv: list[str]) -> None:
    event = os.environ.get("EVENT", "")
    files = changed_files(argv) if event == "pull_request" else None
    merge_queue = os.environ.get("MERGE_QUEUE", "").strip().lower() == "true"
    lane, matrix = plan(event, files, merge_queue, os.environ.get("OS_INPUT", ""))
    print(f"lane={lane}")
    print(f"matrix={json.dumps(matrix)}")
    print(f"{event}: lane {lane}, {len(matrix)} job(s)", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
