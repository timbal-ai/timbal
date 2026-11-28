# ruff: noqa: T201
"""Decide what CI runs for this event: the fast lane, the full matrix, or nothing.

Run by the `plan` job of `.github/workflows/ci.yml`; prints `lane=` and `matrix=` for `$GITHUB_OUTPUT`.

- **full**: every OS x Python. The merge queue (`merge_group`), manual runs (`workflow_dispatch`, narrowed
  by its `os` input), and pull requests while the merge queue is not on (`CI_MERGE_QUEUE` repo variable unset).
- **fast**: Ubuntu with the lowest supported Python (what a newer local Python hides most often). Pull
  requests into main once `CI_MERGE_QUEUE=true`: the merge queue then runs the full matrix before anything
  lands. Not pull requests into `release/*`: GitHub has no merge queue for a branch pattern, so nothing would
  run the full matrix before a backport lands. They run it on every push.
- **none**: a pull request, or a merge queue group, that changes nothing the tests read (docs, other workflows,
  release scripts). `CI result` passes. A merge queue group is judged by everything it would merge (its base to
  its head), so a group holding one PR that does touch tested files runs the full lane.

Environment: EVENT, PR, BASE (the pull request's target branch), GROUP_BASE / GROUP_HEAD (a merge queue
group's commits), REPO, MERGE_QUEUE, OS_INPUT (from the workflow), GH_TOKEN for `gh`.
`--files a b c` replaces the PR's changed files (local testing).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

# Linux on ubuntu-24.04, not ubuntu-latest (ubuntu-26.04 since November 2026): on image 20261122 the hosted
# runner was shut down near the end of a full run, 3 of its 86 test jobs ("The runner has received a shutdown
# signal", exit 143, no test failed), and none of the 320 on earlier 26.04 images. actions/runner-images#15179
# reports the same shutdowns on 26.04 only. Move on once 26.04 runs our suite to the end.
LINUX = "ubuntu-24.04"
OSES = [LINUX, "macos-latest", "windows-latest"]
PYTHONS = ["3.11", "3.12", "3.13"]
FAST = {"os": LINUX, "python": "3.11"}
COVERAGE = {"os": LINUX, "python": "3.13"}  # one cell of the full matrix measures coverage

# Files the tests (or the CI job itself) depend on. Anything else (docs, other workflows) skips the tests.
TESTED = re.compile(
    r"^(python/|docs/models/|pyproject\.toml$|uv\.lock$|scripts/(audit_models|generate_models|ci_plan|ci_flaky)\.py$"
    r"|\.github/workflows/ci\.yml$)"
)


# GitHub's compare API lists at most this many files; a group that changes more runs the full lane.
COMPARE_FILE_LIMIT = 300


def gh(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], check=True, capture_output=True, text=True).stdout


def changed_files(argv: list[str], event: str) -> list[str] | None:
    """The files the change touches: a pull request's, or everything a merge queue group would merge. None:
    not known (a manual run, or more files than the compare API lists), which plans the full lane."""
    if "--files" in argv:
        return argv[argv.index("--files") + 1 :]
    repo = os.environ.get("REPO", "")
    if event == "pull_request":
        out = gh(f"repos/{repo}/pulls/{os.environ['PR']}/files", "--paginate", "--jq", ".[].filename")
        return [line for line in out.splitlines() if line]
    if event == "merge_group":
        base, head = os.environ.get("GROUP_BASE", ""), os.environ.get("GROUP_HEAD", "")
        if not base or not head:
            return None
        files = [
            entry["filename"] for entry in json.loads(gh(f"repos/{repo}/compare/{base}...{head}")).get("files", [])
        ]
        return None if len(files) >= COMPARE_FILE_LIMIT else files
    return None


def plan(
    event: str, files: list[str] | None, merge_queue: bool, os_input: str, base: str = ""
) -> tuple[str, list[dict]]:
    if event in ("pull_request", "merge_group") and files is not None and not any(TESTED.match(f) for f in files):
        return "none", []
    if event == "pull_request":
        lane = "fast" if merge_queue and not base.startswith("release/") else "full"
    else:  # merge_group, workflow_dispatch
        lane = "full"
    if lane == "fast":
        cells = [FAST]
    else:
        oses = [os_input] if os_input and os_input != "all" else OSES
        cells = [{"os": o, "python": p} for o in oses for p in PYTHONS]
    return lane, [{**c, "coverage": c == COVERAGE} for c in cells]


def main(argv: list[str]) -> None:
    event = os.environ.get("EVENT", "")
    files = changed_files(argv, event)
    merge_queue = os.environ.get("MERGE_QUEUE", "").strip().lower() == "true"
    lane, matrix = plan(event, files, merge_queue, os.environ.get("OS_INPUT", ""), os.environ.get("BASE", ""))
    print(f"lane={lane}")
    print(f"matrix={json.dumps(matrix)}")
    print(f"{event}: lane {lane}, {len(matrix)} job(s)", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
