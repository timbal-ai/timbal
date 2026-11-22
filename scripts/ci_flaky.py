# ruff: noqa: T201
"""The tests that passed only on retry in a pytest log (pytest-rerunfailures), as a Markdown list.

Used by the `Run tests` step of `.github/workflows/ci.yml`: prints nothing when every test passed on its first
try, else a heading and one line per flaky test, for the job summary. A test that also failed its retry (or
errored in setup/teardown again) is a failure, not a flake, and is left out (the job fails on it anyway).

Reads both report formats: `[gw0] [ 45%] RERUN <id>` (pytest-xdist) and `<id> RERUN [ 45%]` (one process).
"""

from __future__ import annotations

import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TEST_ID = r"(python/tests/\S+::\S+)"
RERUN = re.compile(rf"RERUN {TEST_ID}|{TEST_ID} RERUN\b")
# FAILED (the test) or ERROR (its setup or teardown): either way it did not pass its retry.
FAILED = re.compile(rf"^(?:\[gw\d+\] \[\s*\d+%\] )?(?:FAILED|ERROR) {TEST_ID}|{TEST_ID} (?:FAILED|ERROR)\b")


def flaky(log: str) -> list[str]:
    reran: set[str] = set()
    failed: set[str] = set()
    for line in ANSI.sub("", log).splitlines():
        line = line.strip()
        if match := RERUN.search(line):
            reran.add(match.group(1) or match.group(2))
        if match := FAILED.search(line):
            failed.add(match.group(1) or match.group(2))
    return sorted(reran - failed)


def main(path: str) -> None:
    with open(path, encoding="utf-8", errors="replace") as fh:
        tests = flaky(fh.read())
    if tests:
        print("### Passed only on retry (flaky)")
        for test in tests:
            print(f"- `{test}`")


if __name__ == "__main__":
    main(sys.argv[1])
