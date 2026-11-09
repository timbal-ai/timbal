"""Fail when this checkout executes more instructions per run than a base revision.

    uv run python benchmarks/instructions/gate.py --base origin/main
    uv run python benchmarks/instructions/gate.py --base-path ../timbal-main --threshold 0.5

Runs ``count.py`` from this checkout against both revisions' ``python/`` (same interpreter, same
dependencies, same harness; only ``PYTHONPATH`` differs) and compares per case. Counts repeat to about
0.01%, so the default threshold of 1% only trips on real work added to a run. Writes a markdown table to
``$GITHUB_STEP_SUMMARY`` when set.

Exit codes: 0 no case over the threshold; 1 a case over it; 2 the gate could not measure (a revision did
not run, imported the wrong package, or a case failed on this checkout). A case only the base cannot run
(its API is newer than the base) is listed, not compared.
"""

# ruff: noqa: T201 - prints its report
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COUNT = Path(__file__).with_name("count.py")


class GateError(Exception):
    """The gate could not measure: never a pass."""


def _count(source_root: Path) -> dict:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(PYTHONPATH=str(source_root / "python"), PYTHONHASHSEED="0")
    out = subprocess.run([sys.executable, str(COUNT)], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise GateError(f"count.py failed for {source_root}:\n{out.stderr[-2000:]}")
    try:
        report = json.loads(out.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as e:
        raise GateError(f"count.py printed no report for {source_root}: {e}\n{out.stdout[-2000:]}") from e
    expected = (source_root / "python" / "timbal").resolve()
    if Path(report["timbal"]).resolve().parent != expected:
        raise GateError(f"measured {report['timbal']}, expected the package in {expected}")
    return report


def _table(base: dict, head: dict, threshold: float) -> tuple[str, list[str]]:
    lines = ["| case | base | head | change |", "|---|---:|---:|---:|"]
    regressed = []
    base_n_by_case, base_errors = base["instructions"], base.get("errors", {})
    for case, head_n in head["instructions"].items():
        base_n = base_n_by_case.get(case)
        if base_n is None:
            why = "base cannot run it" if case in base_errors else "new"
            lines.append(f"| {case} | - | {head_n:,.0f} | {why} |")
            continue
        change = (head_n / base_n - 1) * 100
        mark = " :x:" if change > threshold else ""
        lines.append(f"| {case} | {base_n:,.0f} | {head_n:,.0f} | {change:+.2f}%{mark} |")
        if change > threshold:
            regressed.append(case)
    return "\n".join(lines), regressed


def _report(text: str) -> None:
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="git ref to compare against (checked out in a temporary worktree)")
    source.add_argument("--base-path", type=Path, help="an existing checkout of the base revision")
    parser.add_argument("--threshold", type=float, default=1.0, help="max allowed increase per case, in %%")
    args = parser.parse_args()

    title = f"### Instructions per run (Python {sys.version.split()[0]})\n"
    with tempfile.TemporaryDirectory() as tmp:
        if args.base_path is not None:
            base_root = args.base_path.resolve()
        else:
            base_root = Path(tmp) / "base"
            subprocess.run(
                ["git", "worktree", "add", "--detach", "-q", str(base_root), args.base], cwd=ROOT, check=True
            )
        try:
            base = _count(base_root)
            head = _count(ROOT)
        except GateError as e:
            _report(f"{title}\n**The gate could not measure.**\n\n```\n{e}\n```\n")
            return 2
        finally:
            if args.base_path is None:
                subprocess.run(["git", "worktree", "remove", "--force", str(base_root)], cwd=ROOT, check=False)

    if head.get("errors"):
        failed = "\n".join(f"- `{case}`: {error}" for case, error in head["errors"].items())
        _report(f"{title}\n**Cases failed on this checkout:**\n\n{failed}\n")
        return 2
    table, regressed = _table(base, head, args.threshold)
    verdict = (
        f"**{len(regressed)} case(s) over +{args.threshold:g}%:** {', '.join(regressed)}"
        if regressed
        else f"No case over +{args.threshold:g}%."
    )
    _report(f"{title}\n{table}\n\n{verdict}\n")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
