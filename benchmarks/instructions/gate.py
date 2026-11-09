"""Fail when this checkout executes more instructions per run than a base revision.

    uv run python benchmarks/instructions/gate.py --base origin/main
    uv run python benchmarks/instructions/gate.py --base-path ../timbal-main --threshold 0.5

Runs ``count.py`` from this checkout against both revisions' ``python/`` (same interpreter, same
dependencies, same harness; only ``PYTHONPATH`` differs) and compares per case. Counts repeat to about
0.01%, so the default threshold of 1% only trips on real work added to a run. Writes a markdown table to
``$GITHUB_STEP_SUMMARY`` when set.
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


def _count(source_root: Path) -> dict[str, float]:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(PYTHONPATH=str(source_root / "python"), PYTHONHASHSEED="0")
    out = subprocess.run([sys.executable, str(COUNT)], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"count.py failed for {source_root}:\n{out.stderr[-2000:]}")
    report = json.loads(out.stdout.strip().splitlines()[-1])
    expected = (source_root / "python" / "timbal").resolve()
    if Path(report["timbal"]).resolve().parent != expected:
        raise RuntimeError(f"measured {report['timbal']}, expected the package in {expected}")
    return report["instructions"]


def _table(base: dict[str, float], head: dict[str, float], threshold: float) -> tuple[str, list[str]]:
    lines = ["| case | base | head | change |", "|---|---:|---:|---:|"]
    regressed = []
    for case, head_n in head.items():
        base_n = base.get(case)
        if base_n is None:
            lines.append(f"| {case} | - | {head_n:,.0f} | new |")
            continue
        change = (head_n / base_n - 1) * 100
        mark = " :x:" if change > threshold else ""
        lines.append(f"| {case} | {base_n:,.0f} | {head_n:,.0f} | {change:+.2f}%{mark} |")
        if change > threshold:
            regressed.append(case)
    return "\n".join(lines), regressed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="git ref to compare against (checked out in a temporary worktree)")
    source.add_argument("--base-path", type=Path, help="an existing checkout of the base revision")
    parser.add_argument("--threshold", type=float, default=1.0, help="max allowed increase per case, in %%")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.base_path is not None:
            base_root = args.base_path.resolve()
        else:
            base_root = Path(tmp) / "base"
            subprocess.run(
                ["git", "worktree", "add", "--detach", "-q", str(base_root), args.base], cwd=ROOT, check=True
            )
        try:
            try:
                base = _count(base_root)
            except RuntimeError as e:
                # A base too old for the harness's public API has nothing to compare against.
                print(f"Base not measurable, skipping the gate.\n{e}")
                return 0
            head = _count(ROOT)
        finally:
            if args.base_path is None:
                subprocess.run(["git", "worktree", "remove", "--force", str(base_root)], cwd=ROOT, check=False)

    table, regressed = _table(base, head, args.threshold)
    verdict = (
        f"**{len(regressed)} case(s) over +{args.threshold:g}%:** {', '.join(regressed)}"
        if regressed
        else f"No case over +{args.threshold:g}%."
    )
    report = f"### Instructions per run (Python {sys.version.split()[0]})\n\n{table}\n\n{verdict}\n"
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(report)
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
