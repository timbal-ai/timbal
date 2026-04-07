"""Install benchmark — measures installation time and disk footprint per extras profile.

Spins up an isolated venv for each profile, installs the local timbal package,
then reports install time and installed size. Run from the repo root:

    python benchmarks/install/bench_install.py

Requirements: uv must be on PATH.

Profiles measured:
    bare    — pip install .
    codegen — pip install .[codegen]
    all     — pip install .[all]
"""

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.resolve()

PROFILES: list[tuple[str, str]] = [
    ("bare",      "."),
    ("codegen",   ".[codegen]"),
    ("documents", ".[documents]"),
    ("evals",     ".[evals]"),
    ("server",    ".[server]"),
    ("all",       ".[all]"),
]

# Packages to break down individually in the size report.
TRACKED_PACKAGES = [
    "anthropic",
    "openai",
    "httpx",
    "mcp",
    "pydantic",
    "structlog",
    "rich",
    "fastapi",
    "uvicorn",
    "libcst",
    "ruff",
    "pymupdf",
    "openpyxl",
    "docx",         # python-docx
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def dir_size_mb(path: Path) -> float:
    """Return total size of a directory tree in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def site_packages(venv: Path) -> Path:
    candidates = list(venv.glob("lib/python*/site-packages"))
    if not candidates:
        raise RuntimeError(f"No site-packages found in {venv}")
    return candidates[0]


def package_size_mb(sp: Path, name: str) -> float | None:
    """Return installed size of a package directory in MB, or None if absent."""
    pkg_dir = sp / name
    if pkg_dir.exists():
        return dir_size_mb(pkg_dir)
    # Some packages install as a single .py file
    single = sp / f"{name}.py"
    if single.exists():
        return single.stat().st_size / (1024 * 1024)
    return None


def format_mb(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.1:
        return f"{value * 1024:.0f} KB"
    return f"{value:.1f} MB"


def col(text: str, width: int, align: str = "<") -> str:
    return f"{text:{align}{width}}"


# ── Core benchmark ────────────────────────────────────────────────────────────

def bench_profile(name: str, install_spec: str, tmp_dir: Path) -> dict:
    venv = tmp_dir / name

    # Create venv
    run(["uv", "venv", str(venv), "--python", sys.executable])

    # Time the install
    t0 = time.perf_counter()
    run([
        "uv", "pip", "install",
        "--python", str(venv / "bin" / "python"),
        str(REPO_ROOT / install_spec) if install_spec == "." else str(REPO_ROOT) + install_spec[1:],
    ])
    elapsed = time.perf_counter() - t0

    sp = site_packages(venv)
    total_mb = dir_size_mb(sp)

    package_sizes = {pkg: package_size_mb(sp, pkg) for pkg in TRACKED_PACKAGES}

    return {
        "name": name,
        "elapsed": elapsed,
        "total_mb": total_mb,
        "packages": package_sizes,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    names = [r["name"] for r in results]
    bare = next((r for r in results if r["name"] == "bare"), results[0])

    # ── Summary table ──────────────────────────────────────────────────────────
    W = [10, 12, 14, 14]
    sep = "─" * (sum(W) + len(W) * 3 + 1)

    print()
    print("  Install benchmark — timbal extras profiles")
    print(f"  {sep}")
    header = (
        f"  {col('Profile', W[0])} │ "
        f"{col('Time (s)', W[1], '>')} │ "
        f"{col('Size (total)', W[2], '>')} │ "
        f"{col('Δ bare', W[3], '>')}"
    )
    print(header)
    print(f"  {sep}")

    for r in results:
        delta = r["total_mb"] - bare["total_mb"]
        delta_str = f"+{format_mb(delta)}" if delta > 0 else "—"
        elapsed_str = f"{r['elapsed']:.1f}s"
        print(
            f"  {col(r['name'], W[0])} │ "
            f"{col(elapsed_str, W[1], '>')} │ "
            f"{col(format_mb(r['total_mb']), W[2], '>')} │ "
            f"{col(delta_str, W[3], '>')}"
        )

    print(f"  {sep}")

    # ── Per-package breakdown ──────────────────────────────────────────────────
    print()
    print("  Per-package size breakdown")
    PW = [14] + [12] * len(results)
    psep = "─" * (sum(PW) + len(PW) * 3 + 1)
    print(f"  {psep}")
    pkg_header = f"  {col('Package', PW[0])}"
    for r in results:
        pkg_header += f" │ {col(r['name'], PW[results.index(r) + 1], '>')}"
    print(pkg_header)
    print(f"  {psep}")

    for pkg in TRACKED_PACKAGES:
        row = f"  {col(pkg, PW[0])}"
        any_present = any(r["packages"].get(pkg) is not None for r in results)
        if not any_present:
            continue
        for i, r in enumerate(results):
            size = r["packages"].get(pkg)
            row += f" │ {col(format_mb(size), PW[i + 1], '>')}"
        print(row)

    print(f"  {psep}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not shutil.which("uv"):
        print("error: uv not found on PATH. Install from https://github.com/astral-sh/uv", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking install profiles for: {REPO_ROOT}")
    print(f"Python: {sys.executable}")
    print()

    results = []
    with tempfile.TemporaryDirectory(prefix="timbal_bench_") as tmp:
        tmp_dir = Path(tmp)
        for name, spec in PROFILES:
            print(f"  [{name}] installing '{spec}' ...", end=" ", flush=True)
            try:
                result = bench_profile(name, spec, tmp_dir)
                results.append(result)
                print(f"done ({result['elapsed']:.1f}s, {format_mb(result['total_mb'])})")
            except subprocess.CalledProcessError as e:
                print(f"FAILED\n{e.stderr}")
                sys.exit(1)

    print_report(results)


if __name__ == "__main__":
    main()
