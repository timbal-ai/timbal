#!/usr/bin/env python3
"""Regenerate the Model Literal in models.py from models.yaml.

Usage (run from the repo root):
    python scripts/generate_models.py
"""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
MODELS_YAML = ROOT / "python/timbal/models.yaml"
MODELS_PY = ROOT / "python/timbal/core/models.py"

# Markers that delimit the auto-generated block inside models.py.
_START = "# Model type with provider prefixes"
_LITERAL_PATTERN = re.compile(
    r"(# Model type with provider prefixes\n)Model = Literal\[.*?\]",
    re.DOTALL,
)


def main() -> None:
    if not MODELS_YAML.exists():
        print(f"error: {MODELS_YAML} not found", file=sys.stderr)
        sys.exit(1)

    with MODELS_YAML.open() as f:
        data = yaml.safe_load(f)

    models = data.get("models", [])
    if not models:
        print("error: no models found in models.yaml", file=sys.stderr)
        sys.exit(1)

    model_ids = [m["id"] for m in models]

    # Build the replacement Literal block.
    lines = ["Model = Literal[\n"]
    for mid in model_ids:
        lines.append(f'    "{mid}",\n')
    lines.append("]")
    new_block = "".join(lines)

    source = MODELS_PY.read_text()

    if not _LITERAL_PATTERN.search(source):
        print(
            f"error: could not find 'Model = Literal[...]' block in {MODELS_PY}",
            file=sys.stderr,
        )
        sys.exit(1)

    new_source = _LITERAL_PATTERN.sub(r"\g<1>" + new_block, source)

    MODELS_PY.write_text(new_source)
    print(f"Updated {MODELS_PY} with {len(model_ids)} models.")


if __name__ == "__main__":
    main()
