#!/usr/bin/env python3
"""Regenerate the Model Literal in llm_router.py from models.yaml.

Usage (run from the repo root):
    python scripts/generate_models.py
"""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
MODELS_YAML = ROOT / "python/timbal/models.yaml"
LLM_ROUTER = ROOT / "python/timbal/core/llm_router.py"

# Markers that delimit the auto-generated block inside llm_router.py.
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

    source = LLM_ROUTER.read_text()

    if not _LITERAL_PATTERN.search(source):
        print(
            f"error: could not find 'Model = Literal[...]' block in {LLM_ROUTER}",
            file=sys.stderr,
        )
        sys.exit(1)

    new_source = _LITERAL_PATTERN.sub(r"\g<1>" + new_block, source)

    LLM_ROUTER.write_text(new_source)
    print(f"Updated {LLM_ROUTER} with {len(model_ids)} models.")


if __name__ == "__main__":
    main()
