#!/usr/bin/env python3
"""Regenerate the VideoModel Literal in video_models.py from video_models.yaml.

Usage (run from the repo root):
    uv run python scripts/generate_video_models.py
"""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
VIDEO_MODELS_YAML = ROOT / "python/timbal/video_models.yaml"
VIDEO_MODELS_PY = ROOT / "python/timbal/core/video_models.py"

_LITERAL_PATTERN = re.compile(
    r"(# VideoModel type — direct models only \(fal endpoints are dynamic\)\n)VideoModel = Literal\[.*?\]",
    re.DOTALL,
)


def main() -> None:
    if not VIDEO_MODELS_YAML.exists():
        print(f"error: {VIDEO_MODELS_YAML} not found", file=sys.stderr)
        sys.exit(1)

    with VIDEO_MODELS_YAML.open() as f:
        data = yaml.safe_load(f)

    models = data.get("models", [])
    if not models:
        print("error: no models found in video_models.yaml", file=sys.stderr)
        sys.exit(1)

    model_ids = [m["id"] for m in models if m.get("id")]

    lines = ["VideoModel = Literal[\n"]
    for mid in model_ids:
        lines.append(f'    "{mid}",\n')
    lines.append("]")
    new_block = "".join(lines)

    source = VIDEO_MODELS_PY.read_text()
    if not _LITERAL_PATTERN.search(source):
        print(
            f"error: could not find 'VideoModel = Literal[...]' block in {VIDEO_MODELS_PY}",
            file=sys.stderr,
        )
        sys.exit(1)

    new_source = _LITERAL_PATTERN.sub(r"\g<1>" + new_block, source)
    VIDEO_MODELS_PY.write_text(new_source)
    print(f"Updated {VIDEO_MODELS_PY} with {len(model_ids)} video models.")


if __name__ == "__main__":
    main()
