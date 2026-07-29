"""Ambient background audio presets: CDN-hosted, lazily downloaded.

Preset tracks live at ``{base_url}/{name}.wav`` (default
https://timbalusercontent.com/assets/voice/ambience) and are downloaded on
first use into ``~/.cache/timbal/ambience``, verified against the sha256
pinned in ``PRESETS``. CDN assets are immutable — a changed track must be
uploaded under a new name, or every install pinning the old hash breaks.

A custom track is any readable audio file path in
``voice_config["ambient"]["source"]``.

Import-light on purpose — the server imports this (via config) at boot.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

DEFAULT_BASE_URL = "https://timbalusercontent.com/assets/voice/ambience"

# name → sha256 of {base_url}/{name}.wav. Regenerate assets with
# scripts/prepare_ambience.py (it prints the hash); provenance in
# ambience/ATTRIBUTIONS.md. Voiceless (or unintelligible murmur) only:
# ambience that leaks back through the caller's mic must not be transcribable.
PRESETS: dict[str, str] = {
    "office": "2b885fd5ad410a411542ea1e8788e8b064cbdca83287dd591776ccd6f34a1659",
    "call-center": "ca8b789d6547f48bb9c8a7cc3d317d6c7b469d27756a2f69b1088a5f4d1a33ac",
    "cafe": "f2ea5badf8b016c5c412a8aa64050de3acca72c99c119ae46398919d3094fd98",
    "city": "36bee8df55867e5143991aa2fbd4d9e8abb220db260463e95ffc99dfd348227c",
    "typing": "248a29b392b009df1435c18666587086285b694eb1338e8802687da3bcf49aa8",
}


def base_url() -> str:
    return os.environ.get("TIMBAL_AMBIENCE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "timbal" / "ambience"


def validate_ambient_source(source: str) -> None:
    """Offline check that ``source`` is a known preset or an existing file.

    Raises ``ValueError`` so config validation fails at server boot, not on
    the first session. Preset downloads are deferred to first use.
    """
    name = source.strip()
    if not name:
        raise ValueError("ambient source must be a preset name or file path")
    if name.lower() in PRESETS:
        return
    if not Path(name).expanduser().is_file():
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"ambient source {source!r} is neither a preset ({known}) nor an existing file")


def ensure_ambient_source(source: str) -> Path:
    """Local path for ``source``, downloading a preset on first use.

    Blocking (network + disk) — call off the event loop.
    """
    name = source.strip()
    if name.lower() in PRESETS:
        return _ensure_preset(name.lower())
    return Path(name).expanduser()


def _fetch(url: str) -> bytes:
    import httpx

    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _ensure_preset(name: str) -> Path:
    cached = cache_dir() / f"{name}.wav"
    if cached.is_file():
        return cached
    url = f"{base_url()}/{name}.wav"
    data = _fetch(url)
    digest = hashlib.sha256(data).hexdigest()
    if digest != PRESETS[name]:
        raise RuntimeError(f"ambience preset {name!r} from {url} failed checksum verification (got {digest})")
    cached.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cached.parent, suffix=".part")
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(data)
    os.replace(tmp_path, cached)  # atomic — concurrent downloads race harmlessly
    return cached
