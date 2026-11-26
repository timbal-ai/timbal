"""Voice catalog, TTS preview and LLM list for the standalone playground.

Stdlib only, like the launcher. Keys come from the launcher's env, then the
launch directory's ``.env`` — the same lookup as the LiveKit credentials. The
launcher binds to localhost and every preview spends the developer's own TTS
credit, so this is a dev tool: deployed servers never mount it.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import wave
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

_CACHE_TTL_SECS = 3600.0
_cache: dict[tuple[str, ...], tuple[float, Any]] = {}
_cache_lock = threading.Lock()

PREVIEW_MAX_CHARS = 400
_VOICE_ID = re.compile(r"^[A-Za-z0-9._-]{1,96}$")
_PREVIEW_RATE = 24000

#: ElevenLabs library pages fetched per language (100 voices each).
_EL_LIBRARY_PAGES = 3
#: Extra library queries per language. Sorting by usage buries regional
#: accents under the largest locale, so they are asked for by name.
_EL_LIBRARY_EXTRA: dict[str, tuple[dict[str, str], ...]] = {
    "es": (
        {"locale": "es-ES"},
        {"accent": "andalusian"},
        {"search": "canario"},
        {"search": "gallego"},
        {"search": "catalán"},
        {"search": "vasco"},
        {"search": "valenciano"},
    ),
}

#: Provider -> env vars holding its key, in lookup order.
_TTS_KEYS = {
    "elevenlabs": ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY"),
    "deepgram": ("DEEPGRAM_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}
#: Gemini prebuilt voices: (name, gender, Google's one-word character). They speak any language.
_GEMINI_VOICES = (
    ("Fenrir", "male", "excitable"), ("Puck", "male", "upbeat"), ("Charon", "male", "informative"),
    ("Orus", "male", "firm"), ("Algieba", "male", "smooth"), ("Sadaltager", "male", "knowledgeable"),
    ("Achird", "male", "friendly"), ("Zubenelgenubi", "male", "casual"), ("Schedar", "male", "even"),
    ("Iapetus", "male", "clear"), ("Umbriel", "male", "easy-going"), ("Enceladus", "male", "breathy"),
    ("Algenib", "male", "gravelly"), ("Rasalgethi", "male", "informative"), ("Alnilam", "male", "firm"),
    ("Sadachbia", "male", "lively"),
    ("Kore", "female", "firm"), ("Zephyr", "female", "bright"), ("Leda", "female", "youthful"),
    ("Aoede", "female", "breezy"), ("Callirrhoe", "female", "easy-going"), ("Autonoe", "female", "bright"),
    ("Despina", "female", "smooth"), ("Erinome", "female", "clear"), ("Laomedeia", "female", "upbeat"),
    ("Achernar", "female", "soft"), ("Gacrux", "female", "mature"), ("Pulcherrima", "female", "forward"),
    ("Vindemiatrix", "female", "gentle"), ("Sulafat", "female", "warm"),
)  # fmt: skip
_OPENAI_VOICES = (
    "marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
)  # fmt: skip
#: Spoken-preview speed is native on these ElevenLabs models; v4 ignores it.
ELEVENLABS_NATIVE_SPEED_MODELS = ("eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_v3")

#: Accent words that place a Spanish voice in a region, checked against the
#: library's accent, name and description (lowercased).
_ES_REGIONS = (
    ("andalusian", ("andalu", "sevill", "málaga", "malaga", "cádiz", "cadiz", "granad", "southern spain", "sur de españa")),
    ("canarian", ("canari", "canary")),
    ("catalan", ("catal", "barcelon", "valencian", "valencia")),
    ("galician", ("galic", "gallego", "gallega")),
    ("basque", ("basque", "vasco", "euskad", "bilbao")),
    ("castilian", ("peninsular", "castil", "madrid", "spain", "españa", "castellano")),
)  # fmt: skip
_LATAM_LOCALES = ("es-MX", "es-AR", "es-CO", "es-CL", "es-PE", "es-VE", "es-US", "es-419")


class PlaygroundApiError(Exception):
    """A provider call failed. ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


def _env_key(names: tuple[str, ...], launch_dir: Path | None = None) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    from .playground import _read_dotenv

    file_env = _read_dotenv((launch_dir or Path.cwd()) / ".env")
    for name in names:
        if file_env.get(name):
            return file_env[name]
    return ""


def _request(method: str, url: str, headers: dict[str, str], body: dict | None = None, timeout: float = 30.0) -> bytes:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=data, method=method, headers={**headers, "Accept": "*/*"})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        raise PlaygroundApiError(f"{e.code} from provider: {detail}", status=502) from e
    except URLError as e:
        raise PlaygroundApiError(f"provider unreachable: {e.reason}", status=502) from e


def _cached(key: tuple[str, ...], build: Any) -> Any:
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL_SECS:
            return hit[1]
    value = build()
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def spanish_region(accent: str, locale: str, text: str) -> str:
    """Coarse region label for a Spanish voice: ``castilian``, ``andalusian``, … or ``latam``."""
    if locale in _LATAM_LOCALES:
        return "latam"
    blob = f"{accent} {text}".lower()
    for region, needles in _ES_REGIONS:
        if any(n in blob for n in needles):
            return region
    if locale == "es-ES":
        return "castilian"
    if any(w in blob for w in ("latin", "mexic", "argentin", "colombia", "venezol", "chile", "peru", "caribbean")):
        return "latam"
    return ""


def _el_voice(v: dict, source: str) -> dict:
    labels = v.get("labels") or {}
    accent = str(v.get("accent") or labels.get("accent") or "")
    locale = str(v.get("locale") or "")
    description = str(v.get("description") or labels.get("description") or "")
    name = str(v.get("name") or "")
    language = str(v.get("language") or labels.get("language") or "")
    return {
        "provider": "elevenlabs",
        "id": str(v.get("voice_id") or ""),
        "name": name,
        "accent": accent,
        "locale": locale,
        "language": language,
        "region": spanish_region(accent, locale, f"{name} {description}") if language.startswith("es") else "",
        "gender": str(v.get("gender") or labels.get("gender") or ""),
        "age": str(v.get("age") or labels.get("age") or "").replace("_", " "),
        "use_case": str(v.get("use_case") or labels.get("use_case") or "").replace("_", " "),
        "description": description[:240],
        "preview_url": str(v.get("preview_url") or ""),
        "source": source,
        "popularity": int(v.get("usage_character_count_1y") or v.get("cloned_by_count") or 0),
    }


def _elevenlabs_voices(key: str, language: str) -> list[dict]:
    headers = {"xi-api-key": key}
    out: dict[str, dict] = {}
    account = json.loads(_request("GET", "https://api.elevenlabs.io/v2/voices?page_size=100", headers))
    for v in account.get("voices", []):
        item = _el_voice(v, "account")
        if item["id"]:
            out[item["id"]] = item

    def library(query: dict[str, Any]) -> bool:
        params = {"language": language, "page_size": 100, "sort": "usage_character_count_1y", **query}
        url = f"https://api.elevenlabs.io/v1/shared-voices?{urlencode(params)}"
        data = json.loads(_request("GET", url, headers))
        for v in data.get("voices", []):
            if v.get("use_case") == "characters_animation":
                continue
            item = _el_voice(v, "library")
            if item["id"] and item["id"] not in out:
                out[item["id"]] = item
        return bool(data.get("has_more"))

    if language:
        for page in range(_EL_LIBRARY_PAGES):
            if not library({"page": page}):
                break
        for query in _EL_LIBRARY_EXTRA.get(language, ()):
            library(dict(query))
    voices = list(out.values())
    if language:
        voices = [v for v in voices if v["source"] == "library" or v["language"].startswith(language)] or voices
    return voices


def _deepgram_voices(key: str, language: str) -> list[dict]:
    data = json.loads(_request("GET", "https://api.deepgram.com/v1/models", {"Authorization": f"Token {key}"}))
    out = []
    for m in data.get("tts", []):
        langs = [str(lang) for lang in m.get("languages") or []]
        if language and not any(lang.split("-")[0] == language for lang in langs):
            continue
        md = m.get("metadata") or {}
        accent = str(md.get("accent") or "")
        locale = next((lang for lang in langs if "-" in lang), langs[0] if langs else "")
        tags = ", ".join(str(t) for t in md.get("tags") or [])
        out.append(
            {
                "provider": "deepgram",
                "id": str(m.get("canonical_name") or ""),
                "name": str(m.get("name") or m.get("canonical_name") or "").title(),
                "accent": accent,
                "locale": locale,
                "language": langs[0] if langs else "",
                "region": spanish_region(accent, locale, tags) if language == "es" else "",
                "gender": "",
                "age": "",
                "use_case": tags,
                "description": tags,
                "preview_url": str(md.get("sample") or ""),
                "source": "catalog",
                "popularity": 0,
            }
        )
    return [v for v in out if v["id"]]


def list_voices(provider: str, language: str = "", launch_dir: Path | None = None) -> list[dict]:
    """Voices ``provider`` can speak, optionally only those for ``language`` (``es``, ``en``…)."""
    provider = provider.strip().lower()
    language = language.strip().lower()[:8]
    if provider not in _TTS_KEYS:
        raise PlaygroundApiError(f"unknown TTS provider {provider!r}", status=400)
    key = _env_key(_TTS_KEYS[provider], launch_dir)
    if not key:
        raise PlaygroundApiError(f"no API key for {provider} in the launcher env or .env", status=424)
    if provider == "openai":
        return [
            {"provider": "openai", "id": v, "name": v.title(), "accent": "", "locale": "", "language": "",
             "region": "", "gender": "", "age": "", "use_case": "", "description": "", "preview_url": "",
             "source": "catalog", "popularity": 0}
            for v in _OPENAI_VOICES
        ]  # fmt: skip
    if provider == "gemini":
        return [
            {"provider": "gemini", "id": n, "name": n, "accent": "", "locale": "", "language": "",
             "region": "", "gender": g, "age": "", "use_case": "", "description": d.capitalize(),
             "preview_url": "", "source": "catalog", "popularity": 0}
            for n, g, d in _GEMINI_VOICES
        ]  # fmt: skip
    build = _elevenlabs_voices if provider == "elevenlabs" else _deepgram_voices
    voices = _cached((provider, language), lambda: build(key, language))
    return sorted(voices, key=lambda v: (v["source"] != "account", -v["popularity"], v["name"]))


def _wav(pcm: bytes, rate: int = _PREVIEW_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def tts_preview(
    provider: str,
    voice: str,
    text: str,
    *,
    model: str = "",
    speed: float | None = None,
    accent: str = "",
    launch_dir: Path | None = None,
) -> bytes:
    """One synthesized line as WAV, for auditioning a voice before a call.

    ``accent`` is Gemini's accent line (``andaluz marcado…``), the same text a call sends as ``tts_extra.accent``;
    the other providers take their accent from the voice and ignore it.
    """
    provider = provider.strip().lower()
    text = text.strip()[:PREVIEW_MAX_CHARS]
    if provider not in _TTS_KEYS:
        raise PlaygroundApiError(f"unknown TTS provider {provider!r}", status=400)
    if not text:
        raise PlaygroundApiError("text is required", status=400)
    if not _VOICE_ID.match(voice or ""):
        raise PlaygroundApiError("invalid voice id", status=400)
    if model and not _VOICE_ID.match(model):
        raise PlaygroundApiError("invalid model id", status=400)
    if speed is not None and not 0.5 <= speed <= 2.0:
        raise PlaygroundApiError("speed must be between 0.5 and 2.0", status=400)
    key = _env_key(_TTS_KEYS[provider], launch_dir)
    if not key:
        raise PlaygroundApiError(f"no API key for {provider} in the launcher env or .env", status=424)
    if provider == "elevenlabs":
        model = model or "eleven_v4_turbo"
        body: dict[str, Any] = {"text": text, "model_id": model}
        if speed is not None and model in ELEVENLABS_NATIVE_SPEED_MODELS:
            body["voice_settings"] = {"speed": max(0.7, min(1.2, speed))}
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{quote(voice, safe='')}?output_format=pcm_{_PREVIEW_RATE}"
        return _wav(_request("POST", url, {"xi-api-key": key}, body, timeout=60))
    if provider == "deepgram":
        params = urlencode({"model": voice, "encoding": "linear16", "sample_rate": _PREVIEW_RATE, "container": "none"})
        pcm = _request(
            "POST", f"https://api.deepgram.com/v1/speak?{params}", {"Authorization": f"Token {key}"}, {"text": text}
        )
        return _wav(pcm)
    if provider == "gemini":
        return _gemini_preview(key, model, voice, text, accent.strip()[:300])
    body = {"model": model or "gpt-4o-mini-tts", "voice": voice, "input": text, "response_format": "pcm"}
    if speed is not None:
        body["speed"] = max(0.25, min(4.0, speed))
    return _wav(_request("POST", "https://api.openai.com/v1/audio/speech", {"Authorization": f"Bearer {key}"}, body))


def _gemini_preview(key: str, model: str, voice: str, text: str, accent: str = "") -> bytes:
    """One line through the same prompt and voice rules the live adapter uses (single request, no streaming)."""
    from ..voice.gemini_tts import DEFAULT_TTS_MODEL, GEMINI_RATE, GEMINI_VOICES, build_prompt

    if voice.lower() not in GEMINI_VOICES:
        raise PlaygroundApiError(f"{voice!r} is not a Gemini voice", status=400)
    model = model if model.startswith("gemini-") else DEFAULT_TTS_MODEL
    body = {
        "contents": [{"role": "user", "parts": [{"text": build_prompt(text, accent=accent or None)}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice.capitalize()}}},
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='.-')}:generateContent"
    data = json.loads(_request("POST", url, {"x-goog-api-key": key}, body, timeout=60))
    for cand in data.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                return _wav(base64.b64decode(inline["data"]), GEMINI_RATE)
    raise PlaygroundApiError("Gemini returned no audio for this line", status=502)


def llm_models(launch_dir: Path | None = None) -> list[dict]:
    """Chat models the framework knows, with how they would be reached.

    ``access`` is ``"key"`` (the provider's own key is set), ``"proxy"`` (the
    Timbal platform proxy, ``TIMBAL_API_KEY`` + ``TIMBAL_ORG_ID``) or ``None``.
    Only presence is checked, not validity. Empty when the framework's model
    registry is not importable in the launcher's environment.
    """
    try:
        import typing

        from ..core.llm.registry import _PROVIDERS
        from ..core.models import Model
    except ImportError:
        return []
    proxy = bool(_env_key(("TIMBAL_API_KEY",), launch_dir) and _env_key(("TIMBAL_ORG_ID",), launch_dir))
    out = []
    for model in typing.get_args(Model):
        provider = model.split("/", 1)[0]
        if any(tag in model for tag in ("-tts", "-image", "embedding", "transcribe", "realtime")):
            continue
        cfg = _PROVIDERS.get(provider)
        if cfg is None:
            continue
        access = None
        if not cfg.platform_only and _env_key((cfg.env_key,), launch_dir):
            access = "key"
        elif proxy and cfg.supports_platform_proxy:
            access = "proxy"
        out.append({"id": model, "provider": provider, "access": access})
    return out
