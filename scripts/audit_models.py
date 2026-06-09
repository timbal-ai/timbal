#!/usr/bin/env python3
"""Audit models.yaml against provider APIs and internal consistency rules.

Usage (from repo root):
    uv run python scripts/audit_models.py              # live audit (informational; exit 0)
    uv run python scripts/audit_models.py --offline      # structural checks only (CI-safe)
    uv run python scripts/audit_models.py --strict     # exit 1 on unexpected DEAD models

Loads API keys from the environment (and optionally a repo-root `.env` file).
Models marked ``requires_activation: true`` are allowed to return activation/not-open
errors until the account has enabled them in the provider console.

Models marked ``dedicated_only: true`` are allowed to return non-serverless / dedicated
endpoint errors from providers such as TogetherAI.

Live mode is intentionally lenient by default: billing limits, specialized endpoints
(TTS, native audio, OpenAI Responses-only), and probe limitations are classified as
SKIPPED or IN_CATALOG rather than DEAD. Use ``--strict`` when you want a hard gate on
genuinely unreachable model IDs (mostly useful for OSS provider hygiene).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).parent.parent
MODELS_YAML = ROOT / "python/timbal/models.yaml"

STANDARD_CAPABILITIES = frozenset(
    {"vision", "tools", "reasoning", "audio", "video", "image_generation"}
)

_ACTIVATION_MARKERS = (
    "activate",
    "activation",
    "modelnotopen",
    "not activated",
    "has not activated",
    "model_not_available",
    "dedicated endpoint",
    "non-serverless",
)

_BILLING_MARKERS = (
    "payment_required",
    "credits_exhausted",
    "credit exhausted",
    "run out of credits",
    "balance_units",
)

_NON_CHAT_MARKERS = (
    "not a chat model",
    "only supported in v1/responses",
    "v1/responses and not in v1/chat/completions",
)

_NON_TEXT_MARKERS = (
    "response modalities",
    "accepts the following combination of response modalities",
)

# Providers using non-OpenAI-chat-completions APIs — skipped in live probes.
_SKIP_LIVE_PROVIDERS = frozenset({"anthropic"})

# OpenAI models that do not accept chat/completions text probes.
_OPENAI_NON_CHAT_MODELS = frozenset(
    {
        "gpt-5.1-codex",
        "gpt-5.2-pro",
        "o3-pro",
        "o3-deep-research",
        "o4-mini-deep-research",
    }
)

_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
}


class Status(str, Enum):
    OK = "ok"
    IN_CATALOG = "in_catalog"
    ACTIVATION_REQUIRED = "activation_required"
    DEDICATED_ONLY = "dedicated_only"
    DEAD = "dead"
    NO_KEY = "no_key"
    SKIPPED = "skipped"


@dataclass
class ModelResult:
    model_id: str
    provider: str
    api_name: str
    status: Status
    detail: str = ""


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_models() -> list[dict]:
    with MODELS_YAML.open() as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def _offline_audit(models: list[dict]) -> list[str]:
    """Return a list of structural error messages."""
    errors: list[str] = []
    from timbal.core.llm_router import _PROVIDERS

    seen: set[str] = set()
    for m in models:
        mid = m["id"]
        if mid in seen:
            errors.append(f"duplicate model id: {mid}")
        seen.add(mid)

        if "/" not in mid:
            errors.append(f"model id missing provider prefix: {mid}")
            continue

        provider, _ = mid.split("/", 1)
        if m.get("provider") != provider:
            errors.append(f"provider mismatch for {mid}: yaml={m.get('provider')} id={provider}")

        if provider not in _PROVIDERS:
            errors.append(f"unknown provider '{provider}' for {mid} (not in llm_router._PROVIDERS)")

        for cap in m.get("capabilities", []):
            if cap not in STANDARD_CAPABILITIES:
                errors.append(f"non-standard capability '{cap}' on {mid}")

        if m.get("requires_activation") and not m.get("notes"):
            errors.append(f"requires_activation set but notes missing on {mid}")

        if m.get("dedicated_only") and not m.get("notes"):
            errors.append(f"dedicated_only set but notes missing on {mid}")

        if m.get("requires_activation") and m.get("dedicated_only"):
            errors.append(f"model has both requires_activation and dedicated_only: {mid}")

    return errors


def _skip_probe_reason(m: dict) -> str | None:
    """Return a human reason when a generic text chat probe is inappropriate."""
    api_name = m["id"].split("/", 1)[1]
    provider = m["provider"]
    caps = set(m.get("capabilities") or [])

    if provider == "openai" and api_name in _OPENAI_NON_CHAT_MODELS:
        return "specialized OpenAI endpoint (not chat/completions)"

    if provider == "google":
        lowered = api_name.lower()
        if "tts" in lowered or "native-audio" in lowered:
            return "audio/TTS model (non-text probe)"
        if caps == {"audio"}:
            return "audio-only model (non-text probe)"

    return None


async def _fetch_catalog(client: httpx.AsyncClient, base_url: str, api_key: str) -> set[str]:
    r = await client.get(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"})
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, list):
        return {item.get("id", item) if isinstance(item, dict) else item for item in payload}
    return {item["id"] for item in payload.get("data", [])}


def _classify_error(
    body: str,
    status_code: int,
    *,
    requires_activation: bool,
    dedicated_only: bool,
    catalog: set[str] | None,
    api_name: str,
) -> Status:
    lower = body.lower()

    if any(m in lower for m in _BILLING_MARKERS):
        return Status.SKIPPED

    if any(m in lower for m in _NON_CHAT_MARKERS):
        if catalog is not None and api_name in catalog:
            return Status.IN_CATALOG
        return Status.SKIPPED

    if any(m in lower for m in _NON_TEXT_MARKERS):
        if catalog is not None and api_name in catalog:
            return Status.IN_CATALOG
        return Status.SKIPPED

    if requires_activation and status_code in (403, 404) and any(m in lower for m in _ACTIVATION_MARKERS):
        return Status.ACTIVATION_REQUIRED

    if status_code == 400 and any(m in lower for m in ("non-serverless", "dedicated endpoint", "model_not_available")):
        return Status.DEDICATED_ONLY if dedicated_only else Status.DEAD

    if status_code == 402:
        return Status.SKIPPED

    if status_code in (404, 410):
        if requires_activation:
            return Status.ACTIVATION_REQUIRED
        return Status.DEAD

    if status_code == 200:
        return Status.OK

    return Status.DEAD


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    payload: dict,
) -> httpx.Response:
    return await client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )


async def _probe_chat_completions(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    api_name: str,
    *,
    use_max_completion_tokens: bool,
) -> httpx.Response:
    payload: dict = {
        "model": api_name,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    if use_max_completion_tokens:
        payload["max_completion_tokens"] = 3
    else:
        payload["max_tokens"] = 3
    return await _post_json(client, f"{base_url.rstrip('/')}/chat/completions", api_key, payload)


async def _probe_openai_responses(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    api_name: str,
) -> httpx.Response:
    payload = {
        "model": api_name,
        "input": "ping",
        "max_output_tokens": 3,
        "store": False,
    }
    return await _post_json(client, f"{base_url.rstrip('/')}/responses", api_key, payload)


async def _probe_completion(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    provider: str,
    api_name: str,
    *,
    requires_activation: bool,
    dedicated_only: bool,
    catalog: set[str] | None,
) -> tuple[Status, str]:
    # Match llm_router: prefer max_completion_tokens on chat/completions.
    r = await _probe_chat_completions(
        client, base_url, api_key, api_name, use_max_completion_tokens=True
    )
    if r.status_code == 200:
        return Status.OK, "completion ok"

    body = r.text
    lower = body.lower()

    # Legacy providers still expect max_tokens.
    if "max_completion_tokens" in lower and "unsupported" in lower:
        r = await _probe_chat_completions(
            client, base_url, api_key, api_name, use_max_completion_tokens=False
        )
        if r.status_code == 200:
            return Status.OK, "completion ok"
        body = r.text
        lower = body.lower()

    # OpenAI reasoning models reject max_tokens — retry was already completion_tokens first.
    # Responses-only models: try /v1/responses (Timbal's default OpenAI path).
    if provider == "openai" and any(m in lower for m in _NON_CHAT_MARKERS):
        r = await _probe_openai_responses(client, base_url, api_key, api_name)
        if r.status_code == 200:
            return Status.OK, "responses ok"
        body = r.text

    status = _classify_error(
        body,
        r.status_code,
        requires_activation=requires_activation,
        dedicated_only=dedicated_only,
        catalog=catalog,
        api_name=api_name,
    )
    return status, body[:240].replace("\n", " ")


def _resolve_base_url(provider: str, config) -> str | None:
    if config.default_base_url:
        return config.default_base_url
    return _DEFAULT_BASE_URLS.get(provider)


async def _audit_model(
    client: httpx.AsyncClient,
    m: dict,
    provider: str,
    base_url: str,
    api_key: str,
    catalog: set[str] | None,
) -> ModelResult:
    api_name = m["id"].split("/", 1)[1]
    requires_activation = bool(m.get("requires_activation"))
    dedicated_only = bool(m.get("dedicated_only"))

    skip_reason = _skip_probe_reason(m)
    if skip_reason:
        if catalog is not None and api_name in catalog:
            return ModelResult(m["id"], provider, api_name, Status.IN_CATALOG, f"in catalog ({skip_reason})")
        return ModelResult(m["id"], provider, api_name, Status.SKIPPED, skip_reason)

    try:
        status, detail = await _probe_completion(
            client,
            base_url,
            api_key,
            provider,
            api_name,
            requires_activation=requires_activation,
            dedicated_only=dedicated_only,
            catalog=catalog,
        )
        if status == Status.OK and catalog is not None and api_name in catalog:
            status = Status.IN_CATALOG
        return ModelResult(m["id"], provider, api_name, status, detail)
    except httpx.TimeoutException:
        return ModelResult(m["id"], provider, api_name, Status.SKIPPED, "probe timed out")
    except Exception as exc:
        return ModelResult(m["id"], provider, api_name, Status.SKIPPED, f"probe error: {exc}")


async def _live_audit(models: list[dict]) -> list[ModelResult]:
    from timbal.core.llm_router import _PROVIDERS

    by_provider: dict[str, list[dict]] = {}
    for m in models:
        by_provider.setdefault(m["provider"], []).append(m)

    results: list[ModelResult] = []
    sem = asyncio.Semaphore(8)

    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:

        async def _bounded(coro):
            async with sem:
                return await coro

        for provider, provider_models in sorted(by_provider.items()):
            config = _PROVIDERS.get(provider)
            if config is None:
                continue

            if provider in _SKIP_LIVE_PROVIDERS:
                for m in provider_models:
                    results.append(
                        ModelResult(
                            m["id"],
                            provider,
                            m["id"].split("/", 1)[1],
                            Status.SKIPPED,
                            "live audit skipped (non-chat-completions API)",
                        )
                    )
                continue

            api_key = os.getenv(config.env_key)
            if not api_key:
                for m in provider_models:
                    results.append(
                        ModelResult(
                            m["id"],
                            provider,
                            m["id"].split("/", 1)[1],
                            Status.NO_KEY,
                            f"{config.env_key} not set",
                        )
                    )
                continue

            base_url = _resolve_base_url(provider, config)
            if not base_url:
                for m in provider_models:
                    results.append(
                        ModelResult(
                            m["id"],
                            provider,
                            m["id"].split("/", 1)[1],
                            Status.SKIPPED,
                            "no base URL for catalog/probe audit",
                        )
                    )
                continue

            catalog: set[str] | None
            try:
                catalog = await _fetch_catalog(client, base_url, api_key)
            except Exception as exc:
                catalog = None
                for m in provider_models:
                    results.append(
                        ModelResult(
                            m["id"],
                            provider,
                            m["id"].split("/", 1)[1],
                            Status.SKIPPED,
                            f"catalog fetch failed: {exc}",
                        )
                    )
                continue

            probe_results = await asyncio.gather(
                *[
                    _bounded(_audit_model(client, m, provider, base_url, api_key, catalog))
                    for m in provider_models
                ]
            )
            results.extend(probe_results)

    return results


def _print_results(results: list[ModelResult]) -> None:
    by_status: dict[Status, list[ModelResult]] = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)

    for status in Status:
        items = by_status.get(status, [])
        if not items:
            continue
        print(f"\n{status.value.upper()} ({len(items)})")
        for r in sorted(items, key=lambda x: x.model_id):
            suffix = f" — {r.detail}" if r.detail else ""
            print(f"  {r.model_id}{suffix}")


def _is_expected_live_result(r: ModelResult, models_by_id: dict[str, dict]) -> bool:
    m = models_by_id[r.model_id]
    if r.status in (Status.OK, Status.IN_CATALOG, Status.ACTIVATION_REQUIRED):
        return True
    if r.status == Status.DEDICATED_ONLY and m.get("dedicated_only"):
        return True
    if r.status == Status.SKIPPED:
        return True
    if r.status == Status.NO_KEY:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit models.yaml")
    parser.add_argument("--offline", action="store_true", help="Structural checks only (no network)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on DEAD models or DEDICATED_ONLY without dedicated_only flag",
    )
    args = parser.parse_args()

    if not MODELS_YAML.exists():
        print(f"error: {MODELS_YAML} not found", file=sys.stderr)
        return 1

    models = _load_models()
    if not models:
        print("error: no models in models.yaml", file=sys.stderr)
        return 1

    errors = _offline_audit(models)
    if errors:
        print("OFFLINE ERRORS:")
        for err in errors:
            print(f"  ✗ {err}")
    else:
        print(f"Offline checks passed ({len(models)} models)")

    exit_code = 1 if errors else 0

    if args.offline:
        return exit_code

    _load_dotenv()
    try:
        results = asyncio.run(_live_audit(models))
    except Exception as exc:
        print(f"live audit failed: {exc}", file=sys.stderr)
        return 1
    _print_results(results)

    models_by_id = {m["id"]: m for m in models}
    dead: list[ModelResult] = []
    unexpected_dedicated: list[ModelResult] = []

    for r in results:
        if r.status == Status.DEAD:
            dead.append(r)
            if args.strict:
                print(f"\nerror: dead model {r.model_id}", file=sys.stderr)
        elif r.status == Status.DEDICATED_ONLY and not models_by_id[r.model_id].get("dedicated_only"):
            unexpected_dedicated.append(r)
            print(f"\nwarning: dedicated-only model listed as serverless: {r.model_id}", file=sys.stderr)
            if args.strict:
                exit_code = 1

    keyed = [r for r in results if r.status != Status.NO_KEY]
    if not keyed:
        print("\nNo provider API keys set — live audit skipped (offline checks only).")
    else:
        okish = sum(1 for r in results if _is_expected_live_result(r, models_by_id))
        print(f"\nLive summary: {okish}/{len(keyed)} models ok, skipped, or expected ({len(keyed)} with keys)")
        if dead:
            print(f"  {len(dead)} DEAD — review IDs or run with context above")
        if unexpected_dedicated:
            print(f"  {len(unexpected_dedicated)} DEDICATED_ONLY without dedicated_only flag in yaml")

    if args.strict and dead:
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
