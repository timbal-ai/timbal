"""Telephony routes — phone calls bridged into voice sessions.

Providers (Twilio, Telnyx) answer a call, then open a WebSocket to this server
and stream the caller's audio as base64 G.711 μ-law at 8kHz inside JSON
frames. The bridge decodes/resamples into a regular :class:`VoiceSession` and
sends TTS audio back the same way — barge-in maps to the provider's ``clear``
message, playback tracking to ``mark`` echoes.

Routes (per provider ``{p}`` in ``twilio`` | ``telnyx``):

- ``POST /voice/{p}/incoming`` — the number's voice webhook. Returns
  TwiML/TeXML that connects the call to the media WebSocket below.
- ``WS /voice/{p}/stream`` — the bidirectional media stream.

The two providers speak near-identical protocols with different spellings
(Twilio camelCase / ``streamSid``; Telnyx snake_case / ``stream_id``), so one
handler runs both via a small dialect table.

Signature validation: set ``TWILIO_AUTH_TOKEN`` (HMAC-SHA1 of the webhook URL
+ params) and/or ``TELNYX_PUBLIC_KEY`` (Ed25519 over ``timestamp|body``) to
enforce webhook authenticity; without them webhooks are accepted with a
warning (dev mode).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from contextlib import aclosing
from typing import Any
from urllib.parse import parse_qsl
from xml.sax.saxutils import quoteattr

import structlog
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

from ..voice.config import VoiceConfig

logger = structlog.get_logger("timbal.server.telephony")

router = APIRouter(prefix="/voice", tags=["telephony"])

# Custom <Parameter> names a webhook may pass through to the session config.
# Same trust level as the browser hello, but telephony parameters are set by
# whoever controls the TwiML/TeXML — still allowlist to names, never callables.
_CONFIG_PARAM_KEYS = ("turn_detector", "stt_provider", "stt_model", "tts_model", "model", "language", "voice")

_ULAW_ENCODINGS = {"audio/x-mulaw", "pcmu"}

# Telnyx rejects RTP media chunks under 20ms; Twilio takes any size. Batch
# the downlink to >=20ms for both (fewer frames is also just cheaper).
_MIN_MEDIA_BYTES = 160  # 20ms of mulaw @ 8kHz


# ---------------------------------------------------------------------------
# Provider dialects
# ---------------------------------------------------------------------------


class _TwilioDialect:
    """Twilio Media Streams: camelCase frames keyed by ``streamSid``."""

    name = "twilio"

    def stream_id(self, frame: dict) -> str | None:
        return frame.get("streamSid")

    def start_info(self, frame: dict) -> dict[str, Any]:
        start = frame.get("start") or {}
        media_format = start.get("mediaFormat") or {}
        custom = start.get("customParameters") or {}
        return {
            "call_id": start.get("callSid"),
            # Twilio's start frame has no From/To; our TwiML forwards them
            # as custom parameters (see twilio_incoming).
            "from": custom.get("from"),
            "to": custom.get("to"),
            "custom": custom,
            "encoding": media_format.get("encoding"),
            "sample_rate": media_format.get("sampleRate"),
        }

    def media_bytes(self, frame: dict) -> bytes | None:
        media = frame.get("media") or {}
        track = media.get("track")
        if track not in (None, "inbound"):
            return None
        payload = media.get("payload")
        return base64.b64decode(payload) if payload else None

    def mark_name(self, frame: dict) -> str | None:
        return (frame.get("mark") or {}).get("name")

    def dtmf_digit(self, frame: dict) -> str | None:
        return (frame.get("dtmf") or {}).get("digit")

    def media_frame(self, stream_id: str, payload_b64: str) -> dict:
        return {"event": "media", "streamSid": stream_id, "media": {"payload": payload_b64}}

    def mark_frame(self, stream_id: str, name: str) -> dict:
        return {"event": "mark", "streamSid": stream_id, "mark": {"name": name}}

    def clear_frame(self, stream_id: str) -> dict:
        return {"event": "clear", "streamSid": stream_id}


class _TelnyxDialect:
    """Telnyx bidirectional streaming: snake_case frames keyed by ``stream_id``.

    Client → Telnyx frames carry no stream id (one stream per socket).
    """

    name = "telnyx"

    def stream_id(self, frame: dict) -> str | None:
        return frame.get("stream_id")

    def start_info(self, frame: dict) -> dict[str, Any]:
        start = frame.get("start") or {}
        media_format = start.get("media_format") or {}
        return {
            "call_id": start.get("call_control_id"),
            "from": start.get("from"),
            "to": start.get("to"),
            "custom": start.get("custom_parameters") or {},
            "encoding": media_format.get("encoding"),
            "sample_rate": media_format.get("sample_rate"),
        }

    def media_bytes(self, frame: dict) -> bytes | None:
        media = frame.get("media") or {}
        track = media.get("track")
        if track not in (None, "inbound"):
            return None
        payload = media.get("payload")
        return base64.b64decode(payload) if payload else None

    def mark_name(self, frame: dict) -> str | None:
        return (frame.get("mark") or {}).get("name")

    def dtmf_digit(self, frame: dict) -> str | None:
        dtmf = frame.get("dtmf") or frame.get("payload") or {}
        return dtmf.get("digit") if isinstance(dtmf, dict) else None

    def media_frame(self, stream_id: str, payload_b64: str) -> dict:  # noqa: ARG002
        return {"event": "media", "media": {"payload": payload_b64}}

    def mark_frame(self, stream_id: str, name: str) -> dict:  # noqa: ARG002
        return {"event": "mark", "mark": {"name": name}}

    def clear_frame(self, stream_id: str) -> dict:  # noqa: ARG002
        return {"event": "clear"}


_TWILIO = _TwilioDialect()
_TELNYX = _TelnyxDialect()


# ---------------------------------------------------------------------------
# Webhooks (call answer → connect the media stream)
# ---------------------------------------------------------------------------


def _external_base(request: Request) -> tuple[str, str]:
    """(scheme, host) as the outside world sees this server (proxy-aware)."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return proto, host


def _external_url(request: Request) -> str:
    proto, host = _external_base(request)
    url = f"{proto}://{host}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"
    return url


def _stream_ws_url(request: Request, path: str) -> str:
    proto, host = _external_base(request)
    ws_proto = "wss" if proto == "https" else "ws"
    return f"{ws_proto}://{host}{path}"


def _stream_xml(ws_url: str, params: dict[str, str], *, telnyx: bool) -> str:
    """TwiML/TeXML that answers the call into the bidirectional media WS."""
    attrs = f"url={quoteattr(ws_url)}"
    if telnyx:
        # Default TeXML mode is mp3 playback; RTP is the raw-audio mode. Pin
        # PCMU both ways so the bridge never has to sniff codecs.
        attrs += ' bidirectionalMode="rtp" codec="PCMU" bidirectionalCodec="PCMU"'
    children = "".join(
        f"<Parameter name={quoteattr(k)} value={quoteattr(v)} />" for k, v in params.items() if v
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Connect><Stream {attrs}>{children}</Stream></Connect></Response>"
    )


def _twilio_signature_ok(url: str, params: dict[str, str], signature: str, auth_token: str) -> bool:
    """Twilio request validation: base64(HMAC-SHA1(token, url + sorted params))."""
    payload = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature)


def _telnyx_signature_ok(body: bytes, timestamp: str, signature_b64: str, public_key_b64: str) -> bool:
    """Telnyx webhook validation: Ed25519 over ``{timestamp}|{raw body}``."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        logger.warning("telnyx_signature_skipped", hint="Ed25519 verification requires the cryptography package")
        return True
    try:
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
        key.verify(base64.b64decode(signature_b64), f"{timestamp}|".encode() + body)
        return True
    except (InvalidSignature, ValueError):
        return False


@router.post("/twilio/incoming")
async def twilio_incoming(request: Request) -> Response:
    body = await request.body()
    params = dict(parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True))
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if auth_token:
        signature = request.headers.get("x-twilio-signature", "")
        if not _twilio_signature_ok(_external_url(request), params, signature, auth_token):
            logger.warning("twilio_webhook_rejected", reason="bad signature")
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    else:
        logger.warning("twilio_webhook_unverified", hint="set TWILIO_AUTH_TOKEN to verify webhook signatures")
    logger.info("twilio_incoming_call", call_sid=params.get("CallSid"), from_=params.get("From"), to=params.get("To"))
    xml = _stream_xml(
        _stream_ws_url(request, "/voice/twilio/stream"),
        # Twilio's WS start frame carries no caller metadata; tunnel it.
        {"call_sid": params.get("CallSid", ""), "from": params.get("From", ""), "to": params.get("To", "")},
        telnyx=False,
    )
    return Response(content=xml, media_type="application/xml")


@router.post("/telnyx/incoming")
async def telnyx_incoming(request: Request) -> Response:
    body = await request.body()
    public_key = os.environ.get("TELNYX_PUBLIC_KEY")
    if public_key:
        signature = request.headers.get("telnyx-signature-ed25519", "")
        timestamp = request.headers.get("telnyx-timestamp", "")
        if not _telnyx_signature_ok(body, timestamp, signature, public_key):
            logger.warning("telnyx_webhook_rejected", reason="bad signature")
            raise HTTPException(status_code=403, detail="Invalid Telnyx signature")
    else:
        logger.warning("telnyx_webhook_unverified", hint="set TELNYX_PUBLIC_KEY to verify webhook signatures")
    params = dict(parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True))
    logger.info("telnyx_incoming_call", call_sid=params.get("CallSid"), from_=params.get("From"), to=params.get("To"))
    xml = _stream_xml(_stream_ws_url(request, "/voice/telnyx/stream"), {}, telnyx=True)
    return Response(content=xml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Media WebSocket bridge
# ---------------------------------------------------------------------------


@router.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket) -> None:
    await _media_ws(ws, _TWILIO)


@router.websocket("/telnyx/stream")
async def telnyx_stream(ws: WebSocket) -> None:
    await _media_ws(ws, _TELNYX)


async def _media_ws(ws: WebSocket, dialect: Any) -> None:
    from ..core.agent import Agent

    await ws.accept()
    logger.info("telephony_ws_connected", provider=dialect.name)

    runnable = ws.app.state.runnable
    if not isinstance(runnable, Agent):
        logger.error("telephony_ws_rejected", reason="runnable is not an Agent", type=type(runnable).__name__)
        await ws.close(code=1008, reason="Voice requires an Agent runnable")
        return

    guard = getattr(ws.app.state, "single_session_guard", None)
    if guard is None:
        await _serve_media_ws(ws, runnable, dialect)
        return

    if not guard.claim():
        logger.info("telephony_ws_rejected", reason="single-session server already served its session")
        await ws.close(code=1008, reason="Single-session server: a voice session was already served")
        return
    guard.mark_connected()
    try:
        await _serve_media_ws(ws, runnable, dialect)
    finally:
        await guard.finish()


async def _safe_close(ws: WebSocket, code: int, reason: str) -> None:
    try:
        await ws.close(code=code, reason=reason)
    except Exception:
        # Already disconnected — the close was only best-effort courtesy.
        pass


async def _serve_media_ws(ws: WebSocket, runnable: Any, dialect: Any) -> None:
    """One phone call: start frame → session → μ-law media both ways, until stop."""
    try:
        from ..voice import AudioOutput
        from ..voice.telephony import (
            TELEPHONY_SAMPLE_RATE,
            ULAW_SILENCE,
            PcmResampler,
            TelephonyPlaybackTracker,
            ulaw_decode,
            ulaw_encode,
        )
    except ImportError as e:
        logger.error("telephony_unavailable", error=str(e), hint="telephony requires the timbal[voice] extra")
        await _safe_close(ws, 1011, "telephony requires timbal[voice]")
        return

    from .voice import build_voice_session, merge_client_voice_overrides

    # -- Handshake: providers send `connected` then `start`; media only after.
    start_frame: dict | None = None
    deadline = time.monotonic() + 10.0
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=remaining)
            try:
                frame = json.loads(raw)
            except ValueError:
                continue
            if isinstance(frame, dict) and frame.get("event") == "start":
                start_frame = frame
                break
    except (TimeoutError, WebSocketDisconnect, RuntimeError):
        pass
    if start_frame is None:
        logger.warning("telephony_no_start_frame", provider=dialect.name)
        await _safe_close(ws, 1002, "expected a start frame")
        return

    stream_id = dialect.stream_id(start_frame) or ""
    info = dialect.start_info(start_frame)
    encoding = (info.get("encoding") or "audio/x-mulaw").lower()
    line_rate = int(info.get("sample_rate") or TELEPHONY_SAMPLE_RATE)
    if encoding not in _ULAW_ENCODINGS:
        # v1 speaks G.711 μ-law only; both webhooks pin it, so this means a
        # stream started outside our TwiML/TeXML with another codec.
        logger.error("telephony_unsupported_encoding", provider=dialect.name, encoding=encoding)
        await _safe_close(ws, 1003, f"unsupported media encoding: {encoding}")
        return
    logger.info(
        "telephony_call_started",
        provider=dialect.name,
        call_id=info.get("call_id"),
        from_=info.get("from"),
        to=info.get("to"),
        line_rate=line_rate,
    )

    custom = info.get("custom") or {}
    client_config = {k: v for k, v in custom.items() if k in _CONFIG_PARAM_KEYS and isinstance(v, str) and v}

    defaults: VoiceConfig = getattr(ws.app.state, "voice_config", None) or VoiceConfig()
    session_rate = int(merge_client_voice_overrides(defaults, client_config).sample_rate)

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    send_lock = asyncio.Lock()

    async def _send_frame(frame: dict) -> None:
        try:
            async with send_lock:
                await ws.send_json(frame)
        except Exception as e:
            logger.debug("telephony_send_skipped", provider=dialect.name, error=str(e))

    # -- Playback tracking: cumulative μ-law bytes sent ride as mark names;
    # a mark echo means everything before it has played (8 bytes/ms @ 8kHz).
    # After a clear, providers echo all outstanding marks for *unplayed*
    # audio — those must not count as acks.
    pending_marks: set[str] = set()
    ignored_marks: set[str] = set()
    clear_tasks: set[asyncio.Task] = set()
    out_ulaw = bytearray()  # downlink batch buffer (see _MIN_MEDIA_BYTES)

    def _on_clear() -> None:
        # Called by the session (same loop) at barge-in, before it resumes
        # listening — schedule the provider-side buffer drop right away.
        ignored_marks.update(pending_marks)
        pending_marks.clear()
        out_ulaw.clear()
        task = asyncio.get_running_loop().create_task(_send_frame(dialect.clear_frame(stream_id)))
        clear_tasks.add(task)
        task.add_done_callback(clear_tasks.discard)

    tracker = TelephonyPlaybackTracker(bytes_per_second=session_rate * 2, on_clear=_on_clear)
    session, meta = build_voice_session(runnable, defaults, client_config, playback_tracker=tracker)
    meta = {
        "transport": dialect.name,
        "call_id": info.get("call_id"),
        "from": info.get("from"),
        "to": info.get("to"),
        **meta,
    }
    session.recording_meta = meta

    up_resampler = PcmResampler(line_rate, session_rate) if line_rate != session_rate else None
    down_resampler = PcmResampler(session_rate, TELEPHONY_SAMPLE_RATE) if session_rate != TELEPHONY_SAMPLE_RATE else None

    sent_ulaw_bytes = 0

    async def _send_media(payload: bytes) -> None:
        nonlocal sent_ulaw_bytes
        await _send_frame(dialect.media_frame(stream_id, base64.b64encode(payload).decode("ascii")))
        sent_ulaw_bytes += len(payload)
        name = str(sent_ulaw_bytes)
        pending_marks.add(name)
        await _send_frame(dialect.mark_frame(stream_id, name))

    async def _push_downlink(pcm: bytes) -> None:
        data = down_resampler.process(pcm) if down_resampler else pcm
        out_ulaw.extend(ulaw_encode(data))
        if len(out_ulaw) >= _MIN_MEDIA_BYTES:
            payload = bytes(out_ulaw)
            out_ulaw.clear()
            await _send_media(payload)

    async def _flush_downlink() -> None:
        if not out_ulaw:
            return
        # Pad short tails to the provider minimum with μ-law silence.
        payload = bytes(out_ulaw).ljust(_MIN_MEDIA_BYTES, ULAW_SILENCE)
        out_ulaw.clear()
        await _send_media(payload)

    async def _recv_loop() -> None:
        """Provider frames → PCM into the session; mark echoes → playback acks."""
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                raw = msg.get("text")
                if not raw:
                    continue
                try:
                    frame = json.loads(raw)
                except ValueError:
                    continue
                event = frame.get("event")
                if event == "media":
                    payload = dialect.media_bytes(frame)
                    if payload:
                        pcm = ulaw_decode(payload)
                        if up_resampler:
                            pcm = up_resampler.process(pcm)
                        if pcm:
                            await audio_queue.put(pcm)
                elif event == "mark":
                    name = dialect.mark_name(frame)
                    if not name:
                        continue
                    if name in ignored_marks:
                        ignored_marks.discard(name)
                        continue
                    pending_marks.discard(name)
                    try:
                        played_ms = int(name) / (TELEPHONY_SAMPLE_RATE / 1000)
                    except ValueError:
                        continue
                    session.playback.on_playback_ack(played_ms)
                elif event == "stop":
                    logger.info("telephony_call_stopped", provider=dialect.name, call_id=info.get("call_id"))
                    break
                elif event == "dtmf":
                    logger.info("telephony_dtmf", provider=dialect.name, digit=dialect.dtmf_digit(frame))
                elif event == "error":
                    logger.warning("telephony_provider_error", provider=dialect.name, detail=str(frame)[:300])
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("telephony_recv_error", provider=dialect.name, error=str(e))
        finally:
            await audio_queue.put(b"")
            # Caller hung up (or the stream stopped): end the session now
            # instead of waiting for the STT silence timeout.
            await session.close()

    async def _phone_stream():
        while True:
            chunk = await audio_queue.get()
            if not chunk:
                break
            yield chunk

    recv_task = asyncio.create_task(_recv_loop())
    try:
        async with aclosing(session.run(_phone_stream())) as event_iter:
            async for event in event_iter:
                if isinstance(event, AudioOutput):
                    await _push_downlink(event.data)
                else:
                    # Any non-audio event marks a pause in synthesis: flush the
                    # sub-minimum tail so turn endings aren't clipped.
                    await _flush_downlink()
    finally:
        if not recv_task.done():
            recv_task.cancel()
        await asyncio.gather(recv_task, *clear_tasks, return_exceptions=True)
        try:
            await session.close()
        except Exception as e:
            logger.debug("telephony_session_close_suppressed", error=str(e))
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("telephony_ws_disconnected", provider=dialect.name)
