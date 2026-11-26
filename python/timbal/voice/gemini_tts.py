"""Gemini TTS over ``streamGenerateContent`` (SSE), no provider SDK required.

Gemini speech models take a whole prompt, not a live text stream, and keep no
context between requests. So each *clause* is its own request and its own
intonation. A reply is cut at sentence ends and at commas (see
:func:`split_clauses`), the next clauses are requested while the current one
plays (``prefetch``), and audio is handed to the session in order.

The prompt is a director's brief followed by the transcript. The brief says
who speaks, the scene, the pace and the accent. Keep the layout: telling the
model to "read only the transcript" makes it read the notes aloud in a large
share of calls, while a brief with a ``TRANSCRIPCIÓN`` header leaked none in
testing.

Voices are Gemini prebuilt names (``Fenrir``, ``Puck`` ...). A voice or model
id from another provider is ignored, so a session that switches provider on a
server configured for ElevenLabs does not send an ElevenLabs id to Google.

``AudioOutputConfig.extra`` knobs (all optional):

``accent``
    Free text for the brief's accent line. Default: peninsular Spanish.
``director_notes``
    Replaces the whole brief, keeping the transcript header.
``prefetch``
    Clause requests in flight at once (default 2).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
from collections.abc import AsyncIterator

import httpx
import structlog
from pydantic import SecretStr

from .providers import AudioOutputConfig, TextToSpeech, TTSStream

logger = structlog.get_logger("timbal.voice.gemini_tts")

DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_VOICE = "Fenrir"
GEMINI_VOICES = frozenset(
    {
        "zephyr", "puck", "charon", "kore", "fenrir", "leda", "orus", "aoede", "callirrhoe", "autonoe",
        "enceladus", "iapetus", "umbriel", "algieba", "despina", "erinome", "algenib", "rasalgethi",
        "laomedeia", "achernar", "alnilam", "schedar", "gacrux", "pulcherrima", "achird", "zubenelgenubi",
        "vindemiatrix", "sadachbia", "sadaltager", "sulafat",
    }
)  # fmt: skip
"""The 30 prebuilt voices, lower-cased. Only used to tell a Gemini voice from another provider's id."""

GEMINI_RATE = 24_000
"""Gemini returns 24 kHz PCM16 mono."""

_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
_MIN_CLAUSE_CHARS = 18
_TERMINALS = ".!?…;:"
_FRAME_SECS = 0.1
"""Audio leaves in 100 ms frames so a barge-in cuts within 100 ms, not at the end of a clause."""
_SPLIT = re.compile(r"(?<=[.!?…;:])\s+|(?<=,)\s+(?=\S{3,})")

DEFAULT_ACCENT = "castellano peninsular de España, natural, sin acento latinoamericano"
_BRIEF = """# PERFIL DE AUDIO: agente de atención al cliente en España
## ESCENA: llamada telefónica real a un servicio de atención al cliente.
### NOTAS DEL DIRECTOR
Estilo: natural, cercano y profesional, como una persona real al teléfono. Sin tono de locutor ni dramatismo.
Ritmo: ágil y fluido, pausas breves. Las dudas escritas ('eh...', 'mm...', 'a ver,') se dicen tal cual, como alguien que piensa en voz alta, sin exagerar.
Acento: {accent}"""
_TRANSCRIPT_HEADER = "#### TRANSCRIPCIÓN"


def effective_tts_model(config: AudioOutputConfig) -> str:
    """The Gemini model to call; another provider's ``tts_model`` default is ignored."""
    m = (config.model or "").strip()
    return m if m.startswith("gemini-") else DEFAULT_TTS_MODEL


def effective_voice(config: AudioOutputConfig) -> str:
    """The prebuilt voice to use; an id that is not a Gemini voice name falls back to the default."""
    v = (config.voice or "").strip()
    return v.capitalize() if v.lower() in GEMINI_VOICES else DEFAULT_VOICE


def build_prompt(text: str, *, accent: str | None = None, director_notes: str | None = None) -> str:
    """The director's brief plus the transcript, in the layout that does not leak the notes."""
    brief = (director_notes or "").strip() or _BRIEF.format(accent=(accent or "").strip() or DEFAULT_ACCENT)
    return f"{brief}\n{_TRANSCRIPT_HEADER}\n{text}"


def split_clauses(text: str, *, min_chars: int = _MIN_CLAUSE_CHARS) -> tuple[list[str], str]:
    """Split off what can be spoken now. Returns ``(complete clauses, unfinished rest)``."""
    parts = _SPLIT.split(text)
    if len(parts) == 1:
        return [], text
    out: list[str] = []
    carry = ""
    for p in parts[:-1]:
        carry = f"{carry} {p}".strip() if carry else p
        if len(carry) >= min_chars:
            out.append(carry)
            carry = ""
    rest = f"{carry} {parts[-1]}".strip() if carry else parts[-1]
    return out, rest


def _resolve_api_key(explicit: str | SecretStr | None) -> str:
    if explicit is not None:
        key = explicit.get_secret_value() if isinstance(explicit, SecretStr) else str(explicit)
        if key:
            return key
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not key:
        raise ValueError("Gemini TTS needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")
    return key


class GeminiStreamTTS(TextToSpeech):
    """Gemini TTS: one SSE request per clause, ordered playback, bounded read-ahead."""

    provider_id = "gemini"

    def __init__(
        self, api_key: str | SecretStr | None = None, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._api_key_explicit = api_key
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._model = DEFAULT_TTS_MODEL
        self._voice = DEFAULT_VOICE
        self._accent: str | None = None
        self._notes: str | None = None
        self._prefetch = 2
        self._rate = GEMINI_RATE
        self._streams: set[_GeminiTTSStream] = set()

    async def connect(self, config: AudioOutputConfig) -> None:
        if config.encoding not in ("pcm_s16le", "linear16", ""):
            raise ValueError("Gemini TTS voice sessions require pcm_s16le output.")
        await self.close()
        key = _resolve_api_key(self._api_key_explicit)
        extra = config.extra
        self._model = effective_tts_model(config)
        self._voice = effective_voice(config)
        self._accent = str(extra["accent"]) if extra.get("accent") else None
        self._notes = str(extra["director_notes"]) if extra.get("director_notes") else None
        self._prefetch = max(1, int(extra.get("prefetch", 2)))
        self._rate = int(config.sample_rate)
        # The key travels in a header: httpx logs URLs, and a key in the query string would land in the logs.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=5.0),
            headers={"x-goog-api-key": key},
            transport=self._transport,
        )

    async def close(self) -> None:
        await asyncio.gather(*(s.abort() for s in list(self._streams)))
        if self._client is not None:
            client, self._client = self._client, None
            with contextlib.suppress(Exception):
                await client.aclose()

    def open_stream(self) -> TTSStream:
        if self._client is None:
            raise RuntimeError("Call connect() before open_stream().")
        stream = _GeminiTTSStream(self)
        self._streams.add(stream)
        return stream

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """One whole text in a single request (fixed lines such as greetings)."""
        if not text.strip():
            return
        async for frame in self._clause_with_retry(text.strip()):
            yield frame

    def _frames(self, pcm: bytes) -> list[bytes]:
        step = max(2, int(self._rate * _FRAME_SECS) * 2)
        return [pcm[i : i + step] for i in range(0, len(pcm), step)]

    async def _clause_audio(self, text: str) -> AsyncIterator[bytes]:
        """Audio for one text at the session rate. Raises if Gemini fails or sends no audio."""
        if self._client is None:
            raise RuntimeError("Call connect() before synthesizing.")
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_prompt(text, accent=self._accent, director_notes=self._notes)}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}},
            },
        }
        resampler = None
        if self._rate != GEMINI_RATE:
            from .telephony import PcmResampler

            resampler = PcmResampler(GEMINI_RATE, self._rate)
        carry = b""
        got = 0
        async with self._client.stream("POST", _URL.format(model=self._model), params={"alt": "sse"}, json=body) as r:
            if r.status_code >= 400:
                detail = (await r.aread()).decode(errors="replace")[:300]
                raise RuntimeError(f"Gemini TTS HTTP {r.status_code}: {detail}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                for cand in data.get("candidates") or []:
                    for part in (cand.get("content") or {}).get("parts") or []:
                        inline = part.get("inlineData") or part.get("inline_data")
                        if not inline:
                            continue
                        pcm = carry + base64.b64decode(inline["data"])
                        cut = len(pcm) - len(pcm) % 2
                        pcm, carry = pcm[:cut], pcm[cut:]
                        if not pcm:
                            continue
                        got += len(pcm)
                        if resampler is not None:
                            pcm = resampler.process(pcm)
                        for frame in self._frames(pcm):
                            yield frame
        if not got:
            raise RuntimeError("Gemini TTS returned no audio")
        if resampler is not None:
            for frame in self._frames(resampler.flush()):
                yield frame

    async def _clause_with_retry(self, text: str) -> AsyncIterator[bytes]:
        """One retry when a request fails before any audio (a 5xx, or a text-only answer Gemini gives now and then)."""
        for attempt in (1, 2):
            sent = False
            try:
                async for frame in self._clause_audio(text):
                    sent = True
                    yield frame
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if sent or attempt == 2:
                    raise
                logger.warning("gemini_tts_retry", error=str(exc)[:200], text_preview=text[:60])


class _GeminiTTSStream(TTSStream):
    """One reply: every closed clause is requested as soon as it arrives and plays in order."""

    def __init__(self, tts: GeminiStreamTTS) -> None:
        self._tts = tts
        self._buf = ""
        self._jobs: asyncio.Queue[asyncio.Queue[bytes | Exception | None] | None] = asyncio.Queue()
        self._queues: list[asyncio.Queue[bytes | Exception | None]] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._sem = asyncio.Semaphore(tts._prefetch)
        self._ended = False
        self._aborted = False
        self._audio_done = False
        self._played = 0
        self._progress = asyncio.Event()

    def _launch(self, text: str) -> None:
        out: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        self._queues.append(out)
        self._jobs.put_nowait(out)

        async def run() -> None:
            try:
                async with self._sem:
                    async for frame in self._tts._clause_with_retry(text):
                        if self._aborted:
                            return
                        out.put_nowait(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - handed to audio(), which raises it in the consumer's task
                out.put_nowait(exc)
            finally:
                out.put_nowait(None)

        self._tasks.append(asyncio.create_task(run()))

    async def feed(self, text: str) -> None:
        if self._ended or self._aborted or not text:
            return
        self._buf += text
        done, self._buf = split_clauses(self._buf)
        # The session feeds whole sentences: they end in punctuation and the next space arrives with the
        # next sentence. Waiting for that space would delay the first audio, so a long-enough tail that ends
        # a clause is spoken now.
        rest = self._buf.strip()
        if len(rest) >= _MIN_CLAUSE_CHARS and rest[-1] in _TERMINALS:
            done.append(rest)
            self._buf = ""
        for clause in done:
            self._launch(clause)

    async def flush(self) -> bool:
        if not self._ended and not self._aborted and self._buf.strip():
            self._launch(self._buf.strip())
        self._buf = ""
        return True

    async def drain(self) -> bool:
        """Wait until everything requested so far has played out (text spoken before a tool call)."""
        await self.flush()
        target = len(self._queues)
        while self._played < target and not self._aborted and not self._audio_done:
            self._progress.clear()
            await self._progress.wait()
        return not self._aborted and not self._audio_done

    async def end(self) -> None:
        if self._ended or self._aborted:
            return
        await self.flush()
        self._ended = True
        self._jobs.put_nowait(None)

    async def abort(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        for t in self._tasks:
            t.cancel()
        for q in self._queues:
            q.put_nowait(None)
        self._jobs.put_nowait(None)
        self._progress.set()
        self._tts._streams.discard(self)

    async def audio(self) -> AsyncIterator[bytes]:
        try:
            while not self._aborted:
                q = await self._jobs.get()
                if q is None or self._aborted:
                    return
                while True:
                    item = await q.get()
                    if self._aborted:
                        return
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise RuntimeError(f"Gemini TTS failed: {item}") from item  # noqa: TRY004
                    yield item
                self._played += 1
                self._progress.set()
        finally:
            self._audio_done = True
            self._progress.set()
            self._tts._streams.discard(self)
