"""Gemini TTS over ``streamGenerateContent`` (SSE), no provider SDK required.

Gemini speech models take a whole prompt, not a live text stream, and keep no
context between requests: each request is a new take, with its own pitch and
energy. So a reply is spoken in as few requests as its text allows. Text is
cut after a sentence, never at a comma (see :func:`split_sentences`). A
sentence waits ``hold_ms`` for the ones right behind it, and the sentences that
arrive while a request is generating wait for it and go together. With a model
that writes a reply in one burst that is one request per reply; with a slow one
it is the first sentence and then the rest. Audio is handed to the session in
order.

The prompt is the text to speak and nothing else. These models perform what
surrounds the text. Given a director's brief (a profile, a scene, style notes)
they say words of their own before the line ("vale, a ver, déjame que lo
mire...") or read the notes aloud. Measured on ``gemini-3.8-flash-tts`` with
``benchmarks/voice/tts_fidelity.py``: with the brief this adapter used to
send, 10 of 147 requests said words that were not in the text, up to 87 of
them; with the text alone, 0 of 950, questions and orders ("write a poem")
included. No wording of the brief tried got to zero. Language and accent are a
request field instead (``language_code``), which cannot be read aloud.

Voices are Gemini prebuilt names (``Fenrir``, ``Puck`` ...). A voice or model
id from another provider is ignored, so a session that switches provider on a
server configured for ElevenLabs does not send an ElevenLabs id to Google.

``AudioOutputConfig.extra`` knobs (all optional):

``language_code``
    BCP-47 tag sent as ``speechConfig.languageCode`` (``"es-ES"``, ``"ca-ES"``,
    ``"en-US"``). Without it Gemini takes the language from the text, and the
    accent then changes from one request to the next: Spanish came out
    Latin American in about 3 requests of 10. Name the region: a bare ``"es"``
    does not pin the accent either. Text in another language is still read in
    its own. A tag Gemini refuses is dropped for the rest of the call.
``accent`` / ``director_notes``
    Send a director's brief before the text: ``accent`` is the accent line of
    the built-in brief (the way to a regional accent no tag names, such as
    Andalusian), ``director_notes`` replaces the whole brief. Either one brings
    back the added words measured above.
``hold_ms``
    How long a sentence waits for the next before it is requested alone
    (default 200; 0 = not at all). The end of the reply ends the wait, so a
    reply that is already whole is not held. This is the most a reply's first
    audio is delayed, in exchange for one voice from its start to its end.
``prefetch``
    Requests in flight at once (default 2).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator

import httpx
import structlog
from pydantic import SecretStr

from .language import normalize_language_tag
from .providers import AudioOutputConfig, TextToSpeech, TTSStream
from .segments import sentence_boundary_ends

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
_MIN_REQUEST_CHARS = 18
"""A shorter opener ("Perfecto, Marta.") waits for the sentence after it: one take, not two."""
_SENTENCE_ENDS = ".!?\u061f\uff1f\uff01"
"""What closes a sentence here. A colon or a semicolon does not: what follows belongs to the same take."""
_FRAME_SECS = 0.1
"""Audio leaves in 100 ms frames so a barge-in cuts within 100 ms, not at the end of a request."""

_BRIEF = """# PERFIL DE AUDIO: agente de atención al cliente en España
## ESCENA: llamada telefónica real a un servicio de atención al cliente.
### NOTAS DEL DIRECTOR
Estilo: natural, cercano y profesional, como una persona real al teléfono. Sin tono de locutor ni dramatismo.
Ritmo: ágil y fluido, pausas breves. Las dudas escritas ('eh...', 'mm...', 'a ver,') se dicen tal cual, como alguien que piensa en voz alta, sin exagerar.
Acento: {accent}"""
_TRANSCRIPT_HEADER = "#### TRANSCRIPCIÓN"


class GeminiTTSError(RuntimeError):
    """Gemini answered a synthesis request with an HTTP error."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Gemini TTS HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def effective_tts_model(config: AudioOutputConfig) -> str:
    """The Gemini model to call; another provider's ``tts_model`` default is ignored."""
    m = (config.model or "").strip()
    return m if m.startswith("gemini-") else DEFAULT_TTS_MODEL


def effective_voice(config: AudioOutputConfig) -> str:
    """The prebuilt voice to use; an id that is not a Gemini voice name falls back to the default."""
    v = (config.voice or "").strip()
    return v.capitalize() if v.lower() in GEMINI_VOICES else DEFAULT_VOICE


def build_prompt(text: str, *, accent: str | None = None, director_notes: str | None = None) -> str:
    """What one request sends: the text alone, or a brief and then the text when the agent asked for a brief."""
    notes = (director_notes or "").strip()
    accent = (accent or "").strip()
    if not notes and not accent:
        return text
    return f"{notes or _BRIEF.format(accent=accent)}\n{_TRANSCRIPT_HEADER}\n{text}"


def split_sentences(text: str, *, min_chars: int = _MIN_REQUEST_CHARS) -> tuple[str, str]:
    """Split off what can be spoken now. Returns ``(the complete sentences, the unfinished rest)``.

    The sentences come back as one text, for one request. Nothing is cut inside a sentence: not at a comma, not at
    a colon, and not at the period of ``Sr.`` or ``3.14`` (the session's own boundary rules). Sentences that do not
    add up to ``min_chars`` are not ready yet.
    """
    body = text.rstrip()
    # A word after the text gives its last terminator the lookahead a sentence boundary needs.
    ends = [end for end in sentence_boundary_ends(f"{body} x") if body[:end].rstrip()[-1:] in _SENTENCE_ENDS]
    if not ends:
        return "", text
    cut = min(ends[-1], len(body))
    ready = body[:cut].strip()
    if len(ready) < min_chars:
        return "", text
    return ready, (text[cut:].lstrip() if cut < len(body) else "")


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
    """Gemini TTS: one SSE request per group of sentences, ordered playback, bounded read-ahead."""

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
        self._language_code: str | None = None
        self._prefetch = 2
        self._hold = 0.2
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
        self._language_code = (
            normalize_language_tag(str(extra["language_code"])) if extra.get("language_code") else None
        )
        if extra.get("language_code") and self._language_code is None:
            logger.warning("gemini_tts_language_code_ignored", reason="not a language tag")
        self._prefetch = max(1, int(extra.get("prefetch", 2)))
        self._hold = max(0.0, float(extra.get("hold_ms", 200))) / 1000
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
        text = " ".join(text.split())
        if not text:
            return
        async for frame in self._request_with_retry(text):
            yield frame

    def _frames(self, pcm: bytes) -> list[bytes]:
        step = max(2, int(self._rate * _FRAME_SECS) * 2)
        return [pcm[i : i + step] for i in range(0, len(pcm), step)]

    async def _request_audio(self, text: str) -> AsyncIterator[bytes]:
        """Audio for one text at the session rate. Raises if Gemini fails or sends no audio."""
        if self._client is None:
            raise RuntimeError("Call connect() before synthesizing.")
        speech: dict[str, object] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}}
        if self._language_code:
            speech["languageCode"] = self._language_code
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_prompt(text, accent=self._accent, director_notes=self._notes)}],
                }
            ],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": speech},
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
                raise GeminiTTSError(r.status_code, detail)
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

    async def _request_with_retry(self, text: str) -> AsyncIterator[bytes]:
        """One retry when a request fails before any audio (a 5xx, or a text-only answer Gemini gives now and then).

        A language tag Gemini refuses is not a reason to lose the call's voice: the request goes again without the
        tag, and so does every later one.
        """
        for attempt in (1, 2):
            sent = False
            try:
                async for frame in self._request_audio(text):
                    sent = True
                    yield frame
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if sent or attempt == 2:
                    raise
                if isinstance(exc, GeminiTTSError) and exc.status == 400 and "language_code" in exc.detail:
                    logger.warning("gemini_tts_language_code_refused", language_code=self._language_code)
                    self._language_code = None
                    continue
                logger.warning("gemini_tts_retry", error=str(exc)[:200], text_preview=text[:60])


class _GeminiTTSStream(TTSStream):
    """One reply: its complete sentences, in as few requests as the pace of the text allows, played in order."""

    def __init__(self, tts: GeminiStreamTTS) -> None:
        self._tts = tts
        self._buf = ""
        self._jobs: asyncio.Queue[asyncio.Queue[bytes | Exception | None] | None] = asyncio.Queue()
        self._queues: list[asyncio.Queue[bytes | Exception | None]] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._sem = asyncio.Semaphore(tts._prefetch)
        self._generating = 0
        self._hold_timer: asyncio.TimerHandle | None = None
        self._first_text_at: float | None = None
        self._ended = False
        self._aborted = False
        self._audio_done = False
        self._played = 0
        self._progress = asyncio.Event()

    def _launch(self, text: str) -> None:
        text = " ".join(text.split())  # one line: a reply's line breaks are not part of what is said
        if not text:
            return
        out: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        self._queues.append(out)
        self._jobs.put_nowait(out)
        logger.debug(
            "gemini_tts_request",
            request=len(self._queues),
            chars=len(text),
            since_first_text_ms=round((time.monotonic() - (self._first_text_at or time.monotonic())) * 1000),
        )

        async def run() -> None:
            try:
                async with self._sem:
                    async for frame in self._tts._request_with_retry(text):
                        if self._aborted:
                            return
                        out.put_nowait(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - handed to audio(), which raises it in the consumer's task
                out.put_nowait(exc)
            finally:
                out.put_nowait(None)
                self._generating -= 1
                self._launch_ready()

        self._generating += 1
        self._tasks.append(asyncio.create_task(run()))

    def _launch_ready(self) -> None:
        """Request the sentences that are complete, unless a request is still generating.

        Audio is generated about three times faster than it plays, so the sentences that arrive while one request
        runs can wait for it and go together: what follows a reply's first request is one more take, however many
        sentences it has, and no audio is later for it.
        """
        self._stop_hold()
        if self._generating or self._ended or self._aborted:
            return
        ready, self._buf = split_sentences(self._buf)
        if ready:
            self._launch(ready)

    async def feed(self, text: str) -> None:
        if self._ended or self._aborted or not text:
            return
        # The session feeds a reply in segments it has already cut where speech can break. What is complete
        # goes in one request, as it came: cutting it again at each sentence, or at a comma, gave every piece a
        # take of its own, and the voice changed from one to the next.
        if self._first_text_at is None:
            self._first_text_at = time.monotonic()
        self._buf += text
        if self._generating or self._hold_timer is not None or not split_sentences(self._buf)[0]:
            return  # it goes with what is waiting, or it is not a sentence yet
        if self._tts._hold <= 0:
            self._launch_ready()
        else:
            # A model writes the sentences of a short reply within milliseconds of each other. Asked for one by
            # one they are as many takes; held this long they are one, and ``flush()`` (the reply's end, a tool
            # call) cuts the wait short.
            self._hold_timer = asyncio.get_running_loop().call_later(self._tts._hold, self._launch_ready)

    def _stop_hold(self) -> None:
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None

    async def flush(self) -> bool:
        self._stop_hold()
        if not self._ended and not self._aborted:
            self._launch(self._buf)
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
        self._stop_hold()
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
