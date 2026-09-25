# Provider-reported voice usage

OpenAI voice adapters expose `VoiceUsageEvent` independently of audio and
transcript delivery. The framework reports usage, not prices or invoice totals.
A billing consumer applies its model-specific rates and persists the records.

```python
import asyncio
from timbal.voice import VoiceUsageEvent

usage_queue: asyncio.Queue[VoiceUsageEvent] = asyncio.Queue()
stt.add_usage_listener(usage_queue.put_nowait)
tts.add_usage_listener(usage_queue.put_nowait)
# Register before connect(); persist events from a separate queue consumer.
# Remove listeners when the application is finished with these adapters.
```

Listeners are synchronous and must not block. Exceptions are logged and isolated
from speech. Delivery is in-process, not durable; a billing application must drain
and persist its queue even after the voice transport disconnects. Registering the
same listener twice is harmless. Each listener receives its own event copy.

`VoiceSession.run()` also yields usage events, and the shared server transport
mapper forwards them as JSON with `type: "voice_usage"`. Session cleanup reports
known unfinished operations before `session_ended` when the iterator is consumed
to completion. For billing, use adapter listeners rather than relying on a client
connection surviving teardown. Other providers and custom adapters keep their
existing behavior; they do not emit usage unless they implement this capability.

## Event contract

- `provider`, `operation` (`stt` or `tts`), and `model` identify the resolved
  provider/model, including cross-provider default resolution.
- `usage_id` identifies one operation. Deduplicate persisted records by this key;
  STT retransmissions have the same key. Prefer a complete report over an earlier
  incomplete report. Keys are scoped to an adapter connection for STT and to an
  individual synthesis request for TTS.
- `request_id` contains the Speech API's `x-request-id`, when received. `item_id`
  contains the Realtime transcription item ID. Transcription content index is
  included in `usage_id`; it is not an LLM turn ID.
- `status: "complete"` means the provider supplied its terminal usage counters.
  `usage` preserves the provider object, including any token detail fields.
- `status: "incomplete"` means counters are missing, invalid, or unavailable after
  cancellation/error/closure. It is **not zero usage**. A partial or invalid
  provider object is retained for inspection; no token counts are manufactured.

OpenAI STT forwards `usage` from
`conversation.item.input_audio_transcription.completed`, even for empty or
out-of-order transcripts. Token-priced models report `input_tokens`,
`output_tokens`, `total_tokens`, and `input_token_details` (`text_tokens` and
`audio_tokens`). Duration-priced models report `type: "duration"` and `seconds`.
Input audio and text have distinct rates; do not price `total_tokens` at one rate.
The consumer must check the breakdown needed by its rate card before pricing.

OpenAI mini-TTS uses `stream_format: "sse"`. Audio deltas are decoded and resampled
as they arrive; `speech.audio.done.usage` supplies input text and output audio
token counts. The usage object has no `type` discriminator in this API. A done
event with missing usage is incomplete. EOF before done or an explicit provider
error fails synthesis and reports incomplete usage. Audio already generated may
be billed even when the caller never hears it.

Legacy `tts-1` / `tts-1-hd` keep raw PCM streaming and their existing character
metering path; they do not support SSE. This change does not add estimated
character counts to provider-reported events or change other speech providers.

Known committed STT items without final usage are reported incomplete at receiver
shutdown. Audio that never acquired a provider item ID cannot be attributed this
way. Cancelled TTS requests are reported incomplete without draining extra audio
just to obtain usage. Network failures can prevent authoritative counters; these
records require an explicit fallback policy. Provider Costs APIs can reconcile
aggregate spend but do not supply exact per-Timbal-session receipts.

## Composer integration

A separate Composer update must subscribe to these events, persist/deduplicate
usage, price the appropriate token categories or reported duration, and replace
its duration estimates when complete quantities are available. Do not charge both
the existing estimate and the reported usage. Retain estimates explicitly marked
as such for incomplete or unsupported operations. No pricing, ledger migration,
Composer repin, or provider invoice reconciliation is performed by this change.

References:

- [OpenAI transcription usage](https://developers.openai.com/api/docs/guides/voice-latency-cost#input-transcription-costs)
- [Speech streaming request](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create)
- [Speech completion usage](https://platform.openai.com/docs/api-reference/audio#speech-audio-done-event)
- [Organization Costs API](https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage/methods/costs)
