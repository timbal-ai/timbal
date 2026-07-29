# Ambience preset provenance

All presets are CC0 (attribution-free), mono 16 kHz PCM16 WAV,
loudness-normalized to −30 dBFS RMS, loop-ready (either recorded seamless or
with the end→start crossfade baked in by `scripts/prepare_ambience.py`).

Assets are not bundled in the package — they're hosted at
`https://timbalusercontent.com/assets/voice/ambience/{name}.wav` and
downloaded lazily on first use, verified against the sha256 pinned in
`timbal.voice.ambience.PRESETS`. Treat CDN files as immutable: a changed
track gets a new name.

| File | Source | License | Processing |
|---|---|---|---|
| `office.wav` | ["Office_Ambience_Interior_quiet" by joseegn](https://freesound.org/people/joseegn/sounds/752611/) + ["quiet typing on laptop computer" by lmz36](https://freesound.org/people/lmz36/sounds/721033/) | CC0 + CC0 | typing layered at −5 dB under the room bed, peak-capped, crossfade baked |
| `call-center.wav` | ["Busy Office No People Loop" by Fupicat](https://freesound.org/people/Fupicat/sounds/534123/) (itself built from CC0 clips) | CC0 | seamless source, resample + normalize only |
| `cafe.wav` | ["Restaurant walla" by hz37](https://freesound.org/people/hz37/sounds/792481/) | CC0 | low-passed at 2.5 kHz to smear speech intelligibility, crossfade baked |
| `city.wav` | ["Seamless City Loop" by qubodup](https://freesound.org/people/qubodup/sounds/223093/) | CC0 (page badge; description mentions CC BY — attribution given anyway) | seamless source, resample + normalize only |
| `typing.wav` | ["Keyboard Typing & Clicking Ambience" by Sayaka04](https://freesound.org/people/Sayaka04/sounds/851269/) | CC0 | 45 s excerpt, peak-capped, crossfade baked |

When adding recordings: CC0 only (attribution-free), voiceless or unintelligible
murmur (intelligible chatter leaking back through the caller's mic gets
transcribed — low-pass murmur at ~2.5 kHz), processed through
`scripts/prepare_ambience.py`, and listed here with source URL and license.
