"""
audio_async_bytes.py

- mic_byte_stream(...): generador asíncrono que emite bytes del micrófono.
- play_from_async_byte_source(...): reproduce bytes en tiempo real en los altavoces.
- demo_loopback(): ejemplo que conecta micrófono -> altavoces (usa cascos para evitar acople).

Probado con Python 3.9+.
"""

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import wave
from scipy.signal import resample
from audiomentations import AddBackgroundNoise

import numpy as np
import sounddevice as sd

from audio_local import VoiceAgent



# -----------------------------
# 1) Generador asíncrono de MIC
# -----------------------------
async def mic_byte_stream(
    samplerate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",     # "int16", "int24", "int32", "float32", etc.
    blocksize: int = 320,    
    device: Optional[int] = None,
) -> AsyncIterator[bytes]:
    """
    Captura el micrófono en tiempo real y va emitiendo 'bytes' por bloques.
    Usa RawInputStream para obtener bytes directamente.

    Yields:
        bytes: bloque crudo de audio con tamaño ≈ blocksize * channels * bytes_per_sample
    """
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)  # pequeño búfer anti-jitter

    def callback(indata, frames, time_info, status):
        if status:
            pass
        data = bytes(indata)
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    stream = sd.RawInputStream(
        samplerate=samplerate,
        channels=channels,
        dtype=dtype,
        blocksize=blocksize,
        callback=callback,
        device=device,
    )

    with stream:
        while True:
            chunk = await queue.get()
            yield chunk


# ----------------------------------------------
# 2) Reproductor asíncrono desde un async source
# ----------------------------------------------
@dataclass
class _ByteAccumulator:
    """
    Simple buffer that keeps latency low by capping maximum size.
    Assumes continuous data flow after initial startup.
    """
    def __init__(self, max_buffer_bytes: int = 1600):
        self._buf = bytearray()
        self._max_buffer_bytes = max_buffer_bytes

    def push(self, data: bytes) -> None:
        """Add data, dropping oldest if buffer gets too large."""
        self._buf.extend(data)
        
        if len(self._buf) > self._max_buffer_bytes:
            excess = len(self._buf) - self._max_buffer_bytes
            del self._buf[:excess]

    def pull_exact(self, n: int) -> bytes:
        """Return exactly n bytes (padding only during startup/end)."""
        if len(self._buf) >= n:
            out = bytes(self._buf[:n])
            del self._buf[:n]
            return out
        else:
            out = bytes(self._buf)
            self._buf.clear()
            out += b"\x00" * (n - len(out))
            return out

    def __len__(self):
        return len(self._buf)


async def play_from_async_byte_source(
    source: AsyncIterator[bytes],
    samplerate,
    channels,
    dtype,
    blocksize,
    device: int | None = None,
) -> None:
    """
    Lee bytes de un generador asíncrono (source) y los reproduce en tiempo real.
    Usa RawOutputStream y un callback que va consumiendo un búfer de bytes compartido.
    """
    bytes_per_sample = np.dtype(dtype).itemsize
    bytes_per_frame = bytes_per_sample * channels
    finished = False

    acc = _ByteAccumulator()
    lock = threading.Lock()   # protegeremos el búfer entre el hilo de audio y el feeder async

    def audio_callback(outdata, frames, time_info, status):
        if status:
            pass
        need = frames * bytes_per_frame
        with lock:
            chunk = acc.pull_exact(need)
        outdata[:] = chunk

    async def feeder():
        nonlocal finished
        try:
            async for chunk in source:
                if not chunk:
                    continue
                with lock:
                    acc.push(chunk)
        finally:
            finished = True


    feeder_task = asyncio.create_task(feeder())

    with sd.RawOutputStream(
        samplerate=samplerate,
        channels=channels,
        dtype=dtype,
        blocksize=blocksize,
        callback=audio_callback,
        device=device,
    ):
        while True:
            await asyncio.sleep(0.0)
            with lock:
                remaining = len(acc)
            if finished and remaining == 0:
                break

    await feeder_task


async def demo_loopback(samplerate, channels, dtype, blocksize):
    """
    Conecta el micrófono a los altavoces en tiempo real.
    Usa cascos para evitar acople (feedback).
    Pulsa Ctrl+C para salir.
    """
    async def mic_source():
        async for b in mic_byte_stream(
            samplerate=samplerate, channels=channels, dtype=dtype, blocksize=blocksize
        ):
            yield b        

    async def pcm_chunk_generator(wave_file: str, chunk_size=blocksize):
        with wave.open(wave_file, "rb") as wf:
            sampwidth = wf.getsampwidth()

            while True:
                chunk = wf.readframes(chunk_size // sampwidth)
                if not chunk:
                    break
                yield chunk
                await asyncio.sleep(0)  # Yield control to event loop

    def check_audio(file1: str, file2: str):
        w1 = wave.open(file1, 'rb')
        w2 = wave.open(file2, 'rb')

        # Verificar que tengan el mismo formato
        assert w1.getframerate() == w2.getframerate(), "Sample rates diferentes"
        assert w1.getsampwidth() == w2.getsampwidth(), "Profundidad de bits diferente"
        assert w1.getnchannels() == w2.getnchannels(), "Número de canales diferente"


    input_stream = mic_source()


    voice_agent = VoiceAgent(
        name="local_voice_agent",
        model="openai/gpt-4.1-mini",
        # system_prompt=system_prompt,
        language="en",  
        audio_format="pcm16", 
        vad_prefix_padding_ms=250,
        vad_silence_duration_ms=250,
        elevenlabs_voice_type="21m00Tcm4TlvDq8ikWAM",  
        elevenlabs_output_format="pcm_16000",
    )
    
    voice_agent._twilio_ws = None
    voice_agent._twilio_stream_sid = None

    async with voice_agent.session(input_stream=input_stream) as agent_output:
            await play_from_async_byte_source(agent_output, samplerate=samplerate, channels=channels, dtype=dtype, blocksize=blocksize)

# -----------------------------
# 4) Ejecución directa
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo asíncrona audio bytes")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="int16")
    parser.add_argument("--blocksize", type=int, default=320)
    parser.add_argument("--mode", choices=["loopback"], default="loopback")
    args = parser.parse_args()

    try:
        if args.mode == "loopback":
            asyncio.run(
                demo_loopback(
                    samplerate=args.samplerate,
                    channels=args.channels,
                    dtype=args.dtype,
                    blocksize=args.blocksize,
                )
            )
    except KeyboardInterrupt:
        print("\n👋 Salida limpia.")
