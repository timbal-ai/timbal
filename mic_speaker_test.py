#!/usr/bin/env python3
"""Record for 3 seconds, wait, then playback."""

import asyncio
import pyaudio

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 5

async def record_audio() -> bytes:
    """Record audio for 3 seconds."""
    audio = pyaudio.PyAudio()

    # Find microphone device
    print("Available audio devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']})")

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print(f"🎤 Recording for {RECORD_SECONDS} seconds... SPEAK NOW!")

    frames = []
    chunks_to_record = int(SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)

    for i in range(chunks_to_record):
        audio_data = stream.read(CHUNK_SIZE)
        frames.append(audio_data)
        # Small async yield to keep it responsive
        if i % 10 == 0:
            await asyncio.sleep(0.001)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("✅ Recording complete!")
    return b''.join(frames)

async def play_audio(audio_data: bytes):
    """Play recorded audio to speakers."""
    audio = pyaudio.PyAudio()

    # Find speaker device
    print("\nAvailable output devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  {i}: {info['name']} (outputs: {info['maxOutputChannels']})")

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("🔊 Playing back recorded audio...")

    # Add a small silence at the beginning to avoid click
    silence = b'\x00\x00' * CHUNK_SIZE
    stream.write(silence)

    # Split audio data into chunks and play
    for i in range(0, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[i:i + CHUNK_SIZE]
        # Pad chunk if it's shorter than CHUNK_SIZE
        if len(chunk) < CHUNK_SIZE:
            chunk += b'\x00\x00' * ((CHUNK_SIZE - len(chunk)) // 2)
        stream.write(chunk)
        await asyncio.sleep(0.001)  # Small yield

    # Add small silence at the end
    stream.write(silence)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("✅ Playback complete!")

async def main():
    """Main function - record, wait, then play."""
    print("🎵 Starting record and playback test...")

    # Record for 3 seconds
    recorded_audio = await record_audio()

    # Add delay
    print("⏳ Waiting 2 seconds before playback...")
    await asyncio.sleep(2)

    # Play the recorded audio
    await play_audio(recorded_audio)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")