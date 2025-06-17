import asyncio
import httpx
import time


async def main():
    voice_id = "56AoDkrOh6qfVPDXZ7Pt"
    api_key = "sk_636eb93a7e6d459bd5bea5c5d365370fabf34564cabbbfcb" 

    all_text_segments = [
        "In the bustling heart of the city, "
        "beneath skyscrapers that touch the clouds and among streets alive with the rhythm of footsteps and distant car horns, "
        "a gentle breeze carries the scent of roasted coffee and blooming flowers. ",
        "People from all walks of life navigate the sunlit avenues, ",
        "their stories weaving together in a tapestry of laughter, conversation, and quiet contemplation. ",
        # "As the day unfolds, the city reveals its hidden gems: cozy bookshops tucked ",
        # "between lively markets, verdant parks offering sanctuary from the urban rush, and charming cafés where ",
        # "friends gather to share dreams and memories. Above it all, the sky shifts from brilliant blue to shades ",
        # "of gold and rose, promising the serenity of dusk. In this symphony of life, every moment is a note, every ",
        # "voice a melody, and every experience a verse in the never-ending song of the city.",
    ]
    
    processed_segments_for_context = []
    num_chunks = 0
    time_to_first_chunk = None

    async with httpx.AsyncClient() as client:
        t0 = time.time()
        # all_text_segments = ["".join(all_text_segments)]
        for current_segment_text in all_text_segments:
            previous_text_context = " ".join(processed_segments_for_context) or None

            headers = {"xi-api-key": api_key}
            payload = {
                "text": current_segment_text,
                "model_id": "eleven_multilingual_v2",  # Or your preferred model
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            if previous_text_context:
                payload["previous_text"] = previous_text_context

            res = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                json=payload,
                headers=headers,
                timeout=120,
            )
            res.raise_for_status()
            audio_content = await res.aread()

            # Instead of creating a temporary file for each chunk, append all chunks to 'audio.mp3'
            with open("audio_chunks.mp3", "ab") as audio_file:
                audio_file.write(audio_content)

            num_chunks += 1
            if time_to_first_chunk is None:
                time_to_first_chunk = time.time() - t0

            print(f"Chunk {num_chunks}: received at {time.time() - t0:.2f} seconds")

            processed_segments_for_context.append(current_segment_text)

        total_time = time.time() - t0
        print(f"Time to first audio chunk: {time_to_first_chunk:.2f} seconds")
        print(f"Number of audio chunks: {num_chunks}")
        print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())