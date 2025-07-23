import pytest
import os
from pathlib import Path

from timbal import Agent
from timbal.state.savers import InMemorySaver
from timbal.types import File
from timbal.steps.elevenlabs import tts # , stt

from timbal.state.context import RunContext


# No need to run these explicitly, since they'll always be run internally by all the agents tests.
# @pytest.mark.asyncio
# async def test_stt():
#     audio_file = File.validate("https://cdn.openai.com/API/docs/audio/alloy.wav")

#     res = await stt(audio_file=audio_file)
#     print(res)


# @pytest.mark.asyncio
# async def test_tts():
#     text = "Hello, how are you?"
#     voice_id = "56AoDkrOh6qfVPDXZ7Pt"

#     res = await tts(text=text, voice_id=voice_id)
#     run_context = RunContext.model_validate({
#         "id": "test",
#         "timbal_platform_config": {
#             "host": "dev.timbal.ai",
#             "auth_config": {
#                 "type": "bearer",
#                 "token": os.getenv("TIMBAL_API_TOKEN"),
#             },
#             "app_config": {
#                 "org_id": "1",
#                 "app_id": "18",
#                 "version_id": "85",
#             },
#         },
#     })
#     res_ser = File.serialize(res, run_context) 
#     print(res_ser)


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "alloy.wav",
    "https://cdn.openai.com/API/docs/audio/alloy.wav",
])
def audio(request):
    return request.param


# TODO Test all of this with multiple models.
@pytest.mark.asyncio
async def test_agent_stt(audio):
    audio_file = File.validate(audio)

    agent = Agent(model="gpt-4.1-nano")

    res = await agent.complete(prompt=audio_file)
    print(res)


# TODO Change this.
@pytest.mark.asyncio
async def test_agent_stt_tts():
    # audio_file = File.validate("https://cdn.openai.com/API/docs/audio/alloy.wav")

    agent = Agent(
        model="gpt-4o-mini",
        tools=[{
            "runnable": tts,
            "force_exit": True
        }],
        system_prompt=(
            "You are a helpful assistant and you must always respond in audio format. "
            "Always use '56AoDkrOh6qfVPDXZ7Pt' as the voice_id for the TTS model."
        ),
        state_saver=InMemorySaver(),
    )

    run_context = RunContext.model_validate({
        "id": "test",
        "timbal_platform_config": {
            "host": "dev.timbal.ai",
            "auth_config": {
                "type": "bearer",
                "token": os.getenv("TIMBAL_API_TOKEN"),
            },
            "app_config": {
                "org_id": "1",
                "app_id": "18",
                "version_id": "85",
            },
        },
    })

    res = await agent.complete(prompt="Hello how are you?", context=run_context)
