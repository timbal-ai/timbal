import pytest
import os

from timbal import Agent
from timbal.types import File
# from timbal.steps.elevenlabs import stt, tts

# from timbal.state.context import RunContext


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


# TODO Test this with multiple models.
@pytest.mark.asyncio
async def test_agent_stt():
    audio_file = File.validate("https://cdn.openai.com/API/docs/audio/alloy.wav")

    agent = Agent(model="claude-3-5-sonnet-20241022", max_tokens=2048)

    res = await agent.complete(prompt=audio_file)
