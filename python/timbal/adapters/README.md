Agent -> Adapters: [slack, discord]

adapter
Agent.run
adapter


problem? we have a stream input. not a full message for the agent



class TwilioCallAdapter
class SlackAdapter
class DiscordAdapter

HAN DE TENIR MEMORIA I ALTRES DADES (STATEFUL)


Agent(
    model="
    tools
    instructions
    adapters=[TwilioCallAdapter(...), SlackAdapter(...), DiscordAdapter(...)]
    # some config about conversation ending -> webhook, or ws event, or X time without interaction
)


TwilioCallAdapter(
    stt={}
    tts={
        platform="elevenlabs,
        voice_id=""
    }
    vad={}
)


class BaseAdapter(ABC):
    pass


class TwilioCallAdapter(BaseAdapter):
    pass
    # aqui fem override


sales_expert = Agent()
meta_agent = Agent(tools=[sales_expert])
