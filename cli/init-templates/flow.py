

from timbal import Flow
from timbal.state.savers import InMemorySaver


flow = (
    Flow()
    .add_llm(model="gpt-4o-mini", memory_id="llm")
    .set_data_map("llm.prompt", "prompt")
    .set_output("response", "llm.return")
    .compile(state_saver=InMemorySaver())
)
