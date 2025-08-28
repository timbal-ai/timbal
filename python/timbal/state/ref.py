import re
from collections.abc import Callable
from typing import Any

REF_KEY_PATTERN = re.compile(r"^(?:\.{1,2}|\w+)\.(?:input|output|shared)(?:\.\w+)*$")


# TODO Add default
# TODO Add transformation function
def ref(key: str) -> Callable[[], Any]:
    """"""
    from . import get_run_context
    if not REF_KEY_PATTERN.match(key):
        raise ValueError(f"Invalid ref key format: {key}")
    # TODO Validate the key
    def resolve() -> Any:
        context = get_run_context()
        if not context:
            raise RuntimeError("Cannot resolve reference outside of a Runnable execution context.")
        return context.get_data(key)
    return resolve
