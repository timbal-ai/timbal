import re

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ...state.tracing.trace import Trace
from .any import AnyValidator
from .base import BaseValidator

any_pattern = re.compile(r"^(\d+|\.)\.(\d+|\.)$")


class SeqValidator(BaseValidator):
    def __init__(self, steps, **kwargs) -> None:
        super().__init__(type="seq", **kwargs)
        self.steps = []
        if not isinstance(steps, list):
            raise ValueError(f"seq! validator expects a list. Got {type(steps)}")
        for step in steps:
            if isinstance(step, str):
                any_match = any_pattern.match(step)
                if any_match:
                    any_min, any_max = any_match.groups()
                    any_validator = AnyValidator(min=any_min, max=any_max, **kwargs)
                    self.steps.append(any_validator)
                else:
                    # Is this the same as AnyValidator(min=1, max=1, contains=[step]) ???
                    self.steps.append(step)
            elif isinstance(step, dict):
                raise NotImplementedError("seq! validator with dicts")
            else:
                raise ValueError(f"seq! validator expects a list of strings or dicts. Got item of type {type(step)}")

    @override
    async def run(self, trace: Trace, **kwargs) -> bool:
        # TODO Implement sequence validation
        i, j = 0, 0
        spans = trace.get_level(self.path)
        for step in self.steps:
            print(step)

        # for span in trace.get_level(self.path):
        #     print(span.model_dump())
        return True
