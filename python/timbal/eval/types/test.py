from pydantic import BaseModel, ConfigDict, model_validator

from .turn import Turn


class Test(BaseModel):
    """Represents a single evaluation test case, identified by a unique name."""
    __test__ = False # This attribute explicitly tells pytest's discover mechanism to skip these classes.
    model_config = ConfigDict(extra="ignore")

    name: str
    """The name for the test.
    This name should be unique within the list of tests.
    """
    description: str | None = None
    """Additional description for the test."""
    turns: list[Turn]
    """The list of turns to run for this test."""

    @model_validator(mode="after")
    def validate_only_last_turn_has_validators(self):
        """Ensure only the last turn has validators. Previous turns establish context only."""
        if len(self.turns) == 0:
            return self
        
        for i, turn in enumerate(self.turns):
            is_last_turn = (i == len(self.turns) - 1)
            
            if turn.input and turn.input.validators:
                if not is_last_turn:
                    raise ValueError(
                        f"Test '{self.name}': Turn {i + 1} (not the last turn) has input validators. "
                        "Only the last turn can have validators. Previous turns establish context only."
                    )
            
            if turn.output and turn.output.validators:
                if not is_last_turn:
                    raise ValueError(
                        f"Test '{self.name}': Turn {i + 1} (not the last turn) has output validators. "
                        "Only the last turn can have validators. Previous turns establish context only."
                    )
            
            if turn.steps and turn.steps.validators:
                if not is_last_turn:
                    raise ValueError(
                        f"Test '{self.name}': Turn {i + 1} (not the last turn) has steps validators. "
                        "Only the last turn can have validators. Previous turns establish context only."
                    )
        
        return self
