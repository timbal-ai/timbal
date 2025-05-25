from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from .test import Test


class TestSuite(BaseModel):
    """Represents a collection of tests to evaluate."""
    model_config = ConfigDict(extra="ignore")

    tests: list[Test]
    """The list of tests to run."""


    @model_validator(mode="before")
    @classmethod
    def from_list(cls, v: Any) -> Any:
        """Enable initializing a test suite directly with a list of tests."""
        if isinstance(v, list):
            return {"tests": v}
        return v
