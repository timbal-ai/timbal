from typing import Any

from .any import AnyValidator
from .base import BaseValidator
from .seq import SeqValidator


def collect_validators(data: dict[str, Any], path: str) -> list[BaseValidator]:
    validators = []
    validator_keys = [k for k in data.keys() if k.endswith("!")]
    for validator_key in validator_keys:
        validator_name = validator_key.rstrip("!")
        validator_data = data.pop(validator_key)
        print(validator_name, validator_data)
        if validator_name == "seq":
            validator = SeqValidator(validator_data, path=path)
        elif validator_name == "any":
            validator = AnyValidator(path=path, **validator_data)
        else:
            from ..display import print_warning

            print_warning(f"Unknown validator '{validator_name}' will be ignored")
            continue
        validators.append(validator)
    return validators
