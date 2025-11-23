from pathlib import Path

from ...types.file import File


def has_file_extension(s: str) -> bool:
    """Check if a string starts with '@' (file reference)."""
    if not isinstance(s, str):
        return False
    return s.strip().startswith('@')


def resolve_file_path(item: str | File, test_file_dir: Path | None = None) -> File:
    """Resolve a file path, handling relative paths with test_file_dir."""
    if isinstance(item, File):
        if test_file_dir and hasattr(item, 'path') and item.path:
            file_path = Path(item.path)
            if not file_path.is_absolute():
                resolved_path = (test_file_dir / file_path).resolve()
                return File.validate(str(resolved_path))
        return item
    
    file_path_str = item.lstrip('@')
    if test_file_dir:
        file_path = Path(file_path_str)
        if not file_path.is_absolute():
            resolved_path = (test_file_dir / file_path).resolve()
            return File.validate(str(resolved_path))
        return File.validate(file_path_str)
    return File.validate(file_path_str)


def validators_to_dict(validators: list) -> dict:
    """Convert a list of validators to a dictionary format."""
    validators_dict = {}
    for v in validators:
        validator_name = getattr(v, "name", "unknown")
        validator_ref = getattr(v, "ref", None)
        if validator_name in validators_dict:
            if not isinstance(validators_dict[validator_name], list):
                validators_dict[validator_name] = [validators_dict[validator_name]]
            validators_dict[validator_name].append(validator_ref)
        else:
            validators_dict[validator_name] = validator_ref
    return validators_dict

