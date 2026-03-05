from pathlib import Path

TIMBAL_YAML = "timbal.yaml"


def parse_fqn(workspace_path: str | Path):
    """Read timbal.yaml from *workspace_path* and return an ImportSpec for the entry point."""
    import yaml

    from timbal.utils.import_spec import ImportSpec

    workspace_path = Path(workspace_path)
    yaml_path = workspace_path / TIMBAL_YAML
    if not yaml_path.exists():
        raise FileNotFoundError(f"{TIMBAL_YAML} not found in {workspace_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    fqn = (data or {}).get("fqn")
    if not fqn:
        raise ValueError(f"'fqn' field missing in {yaml_path}")

    return ImportSpec.from_fqn(fqn, base_path=workspace_path)
