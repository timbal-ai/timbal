from pathlib import Path


def discover_files(path: Path) -> list[Path]:
    """Discover all eval files given a path.
    If the path is a directory, it will search recursively for files matching the pattern "eval*.yaml".
    If the path is a file, it'll simply check the file is .yaml and return it.
    """
    files = []
    
    # If path doesn't exist, return empty list
    if not path.exists():
        return files
    
    if path.is_dir():
        for file in path.rglob("eval*.yaml"):
            files.append(file)
    else:
        if not path.name.endswith(".yaml"):
            raise ValueError(f"Invalid evals path: {path}")
        files.append(path)

    return files
