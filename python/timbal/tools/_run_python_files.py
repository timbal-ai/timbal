"""Stdlib-only artifact exporter executed inside Modal, never on the host."""

import base64
import json
import os
import stat
import sys


def export_files(root: str, max_files: int, max_bytes: int) -> list[dict]:
    """Read bounded regular files without following links."""
    artifacts = []
    total = 0
    entries = 0

    def visit(directory: int, prefix: str = "") -> None:
        nonlocal total, entries
        with os.scandir(directory) as iterator:
            for entry in iterator:
                entries += 1
                if entries > 1000:
                    raise ValueError("Output directory exceeds 1000 entries.")
                name = prefix + entry.name
                if len(name.encode()) > 240 or "\\" in name:
                    raise ValueError("Output paths must be at most 240 UTF-8 bytes and use forward slashes.")
                # Inspect before opening (devices must never be opened). Recheck the
                # descriptor below to handle files replaced by background processes.
                mode = entry.stat(follow_symlinks=False).st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError(f"Output is not a regular file or directory: {name}")
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if stat.S_ISDIR(mode):
                    flags |= os.O_DIRECTORY
                fd = os.open(entry.name, flags, dir_fd=directory)
                try:
                    info = os.fstat(fd)
                    if stat.S_ISDIR(info.st_mode):
                        visit(fd, name + "/")
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError(f"Output is not a regular file: {name}")
                    if len(artifacts) >= max_files:
                        raise ValueError("Outputs exceed max_files.")
                    remaining = max_bytes - total
                    if info.st_size > remaining:
                        raise ValueError("Outputs exceed max_file_bytes.")
                    with os.fdopen(os.dup(fd), "rb") as stream:
                        data = stream.read(remaining + 1)
                    if len(data) > remaining:
                        raise ValueError("Outputs exceed max_file_bytes.")
                    total += len(data)
                    artifacts.append({"name": name, "data": base64.b64encode(data).decode("ascii")})
                finally:
                    os.close(fd)

    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit(directory)
    finally:
        os.close(directory)
    return artifacts


if __name__ == "__main__":
    try:
        result = {"artifacts": export_files(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])), "error": None}
    except Exception as exc:
        result = {"artifacts": [], "error": {"type": "ArtifactError", "message": str(exc)}}
    sys.stdout.write(json.dumps(result))
