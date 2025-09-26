"""
Bash tool for secure shell command execution with pattern validation.

Supports shell-style wildcards (*, ?, []) for command pattern matching.
Commands must match allowed patterns before execution.

Examples:
    Bash("echo *")  # Allow any echo command
    Bash(["ls *", "pwd"])  # Allow ls and pwd commands
    Bash("cd * && ls *")  # Allow a specific command chain
"""

import asyncio
import re
from typing import Any

import structlog

from ..core.tool import Tool

logger = structlog.get_logger("timbal.tools.bash")


class Bash(Tool):

    def __init__(self, allowed_patterns: str | list[str], **kwargs: Any):
        # Validate and normalize patterns
        if isinstance(allowed_patterns, str):
            allowed_patterns = [allowed_patterns]

        if not allowed_patterns:
            raise ValueError("At least one allowed pattern must be provided")

        # Convert shell patterns to regex patterns
        compiled_patterns = []
        for pattern in allowed_patterns:
            if not isinstance(pattern, str):
                raise TypeError(f"Pattern must be a string, got {type(pattern)}")
            regex_pattern = pattern.strip()
            if not regex_pattern:
                raise ValueError("Pattern cannot be empty or whitespace only")

            # Special case: if pattern is just "*", accept everything
            if regex_pattern == "*":
                compiled_patterns.append(re.compile(r"^.*$"))
                continue

            regex_pattern = regex_pattern.split()
            regex_pattern = [
                part.replace("*", r"""(?:(['"]).*?\1|[\w\/\\\-\.\,\*]+)(?:\s+(?:(['"]).*?\2|[\w\/\\\-\.\,\*]+))*""")
                for part in regex_pattern
            ]
            regex_pattern = r"\s+".join(regex_pattern)
            regex_pattern = f"^{regex_pattern}$"
            compiled_patterns.append(re.compile(regex_pattern))

        async def _execute_command(command: str) -> dict[str, Any]:
            command = command.strip()

            # Check if command matches any allowed pattern
            command_allowed = False
            for compiled_pattern in compiled_patterns:
                if compiled_pattern.match(command):
                    command_allowed = True
                    break

            if not command_allowed:
                # Split by multiple operators: &&, ||, |, ;
                chain_parts = re.split(r'\s*(?:\|\||\&\&|\||\;)\s*', command)
                for part in chain_parts:
                    part_allowed = False
                    for compiled_pattern in compiled_patterns:
                        if compiled_pattern.match(part):
                            part_allowed = True
                            break
                    if not part_allowed:
                        raise ValueError(f"Command '{command}' does not match any allowed patterns: {allowed_patterns}")

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            stdout = stdout.decode("utf-8") if stdout else ""
            stderr = stderr.decode("utf-8") if stderr else ""

            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": process.returncode,
            }

        super().__init__(
            name="bash",
            description=f"Execute a bash command. Allowed patterns: {allowed_patterns}",
            handler=_execute_command,
            **kwargs
        )

        self.allowed_patterns = allowed_patterns
        self.compiled_patterns = compiled_patterns
