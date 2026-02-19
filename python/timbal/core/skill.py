import importlib.util
import sys
from pathlib import Path
from typing import Any

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import yaml
from pydantic import Field, model_validator

from ..state import get_or_create_run_context
from .runnable import Runnable
from .tool import Tool
from .tool_set import ToolSet


class Skill(ToolSet):
    """Skill is a tool set that can be used to provide context to the agent."""

    path: Path
    tools: list[Runnable] = []
    references: dict[str, str] = {}

    @model_validator(mode="after")
    def validate_skill_structure(self) -> "Skill":
        """Validate that the path is a directory with proper skill structure."""
        self.path = Path(self.path).expanduser().resolve()

        if not self.path.exists():
            raise ValueError(f"Skill path does not exist: {self.path}")

        if not self.path.is_dir():
            raise ValueError(f"Skill path is not a directory: {self.path}")

        skill_file = self.path / "SKILL.md"
        if not skill_file.exists():
            raise ValueError(f"Skill directory must contain a SKILL.md file: {self.path}")

        content = skill_file.read_text(encoding="utf-8")
        content = content.strip()
        if not content.startswith("---"):
            raise ValueError(f"SKILL.md must start with a YAML frontmatter: {self.path}")
        end_marker = content.find("---", 3)
        if end_marker == -1:
            raise ValueError(f"SKILL.md must contain a YAML frontmatter: {self.path}")
        metadata = yaml.safe_load(content[3:end_marker].strip())
        # ? We could enforce the name of the skill to be the same as the name of the directory
        self.name = metadata.get("name")
        if not self.name:
            raise ValueError(f"SKILL.md must contain a name: {self.path}")
        self.description = metadata.get("description")
        if not self.description:
            raise ValueError(f"SKILL.md must contain a description: {self.path}")

        # Since we already read it, store the rest of the content
        self.content = content[end_marker + 3 :].strip()

        # Load tools from the tools directory
        tools_dir = self.path / "tools"
        if not tools_dir.exists() or not tools_dir.is_dir():
            return self
        for tool_path in tools_dir.iterdir():
            if not tool_path.is_file() or tool_path.suffix != ".py":
                continue

            # Dynamically load the module
            module_name = f"skill_{self.name}_{tool_path.stem}"

            # Check if already loaded to prevent re-entry
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                module_spec = importlib.util.spec_from_file_location(module_name, tool_path.as_posix())
                if not module_spec or not module_spec.loader:
                    raise ValueError(f"Failed to load module {tool_path}")
                module = importlib.util.module_from_spec(module_spec)
                sys.modules[module_name] = module
                module_spec.loader.exec_module(module)

            # Look for Runnable instances
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, Runnable):
                    self.tools.append(attr)

        return self

    def get_reference(self, name: str) -> str:
        """Get a specific reference file from the skill."""
        if name in self.references:
            return self.references[name]
        path = self.path / name
        if not path.exists():
            raise ValueError(f"Reference file not found: {name}")
        content = path.read_text(encoding="utf-8")
        self.references[name] = content
        return content

    @override
    async def resolve(self) -> list[Runnable]:
        """See base class."""
        session = await get_or_create_run_context().get_session()
        if self.name in session.get("__in_context_skills", []):
            return self.tools
        return []


class ReadSkill(Tool):
    """Read a skill from the skills directory."""

    def __init__(self, agent_path: str, skills: list["Skill"], **kwargs: Any) -> None:
        _agent_path = agent_path
        _skills = {s.name: s for s in skills}

        async def _read_skill(
            name: str,
            reference: str | None = Field(None, description="Referenced file from a skill."),
        ) -> str:
            """Read documentation for a specific skill. Pass an optional reference to read a specific file from the skill."""
            skill = _skills.get(name)
            if not skill:
                available = ", ".join(_skills.keys()) if _skills else "none"
                raise ValueError(f"Skill '{name}' not found. Available skills: {available}")

            session = await get_or_create_run_context().get_session()
            if "__in_context_skills" not in session:
                session["__in_context_skills"] = {}
            if _agent_path not in session["__in_context_skills"]:
                session["__in_context_skills"][_agent_path] = []
                session["__in_context_skills"][_agent_path].append(name)

            if reference:
                return skill.get_reference(reference)
            else:
                return skill.content

        super().__init__(
            name="read_skill",
            description=(
                "Read documentation for a specific skill. "
                "Provide the skill name to read its documentation file or provide reference to read a specific file from the skill."
            ),
            handler=_read_skill,
            **kwargs,
        )
