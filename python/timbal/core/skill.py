import importlib.util
import sys
from pathlib import Path

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import yaml
from pydantic import model_validator

from .runnable import Runnable
from .tool import Tool
from .tool_set import ToolSet


"""


agent = Agent(
    ...
    skills=Path,
    tools=[WebSearch(), Skill(...),],
    tools=[WebSearch(), ToolSet(...)->[],],
)

[WebSearch, ReadSkill]


"""


class Skill(ToolSet):
    """Skill is a tool set that can be used to provide context to the agent."""
    path: str | Path
    tools: list[Tool] = []

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
        yaml_data = yaml.safe_load(content[3:end_marker].strip())
        self.name = yaml_data.get("name")
        if not self.name:
            raise ValueError(f"SKILL.md must contain a name: {self.path}")
        self.description = yaml_data.get("description")
        if not self.description:
            raise ValueError(f"SKILL.md must contain a description: {self.path}")

        # ? We could enforce the name of the skill to be the same as the name of the directory

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


    @override
    async def resolve(self) -> list[Tool]:
        """See base class."""
        # tools = [ReadSkill(self.skills_path)] # TODO Modify read skill to add is in context or something

        # # TODO Other stuff
        # if self.is_in_context:
        #     tools.extend([])

        # return tools
        return []
