import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

# `override` was introduced in Python 3.12; use `typing_extensions` for compatibility with older versions
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import structlog
import yaml
from pydantic import Field, PrivateAttr, model_validator

from ..state import get_or_create_run_context
from .runnable import Runnable
from .tool import Tool
from .tool_set import ToolSet

logger = structlog.get_logger("timbal.core.skill")


# Provider tool-name validators (OpenAI/Anthropic) accept ``[a-zA-Z0-9_-]{1,64}``.
# We slugify the skill name with this in mind and cap the prefixed length to 64.
TOOL_NAME_SEPARATOR = "__"
TOOL_NAME_MAX_LEN = 64
_TOOL_NAME_INVALID_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _slugify_tool_name(s: str) -> str:
    """Replace any character not allowed in OpenAI/Anthropic tool names with ``_``.

    Collapses runs of underscores to keep the prefix readable.
    """
    slug = _TOOL_NAME_INVALID_RE.sub("_", s)
    return re.sub(r"_+", "_", slug).strip("_")


class Skill(ToolSet):
    """A bundle of related tools and documentation gated by ``read_skill``.

    Inner tools are exposed to the LLM only after the agent loads the skill via
    the ``read_skill`` tool. By default, exposed tool names are namespaced as
    ``{skill_name}__{tool_name}`` so they cannot collide with top-level tools and
    so the LLM can't confuse a skill-internal tool with a top-level one (a common
    failure mode on weaker models). Set ``namespace_tools=False`` for legacy
    behaviour where inner tools keep their flat names.
    """

    path: Path
    tools: list[Runnable] = []
    references: dict[str, str] = {}
    namespace_tools: bool = True
    """When True (default), expose inner tools to the LLM as
    ``{skill_name}__{tool_name}`` rather than their bare name. Eliminates name
    collisions with top-level tools and makes skill provenance explicit, which
    reduces skill-vs-tool confusion for weaker LLMs. Set to False to keep flat
    names (legacy behaviour)."""

    _agent_path: str = PrivateAttr()
    _name_prefix: str = PrivateAttr(default="")
    """Slugified skill name applied as a prefix to inner tools when
    ``namespace_tools`` is True; empty string otherwise."""

    # Cross-instance map id(tool) -> original (un-namespaced) name. We need a
    # class-level cache because `sys.modules` caches the imported tool module:
    # the second ``Skill(path=...)`` for the same skill sees the already-renamed
    # Tool object and would otherwise compound the prefix into ``cars__cars__x``.
    # Keyed by id() rather than the Tool itself because Pydantic models have a
    # custom __hash__ and we don't want to keep tools alive longer than needed —
    # tools live for the process lifetime anyway via sys.modules, so id collision
    # after GC is not a concern in practice.
    _SHARED_TOOL_BASE_NAMES: ClassVar[dict[int, str]] = {}

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
                    logger.info(f"Loaded tool {attr_name} from {tool_path}")

        # Namespace inner tools so they cannot collide with top-level tools and
        # so the LLM sees their skill provenance in the name. Tool instances are
        # shared across Skill constructions via Python's module cache, so we
        # always (a) recover each tool's original name from a cross-instance
        # tracker before deriving anything, and (b) reset the name on every
        # construction — opting out of namespacing must restore the flat name
        # even if a previous Skill() with the same path applied a prefix.
        # Length validation happens up front because OpenAI and Anthropic both
        # reject tool names longer than 64 chars at the API edge.
        if self.namespace_tools:
            self._name_prefix = _slugify_tool_name(self.name)
            if not self._name_prefix:
                raise ValueError(
                    f"Skill name {self.name!r} has no characters valid for a tool-name "
                    f"prefix (must contain at least one of [a-zA-Z0-9_-]). Either rename "
                    f"the skill or set namespace_tools=False."
                )

        for tool in self.tools:
            tool_key = id(tool)
            base_name = self.__class__._SHARED_TOOL_BASE_NAMES.get(tool_key)
            if base_name is None:
                base_name = tool.name
                self.__class__._SHARED_TOOL_BASE_NAMES[tool_key] = base_name

            if self.namespace_tools:
                target_name = f"{self._name_prefix}{TOOL_NAME_SEPARATOR}{base_name}"
                if len(target_name) > TOOL_NAME_MAX_LEN:
                    raise ValueError(
                        f"Namespaced tool name {target_name!r} exceeds {TOOL_NAME_MAX_LEN} "
                        f"chars (skill prefix {self._name_prefix!r} + tool name {base_name!r}). "
                        f"Shorten the skill name or the inner tool name, or set "
                        f"namespace_tools=False on this skill."
                    )
            else:
                target_name = base_name

            if tool.name != target_name:
                tool.name = target_name
                # nest() is called later by the agent and will reset _path;
                # keep the local _path consistent so any pre-nest log lines
                # or introspection see the right name.
                tool._path = target_name
                # Drop every cached_property derived from `self.name`. In
                # production these caches are typically empty at this point
                # (the agent hasn't built schemas yet), but the shared
                # Tool object via sys.modules means a second Skill() for the
                # same path with a different `namespace_tools` would otherwise
                # serve stale schemas. Invalidate the full chain:
                #   params_model -> params_model_schema -> _formatted_params_schema
                #   -> {anthropic,openai_chat_completions,openai_responses}_schema
                for cached in (
                    "params_model",
                    "params_model_schema",
                    "_formatted_params_schema",
                    "anthropic_schema",
                    "openai_chat_completions_schema",
                    "openai_responses_schema",
                ):
                    tool.__dict__.pop(cached, None)

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
        if self._agent_path:
            session = await get_or_create_run_context().get_session()
            in_context_skills = session.get("__in_context_skills", {})
            if self.name in in_context_skills.get(self._agent_path, []):
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
            if name not in session["__in_context_skills"][_agent_path]:
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
