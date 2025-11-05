import asyncio
import structlog
import yaml
from pathlib import Path
from pydantic import Field
from ..core.tool import Tool
from ..state import get_run_context

logger = structlog.get_logger("timbal.tools.read_skill")


class ReadSkill(Tool):
    """Read a skill from the skills directory."""

    def __init__(self, skills_path: str | Path, **kwargs):
        
        async def _read_skill(skill_name: str, reference: str | None = Field(None, description="Referenced file from a skill")) -> str:
            """Read documentation for a specific skill.
            Args:
            skill_name: The name of the skill from the YAML frontmatter (e.g., 'timbal')
            reference: The name of the file in the skill directory to read (e.g., 'slack-integration.md')
            """
            # Track read skill
            agent_span = get_run_context().parent_span()
            for tool in agent_span.runnable.tools:
                if hasattr(tool, "name") and tool.name == skill_name:
                    tool.is_in_context = True

            # Reference file
            if reference:
                reference_file = self._skills_path / skill_name / reference
                if not reference_file.exists():
                    return f"Reference file not found: {reference}"
                
                try:
                    return reference_file.read_text(encoding='utf-8')
                except Exception as e:
                    return f"Error reading reference file: {e}"

            # Skill.md file
            skill_file = self._skills_path / skill_name / "SKILL.md"
            if not skill_file.exists():
                return "Skill name incorrect"

            try:
                content = skill_file.read_text(encoding='utf-8')
                if content.startswith('---'):
                    end_marker = content.find('---', 3)
                    if end_marker != -1:
                        # Return content without the YAML frontmatter:
                        doc_content = content[end_marker + 3:].lstrip('\n')
                        return doc_content
                
                return content
            except Exception as e:
                return f"Error reading skill: {e}"

        
        super().__init__(
            name="read_skill",
            description=(
                "Read documentation for a specific skill."
                "Provide the skill name to read its documentation file or provide reference to read a specific file from the skill."
            ),
            handler=_read_skill,
            **kwargs
        )
        self._skills_path = Path(skills_path)


## Skill utils
async def load_skills():
    """Load YAML frontmatter from all SKILL.md files in eve_skills directory."""
    skills_path = Path(get_run_context().current_span().runnable.skills)

    if not skills_path.exists():
        return "No skills available"

    skills_list = []

    for skill_dir in skills_path.iterdir():
        if not skill_dir.is_dir():
            continue

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        content = skill_file.read_text(encoding='utf-8')
        if content.startswith('---'):
            end_marker = content.find('---', 3)
            if end_marker != -1:
                try: 
                    yaml_data = yaml.safe_load(content[3:end_marker].strip())
                    skills_list.append(
                        f"- **{yaml_data.get('name', 'unnamed')}**: {yaml_data.get('description', 'No description')}"
                    )
                except yaml.YAMLError:
                    pass

    return '\n'.join(skills_list) if skills_list else "No skills available"


SKILLS_PROMPT = """
<skills>
Skills provide additional knowledge of a specific topic. The following skills are available:
{timbal::tools::read_skill::load_skills}
In skills documentation, you will encounter references to additional files.
If the file is relevant for the user query, USE the `read_skill` tool to get its content.
</skills>
"""
