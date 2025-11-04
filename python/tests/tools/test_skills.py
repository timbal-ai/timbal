"""
Tests for the ReadSkill tool.

Tests the functionality of the read_skill tool including:
1. Setting the path and properly passing the tool to the agent
2. Using read_skill to read a skill
3. System prompt injection of skills (with <skill> and </skill> tags)
"""

import pathlib
from pathlib import Path

import pytest
from timbal import Agent
from timbal.tools.read_skill import ReadSkill, SKILLS_PROMPT
from timbal.state import set_run_context, RunContext
from timbal.types.message import Message


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory with test skills."""
    skills_path = tmp_path / "skills"
    skills_path.mkdir()
    
    # Create skill1
    skill1_dir = skills_path / "skill1"
    skill1_dir.mkdir()
    skill1_file = skill1_dir / "SKILL.md"
    skill1_file.write_text("""---
name: skill1
description: First test skill
---
# Skill 1 Documentation

This is skill 1 with some documentation.
""", encoding="utf-8")
    
    # Create skill2 with reference files
    skill2_dir = skills_path / "skill2"
    skill2_dir.mkdir()
    skill2_file = skill2_dir / "SKILL.md"
    skill2_file.write_text("""---
name: skill2
description: Second test skill
---
# Skill 2 Documentation

This is skill 2 with some documentation.
Reference: See advanced.md for more details.
""", encoding="utf-8")
    
    # Create reference file
    reference_file = skill2_dir / "advanced.md"
    reference_file.write_text("""# Advanced Skill 2 Reference

This is advanced documentation for skill 2.
""", encoding="utf-8")
    
    return skills_path


@pytest.fixture
def read_skill_tool(skills_dir: Path) -> ReadSkill:
    """Create a ReadSkill tool instance with test skills directory."""
    return ReadSkill(skills_path=str(skills_dir))


# ==============================================================================
# Test 1: Setting the path and passing the tool to the agent
# ==============================================================================

class TestReadSkillToolPath:
    """Test that read_skill tool is properly configured with a path and passed to agents."""
    
    def test_agent_with_skills_path_adds_read_skill_tool(self, skills_dir: Path):
        """Test that creating an agent with skills path automatically adds read_skill tool."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            skills=str(skills_dir)
        )
        
        # Agent should have the read_skill tool
        tool_names = [tool.name for tool in agent.tools]
        assert "read_skill" in tool_names
        
        # Find the read_skill tool
        read_skill_tools = [tool for tool in agent.tools if tool.name == "read_skill"]
        assert len(read_skill_tools) == 1
        assert isinstance(read_skill_tools[0], ReadSkill)
    
    def test_multiple_agents_with_different_skill_paths(self, tmp_path: Path):
        """Test that multiple agents can have different skill paths."""
        # Create two different skill directories
        skills_dir1 = tmp_path / "skills1"
        skills_dir1.mkdir()
        skill1 = skills_dir1 / "skill_a"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("---\nname: skill_a\n---\nSkill A", encoding="utf-8")
        
        skills_dir2 = tmp_path / "skills2"
        skills_dir2.mkdir()
        skill2 = skills_dir2 / "skill_b"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("---\nname: skill_b\n---\nSkill B", encoding="utf-8")
        
        # Create agents with different skill paths
        agent1 = Agent(
            name="agent1",
            model="openai/gpt-4o-mini",
            skills=str(skills_dir1)
        )
        
        agent2 = Agent(
            name="agent2",
            model="openai/gpt-4o-mini",
            skills=str(skills_dir2)
        )
        
        # Both should have read_skill tools
        assert any(tool.name == "read_skill" for tool in agent1.tools)
        assert any(tool.name == "read_skill" for tool in agent2.tools)


# ==============================================================================
# Test 2: Using read_skill tool to read a skill
# ==============================================================================

class TestReadSkillUsage:
    """Test reading skills using the read_skill tool."""
    
    @pytest.mark.asyncio
    async def test_read_skill_basic_skill_file(self, read_skill_tool: ReadSkill):
        """Test reading a basic skill SKILL.md file."""
        result = await read_skill_tool(skill_name="skill1").collect()
        
        assert isinstance(result.output, str)
        assert "Skill 1 Documentation" in result.output
        assert "This is skill 1 with some documentation" in result.output
        # Should not include YAML frontmatter
        assert "---" not in result.output
        assert "name: skill1" not in result.output
    
    @pytest.mark.asyncio
    async def test_read_skill_preserves_markdown_content(self, read_skill_tool: ReadSkill):
        """Test that markdown content is preserved when reading skills."""
        result = await read_skill_tool(skill_name="skill1").collect()
        
        assert result.error is None
        assert "# Skill 1 Documentation" in result.output
    
    @pytest.mark.asyncio
    async def test_read_skill_with_reference_file(self, read_skill_tool: ReadSkill):
        """Test reading a reference file from a skill."""
        result = await read_skill_tool(skill_name="skill2", reference="advanced.md").collect()
        
        assert result.error is None
        assert isinstance(result.output, str)
        assert "Advanced Skill 2 Reference" in result.output
        assert "advanced documentation for skill 2" in result.output
    
    @pytest.mark.asyncio
    async def test_read_skill_nonexistent_skill_error(self, read_skill_tool: ReadSkill):
        """Test error handling when skill name is incorrect."""
        result = await read_skill_tool(skill_name="nonexistent_skill").collect()
        
        # Should handle gracefully
        assert isinstance(result.output, str)
        assert "Skill name incorrect" in result.output or "not found" in result.output.lower()
    
    @pytest.mark.asyncio
    async def test_read_skill_nonexistent_reference_error(self, read_skill_tool: ReadSkill):
        """Test error handling when reference file doesn't exist."""
        result = await read_skill_tool(skill_name="skill2", reference="nonexistent.md").collect()
        
        # Should handle gracefully
        assert isinstance(result.output, str)
        assert "not found" in result.output.lower() or "Reference file" in result.output


# ==============================================================================
# Test 3: System prompt injection of skills
# ==============================================================================

class TestReadSkillSystemPromptInjection:
    """Test that system prompt injection works correctly with skills."""
    
    def test_agent_system_prompt_includes_skills_section(self, skills_dir: Path):
        """Test that agent system prompt includes the <skills> section when skills are configured."""
        original_prompt = "You are a helpful assistant."
        agent = Agent(
            name="skills_agent",
            model="openai/gpt-4o-mini",
            system_prompt=original_prompt,
            skills=str(skills_dir)
        )
        
        # System prompt should have been augmented with skills_prompt
        assert agent.system_prompt is not None
        assert original_prompt in agent.system_prompt
        assert "<skills>" in agent.system_prompt
        assert "</skills>" in agent.system_prompt
    
    
    
    def test_agent_without_system_prompt_only_gets_skills_prompt(self, skills_dir: Path):
        """Test that agent without system prompt gets only the skills prompt."""
        agent = Agent(
            name="no_prompt_agent",
            model="openai/gpt-4o-mini",
            skills=str(skills_dir)
        )
        
        # System prompt should only be the skills_prompt
        assert agent.system_prompt ==  SKILLS_PROMPT
        assert "<skills>" in agent.system_prompt
        assert "</skills>" in agent.system_prompt

    
    def test_read_skill_tool_in_agent_has_access_to_skills(self, skills_dir: Path):
        """Test that the read_skill tool within agent has access to skills directory."""
        agent = Agent(
            name="tool_access_agent",
            model="openai/gpt-4o-mini",
            skills=str(skills_dir)
        )
        
        # Find read_skill tool
        read_skill_tools = [tool for tool in agent.tools if tool.name == "read_skill"]
        assert len(read_skill_tools) == 1
        
        tool = read_skill_tools[0]
        assert isinstance(tool, ReadSkill)
        assert tool._skills_path == pathlib.Path(skills_dir).resolve()


class TestLoadSkillsFunction:
    """Test the load_skills utility function."""
    
    @pytest.mark.asyncio
    async def test_load_skills_parses_multiple_skills(self, skills_dir: Path):
        """Test that load_skills correctly parses multiple skill YAML headers."""
        from timbal.tools.read_skill import load_skills
        from timbal.state import set_run_context, RunContext
        from timbal.core.agent import Agent
        from unittest.mock import Mock, patch
        
        # Create an agent with skills to establish context
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            skills=str(skills_dir)
        )
        
        # Mock the context to return our agent
        mock_context = Mock()
        mock_span = Mock()
        mock_span.runnable.skills = str(skills_dir)
        mock_context.current_span.return_value = mock_span
        
        with patch('timbal.tools.read_skill.get_run_context', return_value=mock_context):
            result = await load_skills()
        
        # Should contain both skills
        assert "skill1" in result or "First test skill" in result
        assert "skill2" in result or "Second test skill" in result
    
    @pytest.mark.asyncio
    async def test_load_skills_empty_directory(self, tmp_path: Path):
        """Test load_skills with empty skills directory."""
        from timbal.tools.read_skill import load_skills
        from unittest.mock import Mock, patch
        
        empty_skills = tmp_path / "empty_skills"
        empty_skills.mkdir()
        
        mock_context = Mock()
        mock_span = Mock()
        mock_span.runnable.skills = str(empty_skills)
        mock_context.current_span.return_value = mock_span
        
        with patch('timbal.tools.read_skill.get_run_context', return_value=mock_context):
            result = await load_skills()
        
        assert "No skills available" in result
    
    @pytest.mark.asyncio
    async def test_load_skills_with_malformed_yaml(self, tmp_path: Path):
        """Test load_skills gracefully handles malformed YAML."""
        from timbal.tools.read_skill import load_skills
        from unittest.mock import Mock, patch
        
        bad_skills = tmp_path / "bad_skills"
        bad_skills.mkdir()
        bad_skill = bad_skills / "bad"
        bad_skill.mkdir()
        
        # Create skill with invalid YAML
        (bad_skill / "SKILL.md").write_text("""---
name: bad_skill
description: This has invalid: yaml: syntax:
---
Content here
""", encoding="utf-8")
        
        mock_context = Mock()
        mock_span = Mock()
        mock_span.runnable.skills = str(bad_skills)
        mock_context.current_span.return_value = mock_span
        
        with patch('timbal.tools.read_skill.get_run_context', return_value=mock_context):
            result = await load_skills()
        
        # Should handle gracefully without crashing
        assert isinstance(result, str)


class TestSkillsIntegration:
    """Test skills integration with agent execution."""
    
    def test_system_prompt_template_resolution_includes_load_skills_pattern(self, skills_dir: Path):
        """Test that system prompt includes the load_skills template pattern."""
        agent = Agent(
            name="template_agent",
            model="openai/gpt-4o-mini",
            skills=str(skills_dir)
        )
        
        # The system prompt should contain the load_skills pattern
        assert "{timbal::tools::read_skill::load_skills}" in agent.system_prompt
    
    @pytest.mark.asyncio
    async def test_yaml_frontmatter_extraction_with_missing_fields(self, tmp_path: Path):
        """Test reading skills with missing YAML fields."""
        skills_path = tmp_path / "skills"
        skills_path.mkdir()
        
        # Create skill with missing description
        skill_dir = skills_path / "minimal_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: minimal_skill
---
# Minimal Skill

No description in YAML.
""", encoding="utf-8")
        
        tool = ReadSkill(skills_path=str(skills_path))
        result = await tool(skill_name="minimal_skill").collect()
        
        assert result.error is None
        assert "# Minimal Skill" in result.output
    
    @pytest.mark.asyncio
    async def test_skill_without_yaml_frontmatter(self, tmp_path: Path):
        """Test reading a skill without YAML frontmatter."""
        skills_path = tmp_path / "skills"
        skills_path.mkdir()
        
        # Create skill without YAML
        skill_dir = skills_path / "no_yaml_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""# Plain Skill

This skill has no YAML frontmatter.
Just plain markdown content.
""", encoding="utf-8")
        
        tool = ReadSkill(skills_path=str(skills_path))
        result = await tool(skill_name="no_yaml_skill").collect()
        
        assert result.error is None
        assert "# Plain Skill" in result.output

    
    @pytest.mark.asyncio
    async def test_multiple_reference_files_in_skill(self, tmp_path: Path):
        """Test skill with multiple reference files."""
        skills_path = tmp_path / "skills"
        skills_path.mkdir()
        
        skill_dir = skills_path / "multi_ref_skill"
        skill_dir.mkdir()
        
        (skill_dir / "SKILL.md").write_text("""---
name: multi_ref_skill
description: Skill with multiple references
---
# Multi Reference Skill
""", encoding="utf-8")
        
        (skill_dir / "advanced.md").write_text("# Advanced Content")
        (skill_dir / "basics.md").write_text("# Basic Content")
        (skill_dir / "examples.md").write_text("# Examples")
        
        tool = ReadSkill(skills_path=str(skills_path))
        
        # Test reading each reference
        adv = await tool(skill_name="multi_ref_skill", reference="advanced.md").collect()
        assert "# Advanced Content" in adv.output
        
        bas = await tool(skill_name="multi_ref_skill", reference="basics.md").collect()
        assert "Basic Content" in bas.output
        
        ex = await tool(skill_name="multi_ref_skill", reference="examples.md").collect()
        assert "Examples" in ex.output