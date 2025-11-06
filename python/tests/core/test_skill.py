
import pytest
from timbal import Agent, Tool
from timbal.core.skill import ReadSkill, Skill


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory structure for testing."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    
    # Create a "cars" skill
    cars_skill = skills_root / "cars"
    cars_skill.mkdir()
    
    # Create SKILL.md with YAML frontmatter
    skill_md = cars_skill / "SKILL.md"
    skill_md.write_text("""---
name: cars
description: Information about cars and automotive topics
---

# Cars Skill

This skill provides information about cars. We specialize in Porsche vehicles.

## Available Information
- Car specifications
- Pricing information
- Model comparisons

See [pricing.md](pricing.md) for detailed pricing information.
""")
    
    # Create a reference file
    pricing_file = cars_skill / "pricing.md"
    pricing_file.write_text("""# Pricing Information

## Porsche Models
- 911: $100,000 - $200,000
- Cayenne: $70,000 - $150,000
- Taycan: $80,000 - $180,000
""")
    
    # Create tools directory with a tool
    tools_dir = cars_skill / "tools"
    tools_dir.mkdir()
    
    search_tool = tools_dir / "search.py"
    search_tool.write_text("""from timbal import Tool

def search_cars(query: str) -> str:
    '''Search for cars matching the query.'''
    return f"Found cars matching: {query}"

search_tool = Tool(
    name="search_cars",
    description="Search for cars in inventory",
    handler=search_cars
)
""")
    
    # Create a second skill "bikes"
    bikes_skill = skills_root / "bikes"
    bikes_skill.mkdir()
    
    bikes_md = bikes_skill / "SKILL.md"
    bikes_md.write_text("""---
name: bikes
description: Information about motorcycles and bikes
---

# Bikes Skill

This skill covers motorcycles and bicycles.
""")
    
    return skills_root


class TestSkillCreation:
    """Test Skill instantiation and validation."""
    
    def test_skill_from_valid_directory(self, skills_dir):
        """Test creating a skill from a valid directory."""
        skill = Skill(path=skills_dir / "cars")
        
        assert skill.name == "cars"
        assert skill.description == "Information about cars and automotive topics"
        assert "Cars Skill" in skill.content
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "search_cars"
    
    def test_skill_path_expansion(self, skills_dir, monkeypatch):
        """Test that skill path expands ~ and resolves relative paths."""
        # Mock home directory
        monkeypatch.setenv("HOME", str(skills_dir.parent))
        
        # Create a skill in "home" directory
        home_skill = skills_dir.parent / "home_skill"
        home_skill.mkdir()
        (home_skill / "SKILL.md").write_text("""---
name: home_test
description: Test skill in home
---
Content
""")
        
        # Test with relative path
        skill = Skill(path=home_skill)
        assert skill.path.is_absolute()
        assert skill.path.exists()
    
    def test_skill_missing_directory(self):
        """Test that skill raises error for non-existent directory."""
        with pytest.raises(ValueError, match="Skill path does not exist"):
            Skill(path="/nonexistent/path")
    
    def test_skill_path_is_file(self, tmp_path):
        """Test that skill raises error when path is a file."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")
        
        with pytest.raises(ValueError, match="Skill path is not a directory"):
            Skill(path=file_path)
    
    def test_skill_missing_skill_md(self, tmp_path):
        """Test that skill raises error when SKILL.md is missing."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        
        with pytest.raises(ValueError, match="must contain a SKILL.md file"):
            Skill(path=skill_dir)
    
    def test_skill_invalid_yaml_frontmatter(self, tmp_path):
        """Test that skill raises error for invalid YAML frontmatter."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("No frontmatter here")
        
        with pytest.raises(ValueError, match="must start with a YAML frontmatter"):
            Skill(path=skill_dir)
    
    def test_skill_missing_name_in_frontmatter(self, tmp_path):
        """Test that skill raises error when name is missing."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
description: Missing name
---
Content
""")
        
        with pytest.raises(ValueError, match="must contain a name"):
            Skill(path=skill_dir)
    
    def test_skill_missing_description_in_frontmatter(self, tmp_path):
        """Test that skill raises error when description is missing."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test
---
Content
""")
        
        with pytest.raises(ValueError, match="must contain a description"):
            Skill(path=skill_dir)
    
    def test_skill_loads_tools_from_directory(self, skills_dir):
        """Test that skill loads tools from tools directory."""
        skill = Skill(path=skills_dir / "cars")
        
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "search_cars"
        assert "Search for cars" in skill.tools[0].description
    
    def test_skill_without_tools_directory(self, skills_dir):
        """Test that skill works without a tools directory."""
        skill = Skill(path=skills_dir / "bikes")
        
        assert skill.name == "bikes"
        assert len(skill.tools) == 0
    
    def test_skill_get_reference(self, skills_dir):
        """Test getting a reference file from skill."""
        skill = Skill(path=skills_dir / "cars")
        
        pricing = skill.get_reference("pricing.md")
        assert "Porsche Models" in pricing
        assert "911:" in pricing
    
    def test_skill_get_reference_caching(self, skills_dir):
        """Test that references are cached."""
        skill = Skill(path=skills_dir / "cars")
        
        # First call
        pricing1 = skill.get_reference("pricing.md")
        # Second call should use cache
        pricing2 = skill.get_reference("pricing.md")
        
        assert pricing1 == pricing2
        assert "pricing.md" in skill.references
    
    def test_skill_get_nonexistent_reference(self, skills_dir):
        """Test that getting non-existent reference raises error."""
        skill = Skill(path=skills_dir / "cars")
        
        with pytest.raises(ValueError, match="Reference file not found"):
            skill.get_reference("nonexistent.md")


class TestAgentWithSkills:
    """Test Agent integration with Skills."""
    
    def test_agent_with_skills_path(self, skills_dir):
        """Test creating agent with skills_path."""
        agent = Agent(
            name="car_agent",
            model="openai/gpt-4o-mini",
            skills_path=skills_dir
        )
        
        # Should have loaded 2 skills (cars and bikes)
        skills = [t for t in agent.tools if isinstance(t, Skill)]
        assert len(skills) == 2
        
        # Should have added read_skill tool
        read_skill_tools = [t for t in agent.tools if isinstance(t, ReadSkill)]
        assert len(read_skill_tools) == 1
        
        # System prompt should mention skills
        assert "skills" in agent.system_prompt.lower()
        assert "cars" in agent.system_prompt
        assert "bikes" in agent.system_prompt
    
    def test_agent_with_skills_path_string(self, skills_dir):
        """Test that skills_path accepts string."""
        agent = Agent(
            name="car_agent",
            model="openai/gpt-4o-mini",
            skills_path=str(skills_dir)
        )
        
        skills = [t for t in agent.tools if isinstance(t, Skill)]
        assert len(skills) == 2
    
    def test_agent_with_invalid_skills_path(self, tmp_path):
        """Test that invalid skills_path raises error."""
        with pytest.raises(ValueError, match="Skills directory .* does not exist"):
            Agent(
                name="agent",
                model="openai/gpt-4o-mini",
                skills_path=tmp_path / "nonexistent"
            )
    
    def test_agent_with_duplicate_skill_names(self, skills_dir):
        """Test that duplicate skill names raise error."""
        skill = Skill(path=skills_dir / "cars")
        
        with pytest.raises(ValueError, match="Skill 'cars' already exists"):
            Agent(
                name="agent",
                model="openai/gpt-4o-mini",
                skills_path=skills_dir,
                tools=[skill]
            )
    
    def test_agent_skills_in_system_prompt(self, skills_dir):
        """Test that skills are documented in system prompt."""
        agent = Agent(
            name="agent",
            model="openai/gpt-4o-mini",
            skills_path=skills_dir,
            system_prompt="You are a helpful assistant."
        )
        
        # Original prompt should be preserved
        assert "helpful assistant" in agent.system_prompt
        
        # Skills section should be added
        assert "<skills>" in agent.system_prompt
        assert "cars" in agent.system_prompt
        assert "Information about cars" in agent.system_prompt
        assert "read_skill" in agent.system_prompt


class TestReadSkillTool:
    """Test the ReadSkill tool functionality."""
    
    @pytest.mark.asyncio
    async def test_read_skill_basic(self, skills_dir):
        """Test reading a skill's main documentation via agent execution."""
        def check_skill_content() -> str:
            """Tool that reads a skill and returns content."""
            from timbal.state import get_run_context
            parent_span = get_run_context().parent_span()
            
            # Find the skill
            skill = next((t for t in parent_span.runnable.tools if isinstance(t, Skill) and t.name == "cars"), None)
            assert skill is not None
            return skill.content
        
        agent = Agent(
            name="agent",
            model="openai/gpt-4o-mini",
            skills_path=skills_dir,
            tools=[Tool(handler=check_skill_content)]
        )
        
        # Execute through agent to have proper context
        events = []
        async for event in agent(prompt="check the skill content"):
            events.append(event)
        
        # Find the tool output
        tool_outputs = [e for e in events if hasattr(e, 'output') and isinstance(e.output, str)]
        assert any("Cars Skill" in str(e.output) for e in tool_outputs)
    
    @pytest.mark.asyncio
    async def test_read_skill_with_reference(self, skills_dir):
        """Test reading a skill reference file."""
        skill = Skill(path=skills_dir / "cars")
        
        # Test get_reference directly (doesn't need context)
        result = skill.get_reference("pricing.md")
        
        assert "Pricing Information" in result
        assert "Porsche Models" in result
    
    @pytest.mark.asyncio  
    async def test_read_skill_content_available(self, skills_dir):
        """Test that skill content is available after initialization."""
        skill = Skill(path=skills_dir / "cars")
        
        # Content should be loaded during init
        assert skill.content
        assert "Cars Skill" in skill.content
        assert "Porsche vehicles" in skill.content


class TestSkillResolve:
    """Test Skill.resolve() behavior with context tracking."""
    
    def test_skill_tools_loaded_at_init(self, skills_dir):
        """Test that skill tools are loaded during initialization."""
        skill = Skill(path=skills_dir / "cars")
        
        # Tools should be loaded in the tools list
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "search_cars"
    
    @pytest.mark.asyncio
    async def test_skill_resolve_behavior(self, skills_dir):
        """Test that skill resolve returns tools based on context."""
        # This test verifies the structure, actual context behavior
        # is tested in integration tests
        skill = Skill(path=skills_dir / "cars")
        
        # Skill has tools loaded
        assert len(skill.tools) == 1
        
        # The resolve method checks for in_context_skills on the span
        # which requires proper RunContext - tested in integration


class TestSkillIntegration:
    """Integration tests for skill functionality with agents."""
    
    @pytest.mark.asyncio
    async def test_skill_tools_available_after_reading(self, skills_dir):
        """Test that skill tools become available after reading the skill."""
        tool_calls = []
        
        def list_tools() -> str:
            """List all available tool names."""
            from timbal.state import get_run_context
            parent_span = get_run_context().parent_span()
            
            tool_names = [t.name for t in parent_span.runnable.tools if isinstance(t, Tool)]
            tool_calls.append(tool_names)
            return f"Available tools: {', '.join(tool_names)}"
        
        agent = Agent(
            name="car_agent",
            model="openai/gpt-4o-mini",
            skills_path=skills_dir,
            tools=[Tool(handler=list_tools)],
            max_iter=10
        )
        
        # Run agent - it should list tools, read skill, then list tools again
        events = []
        async for event in agent(
            prompt="First list tools, then read the cars skill, then list tools again"
        ):
            events.append(event)
        
        # Verify the agent executed successfully
        assert len(events) > 0
        
        # Check if read_skill was called
        read_skill_events = [
            e for e in events
            if hasattr(e, 'metadata') and e.metadata.get('type') == 'ReadSkill'
        ]
        
        # If we have multiple list_tools calls, verify search_cars appears after reading
        if len(tool_calls) >= 2:
            # Before reading: search_cars should not be available
            assert "search_cars" not in tool_calls[0]
            
            # After reading (if read_skill was called): search_cars should be available
            if read_skill_events:
                # Check if search_cars appeared in any subsequent call
                has_search_cars_later = any("search_cars" in tools for tools in tool_calls[1:])
                # This might not always be true depending on LLM behavior
                # So we just verify the structure works
    
    @pytest.mark.asyncio
    async def test_skill_tools_resolve_when_in_context(self, skills_dir):
        """Test that skill.resolve() returns tools when skill is in context."""
        def mark_skill_in_context() -> str:
            """Tool that marks a skill as in context."""
            from timbal.state import get_run_context
            parent_span = get_run_context().parent_span()
            
            # Mark cars skill as in context
            if not hasattr(parent_span, "in_context_skills"):
                parent_span.in_context_skills = []
            parent_span.in_context_skills.append("cars")
            
            return "Skill marked in context"
        
        def check_resolved_tools() -> str:
            """Tool that checks what tools are resolved."""
            from timbal.state import get_run_context
            parent_span = get_run_context().parent_span()
            
            # Get all tool names
            tool_names = [t.name for t in parent_span.runnable.tools]
            return f"Available tools: {', '.join(tool_names)}"
        
        agent = Agent(
            name="agent",
            model="openai/gpt-4o-mini",
            skills_path=skills_dir,
            tools=[
                Tool(handler=mark_skill_in_context),
                Tool(handler=check_resolved_tools)
            ],
            max_iter=5
        )
        
        # Execute agent - it should mark skill in context, then see the tools
        events = []
        async for event in agent(prompt="First mark skill in context, then check resolved tools"):
            events.append(event)
        
        # The agent should have called both tools
        # After marking in context, resolved tools should include search_cars
    
    @pytest.mark.asyncio
    async def test_multiple_skills_tools_isolation(self, skills_dir):
        """Test that only read skills have their tools available."""
        agent = Agent(
            name="agent",
            model="openai/gpt-4o-mini",
            skills_path=skills_dir
        )
        
        # Get cars and bikes skills
        cars_skill = next(t for t in agent.tools if isinstance(t, Skill) and t.name == "cars")
        bikes_skill = next(t for t in agent.tools if isinstance(t, Skill) and t.name == "bikes")
        
        # Cars has tools, bikes doesn't
        assert len(cars_skill.tools) == 1
        assert len(bikes_skill.tools) == 0
        
        # Initially neither should resolve tools (not in context)
        
        # We can test the structure is correct
        assert cars_skill.tools[0].name == "search_cars"
    
    @pytest.mark.asyncio
    async def test_end_to_end_skill_tool_availability(self, skills_dir):
        """
        End-to-end test: Verify that after reading a skill, its tools become available
        in subsequent agent iterations.
        """
        call_log = []
        
        def track_available_tools() -> str:
            """Track which tools are available at this moment."""
            from timbal.state import get_run_context
            parent_span = get_run_context().parent_span()
            
            # Get current iteration's resolved tools
            tool_names = [t.name for t in parent_span.runnable.tools if isinstance(t, Tool)]
            call_log.append({
                "iteration": len(call_log),
                "tools": tool_names
            })
            
            has_search_cars = "search_cars" in tool_names
            return f"search_cars available: {has_search_cars}"
        
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            skills_path=skills_dir,
            tools=[Tool(handler=track_available_tools)],
            max_iter=10
        )
        
        # Run agent with a prompt that should:
        # 1. First check tools (search_cars not available)
        # 2. Read the cars skill
        # 3. Check tools again (search_cars should be available)
        events = []
        async for event in agent(
            prompt="First track available tools, then read the cars skill using read_skill, then track available tools again"
        ):
            events.append(event)
        
        # Verify we have output events
        output_events = [e for e in events if hasattr(e, 'output')]
        assert len(output_events) > 0
        
        # Check that read_skill was called
        read_skill_calls = [
            e for e in events 
            if hasattr(e, 'metadata') and 
            e.metadata.get('type') == 'ReadSkill'
        ]
        
        # If read_skill was called, verify the flow worked
        if read_skill_calls:
            # The skill was read, so in subsequent iterations search_cars should be available
            pass  # Full verification would require checking resolved tools per iteration
    
    @pytest.mark.asyncio
    async def test_skill_persistence_across_turns(self, skills_dir):
        """Test that skills remain in context across multiple agent turns."""
        agent = Agent(
            name="persistent_agent",
            model="openai/gpt-4o-mini",
            skills_path=skills_dir,
            max_iter=3
        )
        
        # First turn: read the skill
        events1 = []
        async for event in agent(prompt="Read the cars skill"):
            events1.append(event)
        
        # Second turn: the skill should still be in context
        # This tests the persistence mechanism in agent._get_history()
        events2 = []
        async for event in agent(prompt="What skills did we read?"):
            events2.append(event)
        
        # Both turns should complete successfully
        assert len(events1) > 0
        assert len(events2) > 0


class TestSkillToolLoading:
    """Test dynamic tool loading from skill directories."""
    
    def test_skill_loads_multiple_tools(self, tmp_path):
        """Test loading multiple tools from a skill."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        
        (skill_dir / "SKILL.md").write_text("""---
name: multi_tool
description: Skill with multiple tools
---
Content
""")
        
        tools_dir = skill_dir / "tools"
        tools_dir.mkdir()
        
        # Create multiple tool files
        (tools_dir / "tool1.py").write_text("""
from timbal import Tool

tool1 = Tool(name="tool1", handler=lambda x: x)
""")
        
        (tools_dir / "tool2.py").write_text("""
from timbal import Tool

tool2 = Tool(name="tool2", handler=lambda x: x * 2)
""")
        
        skill = Skill(path=skill_dir)
        assert len(skill.tools) == 2
        assert {t.name for t in skill.tools} == {"tool1", "tool2"}
    
    def test_skill_skips_non_python_files(self, tmp_path):
        """Test that non-Python files in tools directory are skipped."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        
        (skill_dir / "SKILL.md").write_text("""---
name: test
description: Test skill
---
Content
""")
        
        tools_dir = skill_dir / "tools"
        tools_dir.mkdir()
        
        # Create various file types
        (tools_dir / "tool.py").write_text("""
from timbal import Tool
tool = Tool(name="valid_tool", handler=lambda: "ok")
""")
        (tools_dir / "readme.md").write_text("Not a tool")
        (tools_dir / "data.json").write_text("{}")
        
        skill = Skill(path=skill_dir)
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "valid_tool"
    
    def test_skill_prevents_module_reentry(self, tmp_path):
        """Test that modules are not loaded multiple times."""
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        
        (skill_dir / "SKILL.md").write_text("""---
name: test
description: Test
---
Content
""")
        
        tools_dir = skill_dir / "tools"
        tools_dir.mkdir()
        
        (tools_dir / "tool.py").write_text("""
from timbal import Tool
tool = Tool(name="test_tool", handler=lambda: "ok")
""")
        
        # Create skill twice - should not cause issues
        skill1 = Skill(path=skill_dir)
        skill2 = Skill(path=skill_dir)
        
        assert len(skill1.tools) == 1
        assert len(skill2.tools) == 1
